#!/usr/bin/env python3
"""Standalone Megatron->vLLM refit + NaN reproduction for Nemotron-3 Ultra 550B.

Why this script exists
----------------------
The SWE-teacher run (``examples/configs/ultra/swe_teacher.yaml``) NaNs during
vLLM generation after the Megatron->vLLM weight refit:

    ValueError: Out of range float values are not JSON compliant: nan
    at vllm_worker_async.py create_chat_completion  (the logprobs are NaN)

To investigate this we need to reproduce the NaN in ONE script. The colocated
``tools/refit_verifier.py`` path is memory-bound at 550B (vLLM TP8 ~= 125 GB/GPU
cannot coexist with a Megatron shard on a 184 GB GB200). But the teacher itself
proves vLLM *can* host the 550B when it is NON-COLOCATED (vLLM gets its own
nodes, TP8/EP1/util 0.8, each replica on 2 nodes). So this script reproduces the
teacher's setup exactly:

  * Megatron ``Policy`` on a dedicated ``train`` cluster.
  * ``VllmGeneration`` on a dedicated ``inference`` cluster (its own memory).
  * The exact GRPO NON-COLOCATED refit: init an NCCL collective spanning both
    clusters, then ``refit_policy_generation(..., colocated_inference=False)``
    (which uses ``broadcast_weights_for_collective`` / ``update_weights_from_collective``).
  * Generate on a prompt and check the vLLM logprobs for NaN, cross-checking
    against Megatron logprobs on the same tokens to isolate refit vs inference.

The vLLM config is taken verbatim from the recipe's ``policy.generation`` block
(so ``moe_backend: triton``, ``mamba_ssm_cache_dtype: float32``,
``attention_backend: FLASH_ATTN``, ``num_{first,last}_layers_in_bf16``, precision,
etc. all match the teacher) — only ``colocated`` is forced off and the engine is
run offline (async off) so ``.generate()`` can return logprobs we can inspect
directly. The NaN, if it is a weight/inference-numerics issue, shows up in those
logprobs regardless of the async chat-completion wrapper.

Launch (via ray.sub COMMAND), e.g. 16 train nodes + 2 inference nodes on GB200:

    cd /opt/nemo-rl && uv run --extra mcore python \
        examples/run_standalone_refit.py \
        --config examples/configs/ultra/swe_teacher.yaml \
        --model_name /path/to/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
        --train_nodes 16 --inference_nodes 2 --gpus_per_node 4 \
        --train_tp 8 --train_ep 8 --train_cp 1 --train_pp 1 \
        --max_sequence_length 4096 --max_new_tokens 64
"""

import argparse
import copy

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone non-colocated Megatron->vLLM refit + NaN check"
    )
    p.add_argument(
        "--config",
        required=True,
        help="Recipe YAML (e.g. examples/configs/ultra/swe_teacher.yaml) — source "
        "of policy.megatron_cfg and policy.generation (vLLM) config.",
    )
    p.add_argument("--model_name", required=True, help="HF model path (550B).")
    # Topology
    p.add_argument("--train_nodes", type=int, required=True)
    p.add_argument("--inference_nodes", type=int, required=True)
    p.add_argument("--gpus_per_node", type=int, default=4, help="GB200 = 4.")
    p.add_argument("--segment_size", type=int, default=0, help="0 = None.")
    # Megatron parallelism overrides (default: keep recipe values)
    p.add_argument("--train_tp", type=int, default=0)
    p.add_argument("--train_ep", type=int, default=0)
    p.add_argument("--train_cp", type=int, default=0)
    p.add_argument("--train_pp", type=int, default=0)
    # Generation / sizing
    p.add_argument("--max_sequence_length", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--refit_buffer_size_gb", type=int, default=4)
    p.add_argument(
        "--prompt",
        default="Write a Python function that returns the nth Fibonacci number.",
    )
    return p.parse_args()


def build_configs(args, tokenizer):
    """Load the recipe and produce (policy_config, generation_config).

    policy_config is the full ``policy`` dict (Megatron backend); generation_config
    is ``policy.generation`` forced to non-colocated, offline (async off).
    """
    register_omegaconf_resolvers()  # recipe uses ${mul:...} / interpolations
    cfg = load_config(args.config)
    OmegaConf.set_struct(cfg, False)
    # Drop sections that carry mandatory-missing values (e.g. sif_dir=???) or are
    # irrelevant here, so resolve=True does not raise.
    for k in ("env", "data", "checkpointing", "logger", "grpo", "loss_fn", "sif_dir"):
        if k in cfg:
            del cfg[k]
    cfg = OmegaConf.to_container(cfg, resolve=True)

    policy_config = cfg["policy"]
    policy_config["model_name"] = args.model_name
    policy_config.setdefault("tokenizer", {})["name"] = args.model_name
    policy_config["max_total_sequence_length"] = args.max_sequence_length

    mc = policy_config["megatron_cfg"]
    if args.train_tp:
        mc["tensor_model_parallel_size"] = args.train_tp
    if args.train_ep:
        mc["expert_model_parallel_size"] = args.train_ep
    if args.train_cp:
        mc["context_parallel_size"] = args.train_cp
    if args.train_pp:
        mc["pipeline_model_parallel_size"] = args.train_pp
    mc["train_iters"] = 1

    # Generation: take the recipe's vLLM config verbatim (keeps moe_backend,
    # mamba_ssm_cache_dtype, attention_backend, layers_in_bf16, precision, ...),
    # force non-colocated + offline so .generate() returns inspectable logprobs.
    generation_config = policy_config["generation"]
    generation_config["model_name"] = args.model_name
    generation_config["max_new_tokens"] = args.max_new_tokens
    generation_config["colocated"] = {
        "enabled": False,
        "resources": {
            "gpus_per_node": args.gpus_per_node,
            "num_nodes": args.inference_nodes,
        },
    }
    vc = generation_config["vllm_cfg"]
    vc["async_engine"] = False
    vc["expose_http_server"] = False
    vc["max_model_len"] = args.max_sequence_length
    # http-server-only knobs are meaningless offline; drop if present.
    for k in ("http_server_serving_chat_kwargs", "enable_vllm_metrics_logger"):
        vc.pop(k, None)

    generation_config = configure_generation_config(generation_config, tokenizer)
    return policy_config, generation_config


def main():
    args = parse_args()
    ray.init()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy_config, generation_config = build_configs(args, tokenizer)

    # Two dedicated (non-colocated) clusters — vLLM gets its own memory.
    # NOTE: SLURM/NVLink topology is handled by ray.sub --segment; the baked
    # container's RayVirtualCluster does not accept a segment_size kwarg.
    train_cluster = RayVirtualCluster(
        name="standalone_train_cluster",
        bundle_ct_per_node_list=[args.gpus_per_node] * args.train_nodes,
        use_gpus=True,
        num_gpus_per_node=args.gpus_per_node,
        max_colocated_worker_groups=1,
    )
    inference_cluster = RayVirtualCluster(
        name="standalone_inference_cluster",
        bundle_ct_per_node_list=[args.gpus_per_node] * args.inference_nodes,
        use_gpus=True,
        num_gpus_per_node=args.gpus_per_node,
        max_colocated_worker_groups=1,
    )
    print(
        f"train cluster: {args.train_nodes} nodes; inference cluster: "
        f"{args.inference_nodes} nodes; {args.gpus_per_node} GPUs/node",
        flush=True,
    )

    print("Instantiating Megatron Policy...", flush=True)
    policy = Policy(
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        init_reference_model=False,
        init_optimizer=False,
    )
    print("Instantiating vLLM generation (non-colocated)...", flush=True)
    vllm_generation = VllmGeneration(cluster=inference_cluster, config=generation_config)

    # --- Non-colocated collective init (mirrors grpo.setup) ---
    ip, port = train_cluster.get_master_address_and_port()
    train_world_size = train_cluster.world_size()
    inference_world_size = args.inference_nodes * args.gpus_per_node
    world_size = train_world_size + inference_world_size
    print(
        f"Init collective: ip={ip} port={port} world_size={world_size} "
        f"(train={train_world_size} + inference={inference_world_size})",
        flush=True,
    )
    ray.get(
        policy.init_collective(ip, port, world_size, train_world_size=train_world_size)
        + vllm_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
    )

    # --- Exchange refit metadata + do the refit (non-colocated NCCL path) ---
    state_dict_info = policy.prepare_refit_info()
    vllm_generation.prepare_refit_info(state_dict_info)
    print("\n--- Refitting Megatron -> vLLM (colocated_inference=False) ---", flush=True)
    refit_policy_generation(
        policy,
        vllm_generation,
        colocated_inference=False,
        _refit_buffer_size_gb=args.refit_buffer_size_gb,
    )
    print("Refit complete.", flush=True)

    # --- Generate with vLLM and check for NaN ---
    tok = tokenizer(
        [args.prompt],
        padding=True,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )
    gen_data = BatchedDataDict(
        {
            "input_ids": tok["input_ids"],
            "input_lengths": tok["attention_mask"].sum(dim=1).to(torch.int32),
        }
    )

    print("\n--- vLLM generate ---", flush=True)
    vllm_out = vllm_generation.generate(gen_data, greedy=True)
    vllm_lp = vllm_out["logprobs"]
    vllm_nan = int(torch.isnan(vllm_lp).sum().item())
    print(f"vLLM logprobs shape={tuple(vllm_lp.shape)} NaN_count={vllm_nan}", flush=True)
    print(f"vLLM logprobs sample (last 10): {vllm_lp[0, -10:]}", flush=True)

    # --- Cross-check against Megatron logprobs on the same tokens ---
    print("\n--- Megatron logprobs on vLLM tokens ---", flush=True)
    mg_in = copy.deepcopy(gen_data)
    mg_in["input_ids"] = vllm_out["output_ids"]
    mg_in["input_lengths"] = vllm_out["unpadded_sequence_lengths"]
    policy.prepare_for_lp_inference()
    mg_out = policy.get_logprobs(mg_in)
    mg_lp = mg_out["logprobs"]
    mg_nan = int(torch.isnan(mg_lp).sum().item())
    print(f"Megatron logprobs shape={tuple(mg_lp.shape)} NaN_count={mg_nan}", flush=True)

    # --- Verdict ---
    input_len = int(gen_data["input_lengths"][0].item())
    total_len = vllm_lp.shape[1]
    v = vllm_lp[0, input_len:total_len]
    m = mg_lp[0, input_len:total_len]
    print("\n================ RESULT ================", flush=True)
    if vllm_nan > 0:
        print(
            f"NaN REPRODUCED: vLLM produced {vllm_nan} NaN logprobs after refit "
            f"(Megatron NaN={mg_nan}). The NaN is in the vLLM generation path.",
            flush=True,
        )
    else:
        finite = torch.isfinite(v) & torch.isfinite(m)
        if finite.any():
            diff = torch.abs(v[finite] - m[finite])
            print(
                f"NO vLLM NaN. Mean|vLLM-Megatron| logprob diff = "
                f"{diff.mean().item():.6f}, max = {diff.max().item():.6f}",
                flush=True,
            )
        else:
            print("NO vLLM NaN, but no overlapping finite logprobs to compare.", flush=True)
    print("========================================", flush=True)

    vllm_generation.shutdown()
    print("Script completed.", flush=True)


if __name__ == "__main__":
    main()
