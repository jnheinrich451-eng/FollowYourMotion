#!/usr/bin/env python
"""Generate one video with Wan2.1-T2V-1.3B through FYM's own DiffSynth pipeline.

    --mode backbone   the clean base model: heads not split, no LoRA
    --mode fym        heads split with the clip's head_types.pt under the same
                      split seed used in tuning, then the spatial and temporal
                      LoRAs attached exactly as the tuning script attached them

Both modes share the loader, sampler, seed and settings, so the only thing that
differs between the two rows is FYM.

The repo ships no FYM inference path (examples/wanvideo/inference.py is a plain
diffusers Wan call with a different scheduler and no LoRA), hence this script.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fym_bench.videoio import write_lossless  # noqa: E402

SPATIAL = ["self_attn.q_spatial", "self_attn.k_spatial", "self_attn.v_spatial"]
TEMPORAL = ["self_attn.q_temporal", "self_attn.k_temporal", "self_attn.v_temporal"]


def attach_fym_loras(dit, spatial_path, temporal_path, rank, alpha):
    """Inject both adapters the way EffiVMT_train_wan_t2v_Head.py does, then load them.

    Mirrors the temporal stage of tuning: spatial adapter injected first and kept
    at the base dtype, temporal adapter upcast to fp32. Every LoRA parameter must
    be covered by the two checkpoints and nothing else may be in them -- a silent
    partial load would generate from a half-tuned model.
    """
    import torch
    from peft import LoraConfig, inject_adapter_in_model
    from diffsynth import load_state_dict

    def config(targets):
        return LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights=True, target_modules=targets)

    inject_adapter_in_model(config(SPATIAL), dit, adapter_name="spatial_lora")
    inject_adapter_in_model(config(TEMPORAL), dit, adapter_name="temporal_lora")
    for name, p in dit.named_parameters():
        if "temporal_lora" in name:
            p.data = p.data.to(torch.float32)

    # Anything Lightning adds beside the tensors is ignored; LoRA keys must match exactly.
    sd = {k: v for path in (spatial_path, temporal_path)
          for k, v in load_state_dict(path).items() if "lora_" in k}
    lora_keys = {n for n, _ in dit.named_parameters() if "lora_" in n}
    missing, unexpected = lora_keys - sd.keys(), sd.keys() - lora_keys
    if missing or unexpected:
        raise SystemExit(f"LoRA checkpoints do not match the model: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected, e.g. {sorted(missing | unexpected)[:3]}")
    dit.load_state_dict(sd, strict=False)
    return len(sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fym", "backbone"], required=True)
    ap.add_argument("--models", required=True, help="Wan2.1-T2V-1.3B folder (DiffSynth layout)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="results.mp4 to write")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=21)
    ap.add_argument("--fps", type=float, default=12.0, help="container fps only; metrics are per frame")
    # WanVideoPipeline_Override.__call__ defaults, written out so done.json records them
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--sigma_shift", type=float, default=5.0)
    # fym only
    ap.add_argument("--head_types")
    ap.add_argument("--spatial_lora")
    ap.add_argument("--temporal_lora")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--copy_bias", action="store_true", help="bias-fix variant; must match how the LoRAs were tuned")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=16)
    args = ap.parse_args()

    # The loader reads these while it builds the DiT, so they are set before loading.
    if args.mode == "backbone":
        os.environ["FYM_NO_SPLIT"] = "1"
    else:
        for flag in ("head_types", "spatial_lora", "temporal_lora"):
            if not getattr(args, flag):
                ap.error(f"--mode fym needs --{flag}")
        os.environ["FYM_HEAD_TYPES"] = args.head_types
        os.environ["FYM_SPLIT_SEED"] = str(args.split_seed)
        if args.copy_bias:
            os.environ["FYM_SPLIT_COPY_BIAS"] = "1"

    import numpy as np
    import torch
    from diffsynth import ModelManager, WanVideoPipeline_Override

    m = Path(args.models)
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([str(m / "diffusion_pytorch_model.safetensors"),
                    str(m / "models_t5_umt5-xxl-enc-bf16.pth"),
                    str(m / "Wan2.1_VAE.pth")])
    pipe = WanVideoPipeline_Override.from_model_manager(mm, torch_dtype=torch.bfloat16, device="cuda")
    dit = pipe.denoising_model()
    split = not hasattr(dit.blocks[0].self_attn, "q")
    if split != (args.mode == "fym"):
        raise SystemExit(f"--mode {args.mode} but the DiT heads are {'split' if split else 'not split'}")

    n_lora = 0
    if args.mode == "fym":
        n_lora = attach_fym_loras(dit, args.spatial_lora, args.temporal_lora, args.lora_rank, args.lora_alpha)
    # Whole-model offload between stages; the finer vram-management wrapper would wrap the LoRA layers too.
    pipe.enable_cpu_offload()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    frames = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt, seed=args.seed,
                  height=args.height, width=args.width, num_frames=args.num_frames,
                  cfg_scale=args.cfg_scale, num_inference_steps=args.num_inference_steps,
                  sigma_shift=args.sigma_shift, tiled=True)
    elapsed = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_lossless(np.stack([np.asarray(f) for f in frames]), out, args.fps)
    stats = {"elapsed_s": round(elapsed, 1),
             "torch_peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20),
             "lora_tensors_loaded": n_lora, "heads_split": split,
             "settings": {k: v for k, v in vars(args).items() if k not in ("out", "models")}}
    out.with_suffix(".gen.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
