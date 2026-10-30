# Prints library versions and whether torch can see and use a CUDA GPU.
# Usage: uv run check_env.py   (on the server, set CUDA_VISIBLE_DEVICES first)
import torch
import transformers, peft, trl, accelerate, datasets

print({m.__name__: m.__version__ for m in [transformers, peft, trl, accelerate, datasets, torch]})
print("cuda available:", torch.cuda.is_available(), "| torch cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit(1)

props = torch.cuda.get_device_properties(0)
print(f"device: {props.name}, {props.total_memory / 2**30:.1f} GiB, sm_{props.major}{props.minor}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
print("matmul ok, mean:", (x @ x).float().mean().item())
