from rwkvstic.agnostic.backends import TORCH, TORCH_QUANT
import torch

quantized = {
    "mode": TORCH_QUANT,
    "runtimedtype": torch.bfloat16,
    "useGPU": torch.cuda.is_available(),
    "chunksize": 32,  # larger = more accurate, but more memory
    "target": 100  # your gpu max size, excess vram offloaded to cpu
}

# UNCOMMENT TO SELECT OPTIONS
# Not full list of options, see https://pypi.org/project/rwkvstic/ and https://huggingface.co/BlinkDL/ for more models/modes

# RWKV 1B5 instruct test 1 model
# Approximate
# [Vram usage: 6.0GB]
# [File size: 3.0GB]


config = {
    "path":"https://openmmlab-open.oss-cn-shanghai-internal.aliyuncs.com/model-center/checkpoints/139430/RWKV-4-Pile-7B-Instruct-test2-20230209.pth",
    "mode":TORCH,
    "runtimedtype":torch.bfloat16,
    "useGPU":torch.cuda.is_available(),
    "dtype":torch.bfloat16,
    # "useLogFix":False # When enabled, use BlinkDLs version of the att.
}

title = "RWKV-4 (7B Instruct v2)"

# RWKV 1B5 instruct model quantized
# Approximate
# [Vram usage: 1.3GB]
# [File size: 3.0GB]

# config = {
#     "path": "https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-Instruct-test1-20230124.pth",
#     **quantized
# }

# title = "RWKV-4 (1.5b Instruct Quantized)"

# RWKV 7B instruct pre-quantized (settings baked into model)
# Approximate
# [Vram usage: 7.0GB]
# [File size: 8.0GB]

# config = {
#     "path": "https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-7B-Instruct.pqth"
# }

# title = "RWKV-4 (7b Instruct Quantized)"

# RWKV 14B quantized (latest as of feb 9)
# Approximate
# [Vram usage: 15.0GB]
# [File size: 15.0GB]

# config = {
#     "path": "https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-14B-20230204-7324.pqth"
# }

# title = "RWKV-4 (14b 94% trained, not yet instruct tuned, 8-Bit)"
