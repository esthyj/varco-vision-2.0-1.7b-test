import torch

print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"PyTorch CUDA 버전: {torch.version.cuda}")

from transformers import AutoConfig

config = AutoConfig.from_pretrained("NCSOFT/VARCO-VISION-2.0-1.7B-OCR")
print(f"Max position embeddings: {config.text_config.max_position_embeddings}")