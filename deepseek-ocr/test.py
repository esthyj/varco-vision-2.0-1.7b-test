import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer
import os
import time
import re

#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################

# 전체 시간 측정
total_start = time.time()

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

# GPU 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# 모델 로드
print("\n[1/3] 모델 로딩 중...")
model_start = time.time()

model_name = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    # attn_implementation="sdpa",
    device_map="auto"
)
model = model.eval()

model_time = time.time() - model_start
print(f"    → 모델 로드 완료: {model_time:.2f}초")

# 이미지 로드
print("\n[2/3] 이미지 처리 중...")
image_start = time.time()

image_path = "../images/di1.jpg"  # 본인 이미지 경로로 변경
image = Image.open(image_path).convert("RGB")
print(f"    → 이미지 크기: {image.size}")

image_time = time.time() - image_start
print(f"    → 이미지 처리 완료: {image_time:.2f}초")

# OCR 실행
print("\n[3/3] OCR 추론 중...")
infer_start = time.time()

# 프롬프트 옵션:
# - "<image>\nFree OCR." : 자유 형식 OCR
# - "<image>\n<|grounding|>Convert the document to markdown." : 마크다운 변환
# - "<image>\n<|grounding|>Convert the table to HTML." : 표를 HTML로

prompt = "<image>\nFree OCR."

# 추론
with torch.inference_mode():
    result = model.chat(
        tokenizer=tokenizer,
        image=image,
        prompt=prompt,
        max_new_tokens=4096
    )

infer_time = time.time() - infer_start
print(f"    → OCR 추론 완료: {infer_time:.2f}초")

# 결과 출력
print("\n" + "="*50)
print("OCR 결과")
print("="*50)
print(result)

# 시간 요약
total_time = time.time() - total_start
print("\n" + "="*50)
print("⏱️  실행 시간 요약")
print("="*50)
print(f"  모델 로드:    {model_time:>6.2f}초")
print(f"  이미지 처리:  {image_time:>6.2f}초")
print(f"  OCR 추론:     {infer_time:>6.2f}초")
print("-"*50)
print(f"  총 실행 시간: {total_time:>6.2f}초 ({total_time/60:.1f}분)")
print("="*50)