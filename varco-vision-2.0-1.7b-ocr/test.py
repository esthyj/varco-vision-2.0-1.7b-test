import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re
import os
import time

#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################

# 전체 시간 측정 시작
total_start = time.time()

### GPU 확인 ###
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")
################

# 모델 로드
print("\n[1/4] 모델 로딩 중...")
model_start = time.time()

model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

model_time = time.time() - model_start
print(f"    → 모델 로드 완료: {model_time:.2f}초")

# 이미지 로드
print("\n[2/4] 이미지 처리 중...")
image_start = time.time()

image = Image.open("../images/di1.jpg").convert("RGB")

# 이미지 업스케일
w, h = image.size
target_size = 2304
if max(w, h) < target_size:
    scaling_factor = target_size / max(w, h)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    image = image.resize((new_w, new_h))
    print(f"    → 이미지 업스케일: {w}x{h} → {new_w}x{new_h}")

image_time = time.time() - image_start
print(f"    → 이미지 처리 완료: {image_time:.2f}초")

# OCR 실행
print("\n[3/4] OCR 추론 중...")
infer_start = time.time()

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "<ocr>"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.float16)

generate_ids = model.generate(**inputs, max_new_tokens=8192)
generate_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
]
output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)

infer_time = time.time() - infer_start
print(f"    → OCR 추론 완료: {infer_time:.2f}초")

print("\n=== OCR 원본 출력 ===")
print(output)


# ===== 시각화 함수 =====
def visualize_ocr_result(image, ocr_output, output_path="ocr_result.png"):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    w, h = img_draw.size
    
    pattern = r'<char>(.*?)</char><bbox>([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)</bbox>'
    matches = re.findall(pattern, ocr_output)
    
    print(f"\n=== 인식된 텍스트 ({len(matches)}개) ===")
    
    for i, (text, x1, y1, x2, y2) in enumerate(matches):
        x1_px = int(float(x1) * w)
        y1_px = int(float(y1) * h)
        x2_px = int(float(x2) * w)
        y2_px = int(float(y2) * h)
        
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=2)
        draw.text((x1_px, y1_px - 12), text, fill="blue")
        
        print(f"  [{i+1}] '{text}' → ({x1_px}, {y1_px}, {x2_px}, {y2_px})")
    
    img_draw.save(output_path)
    print(f"\n결과 저장: {output_path}")
    
    return img_draw


# 시각화 실행
print("\n[4/4] 시각화 중...")
vis_start = time.time()

result_image = visualize_ocr_result(image, output, "ocr_result.png")

vis_time = time.time() - vis_start
print(f"    → 시각화 완료: {vis_time:.2f}초")

# 전체 시간 출력
total_time = time.time() - total_start

print("\n" + "="*50)
print("⏱️  실행 시간 요약")
print("="*50)
print(f"  모델 로드:    {model_time:>6.2f}초")
print(f"  이미지 처리:  {image_time:>6.2f}초")
print(f"  OCR 추론:     {infer_time:>6.2f}초")
print(f"  시각화:       {vis_time:>6.2f}초")
print("-"*50)
print(f"  총 실행 시간: {total_time:>6.2f}초 ({total_time/60:.1f}분)")
print("="*50)
