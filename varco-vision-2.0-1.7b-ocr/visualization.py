import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re
import os

#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################

### GPU 확인 ###
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")
################

# 모델 로드
model_name = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

# 이미지 로드
image = Image.open("../images/di1.jpg").convert("RGB")

# 이미지 업스케일
w, h = image.size
target_size = 2304
if max(w, h) < target_size:
    scaling_factor = target_size / max(w, h)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    image = image.resize((new_w, new_h))

# OCR 실행
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

generate_ids = model.generate(**inputs, max_new_tokens=1024)
generate_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
]
output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=False)
print("=== OCR 원본 출력 ===")
print(output)


# ===== 시각화 함수 =====
def visualize_ocr_result(image, ocr_output, output_path="ocr_result.png"):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    w, h = img_draw.size
    
    # 패턴 추출
    pattern = r'<char>(.*?)</char><bbox>([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)</bbox>'
    matches = re.findall(pattern, ocr_output)
    
    print(f"\n=== 인식된 텍스트 ({len(matches)}개) ===")
    
    for i, (text, x1, y1, x2, y2) in enumerate(matches):
        # 좌표 변환
        x1_px = int(float(x1) * w)
        y1_px = int(float(y1) * h)
        x2_px = int(float(x2) * w)
        y2_px = int(float(y2) * h)
        
        # 빨간 박스
        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline="red", width=2)
        
        # 텍스트 라벨
        draw.text((x1_px, y1_px - 12), text, fill="blue")
        
        print(f"  [{i+1}] '{text}' → ({x1_px}, {y1_px}, {x2_px}, {y2_px})")
    
    img_draw.save(output_path)
    print(f"\n결과 저장: {output_path}")
    
    return img_draw


# 시각화 실행
result_image = visualize_ocr_result(image, output, "ocr_result.png")
