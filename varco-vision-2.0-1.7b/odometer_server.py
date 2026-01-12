"""
주행거리(ODO) 감지 FastAPI 서버
- 서버 시작 시 모델을 한 번만 로드
- 이미지 업로드로 주행거리 감지
"""

import torch
import os
import time
import json
import re
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# ============ SSL 검증 비활성화 (필요시) ============
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request

# ============ 설정 ============
SNAPSHOT_PATH = "/root/.cache/huggingface/hub/models--NCSOFT--VARCO-VISION-2.0-1.7B/snapshots/ed09f37445518b1564d1ef3c6e26fbd7c1b2c818"

# ============ 전역 변수 ============
model = None
processor = None


# ============ Pydantic 모델 ============
class OdometerResult(BaseModel):
    odometer_value: Optional[str] = None
    unit: Optional[str] = None
    bounding_box: Optional[list] = None
    confidence: Optional[float] = None
    raw_output: str
    inference_time: float
    tokens_generated: int
    tokens_per_sec: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used: Optional[float] = None


# ============ 유틸리티 함수 ============
def parse_json_from_output(output: str) -> dict | None:
    """모델 출력에서 JSON 파싱"""
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{[^{}]*\})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def draw_bbox_on_image(image: Image.Image, bbox: list, label: str = "") -> Image.Image:
    """바운딩 박스를 이미지에 그리기"""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    x_min, y_min, x_max, y_max = bbox
    img_width, img_height = image.size
    
    # ============ 정규화된 좌표(0~1)를 픽셀 좌표로 변환 ============
    if all(0 <= v <= 1 for v in bbox):
        x_min = int(x_min * img_width)
        y_min = int(y_min * img_height)
        x_max = int(x_max * img_width)
        y_max = int(y_max * img_height)
    else:
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    # 선 두께: 이미지 크기에 비례 (더 잘 보이게)
    line_width = max(4, int(min(img_width, img_height) * 0.006))
    
    # 눈에 띄는 색상 (밝은 녹색)
    box_color = (0, 255, 0)
    
    # 박스 그리기
    draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=line_width)
    
    if label:
        try:
            font_size = max(24, int(min(img_width, img_height) * 0.03))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        text_y = y_min - font_size - 8
        if text_y < 0:  # 위에 공간 없으면 아래에 표시
            text_y = y_max + 4
        
        text_bbox = draw.textbbox((x_min, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x_min, text_y), label, fill=(0, 0, 0), font=font)
    
    return image


def get_prompt() -> str:
    """주행거리 감지용 프롬프트"""
    return """이 자동차 계기판 이미지에서 총 주행거리(ODO/주행적산계)를 찾아주세요.

⚠️ 중요한 구분:
- 총 주행거리 (ODO): 차량이 지금까지 "주행한" 누적 거리 (예: 45,230 km)
- 주행가능거리 (DTE): 남은 연료로 "앞으로 갈 수 있는" 거리 (예: 350 km)
- 트립미터 (TRIP): 구간별 주행거리

→ "주행가능거리"나 "TRIP"이 아닌, "총 주행거리(ODO)"만 찾아주세요.

다음 형식으로 출력해주세요:
{
    "odometer_value": "숫자값",
    "unit": "km 또는 miles",
    "bounding_box": [x_min, y_min, x_max, y_max],
    "confidence": 0.0~1.0
}

힌트:
- ODO, TOTAL, 주행거리 라벨 근처를 확인하세요
- 주행가능거리는 보통 연료 게이지 근처에 표시됩니다
- 총 주행거리는 보통 5~6자리 이상의 큰 숫자입니다"""


# ============ 모델 로딩 ============
def load_model():
    """모델과 프로세서 로드"""
    global model, processor
    
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    
    print("=" * 50)
    print("🚀 모델 로딩 중...")
    start_time = time.perf_counter()
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        SNAPSHOT_PATH,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(SNAPSHOT_PATH, local_files_only=True)
    
    load_time = time.perf_counter() - start_time
    print(f"✅ 모델 로딩 완료! ({load_time:.2f}초)")
    print("=" * 50)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            print(f"  GPU {i} 메모리 사용: {allocated:.2f} GB")


# ============ FastAPI 앱 설정 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""
    # 시작 시: 모델 로드
    load_model()
    yield
    # 종료 시: 정리 (필요시)
    print("서버 종료 중...")


app = FastAPI(
    title="주행거리 감지 API",
    description="자동차 계기판 이미지에서 주행거리(ODO)를 감지하는 API",
    version="1.0.0",
    lifespan=lifespan
)


# ============ API 엔드포인트 ============
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        gpu_memory_used=gpu_memory
    )


@app.post("/detect", response_model=OdometerResult)
async def detect_odometer(
    image: UploadFile = File(..., description="계기판 이미지 파일")
):
    """
    주행거리 감지 (JSON 결과 반환)
    
    - 이미지를 업로드하면 주행거리 정보를 JSON으로 반환
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    # 이미지 로드
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")
    
    # 추론 준비
    prompt = get_prompt()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 전처리
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    # 추론
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_start = time.perf_counter()
    
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=1024)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = time.perf_counter() - inference_start
    
    # 후처리
    generate_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
    
    num_tokens = len(generate_ids_trimmed[0])
    tokens_per_sec = num_tokens / inference_time if inference_time > 0 else 0
    
    # JSON 파싱
    parsed = parse_json_from_output(output)
    
    return OdometerResult(
        odometer_value=parsed.get("odometer_value") if parsed else None,
        unit=parsed.get("unit") if parsed else None,
        bounding_box=parsed.get("bounding_box") if parsed else None,
        confidence=parsed.get("confidence") if parsed else None,
        raw_output=output,
        inference_time=round(inference_time, 3),
        tokens_generated=num_tokens,
        tokens_per_sec=round(tokens_per_sec, 2)
    )


@app.post("/detect/visualize")
async def detect_and_visualize(
    image: UploadFile = File(..., description="계기판 이미지 파일")
):
    """
    주행거리 감지 + 시각화 (바운딩 박스가 그려진 이미지 반환)
    
    - 이미지를 업로드하면 바운딩 박스가 표시된 이미지를 반환
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    # 이미지 로드
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")
    
    # 추론 준비
    prompt = get_prompt()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 전처리 및 추론
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generate_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
    
    # JSON 파싱 및 시각화
    parsed = parse_json_from_output(output)
    
    if parsed and "bounding_box" in parsed:
        bbox = parsed["bounding_box"]
        odometer_value = parsed.get("odometer_value", "")
        unit = parsed.get("unit", "")
        label = f"ODO: {odometer_value} {unit}"
        
        result_image = draw_bbox_on_image(pil_image, bbox, label)
    else:
        result_image = pil_image
    
    # 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    headers = {
        "X-Odometer-Value": "" if not parsed else parsed.get("odometer_value", ""),
        "X-Odometer-Unit": "" if not parsed else parsed.get("unit", ""),
        "X-Raw-Output": (output or "").replace("\n", " ")[:500],
    }

    # 중요: 헤더 값은 무조건 str이어야 함
    headers = {k: "" if v is None else str(v) for k, v in headers.items()}

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers=headers,
    )


@app.post("/detect/full")
async def detect_full(
    image: UploadFile = File(..., description="계기판 이미지 파일"),
    return_image: bool = True
):
    """
    주행거리 감지 (JSON + Base64 이미지 반환)
    
    - JSON 결과와 함께 시각화된 이미지를 Base64로 반환
    """
    import base64
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    # 이미지 로드
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")
    
    # 추론
    prompt = get_prompt()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
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
    
    inference_start = time.perf_counter()
    
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=1024)
    
    inference_time = time.perf_counter() - inference_start
    
    generate_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
    
    num_tokens = len(generate_ids_trimmed[0])
    
    # 결과 구성
    parsed = parse_json_from_output(output)
    
    result = {
        "odometer_value": parsed.get("odometer_value") if parsed else None,
        "unit": parsed.get("unit") if parsed else None,
        "bounding_box": parsed.get("bounding_box") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
        "raw_output": output,
        "inference_time": round(inference_time, 3),
        "tokens_generated": num_tokens,
    }
    
    # 시각화 이미지 추가
    if return_image and parsed and "bounding_box" in parsed:
        bbox = parsed["bounding_box"]
        label = f"ODO: {parsed.get('odometer_value', '')} {parsed.get('unit', '')}"
        result_image = draw_bbox_on_image(pil_image, bbox, label)
        
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        result["visualized_image_base64"] = base64.b64encode(img_byte_arr.getvalue()).decode()
    
    return JSONResponse(content=result)


# ============ 메인 실행 ============
if __name__ == "__main__":
    uvicorn.run(
        "odometer_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 프로덕션에서는 False
        workers=1      # GPU 모델은 단일 워커 권장
    )
    