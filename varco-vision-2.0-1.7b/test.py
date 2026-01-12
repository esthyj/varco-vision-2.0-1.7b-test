import torch
import os
import time
import json
import re
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import textwrap
from glob import glob


#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################


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


def visualize_bbox(image_path: str, bbox: list, label: str = "", output_path: str = None):
    """바운딩 박스를 이미지에 시각화 (다양한 좌표 형식 지원)"""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size
    x_min, y_min, x_max, y_max = bbox
    
    print(f"  🔍 원본 bbox: {bbox}")
    print(f"  🖼️ 이미지 크기: {img_width}x{img_height}")
    
    # 좌표 범위에 따라 변환 방식 결정
    max_val = max(bbox)
    
    if max_val <= 1:
        # 0~1 정규화 좌표
        x_min = int(x_min * img_width)
        x_max = int(x_max * img_width)
        y_min = int(y_min * img_height)
        y_max = int(y_max * img_height)
    elif max_val <= 1000:
        # 0~1000 정규화 좌표
        x_min = int(x_min * img_width / 1000)
        x_max = int(x_max * img_width / 1000)
        y_min = int(y_min * img_height / 1000)
        y_max = int(y_max * img_height / 1000)
    else:
        # 이미 픽셀 좌표
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    print(f"  📍 변환된 bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
    
    # 바운딩 박스 그리기
    box_color = (255, 0, 0)
    draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=3)
    
    # 라벨 그리기
    if label:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # 라벨이 이미지 밖으로 나가지 않도록 조정
        label_y = max(0, y_min - 25)
        text_bbox = draw.textbbox((x_min, label_y), label, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        draw.text((x_min, label_y), label, fill=(255, 255, 255), font=font)
    
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_bbox{ext}"
    
    image.save(output_path)
    return image, output_path


def get_image_files(image_dir: str) -> list:
    """이미지 폴더에서 모든 이미지 파일 가져오기"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
        image_files.extend(glob(os.path.join(image_dir, ext.upper())))
    
    return sorted(image_files)


def process_single_image(model, processor, image_path: str, prompt: str) -> dict:
    """단일 이미지 처리"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
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
    
    print("LLM 출력:", output)
    return {
        "output": output,
        "inference_time": inference_time,
        "num_tokens": num_tokens,
        "parsed": parse_json_from_output(output)
    }


def main():
    SNAPSHOT_PATH = "/root/.cache/huggingface/hub/models--NCSOFT--VARCO-VISION-2.0-1.7B/snapshots/ed09f37445518b1564d1ef3c6e26fbd7c1b2c818"
    IMAGE_DIR = "../images"
    OUTPUT_DIR = "../images/results"
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 이미지 파일 목록 가져오기
    image_files = get_image_files(IMAGE_DIR)
    print(f"발견된 이미지: {len(image_files)}개")
    for img in image_files:
        print(f"  - {os.path.basename(img)}")
    print("=" * 50)
    
    if not image_files:
        print("❌ 이미지가 없습니다!")
        return
    
    # ============ 모델 로딩 ============
    print("모델 로딩 중...")
    load_start = time.perf_counter()
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        SNAPSHOT_PATH,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(SNAPSHOT_PATH, local_files_only=True)
    
    load_time = time.perf_counter() - load_start
    print(f"모델 로딩 시간: {load_time:.3f}초")
    print("=" * 50)
    
    # ============ 프롬프트 ============
    prompt = textwrap.dedent("""
    이 자동차 계기판에서 총 주행거리(ODO)를 찾아주세요.

    규칙: 화면에 거리 값이 여러 개 있으면, **가장 아래쪽에 표시된 값**이 ODO입니다.

    출력:
    {
        "odometer_value": "숫자값",
        "unit": "km 또는 miles",
        "bounding_box": [x_min, y_min, x_max, y_max]
    }
    """).strip()
    
    # ============ 각 이미지 처리 ============
    results = []
    total_inference_time = 0
    
    for idx, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n[{idx}/{len(image_files)}] 처리 중: {image_name}")
        print("-" * 40)
        
        try:
            result = process_single_image(model, processor, image_path, prompt)
            total_inference_time += result["inference_time"]
            
            # 결과 출력
            print(f"  추론 시간: {result['inference_time']:.3f}초")
            print(f"  토큰 수: {result['num_tokens']}개")
            
            if result["parsed"]:
                odometer = result["parsed"].get("odometer_value", "N/A")
                unit = result["parsed"].get("unit", "")
                confidence = result["parsed"].get("confidence", "N/A")
                print(f"  ✅ 주행거리: {odometer} {unit} (신뢰도: {confidence})")
                
                # 바운딩 박스 시각화
                if "bounding_box" in result["parsed"]:
                    bbox = result["parsed"]["bounding_box"]
                    print(f"  🔍 원본 bbox 값: {bbox}")
                    label = f"ODO: {odometer} {unit}"
                    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}_result.png")
                    visualize_bbox(image_path, bbox, label, output_path)
                    print(f"  💾 결과 이미지 저장: {output_path}")
            else:
                print(f"  ⚠️ JSON 파싱 실패")
                print(f"  원본 출력: {result['output'][:200]}...")
            
            results.append({
                "image": image_name,
                "success": result["parsed"] is not None,
                "odometer": result["parsed"].get("odometer_value") if result["parsed"] else None,
                "inference_time": result["inference_time"],
                **result
            })
            
        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")
            results.append({
                "image": image_name,
                "success": False,
                "error": str(e)
            })
    
    # ============ 최종 요약 ============
    print("\n" + "=" * 50)
    print("📊 전체 결과 요약")
    print("=" * 50)
    
    successful = sum(1 for r in results if r.get("success"))
    print(f"  총 이미지: {len(image_files)}개")
    print(f"  성공: {successful}개")
    print(f"  실패: {len(image_files) - successful}개")
    print(f"  모델 로딩 시간: {load_time:.3f}초")
    print(f"  총 추론 시간: {total_inference_time:.3f}초")
    print(f"  평균 추론 시간: {total_inference_time / len(image_files):.3f}초/이미지")
    
    print("\n📋 개별 결과:")
    print("-" * 50)
    for r in results:
        status = "✅" if r.get("success") else "❌"
        odometer = r.get("odometer", "N/A")
        print(f"  {status} {r['image']}: {odometer}")
    
    # 결과 JSON 저장
    results_file = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 결과 저장됨: {results_file}")


if __name__ == "__main__":
    main()