"""
주행거리 감지 API 클라이언트 예제
"""

import requests
import base64
from pathlib import Path
from PIL import Image
import io


BASE_URL = "http://localhost:8000"


def check_health():
    """서버 상태 확인"""
    response = requests.get(f"{BASE_URL}/health")
    print("=== 서버 상태 ===")
    print(response.json())
    return response.json()


def detect_odometer(image_path: str):
    """주행거리 감지 (JSON 결과)"""
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/detect", files=files)
    
    result = response.json()
    print("\n=== 감지 결과 ===")
    print(f"주행거리: {result.get('odometer_value')} {result.get('unit')}")
    print(f"바운딩 박스: {result.get('bounding_box')}")
    print(f"신뢰도: {result.get('confidence')}")
    print(f"추론 시간: {result.get('inference_time')}초")
    print(f"토큰 생성 속도: {result.get('tokens_per_sec')} tokens/sec")
    return result


def detect_and_visualize(image_path: str, output_path: str = None):
    """주행거리 감지 + 시각화 이미지 저장"""
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/detect/visualize", files=files)
    
    # 헤더에서 결과 정보 확인
    print("\n=== 감지 결과 (시각화) ===")
    print(f"주행거리: {response.headers.get('X-Odometer-Value')} {response.headers.get('X-Odometer-Unit')}")
    
    # 이미지 저장
    if output_path is None:
        output_path = str(Path(image_path).stem) + "_result.png"
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"시각화 이미지 저장: {output_path}")
    return output_path


def detect_full(image_path: str, save_image: bool = True):
    """주행거리 감지 (JSON + Base64 이미지)"""
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(
            f"{BASE_URL}/detect/full",
            files=files,
            params={"return_image": True}
        )
    
    result = response.json()
    print("\n=== 감지 결과 (전체) ===")
    print(f"주행거리: {result.get('odometer_value')} {result.get('unit')}")
    print(f"바운딩 박스: {result.get('bounding_box')}")
    print(f"추론 시간: {result.get('inference_time')}초")
    
    # Base64 이미지 디코딩 및 저장
    if save_image and result.get("visualized_image_base64"):
        img_data = base64.b64decode(result["visualized_image_base64"])
        img = Image.open(io.BytesIO(img_data))
        
        output_path = str(Path(image_path).stem) + "_full_result.png"
        img.save(output_path)
        print(f"시각화 이미지 저장: {output_path}")
    
    return result


def batch_detect(image_paths: list):
    """여러 이미지 일괄 처리"""
    results = []
    for path in image_paths:
        print(f"\n처리 중: {path}")
        result = detect_odometer(path)
        results.append({"path": path, "result": result})
    return results


# ============ 사용 예시 ============
if __name__ == "__main__":
    # 1. 서버 상태 확인
    health = check_health()
    
    if not health.get("model_loaded"):
        print("⚠️ 모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도하세요.")
        exit(1)
    
    # 2. 테스트 이미지 경로 (실제 경로로 변경하세요)
    test_image = "../images/dashboard1.jpg"
    
    # 3. JSON 결과만 받기
    result = detect_odometer(test_image)
    
    # 4. 시각화 이미지 받기
    # visualized_path = detect_and_visualize(test_image)
    
    # 5. JSON + Base64 이미지 받기
    # full_result = detect_full(test_image)
    
    # 6. 여러 이미지 일괄 처리
    # images = ["dashboard1.jpg", "dashboard2.jpg", "dashboard3.jpg"]
    # batch_results = batch_detect(images)