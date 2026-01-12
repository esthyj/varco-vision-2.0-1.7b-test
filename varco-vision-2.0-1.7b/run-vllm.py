import os
import time
import json
import re
import textwrap
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams


#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################


class OdometerDetectorVLLM:
    """VLLM 기반 주행거리 감지 클래스"""
    
    _instance = None
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "/root/.cache/huggingface/hub/models--NCSOFT--VARCO-VISION-2.0-1.7B/snapshots/ed09f37445518b1564d1ef3c6e26fbd7c1b2c818"
        
        self.model_path = model_path
        self.llm = None
        self.sampling_params = None
        self.is_loaded = False
        self.load_time = 0
        
        self.prompt = textwrap.dedent("""
            이 자동차 계기판 이미지에서 총 주행거리(ODO/주행적산계)를 찾아주세요.

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
            - 총 주행거리는 보통 5~6자리 이상의 큰 숫자입니다
        """).strip()
    
    @classmethod
    def get_instance(cls, model_path: str = None):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance
    
    def load_model(
        self,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ):
        """
        VLLM 모델 로드
        
        Args:
            tensor_parallel_size: GPU 병렬화 수
            gpu_memory_utilization: GPU 메모리 사용률
            max_model_len: 최대 컨텍스트 길이
        """
        if self.is_loaded:
            print("✅ 모델이 이미 로드되어 있습니다.")
            return
        
        print("=" * 50)
        print("🚀 VLLM 모델 로딩 중...")
        load_start = time.perf_counter()
        
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.80,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="auto",
        )
        
        self.sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.0,  # deterministic
            top_p=1.0,
        )
        
        load_end = time.perf_counter()
        self.load_time = load_end - load_start
        self.is_loaded = True
        
        print(f"모델 로딩 시간: {self.load_time:.3f}초")
        print("=" * 50)
    
    def detect(self, image_path: str, visualize: bool = True, output_path: str = None) -> dict:
        """단일 이미지 감지"""
        results = self.detect_batch([image_path], visualize=visualize)
        return results[0] if results else {}
    
    def detect_batch(
        self, 
        image_paths: list[str], 
        visualize: bool = True,
        output_dir: str = None,
    ) -> list[dict]:
        """
        배치 이미지 감지 (VLLM의 강점!)
        
        Args:
            image_paths: 이미지 경로 리스트
            visualize: 바운딩 박스 시각화 여부
            output_dir: 시각화 결과 저장 디렉토리
            
        Returns:
            list[dict]: 각 이미지의 감지 결과
        """
        if not self.is_loaded:
            self.load_model()
        
        # 유효한 이미지만 필터링
        valid_images = []
        valid_paths = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    valid_images.append(img)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"⚠️ 이미지 로드 실패: {path} - {e}")
            else:
                print(f"⚠️ 이미지 없음: {path}")
        
        if not valid_images:
            return []
        
        # VLLM 입력 준비
        inputs = []
        for img in valid_images:
            inputs.append({
                "prompt": f"<image>\nUser: {self.prompt}\nAssistant:",
                "multi_modal_data": {"image": img},
            })
        
        # 배치 추론
        print(f"🔄 배치 추론 중... ({len(inputs)}개 이미지)")
        inference_start = time.perf_counter()
        
        outputs = self.llm.generate(inputs, self.sampling_params)
        
        inference_time = time.perf_counter() - inference_start
        print(f"배치 추론 시간: {inference_time:.3f}초 (평균 {inference_time/len(inputs):.3f}초/이미지)")
        
        # 결과 처리
        results = []
        for i, (output, image_path) in enumerate(zip(outputs, valid_paths)):
            generated_text = output.outputs[0].text
            parsed = self._parse_json_from_output(generated_text)
            
            result = {
                "image_path": image_path,
                "raw_output": generated_text,
                "parsed": parsed,
                "inference_time": inference_time / len(inputs),  # 평균 시간
                "num_tokens": len(output.outputs[0].token_ids),
            }
            
            # 시각화
            if visualize and parsed and "bounding_box" in parsed:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    base = os.path.basename(image_path)
                    name, ext = os.path.splitext(base)
                    vis_path = os.path.join(output_dir, f"{name}_result{ext}")
                else:
                    base, ext = os.path.splitext(image_path)
                    vis_path = f"{base}_result{ext}"
                
                label = f"ODO: {parsed.get('odometer_value', '')} {parsed.get('unit', '')}"
                self._visualize_bbox(image_path, parsed["bounding_box"], label, vis_path)
                result["output_image"] = vis_path
            
            results.append(result)
        
        return results
    
    def _parse_json_from_output(self, output: str) -> dict | None:
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
    
    def _visualize_bbox(self, image_path: str, bbox: list, label: str, output_path: str):
        """바운딩 박스 시각화"""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        x_min, y_min, x_max, y_max = map(int, bbox)
        box_color = (255, 0, 0)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=3)
        
        if label:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                except:
                    font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((x_min, y_min - 25), label, font=font)
            draw.rectangle(text_bbox, fill=box_color)
            draw.text((x_min, y_min - 25), label, fill=(255, 255, 255), font=font)
        
        image.save(output_path)
        print(f"✅ 저장: {output_path}")


def main():
    detector = OdometerDetectorVLLM.get_instance()
    
    # 모델 로드 (VLLM 설정)
    detector.load_model(
        tensor_parallel_size=1,      # GPU 수
        gpu_memory_utilization=0.9,  # GPU 메모리 90% 사용
        max_model_len=4096,
    )
    
    # 배치 처리 (VLLM의 강점!)
    image_paths = [
        "../images/dashboard1.jpg",
        "../images/dashboard2.png",
        "../images/dashboard3.png",
        "../images/dashboard4.png",
    ]
    
    # 한 번에 배치 처리
    print("\n" + "=" * 50)
    print("🚀 VLLM 배치 처리 시작")
    print("=" * 50)
    
    results = detector.detect_batch(
        image_paths,
        visualize=True,
        output_dir="../images/results/",
    )
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📊 결과 요약")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['image_path']}")
        if result["parsed"]:
            print(f"    주행거리: {result['parsed'].get('odometer_value')} {result['parsed'].get('unit')}")
            print(f"    바운딩박스: {result['parsed'].get('bounding_box')}")
            print(f"    신뢰도: {result['parsed'].get('confidence')}")
        else:
            print(f"    ⚠️ 파싱 실패: {result['raw_output'][:100]}...")
    
    # 전체 통계
    print("\n" + "=" * 50)
    print("📈 전체 통계")
    print("=" * 50)
    print(f"  모델 로딩 시간: {detector.load_time:.3f}초")
    print(f"  처리된 이미지: {len(results)}개")
    if results:
        total_tokens = sum(r["num_tokens"] for r in results)
        avg_time = sum(r["inference_time"] for r in results) / len(results)
        print(f"  평균 추론 시간: {avg_time:.3f}초/이미지")
        print(f"  총 생성 토큰: {total_tokens}개")


# API 서버 예시 (FastAPI)
def create_api_server():
    """FastAPI 서버 예시"""
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    import tempfile
    import shutil
    
    app = FastAPI(title="Odometer Detection API")
    detector = OdometerDetectorVLLM.get_instance()
    
    @app.on_event("startup")
    async def startup():
        detector.load_model()
    
    @app.post("/detect")
    async def detect_odometer(file: UploadFile = File(...)):
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            result = detector.detect(tmp_path, visualize=False)
            return JSONResponse(content={
                "success": True,
                "odometer_value": result["parsed"].get("odometer_value") if result["parsed"] else None,
                "unit": result["parsed"].get("unit") if result["parsed"] else None,
                "bounding_box": result["parsed"].get("bounding_box") if result["parsed"] else None,
            })
        finally:
            os.unlink(tmp_path)
    
    return app


if __name__ == "__main__":
    main()