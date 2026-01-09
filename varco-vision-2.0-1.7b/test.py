import torch
import os
import time
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


#### SSL verification 비활성화 ####
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

import requests
_old_request = requests.sessions.Session.request
def _new_request(self, method, url, **kwargs):
    kwargs["verify"] = False
    return _old_request(self, method, url, **kwargs)
requests.sessions.Session.request = _new_request
###################################


def main():
    SNAPSHOT_PATH = "/root/.cache/huggingface/hub/models--NCSOFT--VARCO-VISION-2.0-1.7B/snapshots/ed09f37445518b1564d1ef3c6e26fbd7c1b2c818"
    
    # ============ 모델 로딩 시간 측정 ============
    print("=" * 50)
    print("모델 로딩 중...")
    load_start = time.perf_counter()
    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        SNAPSHOT_PATH,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(SNAPSHOT_PATH, local_files_only=True)
    
    load_end = time.perf_counter()
    load_time = load_end - load_start
    print(f"모델 로딩 시간: {load_time:.3f}초")
    print("=" * 50)
    
    # ============ 추론 준비 ============
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "../images/di1.jpg"},
                {"type": "text", "text": "OCR 후 HTML 표로 변환해줘."},
            ],
        },
    ]
    
    # ============ 전처리 시간 측정 ============
    print("전처리 중...")
    preprocess_start = time.perf_counter()
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    preprocess_end = time.perf_counter()
    preprocess_time = preprocess_end - preprocess_start
    print(f"전처리 시간: {preprocess_time:.3f}초")
    print("=" * 50)
    
    # ============ 추론 시간 측정 ============
    print("추론 중...")
    
    # GPU 동기화 (정확한 측정을 위해)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_start = time.perf_counter()
    
    generate_ids = model.generate(**inputs, max_new_tokens=1024)
    
    # GPU 동기화 (정확한 측정을 위해)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start
    
    # ============ 후처리 ============
    generate_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
    ]
    output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
    
    # ============ 결과 출력 ============
    print("=" * 50)
    print("생성된 텍스트:")
    print(output)
    print("=" * 50)
    
    # ============ 시간 요약 ============
    total_time = load_time + preprocess_time + inference_time
    num_tokens = len(generate_ids_trimmed[0])
    tokens_per_sec = num_tokens / inference_time if inference_time > 0 else 0
    
    print("\n📊 실행 시간 요약")
    print("=" * 50)
    print(f"  모델 로딩 시간:    {load_time:>8.3f}초")
    print(f"  전처리 시간:       {preprocess_time:>8.3f}초")
    print(f"  추론 시간:         {inference_time:>8.3f}초")
    print("-" * 50)
    print(f"  총 실행 시간:      {total_time:>8.3f}초")
    print("=" * 50)
    print(f"  생성된 토큰 수:    {num_tokens}개")
    print(f"  토큰 생성 속도:    {tokens_per_sec:.2f} tokens/sec")
    print("=" * 50)
    
    # GPU 메모리 사용량 (CUDA 사용 시)
    if torch.cuda.is_available():
        print("\n🖥️  GPU 메모리 사용량")
        print("=" * 50)
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"  GPU {i}:")
            print(f"    할당됨: {allocated:.2f} GB")
            print(f"    예약됨: {reserved:.2f} GB")
        print("=" * 50)


if __name__ == "__main__":
    main()