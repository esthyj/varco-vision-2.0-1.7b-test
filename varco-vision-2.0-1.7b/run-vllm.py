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
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(
    model="NCSOFT/VARCO-VISION-2.0-1.7B",
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
)

# 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
)

# 이미지와 프롬프트 설정
prompt = "<image>\n자동차 계기판에 적힌 내용을 설명해주세요."

# 추론
outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": "../images/dashboard1.png"},
    },
    sampling_params=sampling_params,
)

print(outputs[0].outputs[0].text)