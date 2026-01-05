import torch
import os
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

SNAPSHOT_PATH = "/root/.cache/huggingface/hub/models--NCSOFT--VARCO-VISION-2.0-1.7B/snapshots/ed09f37445518b1564d1ef3c6e26fbd7c1b2c818"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    SNAPSHOT_PATH,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(SNAPSHOT_PATH, local_files_only=True)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "../images/dashboard.jpg"},
            {"type": "text", "text": "계기판에 적힌 내용을 설명해주세요."},
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
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
]
output = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
print(output)