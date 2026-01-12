# vlm-test

## NCSOFT/VARCO-VISION-2.0-1.7B-OCR

- performs both recognition and spatial localization by detecting bounding boxes around each character, enabling structured, layout-aware OCR outputs.
- Update: 2025-09-12

정답:
  ✅ dahsboard1.png: 70045
  ✅ dashboard2.png: 181
  ✅ dashboard3.png: 27
  ✅ dashboard4.png: 5061
  ✅ dashboard5.png: 78593

### prompt1
    prompt = textwrap.dedent("""
    이 자동차 계기판 이미지에서 총 주행거리(ODO/주행적산계)를 찾아주세요.

    중요한 구분:
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
    """).strip()

=> 정확도: 
📋 개별 결과:
  ✅ dahsboard1.png: 70045
  ❌ dashboard2.png: 191 (위치인식오류 - 주행가능거리를 읽음)
  ❌ dashboard3.png: 428 (위치인식오류 - 주행가능거리를 읽음)
  ❌ dashboard4.png: 166.9 (위치인식오류 - 트립미터를 읽음)
  ❌ dashboard5.png: 78993 (OCR 성능문제)

### prompt2
    prompt = textwrap.dedent("""
    이 자동차 계기판 이미지에서 총 주행거리(ODO)를 찾아주세요.

    ## 찾는 방법:
    화면에 여러 거리(km/miles) 값이 표시되어 있다면, 
    **가장 아래쪽에 위치한 거리 값**이 총 주행거리(ODO)입니다.

    ## 단계:
    1. 이미지에서 "km" 또는 "miles" 단위가 붙은 모든 숫자를 찾으세요
    2. 각 숫자의 Y좌표(세로 위치)를 확인하세요
    3. 가장 아래에 있는 값을 선택하세요

    ## 출력:
    {
        "all_distances": [
            {"value": "387", "y_position": "상단"},
            {"value": "45230", "y_position": "하단"}
        ],
        "odometer_value": "가장 아래에 있는 값",
        "unit": "km 또는 miles",
        "bounding_box": [x_min, y_min, x_max, y_max],
        "confidence": 0.0~1.0
    }
    """).strip()

  ❌ dahsboard1.png: 45230 (할루시네이션 - LLM에 45230이 있음)
  ✅ dashboard2.png: 181 
  ❌ dashboard3.png: 45230 (할루시네이션 - LLM에 45230이 있음)
  ❌ dashboard4.png: 5001 (OCR 성능문제)
  ❌ dashboard5.png: None (미출력)


  ### prompt 3
    prompt = textwrap.dedent("""
    이 자동차 계기판에서 총 주행거리(ODO)를 찾아주세요.

    규칙: 화면에 거리 값이 여러 개 있으면, **가장 아래쪽에 표시된 값**이 ODO입니다.

    출력:
    {
        "odometer_value": "숫자값",
        "unit": "km 또는 miles",
        "bounding_box": [x_min, y_min, x_max, y_max],
        "confidence": 0.0~1.0
    }
    """).strip()

  ✅ dahsboard1.png: 70045 
  ❌ dashboard2.png: 191 (위치인식오류 - 주행가능거리를 읽음)
  ❌ dashboard3.png: 428 (위치인식오류 - 주행가능거리를 읽음)
  ❌ dashboard4.png: 5001 (OCR 성능문제)
  ❌ dashboard5.png: 78993 (OCR 성능문제)


