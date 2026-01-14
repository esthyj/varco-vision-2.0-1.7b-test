# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a VLM (Vision Language Model) testing project for vehicle odometer detection using NCSOFT's VARCO-VISION models. The project tests OCR capabilities on car dashboard images to extract total mileage (ODO) values with bounding box localization.

## Models Used

- **VARCO-VISION-2.0-1.7B**: General vision-language model for odometer detection
- **VARCO-VISION-2.0-1.7B-OCR**: Specialized OCR model with character-level bounding boxes

Both models use `LlavaOnevisionForConditionalGeneration` from Hugging Face Transformers.

## Project Structure

```
varco-vision-2.0-1.7b/        # General VLM experiments
├── test.py                   # Batch image processing with Transformers
├── run-vllm.py              # VLLM-based batch inference (faster)
├── odometer_server.py       # FastAPI server for odometer detection
├── client.py                # API client example
└── vllm-test.py             # Simple FastAPI template

varco-vision-2.0-1.7b-ocr/   # OCR-specific experiments
├── test.py                  # OCR with character bounding boxes
└── visualization.py         # OCR result visualization
```

## Running the Code

**Test odometer detection (Transformers):**
```bash
cd varco-vision-2.0-1.7b
python test.py
```

**Test with VLLM (faster batch inference):**
```bash
cd varco-vision-2.0-1.7b
python run-vllm.py
```

**Run FastAPI server:**
```bash
cd varco-vision-2.0-1.7b
python odometer_server.py
# API available at http://localhost:8000
# Endpoints: /health, /detect, /detect/visualize, /detect/full
```

**Run OCR test:**
```bash
cd varco-vision-2.0-1.7b-ocr
python test.py
```

## Key Implementation Details

### Model Loading
Models are loaded from local HuggingFace cache at `/root/.cache/huggingface/hub/`. The code disables SSL verification for HuggingFace Hub connections.

### Bounding Box Coordinate Systems
The code handles multiple coordinate formats:
- Normalized 0-1 coordinates
- Normalized 0-1000 coordinates
- Absolute pixel coordinates

### OCR Output Format
The OCR model outputs structured XML-like format:
```
<char>text</char><bbox>x1, y1, x2, y2</bbox>
```

### Prompt Engineering
The odometer detection prompt distinguishes between:
- **ODO (Total Mileage)**: Cumulative distance driven
- **DTE (Distance To Empty)**: Remaining range on current fuel
- **TRIP**: Trip meter distance

## Test Images

Test dashboard images are stored in `images/` directory with expected odometer values documented in README.md.
