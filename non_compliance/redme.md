# RTMDet-Tiny Person Detection

Minimal person-detection script using **RTMDet-Tiny** from MMDetection.

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended) or CPU fallback
- Packages listed in `requirements.txt`

## Setup

```bash
cd rtmdet_roi_person_detection

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Model Files

The following files must be present in this folder (they are included from
the official MMDetection model zoo):

| File | Purpose |
|------|---------|
| `rtmdet_tiny_8xb32-300e_coco.py` | Model configuration |
| `rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth` | Pretrained weights |

## Run

```bash
# GPU (default)
python main.py --video ../test5.mp4

# CPU fallback
python main.py --video ../test5.mp4 --device cpu

# Adjust confidence threshold
python main.py --video ../test5.mp4 --conf 0.5
```

Press **q** to quit the video window.

## Output

- Only **person** detections are shown (COCO class 0).
- Bounding boxes (green) with confidence scores.
- Real-time FPS overlay and frame counter.
- Average FPS printed to console on exit.
