# Safe Eye — AI Video Monitoring (Selected Source Code)

Safe Eye is a research/prototype computer-vision project combining three independent detection pipelines:

- **Fire detection:** YOLO-based detection with consecutive-frame confirmation.
- **Violence detection:** fine-tuned **MViT v1-B** video classifier distinguishing `Fight` from `NonFight`, operating on 16-frame clips.
- **Suspicious behaviour detection:** YOLO person detection, DeepSORT tracking, MediaPipe pose landmarks and a separately trained TensorFlow/Keras sequence classifier.

`safe_eye_model.py` brings the three pipelines together using worker threads and displays their statuses on a camera feed. Optional event delivery to an HTTP backend is configured using an environment variable. **This repository contains selected source code, not trained weights, datasets, or a validated deployment.**

## Files

| File | Purpose |
| --- | --- |
| `extract_frames.py` | Extract frames from video files into per-video directories |
| `fine_tune.py` | Train and validate the MViT violence classifier |
| `model_utils.py` | Shared architecture, preprocessing, strict checkpoint loading |
| `realtime_camera.py` | Standalone webcam inference for violence classification |
| `safe_eye_model.py` | Unified fire/violence/suspicious-behaviour camera prototype |

## Installation

Use a compatible Python version and PyTorch/torchvision pair for your hardware; install PyTorch according to its official instructions first, then:

```bash
pip install -r requirements.txt
```

Dependencies are listed without tested version pins; environment compatibility and GPU acceleration depend on the host system.

## Training the violence model

Place videos under `dataset/train/Fight`, `dataset/train/NonFight`, `dataset/val/Fight`, and `dataset/val/NonFight` (optional test split with the same classes). Then:

```bash
python extract_frames.py --input dataset --output dataset_frames --step 5
python fine_tune.py --frames dataset_frames --epochs 30
```

This produces `violence_mvit_best.pth`. Class ordering is **0 = Fight, 1 = NonFight**. Validation uses a deterministic center clip per video; training samples a random temporal crop. A separate test split is evaluated if supplied. The original dataset, metrics and checkpoint are **not included**, so no accuracy is claimed here.

## Standalone violence inference

```bash
python realtime_camera.py --weights violence_mvit_best.pth --camera 0
```

Press **q** to exit. Video clips are RGB, 16 frames, 224×224 pixels, normalized with the same mean and standard deviation used during training.

## Unified demo

The unified script additionally requires the original project assets in the working directory (or pass explicit paths):

- `best.pt` — fire detection YOLO weights
- `violence_mvit_best.pth` — compatible MViT weights
- `yolov8n.pt` — person detection YOLO weights
- `pose_landmarker_heavy.task` — MediaPipe Pose asset
- `config.json` and `model.weights.h5` — suspicious behaviour model definition/weights

```bash
python safe_eye_model.py --camera 0
```

To enable optional event POST requests, set `SAFE_EYE_BACKEND_URL` to an endpoint you control. No camera passwords or API credentials should be committed. A private RTSP URL can be supplied at runtime with `--camera` or `SAFE_EYE_CAMERA`.

## Engineering notes and limitations

- The same MViT head definition is shared by training and inference; checkpoint loading uses **strict validation** and fails loudly for incompatible weights. Checkpoints from other architectures or earlier training variants may require migration or retraining.
- Video frame sampling, normalization and label ordering are consistent across the standalone scripts.
- Detection thresholds and confirmation counts in the integrated prototype are experimental settings, **not calibrated safety guarantees**.
- The suspicious-behaviour model is assumed to accept 30 timesteps × 66 two-dimensional pose features; the original model assets are required to verify that assumption.
- No model metrics, real-world reliability, fire/violence safety guarantees or full-system runtime test are claimed. This system is not a substitute for professional fire detection or emergency response systems.
- `safe_eye_model.py` requires compatible versions of several ML frameworks and may require additional environment-specific work.

## Security

Do not commit surveillance footage, private RTSP addresses, credentials, private model assets or personal data. Any credentials exposed in older copies of the source should be rotated before publishing the repository.
