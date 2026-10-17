"""Live MViT violence inference on a local camera or a privately configured RTSP stream."""
import argparse
from collections import deque
import cv2
import torch
from model_utils import CLASSES, CLIP_LEN, clip_from_rgb_frames, frame_transform, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='violence_mvit_best.pth')
    p.add_argument('--camera', default='0', help='Camera index or private stream URL')
    p.add_argument('--stride', type=int, default=4, help='Predict every N new frames')
    args = p.parse_args()
    if args.stride < 1:
        p.error('--stride must be positive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)
    transform = frame_transform()
    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError('Unable to open camera; check --camera')
    buffer = deque(maxlen=CLIP_LEN)
    prediction = 'Collecting frames...'
    confidence = 0.
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(torch.from_numpy(rgb).permute(2, 0, 1))
            count += 1
            if len(buffer) == CLIP_LEN and count % args.stride == 0:
                clip = clip_from_rgb_frames(buffer, transform, device)
                with torch.inference_mode():
                    probabilities = model(clip).softmax(1)[0]
                idx = probabilities.argmax().item()
                prediction, confidence = CLASSES[idx], probabilities[idx].item()
            cv2.putText(frame, f'{prediction} ({confidence:.2f})', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if prediction == 'Fight' else (0, 255, 0), 2)
            cv2.imshow('Safe Eye - Violence Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
