"""Safe Eye integrated fire, violence and pose-based suspicious-behaviour demo.

All model assets must be supplied locally. No camera credentials are embedded.
Event posting is opt-in via SAFE_EYE_BACKEND_URL.
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf
import mediapipe as mp
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from model_utils import CLIP_LEN, clip_from_rgb_frames, frame_transform, load_model

EVENT_CODES = {'normal': 'N', 'fire': 'F', 'violence': 'V', 'suspicious_behaviour': 'R'}


def check_assets(paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('Missing model assets:\n  ' + '\n  '.join(missing))


def post_event(code, url, camera_id):
    if not url:
        print(f'Event changed: {code} (no backend configured)')
        return
    payload = {
        'camera_id': camera_id,
        'event_type': next((k for k, v in EVENT_CODES.items() if v == code), 'normal'),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = requests.post(url, json=payload, timeout=3)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f'[WARN] Event delivery failed: {type(exc).__name__}')


def landmarks_from_box(frame, box, pose_detector):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        return None
    points = [v for lm in result.pose_landmarks[0] for v in (lm.x, lm.y)]
    return np.asarray(points, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description='Safe Eye unified monitoring demo')
    p.add_argument('--camera', default=os.getenv('SAFE_EYE_CAMERA', '0'))
    p.add_argument('--camera-id', default=os.getenv('SAFE_EYE_CAMERA_ID', 'Camera_01'))
    p.add_argument('--fire-weights', default='best.pt')
    p.add_argument('--violence-weights', default='violence_mvit_best.pth')
    p.add_argument('--pose-asset', default='pose_landmarker_heavy.task')
    p.add_argument('--suspicious-config', default='config.json')
    p.add_argument('--suspicious-weights', default='model.weights.h5')
    p.add_argument('--person-weights', default='yolov8n.pt')
    p.add_argument('--backend-url', default=os.getenv('SAFE_EYE_BACKEND_URL', ''))
    args = p.parse_args()
    check_assets((args.fire_weights, args.violence_weights, args.pose_asset, args.suspicious_config, args.suspicious_weights, args.person_weights))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading models...')
    fire_model = YOLO(args.fire_weights)
    violence_model = load_model(args.violence_weights, device)
    with open(args.suspicious_config, encoding='utf-8') as f:
        suspicious_model = tf.keras.models.model_from_json(json.dumps(json.load(f)))
    suspicious_model.load_weights(args.suspicious_weights)
    person_model = YOLO(args.person_weights)
    tracker = DeepSort(max_age=60, n_init=3)
    pose_detector = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.pose_asset),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))

    camera = int(args.camera) if args.camera.isdecimal() else args.camera
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        pose_detector.close()
        raise RuntimeError('Cannot open camera; check --camera or SAFE_EYE_CAMERA')

    frame_lock = threading.Lock()
    results_lock = threading.Lock()
    stop = threading.Event()
    shared = {'frame': None, 'frame_id': -1}
    status = {'fire': False, 'violence': False, 'suspicious_behaviour': False}
    errors = []

    def get_new_frame(last_id):
        with frame_lock:
            if shared['frame'] is None or shared['frame_id'] == last_id:
                return None, last_id
            return shared['frame'].copy(), shared['frame_id']

    def publish(name, detected):
        with results_lock:
            status[name] = bool(detected)

    def fire_worker():
        last_id, streak = -1, 0
        while not stop.is_set():
            frame, last_id_new = get_new_frame(last_id)
            if frame is None:
                stop.wait(0.04)
                continue
            last_id = last_id_new
            predictions = fire_model(frame, conf=0.5, verbose=False)
            found = bool(predictions and len(predictions[0].boxes) > 0)
            streak = streak + 1 if found else 0
            publish('fire', streak >= 5)
            stop.wait(0.1)

    def violence_worker():
        last_id, streak = -1, 0
        buffer = deque(maxlen=CLIP_LEN)
        transform = frame_transform()
        while not stop.is_set():
            frame, new_id = get_new_frame(last_id)
            if frame is None:
                stop.wait(0.03)
                continue
            last_id = new_id
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if len(buffer) == CLIP_LEN:
                clip = clip_from_rgb_frames(buffer, transform, device)
                with torch.inference_mode():
                    probs = violence_model(clip).softmax(dim=1)[0]
                detected = probs[0].item() >= 0.85  # Class 0 = Fight
                streak = streak + 1 if detected else 0
                publish('violence', streak >= 2)
            stop.wait(0.03)

    def suspicious_worker():
        last_id, streak, frozen_until = -1, 0, 0.
        histories = {}
        while not stop.is_set():
            frame, new_id = get_new_frame(last_id)
            if frame is None:
                stop.wait(0.05)
                continue
            last_id = new_id
            if time.monotonic() < frozen_until:
                publish('suspicious_behaviour', False)
                stop.wait(0.1)
                continue
            results = person_model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                if int(box.cls.item()) == 0:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    detections.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(box.conf[0]), 'person'))
            tracks = tracker.update_tracks(detections, frame=frame)
            detected = False
            active = set()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                active.add(tid)
                features = landmarks_from_box(frame, track.to_ltrb(), pose_detector)
                if features is None:
                    continue
                history = histories.setdefault(tid, deque(maxlen=30))
                history.append(features)
                if len(history) == 30:
                    x = np.asarray(history, dtype=np.float32)[None, ...]
                    # Assumes saved model was trained on 30 x 66 pose features.
                    prediction = float(suspicious_model.predict(x, verbose=0)[0][0])
                    detected = detected or prediction > 0.995
            for tid in list(histories):
                if tid not in active:
                    del histories[tid]
            streak = streak + 1 if detected else 0
            confirmed = streak >= 15
            if confirmed:
                frozen_until = time.monotonic() + 240
                streak = 0
            publish('suspicious_behaviour', confirmed)
            stop.wait(0.1)

    def guarded(name, fn):
        try:
            fn()
        except Exception as exc:
            errors.append((name, repr(exc)))
            stop.set()

    threads = [threading.Thread(target=guarded, args=(name, fn), daemon=True)
               for name, fn in [('fire', fire_worker), ('violence', violence_worker), ('suspicious', suspicious_worker)]]
    for thread in threads:
        thread.start()
    previous_code = None
    print('Safe Eye running. Press q to quit.')
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 480))
            with frame_lock:
                shared['frame'] = frame.copy()
                shared['frame_id'] += 1
            with results_lock:
                snapshot = status.copy()
            current = ('fire' if snapshot['fire'] else 'violence' if snapshot['violence']
                       else 'suspicious_behaviour' if snapshot['suspicious_behaviour'] else 'normal')
            code = EVENT_CODES[current]
            if code != previous_code:
                post_event(code, args.backend_url, args.camera_id)
                previous_code = code
            cv2.putText(frame, f'Fire: {snapshot["fire"]}', (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255) if snapshot['fire'] else (0, 255, 0), 2)
            cv2.putText(frame, f'Violence: {snapshot["violence"]}', (12, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255) if snapshot['violence'] else (0, 255, 0), 2)
            cv2.putText(frame, f'Suspicious: {snapshot["suspicious_behaviour"]}', (12, 105), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255) if snapshot['suspicious_behaviour'] else (0, 255, 0), 2)
            cv2.imshow('Safe Eye - Unified Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=3)
        cap.release()
        pose_detector.close()
        cv2.destroyAllWindows()
    if errors:
        raise RuntimeError(f'Detection thread failed: {errors}')


if __name__ == '__main__':
    main()
