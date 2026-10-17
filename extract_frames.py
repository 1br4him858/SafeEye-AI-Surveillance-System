"""Extract sampled frames from train/val/test videos without overwriting other clips."""
import argparse
from pathlib import Path
import cv2

CLASSES = ('Fight', 'NonFight')
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}


def extract_frames(video, output, step=5, size=224, overwrite=False):
    if step < 1:
        raise ValueError('step must be >= 1')
    if output.exists() and any(output.glob('frame_*.jpg')) and not overwrite:
        print(f'[SKIP] {output} already contains frames')
        return 0
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for old in output.glob('frame_*.jpg'):
            old.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video}')
    saved = 0
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                frame = cv2.resize(frame, (size, size))
                path = output / f'frame_{saved+1:06d}.jpg'
                if not cv2.imwrite(str(path), frame):
                    raise IOError(f'Cannot write {path}')
                saved += 1
            index += 1
    finally:
        cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset')
    parser.add_argument('--output', default='dataset_frames')
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.step < 1 or args.size < 1:
        parser.error('--step and --size must be positive')
    root = Path(args.input)
    if not root.is_dir():
        parser.error(f'Dataset folder not found: {root}')
    total = 0
    for split in ('train', 'val', 'test'):
        for cls in CLASSES:
            folder = root / split / cls
            if not folder.is_dir():
                continue
            for video in sorted(folder.iterdir()):
                if video.is_file() and video.suffix.lower() in EXTENSIONS:
                    # Include extension to avoid collisions between clip.mp4 and clip.avi.
                    destination = Path(args.output) / split / cls / f'{video.stem}_{video.suffix[1:].lower()}'
                    count = extract_frames(video, destination, args.step, args.size, args.overwrite)
                    total += count
                    print(f'[OK] {video}: {count} frames')
    print(f'Done. Newly extracted frames: {total}')


if __name__ == '__main__':
    main()
