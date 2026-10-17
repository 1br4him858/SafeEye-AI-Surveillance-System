"""Shared MViT architecture, preprocessing and checkpoint validation."""
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms as T

CLASSES = ('Fight', 'NonFight')
CLIP_LEN = 16
MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)


def build_model(pretrained=False):
    weights = models.video.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None
    model = models.video.mvit_v1_b(weights=weights)
    features = model.head[1].in_features
    # Matches the original fine_tune.py architecture and its checkpoint keys.
    model.head[0] = nn.Sequential(model.head[0], nn.Dropout(p=0.3))
    model.head[1] = nn.Linear(features, len(CLASSES))
    return model


def load_model(checkpoint, device):
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'Missing MViT checkpoint: {path}. Train the model or supply --weights.')
    model = build_model(pretrained=False)
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError as exc:
        raise RuntimeError('Upgrade PyTorch to a version supporting weights_only=True.') from exc
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def frame_transform():
    # Frames arrive as uint8 RGB tensors in [C,H,W].
    return T.Compose([T.Resize((224, 224)), T.ConvertImageDtype(torch.float32), T.Normalize(MEAN, STD)])


def clip_from_rgb_frames(frames, transform, device):
    if len(frames) != CLIP_LEN:
        raise ValueError(f'Expected {CLIP_LEN} frames, received {len(frames)}')
    return torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0).to(device)
