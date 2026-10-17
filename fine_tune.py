"""Fine-tune an MViT-B binary video classifier using per-video frame folders."""
import argparse
import copy
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
from torchvision import transforms as T
from model_utils import CLASSES, CLIP_LEN, MEAN, STD, build_model


class FramesDataset(Dataset):
    def __init__(self, root, transform, training=False):
        self.samples = []
        self.transform = transform
        self.training = training
        for label, cls in enumerate(CLASSES):
            folder = Path(root) / cls
            if not folder.is_dir():
                continue
            for video in sorted(folder.iterdir()):
                if video.is_dir():
                    frames = sorted(p for p in video.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
                    if len(frames) >= CLIP_LEN:
                        self.samples.append((frames, label))
        if not self.samples:
            raise ValueError(f'No clips of >= {CLIP_LEN} frames in {root}. Run extract_frames.py first.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frames, label = self.samples[index]
        max_start = len(frames) - CLIP_LEN
        start = random.randint(0, max_start) if self.training else max_start // 2
        # Use consistent augmentation across the entire video clip.
        imgs = [read_image(str(p)).float().div(255) for p in frames[start:start + CLIP_LEN]]
        clip = torch.stack(imgs, dim=1)  # [C,T,H,W]
        clip = self.transform(clip)
        return clip, label


def make_loader(root, training, batch_size):
    transform = T.Compose([T.Resize((224, 224)), T.Normalize(MEAN, STD)])
    ds = FramesDataset(root, transform, training)
    sampler = None
    if training:
        labels = torch.tensor([label for _, label in ds.samples])
        counts = torch.bincount(labels, minlength=len(CLASSES)).float()
        weights = 1 / counts.clamp_min(1)
        sampler = WeightedRandomSampler(weights[labels].double(), num_samples=len(labels), replacement=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=training and sampler is None, sampler=sampler, num_workers=0)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    with torch.inference_mode():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips)
            loss_sum += criterion(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', default='dataset_frames')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--output', default='violence_mvit_best.pth')
    p.add_argument('--resume', default=None, help='Optional compatible checkpoint')
    p.add_argument('--no-pretrained', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1:
        p.error('epochs, batch-size and patience must be positive')
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = make_loader(Path(args.frames) / 'train', True, args.batch_size)
    val = make_loader(Path(args.frames) / 'val', False, args.batch_size)
    model = build_model(pretrained=not args.no_pretrained)
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc, stale = -1., 0
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0., 0, 0
        for clips, labels in train:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        val_loss, val_acc = evaluate(model, val, criterion, device)
        print(f'Epoch {epoch+1}: train loss={running_loss/total:.4f} train acc={correct/total:.4f} val loss={val_loss:.4f} val acc={val_acc:.4f}')
        if val_acc > best_acc:
            best_acc, stale = val_acc, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, args.output)
            print(f'Saved best checkpoint: {args.output}')
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            print('Early stopping')
            break
    test_dir = Path(args.frames) / 'test'
    if test_dir.is_dir():
        try:
            test = make_loader(test_dir, False, args.batch_size)
            best = torch.load(args.output, map_location='cpu', weights_only=True)
            model.load_state_dict(best, strict=True)
            test_loss, test_acc = evaluate(model, test, criterion, device)
            print(f'Test loss={test_loss:.4f} test acc={test_acc:.4f}')
        except ValueError as e:
            print(f'Skipping test: {e}')
    print(f'Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
