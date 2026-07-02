import numpy as np
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

MANUAL_SEED = 42

np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

datasets_path = Path("./datasets")
datasets_path.exists()

import cv2
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable

def timestamp(path: Path) -> float:
    return float(path.stem.split("-")[1])


def sync_frames(scene_dir: Path) -> list[tuple[float, Path, Path]]:
    depth_frames = sorted(scene_dir.glob("d-*"), key=timestamp)
    rgb_frames = sorted(scene_dir.glob("r-*"), key=timestamp)

    if not depth_frames:
        raise RuntimeError(f"Nenhum frame de profundidade encontrado em '{scene_dir}'.")

    if not rgb_frames:
        raise RuntimeError(f"Nenhum frame RGB encontrado em '{scene_dir}'.")

    rgb_idx = 0
    synced: list[tuple[float, Path, Path]] = []

    for depth in depth_frames:
        t_depth = timestamp(depth)

        while rgb_idx + 1 < len(rgb_frames):
            current_diff = abs(t_depth - timestamp(rgb_frames[rgb_idx]))
            next_diff = abs(t_depth - timestamp(rgb_frames[rgb_idx + 1]))

            if next_diff > current_diff:
                break

            rgb_idx += 1

        synced.append((
            t_depth,
            depth,
            rgb_frames[rgb_idx],
        ))

    return synced


class NYUDepthV2Raw(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        transform: Callable[[np.ndarray], np.ndarray | torch.Tensor] | None = None,
        target_transform: Callable[[np.ndarray], np.ndarray | torch.Tensor] | None = None,
    ):
        self.root = Path(root)

        self.transform = transform
        self.target_transform = target_transform

        self.samples: list[tuple[float, Path, Path]] = []

        self._build_index()

    def _build_index(self) -> None:
        self.samples.clear()

        for scene_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            self.samples.extend(
                sync_frames(scene_dir)
            )

        if not self.samples:
            raise RuntimeError(
                f"Nenhum par RGB/Depth encontrado em '{self.root}'."
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_rgb(path: Path) -> np.ndarray:
        image = np.asarray(Image.open(path).convert("RGB"))

        return image

    @staticmethod
    def load_depth_fixed(path: Path) -> np.ndarray:
        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise IOError(path)

        return depth.astype(np.float32) / 1000.0

    def __getitem__(self, idx: int):
        _, depth_path, rgb_path = self.samples[idx]

        image = self.load_rgb(rgb_path).copy()
        depth = self.load_depth_fixed(depth_path).copy()

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            depth = self.target_transform(depth)

        return image, depth
    
    
from torchvision.transforms import Compose, ToTensor


nyucv2_path = datasets_path / "nyu_depth_v2/preprocessed"

raw_dataset = NYUDepthV2Raw(
    nyucv2_path,
    transform=Compose([
        ToTensor(),
    ]),
    target_transform=Compose([
        lambda x: torch.log1p(torch.from_numpy(x)[None, ...].float()),
    ])
)

print(f"Raw Dataset length: {len(raw_dataset)}")

from torch.utils.data import random_split
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthJointTransform:
    def __init__(
        self,
        resize_size: int | list[int] | None = [224, 224],
        normalization: tuple[list[float], list[float]] | None = (IMAGENET_MEAN, IMAGENET_STD),
        random_crop: bool = True,
        horizontal_flip: bool = True,
        shear: tuple[float, float] | None = (-10, 10),
        brightness: tuple[float, float] | None = (0.8, 1.2),
        contrast: tuple[float, float] | None = (0.8, 1.2),
        saturation: tuple[float, float] | None = (0.8, 1.2),
        hue: tuple[float, float] | None = (-0.1, 0.1),
        exposure: tuple[float, float] | None = (0.8, 1.2)
    ):
        self.resize_size = [resize_size, resize_size] if isinstance(resize_size, int) else resize_size
        self.normalization = normalization
        self.random_crop = random_crop
        self.horizontal_flip = horizontal_flip
        self.shear = shear
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.exposure = exposure

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resize_size is not None and not self.random_crop:
            image = TF.resize(
                image,
                self.resize_size,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True
            )
            depth = TF.resize(
                depth,
                self.resize_size,
                interpolation=T.InterpolationMode.NEAREST,
                antialias=False
            )

        if self.random_crop:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.resize_size)
            image = TF.crop(image, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)

        if self.horizontal_flip and torch.rand(1) < 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        if self.shear:
            shear_x = torch.empty(1).uniform_(*self.shear).item()
            shear_y = torch.empty(1).uniform_(*self.shear).item()
            shear = [shear_x, shear_y]
            image = TF.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=shear
            )
            depth = TF.affine(
                depth,
                angle=0,
                translate=[0, 0],
                scale=1,
                shear=shear
            )

        if self.brightness:
            brightness = torch.empty(1).uniform_(*self.brightness).item()
            image = TF.adjust_brightness(image, brightness)

        if self.contrast:
            contrast = torch.empty(1).uniform_(*self.contrast).item()
            image = TF.adjust_contrast(image, contrast)

        if self.saturation:
            saturation = torch.empty(1).uniform_(*self.saturation).item()
            image = TF.adjust_saturation(image, saturation)

        if self.hue:
            hue = torch.empty(1).uniform_(*self.hue).item()
            image = TF.adjust_hue(image, hue)

        if self.exposure:
            exposure = torch.empty(1).uniform_(*self.exposure).item()
            image = TF.adjust_gamma(image, exposure)

        if self.normalization is not None:
            mean, std = self.normalization
            image = TF.normalize(image, mean=mean, std=std)

        return image, depth


class TransformedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, depth = self.dataset[idx]
        image, depth = self.transform(image, depth)

        return image, depth
    

train_transform = DepthJointTransform(
    random_crop=True,
    horizontal_flip=True,
    shear=(-5, 5),
    brightness=(0.8, 1.2),
    contrast=(0.8, 1.2),
    saturation=(0.8, 1.2),
    hue=(-0.1, 0.1),
    exposure=(0.8, 1.2)
)
val_test_transform = DepthJointTransform(
    random_crop=False,
    horizontal_flip=False,
    shear=None,
    brightness=None,
    contrast=None,
    saturation=None,
    hue=None,
    exposure=None
)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

train_subset, val_subset, test_subset = random_split(
    raw_dataset,
    [train_ratio, val_ratio, test_ratio],
    generator=torch.Generator().manual_seed(MANUAL_SEED)
)

train_dataset = TransformedSubset(train_subset, train_transform)
val_dataset = TransformedSubset(val_subset, val_test_transform)
test_dataset = TransformedSubset(test_subset, val_test_transform)

print(f"Train set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

from torch.utils.data import DataLoader


BATCH_SIZE = 12
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
    

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    

class DepthResNet50UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet50(weights=weights)

        #
        # Encoder
        #

        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        #
        # Decoder
        #

        self.dec4 = DecoderBlock(2048, 1024, 512)

        self.dec3 = DecoderBlock(512, 512, 256)

        self.dec2 = DecoderBlock(256, 256, 128)

        self.dec1 = DecoderBlock(128, 64, 64)

        self.final_up = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),

            ConvBlock(64, 64),
        )

        self.head = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                1,
                kernel_size=1,
            ),

            nn.Softplus()
        )

    def forward(self, x):
        x0 = self.conv1(x)           # 1/2

        x1 = self.layer1(
            self.maxpool(x0)
        )                            # 1/4
        x2 = self.layer2(x1)         # 1/8
        x3 = self.layer3(x2)         # 1/16
        x4 = self.layer4(x3)         # 1/32

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        d1 = self.final_up(d1)

        depth = self.head(d1)
        return depth
    

model = DepthResNet50UNet(pretrained=True).to(device)

summary(model, input_size=(BATCH_SIZE, 3, 224, 224))

from functools import partial
from torch.utils.tensorboard import SummaryWriter


CHECKPOINTS_DIR = Path("./checkpoints")


def train(
    net: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    *,
    suffix: str = "",
    criterion: Callable[..., nn.Module],
    optim: Callable,
    num_epochs: int,
    checkpoint_dir: Path = CHECKPOINTS_DIR,
    device: torch.device = device,
    use_amp: bool = True,
    patience: int = 20,
):
    checkpoint_dir.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=checkpoint_dir / "runs/" / f"run{suffix}")

    optimizer = optim(net.parameters())
    #lr_scheduler = scheduler(optimizer)
    criterion = criterion()

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    global_step = 0

    layers_to_monitor = [
        "head.0",
        "head.2"
    ]

    pbar_postfix = {
        "train_loss": "N/A",
        "val_loss": "N/A",
        "train_batch": "N/A",
        "val_batch": "N/A",
    }
    pbar = tqdm(range(num_epochs), desc="Training", leave=True)

    net.to(device)

    #torch.set_float32_matmul_precision("medium")
    checkpoint_path: Path | None = None
    for epoch in pbar:
        net.train()
        
        losses = []
        for batch_idx, (images, depths) in enumerate(train_dataloader):
            images = images.to(device)
            depths = depths.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=amp_enabled):
                outputs = net(images)
                loss = criterion(outputs, depths)#, images)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=10.0
            )

            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())

            writer.add_scalar(
                "batch/train_loss",
                loss.item(),
                global_step
            )

            writer.add_scalar(
                "batch/grad_norm",
                grad_norm,
                global_step
            )

            global_step += 1

            pbar_postfix["train_batch"] = f"{batch_idx+1}/{len(train_dataloader)}"
            pbar.set_postfix(pbar_postfix)

        train_loss = np.mean(losses)
        history["train_loss"].append(train_loss)
        pbar_postfix["train_loss"] = f"{train_loss:.4f}"

        net.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (images, depths) in enumerate(val_dataloader):
                images = images.to(device)
                depths = depths.to(device)

                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    outputs = net(images)
                    loss = criterion(outputs, depths)#, images)

                losses.append(loss.item())

                pbar_postfix["val_batch"] = f"{batch_idx+1}/{len(val_dataloader)}"
                pbar.set_postfix(pbar_postfix)

        val_loss = np.mean(losses)
        history["val_loss"].append(val_loss)
        pbar_postfix["val_loss"] = f"{val_loss:.4f}"

        writer.add_scalar(
            "epoch/train_loss",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "epoch/val_loss",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "epoch/lr",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        for name, param in net.named_parameters():
            if param.grad is None:
                continue

            if any(layer in name for layer in layers_to_monitor):
                try:
                    writer.add_histogram(
                        f"weights/{name}",
                        param.detach().cpu(),
                        epoch
                    )

                    writer.add_histogram(
                        f"gradients/{name}",
                        param.grad.detach().cpu(),
                        epoch
                    )

                    writer.add_scalar(
                        f"gradient_mean/{name}",
                        param.grad.abs().mean(),
                        epoch
                    )

                    writer.add_scalar(
                        f"weight_mean/{name}",
                        param.abs().mean(),
                        epoch
                    )
                except Exception as e:
                    print(f"Error logging {name}: {e}")

        # lr_scheduler.step(val_loss)

        if val_loss < best_val_loss + 0.1:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / "best_model.pth"

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    #"scheduler_state_dict": lr_scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping after {epoch+1} epochs "
                f"(best val loss={best_val_loss:.5f})"
            )
            checkpoint_path = checkpoint_dir / "early_stopped_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    #"scheduler_state_dict": lr_scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )
            break

        if epoch == num_epochs - 1:
            print(
                f"\nReached maximum epochs ({num_epochs}) "
                f"(best val loss={best_val_loss:.5f})"
            )
            checkpoint_path = checkpoint_dir / "last_epoch_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    #"scheduler_state_dict": lr_scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path
            )

        pbar.set_postfix(pbar_postfix)
    
    writer.close()

    return checkpoint_path, history


from torch.optim import Adam

lr = 1e-4

model = DepthResNet50UNet(pretrained=True)

checkpoint_path, history = train(
    model,
    train_dataloader,
    val_dataloader,
    suffix=f"_resnet_l1_pretrain-{lr}",
    criterion=nn.L1Loss,
    optim=partial(Adam, lr=lr),
    num_epochs=100,
    checkpoint_dir=CHECKPOINTS_DIR
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Validation Loss")
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

fig.tight_layout()
plt.show()
