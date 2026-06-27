"""
dataset.py

Provides DataLoaders for training and validation.

Why separate transforms for train and val?
- Training uses augmentation (RandomHorizontalFlip, RandomRotation) to
  artificially expand the dataset and reduce overfitting. The model sees
  slightly different variations of each image each epoch.
- Validation uses NO augmentation — just resize, ToTensor, and normalize.
  This ensures metrics reflect real-world performance on untouched images.
  If we augmented validation, we'd measure performance on "easier"
  transformed inputs and get inflated accuracy.

Why ImageFolder?
- torchvision.datasets.ImageFolder expects a directory structure like:
    root/class_a/*.jpg
    root/class_b/*.jpg
  It automatically treats each subfolder name as a class label (0, 1, 2...).
  This is exactly what prepare_data.py produced in data/processed/train/
  and data/processed/val/.
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    img_size: int = 224,
) -> tuple:
    """
    Create train and validation DataLoaders with appropriate transforms.

    Args:
        data_dir: Root directory containing 'train' and 'val' subfolders.
        batch_size: Number of images per batch.
        img_size: Target image size (square).

    Returns:
        (train_loader, val_loader, class_names)
    """
    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Training transforms: resize + light augmentation + tensor + normalize
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # Validation transforms: resize + tensor + normalize ONLY (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # ImageFolder loads images from subfolders named by class
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val", transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")