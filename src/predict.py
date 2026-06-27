"""
predict.py

Predict the defect class of a single image using the trained model.
"""
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.model import build_model


# Validation transform constants (must match dataset.py val_transform exactly)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    img_size: int = 224,
) -> tuple[str, float]:
    """
    Predict the class of a single image.

    Args:
        image_path: Path to the image file.
        model: Trained model.
        class_names: List of class names in order.
        device: Torch device.
        img_size: Input image size (square).

    Returns:
        (predicted_class_name, confidence)
    """
    # Same validation transform as dataset.py
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = probs.max(1)

    return class_names[predicted_idx.item()], confidence.item()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    class_names = [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))

    predicted_class, confidence = predict_image(image_path, model, class_names, device)

    print(f"Predicted: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()