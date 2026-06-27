"""
evaluate.py

Evaluate the best trained model on the validation set and print detailed metrics.
"""
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from src.model import build_model
from src.dataset import get_dataloaders


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[float, float]:
    """
    Evaluate model and return (avg_loss, accuracy) plus detailed metrics.

    Prints:
    - Per-class precision, recall, f1-score
    - Confusion matrix
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\nConfusion Matrix (normalized by true class):")
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]
    for i, row in enumerate(cm_norm):
        print(f"  {class_names[i]}: {row}")

    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders()
    print(f"Evaluating on {len(val_loader.dataset)} validation images")

    model = build_model(num_classes=len(class_names)).to(device)

    # Load best checkpoint
    checkpoint_path = "models/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, class_names)

    print(f"\nFinal Validation: loss={val_loss:.4f} accuracy={val_acc:.4f}")


if __name__ == "__main__":
    main()