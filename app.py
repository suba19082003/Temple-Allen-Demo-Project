"""
app.py

A simple web UI for the surface defect classifier. Run this after the
model has been trained (models/best_model.pth must exist) and it opens
a local webpage where you can drag-and-drop an image and see the
predicted defect class and confidence for every class.

Run with:
    python app.py

This is intentionally separate from src/predict.py (the command-line
version) — both share the same prediction logic conceptually, but this
file exists purely to give a no-terminal, click-based way to use the
trained model.
"""
import torch
import torch.nn.functional as F
import gradio as gr
from torchvision import transforms

from src.model import build_model

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "models/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def predict(image):
    """Takes a PIL image (from the Gradio upload widget) and returns a
    dict of {class_name: probability} for every class, which Gradio's
    Label component renders as a ranked bar list automatically."""
    if image is None:
        return None
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a surface image"),
    outputs=gr.Label(num_top_classes=6, label="Predicted defect class"),
    title="Surface Defect Classifier",
    description=(
        "Upload an image of a steel surface to classify it into one of 6 defect "
        "types (NEU-CLS categories: crazing, inclusion, patches, pitted surface, "
        "rolled-in scale, scratches). Built with a fine-tuned ResNet18."
    ),
    examples=None,
)

if __name__ == "__main__":
    demo.launch()