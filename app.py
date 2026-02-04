import streamlit as st                     # web app framework
import torch                               # PyTorch
import torch.nn as nn                      # neural network layers
import torch.nn.functional as F            # softmax
from PIL import Image                      # image handling
import torchvision.transforms as transforms

# -----------------------------
# Model definition (MUST match training)
# -----------------------------
class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)   # input → hidden
        self.fc2 = nn.Linear(128, 10)      # hidden → output (0–9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))        # activation
        x = self.fc2(x)                    # output layer
        return x

# -----------------------------
# Load trained model
# -----------------------------
model = DigitModel()                       # create model
model.load_state_dict(
    torch.load("classifier.pth", map_location="cpu")
)
model.eval()                               # evaluation mode

# -----------------------------
# Image preprocessing (MNIST style)
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),                # convert to grayscale
    transforms.Resize((28, 28)),            # resize to 28x28
    transforms.ToTensor(),                  # convert to tensor
    transforms.Lambda(lambda x: 1 - x)      # invert colors
])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Handwritten Digit Classifier")
st.write("Upload a handwritten digit image (0–9)")

uploaded_file = st.file_uploader(
    "Choose an image", type=["png", "jpg", "jpeg"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)      # open image
    st.image(image, caption="Uploaded Image", width=200)

    image = transform(image)               # preprocess
    image = image.view(1, -1)               # flatten (1, 784)

    with torch.no_grad():                   # no gradients
        outputs = model(image)              # model output
        probs = F.softmax(outputs, dim=1)   # probabilities
        confidence, prediction = torch.max(probs, 1)

    st.subheader(f"Predicted Digit: {prediction.item()}")
    st.subheader(f"Confidence: {confidence.item() * 100:.2f}%")
