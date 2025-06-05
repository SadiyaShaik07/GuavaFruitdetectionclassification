import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

# Flask app
app = Flask(__name__)

# Define class names
class_names = ['Anthracnose', 'Fruit Flies', 'Healthy']

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition (must match the trained architecture)
class CustomVGG16(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomVGG16, self).__init__()
        vgg = models.vgg16(pretrained=False)  # Load without pretrained weights
        vgg.classifier[6] = nn.Linear(4096, num_classes)  # Custom last layer
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the model
model = CustomVGG16(num_classes=3).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        image_path = os.path.join('static', 'uploads', file.filename)
        file.save(image_path)

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = class_names[probs.argmax().item()]

        return render_template('result.html', prediction=predicted_class, image_path=image_path)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    app.run(debug=True)
