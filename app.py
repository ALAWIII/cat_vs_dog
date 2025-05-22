from torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

model = models.efficientnet_b0(pretrained=False)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 2)
)
model.load_state_dict(torch.load("cat_dog_war.pth", map_location="cpu"))
model.eval()



val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])



img = Image.open("/run/media/allawiii/adfea82a-6258-4cbc-969a-9b7bec113569/zed/Python/catdog/PetImages/Cat/1000.jpg").convert("RGB")
input_tensor = val_transform(img).unsqueeze(0)  # Add batch dim

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class = probabilities.argmax().item()

classes = ["cat", "dog"]
print(f"Prediction: {classes[predicted_class]}")
print(f"Probabilities: cat={probabilities[0]:.4f}, dog={probabilities[1]:.4f}")
