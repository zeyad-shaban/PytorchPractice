from torchvision import models, transforms
import torch
from PIL import Image
import json

# Load the model
resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

img = Image.open("./data/bobby.jpg")
img_t = preprocess(img)
batch_t = img_t.unsqueeze(0)

# Get the model output
resnet.eval()
out = resnet(batch_t)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Get the top 10 predictions
top10_prob, top10_catid = torch.topk(probabilities, 10)

# Load ImageNet class labels
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)

# Map the indices to labels
top10_labels = [class_idx[str(catid)][1] for catid in top10_catid]

# Print the top 10 labels and their probabilities
for i in range(10):
    print(f"{top10_labels[i]}: {top10_prob[i].item()}")
