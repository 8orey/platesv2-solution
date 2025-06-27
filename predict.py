import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

model_path = 'best_resnet101_plate_classifier.pth'
test_dir = 'plates/plates/test'
output_csv = 'submission.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class_names = ['clean', 'dirty']
label_map = {'clean': 'cleaned', 'dirty': 'dirty'}

model = torchvision.models.resnet101(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

results = []
for filename in sorted(os.listdir(test_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_id = os.path.splitext(filename)[0]
        img_path = os.path.join(test_dir, filename)
        pred = predict_image(img_path)
        mapped_label = label_map[pred]
        results.append({'id': img_id, 'label': mapped_label})

submission_df = pd.DataFrame(results)
submission_df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")
