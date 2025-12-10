# src/run_quick.py
from PIL import Image
import torchvision, torch
from torchvision import transforms
print("python ok, torchvision", torchvision.__version__)

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# create a dummy tensor image
x = torch.randn(1,3,224,224)
model = torchvision.models.resnet18(pretrained=False)  # use False to avoid download issues
model.eval()
with torch.no_grad():
    out = model(x)
print("inference ok, output shape:", out.shape)
