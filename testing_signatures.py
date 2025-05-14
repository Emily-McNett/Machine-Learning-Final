import torch
from PIL import Image
from Entropy_Train_TwinNN import TwinNeuralNetwork, TwinDataset
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
 
# Set the device
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu or cpu
 
transform=transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
])
 
def process_image(path):
    image = Image.open(path).convert('L') #Set image to grayscale
    return transform(image).unsqueeze(0) # Add Batch Dimension
 
# Load the trained model
 
model = TwinNeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()
 
with torch.no_grad():
    # These are examples from our personally created data
    image0 = process_image("./data/full_data/128/128_05.png").to(device)
    image1 = process_image("./data/full_data/128_forg/128_forg_05.png").to(device)
 
    output = model(image0, image1)
    _, predicted = torch.max(output, 1)
    similarity = "Genuine (1)" if predicted.item() == 1 else "Forged (0)"
 
    print(f"Model Prediction {similarity}")
 
    image0_process = image0.squeeze(0).to(device)
    image1_process = image1.squeeze(0).to(device)
    pair_grid = make_grid([image0_process, image1_process], nrow=2)
 
    plt.imshow(pair_grid.permute(1, 2, 0).cpu(), cmap='gray')
    plt.title(f"Prediction: {similarity}")
    plt.axis("off")
    plt.show()
 
 