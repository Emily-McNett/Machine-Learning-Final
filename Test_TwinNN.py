
import torch
from Entropy_Train_TwinNN import TwinNeuralNetwork, TwinDataset  # Assuming both are defined in Train_TwinNN.py
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import imshow, show_plot
import torchvision.utils
from torch.nn import functional as F
from torchvision.utils import make_grid

# Set the device
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu or cpu

# Load the trained model

model = TwinNeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Prepare test dataset
test_dataset = TwinDataset(
    "data/test_data.csv",
    "data/full_data",
    transform=transforms.Compose([
        transforms.Resize((105, 105)),
        transforms.ToTensor()
    ])
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Visualize a few test predictions
num_pairs_to_show = 10  # Number of image pairs to visualize
shown = 0

with torch.no_grad():
    for batch_idx, (image0, image1, label) in enumerate(test_dataloader):
        image0, image1, label = image0.to(device), image1.to(device), label.to(device)

        outputs = model(image0, image1)
        _, predicted = torch.max(outputs, 1)

        for i in range(image0.size(0)):
            if shown >= num_pairs_to_show:
                break

            img0 = image0[i].cpu()
            img1 = image1[i].cpu()
            lbl = label[i].item()
            pred = predicted[i].item()

            # Create a side-by-side grid of the two images
            pair = make_grid([img0, img1], nrow=2, padding=2)

            plt.figure(figsize=(5, 2.5))
            plt.imshow(pair.permute(1, 2, 0))  # Convert CHW to HWC
            plt.axis('off')
            plt.title(
                f"Predicted: {'Genuine (1)' if pred==1 else 'Forged (0)'} | "
                f"True: {'Genuine (1)' if lbl==1 else 'Forged (0)'}"
            )
            plt.show()

            shown += 1

        if shown >= num_pairs_to_show:
            break

#region Emily don't touch this testiing the image count
# Accuracy counters
correct = 0
total = 0
all_preds = []
all_labels = []

# Evaluation loop
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        image0, image1, label = data
        image0, image1, label = image0.to(device), image1.to(device), label.to(device)

        # Get logits from model
        outputs = model(image0, image1)  # Output is shape [batch_size, 2]
        _, predicted = torch.max(outputs, 1)  # Get predicted class (0 or 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        correct += (predicted == label).sum().item()
        total += label.size(0)

        print(f"Batch {i+1}/{len(test_dataloader)}: Correct: {correct}, Total: {total}")

    # Compute and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Forged (0)", "Genuine (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Accuracy
accuracy = correct / total * 100
print(f"Accuracy on the test set: {accuracy:.2f}%")