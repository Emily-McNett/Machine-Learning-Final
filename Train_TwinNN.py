from PIL import Image  # image loading/manipulation
from torchvision import transforms
import pandas as pd
import numpy as np
import os  # for file path manipulation
import torch
import torchvision.utils
from torch.utils.data import DataLoader, Dataset
from utils import imshow, show_plot
import torch.nn as nn
import multiprocessing
from torch.nn import functional as F

# Class to read in and Process data set
class TwinDataset():
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        # Prepare labels and Images path
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ["image0", "image1", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # Dataloader getting Image Path
        image0_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])

        # Builds the full path to each image in the pair
        image0 = Image.open(image0_path)
        image1 = Image.open(image1_path)
        # Opens both images using Python Imaging Library, and converts them to grayscale ("L"- luminance, values from 0 (black) to 255 (white))
        # This is done to ensure that the images are in the same format for processing
        image0 = image0.convert("L")
        image1 = image1.convert("L")

        # Apply Image Transformations
        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)
        return image0, image1,  int(self.train_df.iat[index, 2])

    # Return the number of samples
    def __len__(self):
        return len(self.train_df)

# Load the train dataset with 32x32 resizing
twin_dataset = TwinDataset("data/train_data.csv", "data/full_data", transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()]))

# View sample of images to check loading
vis_dataloader = DataLoader(twin_dataset, shuffle=True, batch_size=8)
# Wraps the dataset in a DataLoader for batching and shuffling
dataiter = iter(vis_dataloader)

# Grabs a batch of data from the DataLoader
example_batch = next(dataiter)

# Concatenates the two images in the batch along the first dimension (0) to create a single tensor
# This is done to visualize the images side by side
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

# Displays the concatenated images using torchvision's make_grid function
#imshow(torchvision.utils.make_grid(concatenated))

# Prints the labels of the images in the batch as a numpy array
print(example_batch[2].numpy())

# Create Twin Network
class TwinNeuralNetwork(nn.Module):
    def __init__(self):
        super(TwinNeuralNetwork, self).__init__()

       # Shared CNN feature extractor for both images
        self.cnn = nn.Sequential(
            # First conv layer: 1 input channel (grayscale), 32 filters, 3x3 kernel
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Normalizes the output of the previous layer
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduces size: 105x105 -> 52x52

            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 52x52 -> 26x26
        )

        # Fully connected layer to embed each image into a 128-dimensional vector
        self.fc = nn.Sequential(
            nn.Linear(64 * 26 * 26, 128),  # Flattened input size: 128 channels × 13 × 13
            nn.ReLU(),
            nn.Dropout(0.3),                # Dropout to prevent overfitting
            nn.Linear(128, 64)             # Final embedding of size 128
        )

        # Classifier that takes both embeddings and predicts similarity
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 32),  # Concatenated feature vector from both images (128 + 128)
            nn.ReLU(),
            nn.Linear(32, 2)         # Output logits for classes: 0 = same, 1 = different
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, image0, image1):
        x0 = self.forward_once(image0)
        x1 = self.forward_once(image1)
        combined = torch.cat((x0, x1), dim=1)
        # Concatenate features
        output = self.classifier(combined)
        return output

    # Load the dataset as pytorch tensors using DataLoader
train_dataloader = DataLoader(twin_dataset,
                                shuffle=True,
                                num_workers=8,
                                batch_size=16)

    # Train the model
def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(1, 20):
        for i, data in enumerate(train_dataloader, 0):
            if i >= 5:  # <-- LIMIT batches for testing
             break
            image0, image1, label = data
            image0, image1, label = image0.to(device), image1.to(device), label.to(device)
            label = label.to(device).long()
            outputs = net(image0, image1)
            cross_entropy_loss = lossFunction(outputs, label)
            cross_entropy_loss.backward()
            optimizer.step()
            print(f"{i} / {len(train_dataloader)}")
        print("Epoch {}\n Current loss {}\n".format(epoch, cross_entropy_loss.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(cross_entropy_loss.item())
    show_plot(counter, loss)
    return net

# Main code to be executed within `if __name__ == '__main__':`
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Set the device to cpu
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu or cpu

        # Declare Twin Network, Loss Function, and Optimizer
    net = TwinNeuralNetwork().to(device)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

    model = train()
    torch.save(model.state_dict(), "model.pt")
    print("Model Finished Training")