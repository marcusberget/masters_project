import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torchvision import models
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def resize_image_with_aspect_ratio(image, desired_size):

    width, height = image.size
    aspect_ratio = width / height

    new_width, new_height = desired_size
    if new_width / new_height > aspect_ratio:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)
    new_size = (new_width, new_height)
    return new_size

class CNN_Dataset(Dataset):
    def __init__(self, image_indexes, responses, transform=None):
        self.image_indexes = image_indexes
        self.responses = responses
        self.transform = transform
    
    def __len__(self):
        return len(self.image_indexes)
    
    def __getitem__(self, idx):
        img_path_string = f'../scenes/scene_{self.image_indexes[idx]}.jpeg'
        images = Image.open(img_path_string)
        if self.transform:
                images = self.transform(images)

        responses = self.responses[idx, :]

        return images, responses


class mobile_net(nn.Module):
    def __init__(self, pretrained, num_neurons):
        super(mobile_net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pretrained = pretrained
        self.pretrained.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(1000, 512)  
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_neurons)
    
    def forward(self, x):
        # x = self.conv1(x)
        x = self.pretrained(x)
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
        x = self.fc3(x)
        # x = self.fc1(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_neurons):
        super(CNN, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)  # Single input channel
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Maxpool after first layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        # Maxpool after second layer
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 12 * 9, 512)  
        self.fc_relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc_relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, num_neurons)  # Predicting neural response for 50 neurons
    
    def forward(self, x):
        # First convolution layer operations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Maxpool after first layer
        x = self.maxpool1(x)
        
        # Second convolution layer operations
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Maxpool after second layer
        x = self.maxpool2(x)
        
        # Flattening the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_relu1(x)
        x = self.fc2(x)
        x = self.fc_relu2(x)
        x = self.fc3(x)
        
        return x


class my_vgg(nn.Module):
    def __init__(self, num_neurons):
        super(my_vgg, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        print(vgg19)

    def forward(self, x, pupil_data):
        x = self.pretrained(x)
        # print(self.pretrained.requires_grad_())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_pupil_data:
            # x = F.normalize(x, p=2, dim=0)
            x = torch.cat((x, pupil_data), dim=1)
            x = self.fc2(x)
        return x