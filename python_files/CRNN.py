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

class CRNN_Dataset(Dataset):
    def __init__(self, image_indexes, responses, sequence_length, transform=None):
        self.image_indexes = image_indexes
        self.responses = responses
        self.sequence_length = sequence_length
        self.transform = transform
    
    def __len__(self):
        return len(self.image_indexes) - self.sequence_length + 1

    def __getitem__(self, idx):
        image_sequence = []
        image_labels_list = []
        for i in range(self.sequence_length):
            image_labels_list.append(int(self.image_indexes[idx + i]))
            img_path_string = f'../scenes/scene_{self.image_indexes[idx + i]}.jpeg'
            image = Image.open(img_path_string)
            if self.transform:
                image = self.transform(image)
            image_sequence.append(image)
        image_sequence = torch.stack(image_sequence)
        image_labels = torch.tensor(image_labels_list, dtype=torch.int)
        response_sequence = self.responses[idx:idx+self.sequence_length, :]

        return image_sequence, response_sequence, image_labels


class mobile_net(nn.Module):
    def __init__(self, pretrained):
        super(mobile_net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pretrained = pretrained
        self.pretrained.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        for param in self.pretrained.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x = self.conv1(x)
        x = self.pretrained(x)
        # x = self.fc1(x)
        return x

class CRNN(nn.Module):
    def __init__(self, num_neurons, hidden_size, batch_size, sequence_length, pretrained):
        super(CRNN, self).__init__()
        self.cnn = mobile_net(pretrained)#resnet18(pretrained=True)
        self.rnn = nn.LSTM(input_size=1000, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_neurons)
                                # nn.Linear(hidden_size, 100),
                                # nn.Linear(100, num_neurons))
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        # for layer in self.fc:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size()[0] 
        height = x.size()[-2]
        width = x.size()[-1]
        x = x.view(batch_size * self.sequence_length, 1, height, width)
        features = self.cnn(x)  # Input size: (batch_size, sequence_length, channels, height, width)
        rnn_input = features.view(batch_size, self.sequence_length, 1000)
        # features = features.view(batch_size, sequence_length, features.size()[1])
        # features = features.mean(dim=(3, 4))  # Global average pooling

        # RNN sequence modeling
        rnn_out, _ = self.rnn(rnn_input)

        
        # Fully connected layer for prediction
        output = self.fc(rnn_out)#[:, -1, :])  # Taking the last RNN output as prediction
        # output = output.view(self.batch_size, self.sequence_length, output.size()[-1])


        return output
