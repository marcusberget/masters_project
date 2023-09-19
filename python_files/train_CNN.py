import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CNN import resize_image_with_aspect_ratio, CNN_Dataset, CNN, mobile_net
from torch.optim.lr_scheduler import StepLR
import os
from helper_functions import scale_data, save_training_states, load_training_states, get_save_directory
from prepare_data import prepare_data

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = y_true.mean()
    ss_tot = ((y_true - y_true_mean) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2

def train_CNN(num_neurons, batch_size, num_epochs, scaling, save_dir, y_train, train_dataloader, learning_rate, train_proportion):

    cnn_model = CNN(num_neurons)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):

        predictions = np.zeros_like(y_train.T)
        truth = np.zeros_like(y_train.T)

        for i, (batch_images, batch_responses) in enumerate(train_dataloader):
            batch_images = batch_images.to(device)
            batch_responses = batch_responses.to(device)

            batch_images = batch_images.to(torch.float32)
            batch_responses = batch_responses.to(torch.float32)

            optimizer.zero_grad()
            predicted_responses = cnn_model(batch_images)

            loss = criterion(predicted_responses, batch_responses)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], i={i} Loss: {loss.item():.4f}")

            start_idx = (batch_size * i)
            end_idx = start_idx + batch_size

            preds = predicted_responses.detach().cpu().numpy()
            actual_responses = batch_responses.detach().cpu().numpy()

            preds = preds.T
            actual_responses = actual_responses.T

            predictions[:, start_idx:end_idx] = preds
            truth[:, start_idx:end_idx] = actual_responses


        r2_scores = {}
        for neuron in range(num_neurons):
            r2_scores[f'{neuron}'] = r2_score(truth[neuron, :], predictions[neuron, :])
        r2_scores_list = list(r2_scores.values())

        np.save(os.path.join(save_dir, 'predictions.npy'), predictions)

        scheduler.step()

        save_training_states(save_dir, cnn_model, optimizer, scheduler, epoch, loss.item(), train_proportion, num_neurons, scaling, hidden_size=None)
