import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CRNN import resize_image_with_aspect_ratio, CRNN_Dataset, mobile_net, CRNN
from torch.optim.lr_scheduler import StepLR
from helper_functions import scale_data, save_training_states, load_training_states, get_save_directory
import os
from prepare_data import prepare_data

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = y_true.mean()
    ss_tot = ((y_true - y_true_mean) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
    return r2

def train_CRNN(num_neurons, hidden_size, batch_size, num_epochs, sequence_length, scaling, save_dir, y_train, train_dataloader, pretrained, learning_rate, train_proportion):


    crnn_model = CRNN(num_neurons, hidden_size, batch_size, sequence_length, pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crnn_model.to(device)

    optimizer = optim.Adam(crnn_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):

        summed_truth_vals = np.zeros_like(y_train.T)
        summed_predictions = np.zeros_like(y_train.T)
        counts = np.zeros_like(y_train.T)

        for i, (batch_images, batch_responses, image_labels) in enumerate(train_dataloader):

            batch_images = batch_images.to(device)
            batch_responses = batch_responses.to(device)


            batch_images = batch_images.to(torch.float32)
            batch_responses = batch_responses.to(torch.float32)

            optimizer.zero_grad()
            predicted_responses = crnn_model(batch_images)

            loss = criterion(predicted_responses, batch_responses)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], i={i} Loss: {loss.item():.4f}")

            # Calculate R2 scores for the current batch
            predictions = predicted_responses.detach().cpu().numpy()  # Detach from computation graph and move to CPU
            actual_responses = batch_responses.detach().cpu().numpy()  # Move to CPU
            batch_size, sequence_length, num_neurons = actual_responses.shape
            predictions = np.transpose(predictions, (0, 2, 1))
            actual_responses = np.transpose(actual_responses, (0, 2, 1))

            for j in range(batch_size):  # Iterate over batches
                start_idx = j + (batch_size * i)
                end_idx = start_idx + sequence_length

                # Accumulate the predictions and counts for this sequence.
                summed_predictions[:, start_idx:end_idx] += predictions[j]
                summed_truth_vals[:, start_idx:end_idx] += actual_responses[j]
                counts[:, start_idx:end_idx] += 1


        averaged_predictions = np.where(counts != 0, summed_predictions / counts, 0)

        r2_avg_pred = {}
        for neuron in range(num_neurons):
            r2_avg_pred[f'{neuron}'] = r2_score(y_train.T[neuron, :], averaged_predictions[neuron, :])

        avg_scores = list(r2_avg_pred.values())

        np.save(os.path.join(save_dir, 'predictions.npy'), averaged_predictions)

        scheduler.step()

        save_training_states(save_dir, crnn_model, optimizer, scheduler, epoch, loss.item(), train_proportion, num_neurons, scaling, sequence_length, hidden_size)
