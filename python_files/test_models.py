import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from torchvision import models
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from CRNN import resize_image_with_aspect_ratio, CRNN_Dataset, CRNN
from CNN import CNN_Dataset, CNN, mobile_net
from helper_functions import get_save_directory
import os
from prepare_data import prepare_data
from sklearn.metrics import r2_score


def evaluate_CRNN(num_neurons, hidden_size, batch_size, sequence_length, save_dir, y_test, test_dataloader, pretrained):

    crnn_model = CRNN(num_neurons, hidden_size, batch_size, sequence_length, pretrained)

    # Load model state from save directory
    model_state_path = os.path.join(save_dir, 'model_state.pth') # Update path_to_saved_model_state.pth with your filename
    crnn_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    crnn_model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crnn_model.to(device)

    criterion = nn.MSELoss()
    test_loss = 0.0
    summed_truth_vals = np.zeros_like(y_test.T)
    summed_predictions = np.zeros_like(y_test.T)
    counts = np.zeros_like(y_test.T)

    with torch.no_grad(): # Disable gradient computation during testing
        for i, (batch_images, batch_responses, image_labels) in enumerate(test_dataloader):

            batch_images = batch_images.to(device).float()
            batch_responses = batch_responses.to(device).float()

            predicted_responses = crnn_model(batch_images)
            loss = criterion(predicted_responses, batch_responses)

            test_loss += loss.item()

            predictions = predicted_responses.cpu().numpy()
            actual_responses = batch_responses.cpu().numpy()
            batch_size, sequence_length, num_neurons = actual_responses.shape
            predictions = np.transpose(predictions, (0, 2, 1))
            actual_responses = np.transpose(actual_responses, (0, 2, 1))

            for j in range(batch_size):
                start_idx = j + (batch_size * i)
                end_idx = start_idx + sequence_length

                summed_predictions[:, start_idx:end_idx] += predictions[j]
                summed_truth_vals[:, start_idx:end_idx] += actual_responses[j]
                counts[:, start_idx:end_idx] += 1

    averaged_predictions = np.where(counts != 0, summed_predictions / counts, 0)

    r2_avg_pred = {}
    for neuron in range(num_neurons):
        r2_avg_pred[f'{neuron}'] = r2_score(y_test.T[neuron, :], averaged_predictions[neuron, :])

    avg_scores = np.array(list(r2_avg_pred.values()))

    np.save(os.path.join(save_dir, 'r2_scores_test.npy'), avg_scores)

    # Save the predictions
    np.save(os.path.join(save_dir, 'test_predictions.npy'), averaged_predictions)

    print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")


def evaluate_CNN(num_neurons, batch_size, save_dir, y_test, test_dataloader):

    cnn_model = CNN(num_neurons)

    # Load model state from save directory
    model_state_path = os.path.join(save_dir, 'model_state.pth') 
    cnn_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    cnn_model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    criterion = nn.MSELoss()
    test_loss = 0.0
    summed_predictions = np.zeros_like(y_test.T)
    counts = np.zeros_like(y_test.T)

    with torch.no_grad():  # Disable gradient computation during testing
        for i, (batch_images, batch_responses) in enumerate(test_dataloader):

            batch_images = batch_images.to(device).float()
            batch_responses = batch_responses.to(device).float()

            predicted_responses = cnn_model(batch_images)
            loss = criterion(predicted_responses, batch_responses)

            test_loss += loss.item()

            predictions = predicted_responses.cpu().numpy()
            actual_responses = batch_responses.cpu().numpy()

            preds = predictions.T
            actual_responses = actual_responses.T

            start_idx = (batch_size * i)
            end_idx = start_idx + batch_size

            summed_predictions[:, start_idx:end_idx] = preds
            counts[:, start_idx:end_idx] += 1

    r2_avg_pred = {}
    for neuron in range(num_neurons):
        r2_avg_pred[f'{neuron}'] = r2_score(y_test.T[neuron, :], summed_predictions[neuron, :])

    avg_scores = np.array(list(r2_avg_pred.values()))

    np.save(os.path.join(save_dir, 'r2_scores_test.npy'), avg_scores)

    # Save the predictions
    np.save(os.path.join(save_dir, 'test_predictions.npy'), summed_predictions)

    print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")

def evaluate_mobilenet(num_neurons, batch_size, save_dir, y_test, test_dataloader, pretrained):

    mobile_net_model = mobile_net(pretrained, num_neurons)

    # Load model state from save directory
    model_state_path = os.path.join(save_dir, 'model_state.pth')
    mobile_net_model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
    mobile_net_model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobile_net_model.to(device)

    criterion = nn.MSELoss()
    test_loss = 0.0
    summed_predictions = np.zeros_like(y_test.T)

    with torch.no_grad():  # Disable gradient computation during testing
        for i, (batch_images, batch_responses) in enumerate(test_dataloader):

            batch_images = batch_images.to(device).float()
            batch_responses = batch_responses.to(device).float()

            predicted_responses = mobile_net_model(batch_images)
            loss = criterion(predicted_responses, batch_responses)

            test_loss += loss.item()

            predictions = predicted_responses.cpu().numpy()
            actual_responses = batch_responses.cpu().numpy()

            preds = predictions.T
            actual_responses = actual_responses.T

            start_idx = (batch_size * i)
            end_idx = start_idx + batch_size

            summed_predictions[:, start_idx:end_idx] = preds

    r2_avg_pred = {}
    for neuron in range(num_neurons):
        r2_avg_pred[f'{neuron}'] = r2_score(y_test.T[neuron, :], summed_predictions[neuron, :])

    avg_scores = np.array(list(r2_avg_pred.values()))

    np.save(os.path.join(save_dir, 'r2_scores_test.npy'), avg_scores)

    # Save the predictions
    np.save(os.path.join(save_dir, 'test_predictions.npy'), summed_predictions)

    print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")

experiment_id = 501794235

data_dir = f'../data/{experiment_id}'

deconvolve = True
normalize = True

im_labels, responses = prepare_data(experiment_id, deconv=deconvolve, normalize=normalize)

responses = responses.T
num_neurons = responses.shape[1]  


num_classes = 119
batch_size = 8
sequence_length = 5
hidden_size = 256  # RNN hidden state size
learning_rate = 0.001

if deconvolve:
    conv = 'deconvolved'
else:
    conv = 'convolved'
if normalize:
    scaling = 'normalized'
else:
    scaling = 'no_scaling'

save_dir = get_save_directory(experiment_id=experiment_id, model_name='CRNN', conv=conv, scaling=scaling, dir_name='test_run')

train_proportion = 0.8

n_train = int(len(responses.T[0,:]) * train_proportion)
n_test = len(responses.T[0,:]) - n_train

X_train, X_test = im_labels[0:n_train], im_labels[n_train:]
y_train, y_test = responses[0:n_train, :], responses[n_train:, :]


pretrained_CNN_load_path = "../data/pretrained_models/mobilenet_pretrained_model.pth"
loaded_model = models.mobilenet_v2(pretrained=False)  # Create a new MobileNetV2 model with the same architecture

state_dict = torch.load(pretrained_CNN_load_path)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
loaded_model.load_state_dict(state_dict, strict=False)

pretrained = loaded_model

desired_size = (48,48)
img_path_string = f'../scenes/scene_{im_labels[0]}.jpeg'
image = Image.open(img_path_string)

new_size = resize_image_with_aspect_ratio(image, desired_size)

transform = transforms.Compose([
    transforms.Resize(new_size),
    transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])  # Adjust normalization for single-channel
])

training_dataset = CRNN_Dataset(X_train, y_train, sequence_length=sequence_length, transform=transform)
test_dataset = CRNN_Dataset(X_test, y_test, sequence_length=sequence_length, transform=transform)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# evaluate_CRNN(num_neurons, hidden_size, batch_size, sequence_length, save_dir, y_test, test_dataloader, pretrained)

training_dataset = CNN_Dataset(X_train, y_train, transform=transform)
test_dataset = CNN_Dataset(X_test, y_test, transform=transform)
train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

save_dir = get_save_directory(experiment_id=experiment_id, model_name='CNN', conv=conv, scaling=scaling, dir_name='test_run')
evaluate_CNN(num_neurons, batch_size, save_dir, y_test, test_dataloader)
save_dir = get_save_directory(experiment_id=experiment_id, model_name='mobilenet', conv=conv, scaling=scaling, dir_name='test_run')
evaluate_mobilenet(num_neurons, batch_size, save_dir, y_test, test_dataloader, pretrained)