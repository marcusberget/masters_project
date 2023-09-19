import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torchvision import models
from PIL import Image
from CRNN import resize_image_with_aspect_ratio, CRNN_Dataset
from CNN import CNN_Dataset
from helper_functions import get_save_directory
from prepare_data import prepare_data
from train_CRNN import train_CRNN
from train_mobilenet import train_mobilenet
from train_CNN import train_CNN

print(f"Num GPUs Available: {torch.cuda.device_count()}")

experiment_ids = [500964514, 501498760, 501794235, 704826374, 681674286]

for experiment_id in experiment_ids:
    data_dir = f'../data/{experiment_id}'

    deconvolve = False
    normalize = True

    im_labels, responses = prepare_data(experiment_id, deconv=deconvolve, normalize=normalize)

    responses = responses.T
    num_neurons = responses.shape[1]  

    print(f'number of filtered neurons= {num_neurons}')

    if num_neurons < 2:
        continue

    num_classes = 119
    batch_size = 8
    sequence_length = 5
    hidden_size = 256  # RNN hidden state size
    learning_rate = 0.001
    num_epochs = 20
    
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

    train_CRNN(num_neurons, hidden_size, batch_size, num_epochs, sequence_length, scaling, save_dir, y_train, train_dataloader, pretrained, learning_rate, train_proportion)

for experiment_id in experiment_ids:
    data_dir = f'../data/{experiment_id}'

    deconvolve = True
    normalize = True

    im_labels, responses = prepare_data(experiment_id, deconv=deconvolve, normalize=normalize)

    responses = responses.T
    num_neurons = responses.shape[1]  

    print(f'number of filtered neurons= {num_neurons}')

    if num_neurons < 2:
        continue

    num_classes = 119
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 50
    
    if deconvolve:
        conv = 'deconvolved'
    else:
        conv = 'convolved'
    if normalize:
        scaling = 'normalized'
    else:
        scaling = 'no_scaling'
    
    save_dir = get_save_directory(experiment_id=experiment_id, model_name='mobilenet', conv=conv, scaling=scaling, dir_name='test_run')

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

    training_dataset = CNN_Dataset(X_train, y_train, transform=transform)
    test_dataset = CNN_Dataset(X_test, y_test, transform=transform)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_mobilenet(num_neurons, batch_size, num_epochs, scaling, save_dir, y_train, train_dataloader, pretrained, learning_rate, train_proportion)


for experiment_id in experiment_ids:
    data_dir = f'../data/{experiment_id}'

    deconvolve = True
    normalize = True

    im_labels, responses = prepare_data(experiment_id, deconv=deconvolve, normalize=normalize)

    responses = responses.T
    num_neurons = responses.shape[1]  

    print(f'number of filtered neurons= {num_neurons}')

    if num_neurons < 2:
        continue

    num_classes = 119
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 50
    
    if deconvolve:
        conv = 'deconvolved'
    else:
        conv = 'convolved'
    if normalize:
        scaling = 'normalized'
    else:
        scaling = 'no_scaling'
    
    save_dir = get_save_directory(experiment_id=experiment_id, model_name='CNN', conv=conv, scaling=scaling, dir_name='test_run')

    train_proportion = 0.8

    n_train = int(len(responses.T[0,:]) * train_proportion)
    n_test = len(responses.T[0,:]) - n_train

    X_train, X_test = im_labels[0:n_train], im_labels[n_train:]
    y_train, y_test = responses[0:n_train, :], responses[n_train:, :]

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

    training_dataset = CNN_Dataset(X_train, y_train, transform=transform)
    test_dataset = CNN_Dataset(X_test, y_test, transform=transform)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_CNN(num_neurons, batch_size, num_epochs, scaling, save_dir, y_train, train_dataloader, learning_rate, train_proportion)
