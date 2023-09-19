from sklearn.preprocessing import MinMaxScaler
import torch
import datetime
import os

def scale_data(data, scaler_name):
    if scaler_name == 'MinMax':
        scaler = MinMaxScaler()
        normalized_responses = scaler.fit_transform(data)
        data = torch.from_numpy(normalized_responses.copy())
    return data


def get_save_directory(experiment_id, model_name, conv, scaling, dir_name):
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    dir_name = f"../output/{experiment_id}/{model_name}/{conv}/{scaling}/{dir_name}"
    os.makedirs(dir_name, exist_ok=True)  # Create the directory if it doesn't exist
    return dir_name

def save_training_states(save_dir, model, optimizer, scheduler, epoch, loss, train_proportion, num_neurons, scaling=None, sequence_length=None, hidden_size=None):
    # Save model state
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_state.pth'))

    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(save_dir, 'optimizer_state.pth'))

    # Save scheduler state
    torch.save(scheduler.state_dict(), os.path.join(save_dir, 'scheduler_state.pth'))

    # Save other training information as needed
    info = {
        'epoch': epoch,
        'loss': loss,
        'train proportion': train_proportion,
        'num neurons': num_neurons,
        'scaling': scaling,
        'sequence length': sequence_length,
        'hidden size': hidden_size
    }
    torch.save(info, os.path.join(save_dir, 'training_info.pth'))

def load_training_states(model, optimizer, scheduler, load_dir):
    # Load model state
    model.load_state_dict(torch.load(os.path.join(load_dir, 'model_state.pth')))

    # Load optimizer state
    optimizer.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer_state.pth')))

    # Load scheduler state
    scheduler.load_state_dict(torch.load(os.path.join(load_dir, 'scheduler_state.pth')))

    # Load other training information
    info = torch.load(os.path.join(load_dir, 'training_info.pth'))
    return info