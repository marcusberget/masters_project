import numpy as np
import os

def prepare_data(experiment_id, deconv, normalize):

    # Specify the directory path you want to create
    directory_path = f'../data/{experiment_id}/'

    if os.path.exists(directory_path):
        if (deconv == True) & (normalize == True):
            im_labels = np.load(os.path.join(directory_path, 'im_labels.npy'))
            responses = np.load(os.path.join(directory_path, 'deconv_norm_responses.npy'))
            return im_labels, responses
        elif (deconv == True) & (normalize == False):
            im_labels = np.load(os.path.join(directory_path, 'im_labels.npy'))
            responses = np.load(os.path.join(directory_path, 'deconv_responses.npy'))
            return im_labels, responses
        elif (deconv == False) & (normalize == True):
            im_labels = np.load(os.path.join(directory_path, 'im_labels.npy'))
            responses = np.load(os.path.join(directory_path, 'norm_responses.npy'))
            return im_labels, responses
        elif (deconv == False) & (normalize == False):
            im_labels = np.load(os.path.join(directory_path, 'im_labels.npy'))
            responses = np.load(os.path.join(directory_path, 'responses.npy'))
            return im_labels, responses
