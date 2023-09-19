import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os


class Plots:

    def __init__(self, experiment_ids, modelname, deconvolve, normalize, include_datasets=['train', 'test']):
        self.experiment_ids = experiment_ids
        self.modelname = modelname
        self.conv = 'deconvolved' if deconvolve else 'convolved'
        self.norm = 'normalized' if normalize else 'not_normalized'
        self.include_datasets = include_datasets

    def load_data(self, experiment_id):

        data_types = ['train', 'test']
        results = {}
        
        for dt in data_types:
            preds_path = f'../output/{experiment_id}/{self.modelname}/{self.conv}/{self.norm}/test_run/{dt}_predictions.npy'
            
            if not os.path.exists(preds_path):
                print(f"Warning: {dt} predictions not found for Experiment ID {experiment_id}, Model {self.modelname}, Conv {self.conv}, Norm {self.norm}.")
                results[dt] = (None, None)
                continue

            preds = np.load(preds_path)

            directory_path = f'../data/{experiment_id}'

            truth_filename = ''

            if self.conv == 'deconvolved' and self.norm == 'normalized':
                truth_filename = 'deconv_norm_responses.npy'
            elif self.conv == 'deconvolved' and self.norm == 'not_normalized':
                truth_filename = 'deconv_responses.npy'
            elif self.conv == 'convolved' and self.norm == 'normalized':
                truth_filename = 'norm_responses.npy'
            else:
                truth_filename = 'responses.npy'

            truth_path = os.path.join(directory_path, truth_filename)

            if not os.path.exists(preds_path) or not os.path.exists(truth_path):
                print(f"Warning: Data not found for Experiment ID {experiment_id}, Model {self.modelname}, Conv {self.conv}, Norm {self.norm}.")
                return None, None

            preds = np.load(preds_path)
            truth = np.load(truth_path)

            train_proportion = 0.8
            n_train = int(len(truth[0,:]) * train_proportion)
            if dt == 'train':
                truth = truth[:, :n_train]
            else:
                truth = truth[:, n_train:]
            
            results[dt] = (preds, truth)

        return results['train'], results['test']
    
    def plot_histogram(self):
        fixed_lower_limit = -0.3
        for experiment_id in self.experiment_ids:
            (train_preds, train_truth), (test_preds, test_truth) = self.load_data(experiment_id)
            
            data_sets = {'Train': (train_preds, train_truth), 'Test': (test_preds, test_truth)}
            r2_scores_combined = []

            for label, (preds, truth) in data_sets.items():
                if label.lower() not in self.include_datasets:  # skip if the dataset is not in include_datasets
                    continue

                if preds is None or truth is None:
                    continue

                r2_scores = [r2_score(t, p) for t, p in zip(truth, preds)]

                r2_scores = np.clip(r2_scores, fixed_lower_limit, None)

                if np.any(np.isnan(r2_scores)) or np.any(np.isinf(r2_scores)):
                    print(f"Warning: Invalid R^2 values detected for experiment_id {experiment_id}")
                    r2_scores = np.array(r2_scores)[~np.isnan(r2_scores)]
                    r2_scores = np.array(r2_scores)[~np.isinf(r2_scores)]

                r2_scores_combined.extend(r2_scores)
                
            auto_bins = np.histogram_bin_edges(r2_scores_combined, bins='auto')
            bins = np.concatenate(([-0.3, -0.2, -0.1], auto_bins[auto_bins > -0.1]))
            # bins = np.concatenate(([fixed_lower_limit], [-0.1], auto_bins[auto_bins > -0.1]))

            fig, ax = plt.subplots()
            for label, (preds, truth) in data_sets.items():
                if label.lower() not in self.include_datasets:  # skip if the dataset is not in include_datasets
                    continue
                r2_scores = [r2_score(t, p) for t, p in zip(truth, preds)]
                ax.hist(r2_scores, bins=bins, alpha=0.5, label=label, edgecolor='black')

            # ax.hist(r2_scores, bins=bins, edgecolor='black')
            ax.set_title(f'Train Data epoch 20: Distribution of R^2. Model={self.modelname}, {self.conv}, {self.norm}')
            ax.set_xlabel('R^2 Score')
            ax.set_ylabel('Frequency')

            # Define ticks with 0.1 intervals
            min_tick = fixed_lower_limit
            max_tick = np.ceil(max(r2_scores))
            ticks = np.arange(min_tick, max_tick+0.1, 0.1)

            # Adjust the first tick label to indicate values less than or equal to the fixed limit
            formatted_ticks = ['≤' + "{:.2f}".format(fixed_lower_limit) if tick == fixed_lower_limit else "{:.2f}".format(tick) for tick in ticks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(formatted_ticks)
            ax.legend()
            plt.show()
    
    def plot_histogram_sbs(self):
        fixed_lower_limit = -0.3
        for experiment_id in self.experiment_ids:
            (train_preds, train_truth), (test_preds, test_truth) = self.load_data(experiment_id)

            data_sets = {'Train': (train_preds, train_truth), 'Test': (test_preds, test_truth)}
            r2_scores_combined = []

            for label, (preds, truth) in data_sets.items():
                if label.lower() not in self.include_datasets:  # skip if the dataset is not in include_datasets
                    continue

                if preds is None or truth is None:
                    continue

                r2_scores = [r2_score(t, p) for t, p in zip(truth, preds)]
                r2_scores = np.clip(r2_scores, fixed_lower_limit, None)
                r2_scores_combined.extend(r2_scores)

            auto_bins = np.histogram_bin_edges(r2_scores_combined, bins='auto')
            bins = np.concatenate(([-0.3, -0.2, -0.1], auto_bins[auto_bins > -0.1]))

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # creating 1 row and 2 columns of subplots

            for ax, (label, (preds, truth)) in zip(axes, data_sets.items()):  # iterate over both axes and datasets
                if label.lower() not in self.include_datasets:  # skip if the dataset is not in include_datasets
                    continue

                r2_scores = [r2_score(t, p) for t, p in zip(truth, preds)]
                ax.hist(r2_scores, bins=bins, alpha=0.7, label=label, edgecolor='black')
                ax.set_title(f'{label} Data: Distribution of R^2, {self.modelname}')
                ax.set_xlabel('R^2 Score')
                ax.set_ylabel('Frequency')

                # Define ticks with 0.1 intervals
                min_tick = fixed_lower_limit
                max_tick = np.ceil(max(r2_scores))
                ticks = np.arange(min_tick, max_tick+0.1, 0.1)

                # Adjust the first tick label to indicate values less than or equal to the fixed limit
                formatted_ticks = ['≤' + "{:.1f}".format(fixed_lower_limit) if tick == fixed_lower_limit else "{:.1f}".format(tick) for tick in ticks]
                ax.set_xticks(ticks)
                ax.set_xticklabels(formatted_ticks)

            plt.tight_layout()
            plt.show()



    def plot_predictions(self):
        for experiment_id in self.experiment_ids:
            preds, truth = self.load_data(experiment_id)
            
            if preds is None or truth is None:
                continue

            fig, ax = plt.subplots(2,2)
        r2_scores = {}
        for neuron in range(truth.shape[0]):
            r2_scores[f'{neuron}'] = r2_score(truth[neuron, :], preds[neuron, :])
        r2_scores = list(r2_scores.values())

        ax[0,0].plot(truth[4,:], label='traces', color='b')
        ax[0,0].plot(preds[4,], label='predictions', color='r', linestyle='--')
        ax[0,0].set_title(f'Neuron 4, R2 score={r2_scores[4]:.2f}')
        ax[0,0].legend()

        ax[0,1].plot(truth[8,:], label='traces', color='b')
        ax[0,1].plot(preds[8,], label='predictions', color='r', linestyle='--')
        ax[0,1].set_title(f'Neuron 8, R2 score={r2_scores[8]:.2f}')
        ax[0,1].legend()

        ax[1,0].plot(truth[12,:], label='traces', color='b')
        ax[1,0].plot(preds[12,], label='predictions', color='r', linestyle='--')
        ax[1,0].set_title(f'Neuron 12, R2 score={r2_scores[12]:.2f}')
        ax[1,0].legend()

        ax[1,1].plot(truth[16,:], label='traces', color='b')
        ax[1,1].plot(preds[16,], label='predictions', color='r', linestyle='--')
        ax[1,1].set_title(f'Neuron 16, R2 score={r2_scores[16]:.2f}')
        ax[1,1].legend()


        plt.tight_layout()
        plt.show()

    def r2_convergence(self):
        for experiment_id in self.experiment_ids:
            avg_r2 = []
            for epoch in range(50):
                r2_scores_path = f'../output/{experiment_id}/{self.modelname}/{self.conv}/{self.norm}/test_run/r2_scores_epoch_{epoch}.npy'
                
                if not os.path.exists(r2_scores_path):
                    print(f"Warning: R2 scores not found for Epoch {epoch}, Experiment ID {experiment_id}, Model {self.modelname}, {self.conv}, {self.norm}.")
                    continue

                r2_scores = np.load(r2_scores_path, allow_pickle=True)
                avg_r2.append(np.mean(r2_scores))
            plt.title(f'Convergence of R2 scores, model={self.modelname}')
            plt.plot(avg_r2)
            plt.show()

# Usage:
# [500964514, 501498760, 501794235, 704826374, 681674286]
deconv=True
norm=True
plotter = Plots([704826374], 'CRNN', deconv, norm, ['train', 'test'])
plotter.plot_histogram_sbs()
# plotter = Plots([704826374], 'CRNN', deconvolve=False, normalize=True, data_type='train')
# plotter.plot_histogram()
# p1 = Plots([501498760], 'mobilenet', deconvolve=False, normalize=True)
# p2 = Plots([501498760], 'CNN', deconvolve=False, normalize=True)
# p3 = Plots([501498760], 'CRNN', deconvolve=False, normalize=True)
# p1.plot_histogram()
# p2.plot_histogram()
# p3.plot_histogram()
# p1.plot_predictions()
# p2.plot_predictions()
# p3.plot_predictions()
# p1.r2_convergence()
# p2.r2_convergence()
# p3.r2_convergence()

#     # def plot_histogram(self):
#     #     for experiment_id in self.experiment_ids:
#     #         preds, truth = self.load_data(experiment_id)
            
#     #         if preds is None or truth is None:
#     #             continue

#     #         r2_scores = [r2_score(t, p) for t, p in zip(truth, preds)]

#     #         if np.any(np.isnan(r2_scores)) or np.any(np.isinf(r2_scores)):
#     #             print(f"Warning: Invalid R^2 values detected for experiment_id {experiment_id}")
#     #             r2_scores = r2_scores[~np.isnan(r2_scores)]
#     #             r2_scores = r2_scores[~np.isinf(r2_scores)]

#     #         auto_bins = np.histogram_bin_edges(r2_scores, bins='auto')

#     #         # Adding a bin for all values below -0.1
#     #         fixed_lower_limit = -0.3
#     #         bins = np.concatenate(([fixed_lower_limit], [-0.1], auto_bins[auto_bins > -0.1]))

#     #         fig, ax = plt.subplots()
#     #         ax.hist(r2_scores, bins=bins, edgecolor='black')
#     #         ax.set_title(f'Train Data epoch 20: Distribution of R^2. Model={self.modelname}, {self.conv}, {self.norm}')
#     #         ax.set_xlabel('R^2 Score')
#     #         ax.set_ylabel('Frequency')
#     #         current_ticks = plt.xticks()[0]  # Get the current x ticks
#     #         new_ticks = [fixed_lower_limit] + list(current_ticks[1:])  # Replace the first tick
#     #         plt.xticks(new_ticks, ['≤' + str(fixed_lower_limit)] + ["{:.2f}".format(tick) for tick in new_ticks[1:]])
#     #         # ax.set_xlim(-0.3,1)
#     #         plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# import os


# class plots:
#     def __init__(self):

#         self.preds_crnn = np.load(f'../output/{experiment_id}/CRNN/{conv}/{norm}/test_run/predictions.npy')
#         self.preds_mn = np.load(f'../output/{experiment_id}/CNN/{conv}/{norm}/test_run/predictions.npy')
#         self.preds_cnn = np.load(f'../output/{experiment_id}/CNN/{conv}/{norm}/test_run/predictions.npy')

#         train_proportion = 0.8


#         truth = np.load(f'../data/{experiment_id}/norm_responses.npy')
#         n_train = int(len(truth[0,:]) * train_proportion)
#         n_test = len(truth[0,:]) - n_train
#         truth = truth[:, :n_train]

#         pass


#     def plot_histogram(r2_scores):

#         fig, ax = plt.subplots()  # Create a new figure and axis for this epoch
#         ax.hist(r2_scores, bins='auto', edgecolor='black')  
#         ax.set_title(f'Train Data epoch 20: Distribution of R^2 Scores Across {len(r2_scores)} Neurons')
#         ax.set_xlabel('R^2 Score')
#         ax.set_ylabel('Frequency')
#         plt.show()

#     def plot_predictions(truth, pred):

#         fig, ax = plt.subplots(2,2)
#         r2_scores = {}
#         for neuron in range(truth.shape[0]):
#             r2_scores[f'{neuron}'] = r2_score(truth[neuron, :], pred[neuron, :])
#         r2_scores = list(r2_scores.values())

#         ax[0,0].plot(truth[4,:], label='traces', color='b')
#         ax[0,0].plot(pred[4,], label='predictions', color='r', linestyle='--')
#         ax[0,0].set_title(f'Neuron 4, R2 score={r2_scores[4]:.2f}')
#         ax[0,0].legend()

#         ax[0,1].plot(truth[8,:], label='traces', color='b')
#         ax[0,1].plot(pred[8,], label='predictions', color='r', linestyle='--')
#         ax[0,1].set_title(f'Neuron 8, R2 score={r2_scores[8]:.2f}')
#         ax[0,1].legend()

#         ax[1,0].plot(truth[12,:], label='traces', color='b')
#         ax[1,0].plot(pred[12,], label='predictions', color='r', linestyle='--')
#         ax[1,0].set_title(f'Neuron 12, R2 score={r2_scores[12]:.2f}')
#         ax[1,0].legend()

#         ax[1,1].plot(truth[16,:], label='traces', color='b')
#         ax[1,1].plot(pred[16,], label='predictions', color='r', linestyle='--')
#         ax[1,1].set_title(f'Neuron 16, R2 score={r2_scores[16]:.2f}')
#         ax[1,1].legend()


#         plt.tight_layout()
#         plt.show()

#     def r2_convergence(experiment_id):
#         avg_r2 = []
#         for epoch in range(50):
#             r2_scores = np.load(f'../output/{experiment_id}/CNN/{conv}/{norm}/test_run/r2_scores_epoch_{epoch}.npy', allow_pickle=True)
#             avg_r2.append(np.mean(r2_scores))
#         print(len(avg_r2))
#         plt.plot(avg_r2)
#         plt.show()


# experiment_id = 704826374#[500964514, 501498760, 501794235, 704826374, 681674286]

# norm = 'normalized'
# conv = 'deconvolved'
# model = 'mobilenet'


# plot_predictions(truth, preds_mn)

# experiment_ids = [704826374]
# for exp_id in experiment_ids:
#     r2_convergence(exp_id)
