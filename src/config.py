__author__ = 'Bingqing Wei'
import os

data_path = './raw_data' # Store the raw data
work_dir = '.' # Work from where. Default runs at 'src/..''

class Config: pass

"""
during execution:
modifying anything other than the config.mode is not recommended
"""

config_dict = {
    # Data
    'noise_path': os.path.join(work_dir, 'noise'),          # noise dataset directory

    'train_path': os.path.join(work_dir, 'voxceleb', 'train'),     # train dataset directory
    'test_path': os.path.join(work_dir, 'voxceleb', 'test'),       # test dataset directory
    'model_path': os.path.join(work_dir, 'model'),          # save paths
    'infer_path': os.path.join(work_dir, 'infer'),

    # Preprocessing
    'nb_noises': 64,                                        # how many of the noise files to choose
    'max_ckpts': 6,                                         # max checkpoints to keep
    'sr': 8000,                                             # sample rate
    'nfft': 512,                                            # fft kernel size
    'window': 0.025,                                        # window length (ms)
    'hop': 0.01,                                            # hop size (ms)
    'max_frames': 180,                                      # number of max frames
    'min_frames': 140,                                      # number of min frames
    'mels':40,

    # Model
    'nb_hidden': 384,                                       # number of hidden units
    'nb_proj': 128,                                         # number of projection units
    'nb_layers': 3,                                         # number of LSTM_Projection layers
    'loss':'softmax',

    # Session
    'mode': 'test',                                         # train or test
    'N': 16,                                                # number of speakers per batch
    'M': 7,                                                 # number of utterances per speaker
    'lr': 0.01,
    'optim': ['sgd',                                        # type of the optimizer
              {'beta1': 0.5, 'beta2': 0.9}],    # additional parameters
    'nb_iters': 1e5,                                        # max iterations
    'verbose': True,

    # Debug
    'debug': True,                                          # turn on debug info output
}


config = Config()
config.__dict__.update(config_dict)
