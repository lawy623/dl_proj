__author__ = 'Bingqing Wei'
import os

work_dir = '/home/lawy623/dl_proj' # Work from where. Default runs at 'src/..''
data_path = os.path.join(work_dir, 'raw_data') # Main folder storing the raw data

class Config: pass

"""
during execution:
modifying anything other than the config.mode is not recommended
"""

config_dict = {
    # Data
    'train_path': os.path.join(work_dir, 'voxceleb', 'train'),     # train dataset directory
    'valid_path': os.path.join(work_dir, 'voxceleb', 'valid'),     # train dataset directory
    'test_path': os.path.join(work_dir, 'voxceleb', 'test'),       # test dataset directory
    'model_path': os.path.join(work_dir, 'model'),                 # model save paths
    'infer_path': os.path.join(work_dir, 'infer'),      # TODO: What is inferring real doing ??

    # Preprocessing
    'max_ckpts': 6,                                         # max checkpoints to keep. TODO: Any usage here??
    'sr': 16000,                                            # sample rate
    'nfft': 512,                                            # fft kernel size
    'window': 0.025,                                        # window length (ms)
    'hop': 0.01,                                            # hop size (ms)
    'max_frames': 180,                                      # number of max frames
    'min_frames': 140,                                      # number of min frames
    'mels':40,                                              # number of mel banks

    # Model
    'nb_hidden': 384,                                       # number of hidden units
    'nb_proj': 128,                                         # number of projection units
    'nb_layers': 3,                                         # number of LSTM_Projection layers
    'loss':'softmax',                                       # loss function to use. 'softmax' or 'contrast'

    # Session
    'mode': 'train',                                        # train or test
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
