import os

work_dir = '/home/lawy623/dl_proj' # Work from where. Default runs at 'src/..''
data_path = os.path.join(work_dir, 'raw_data') # Main folder storing the raw data

class Config: pass

"""
during execution:
modifying anything other than the config.mode, config.model_path is not recommended
"""

config_dict = {
    # Data
    'train_path': os.path.join(work_dir, 'voxceleb', 'train'),     # train dataset directory
    'valid_path': os.path.join(work_dir, 'voxceleb', 'valid'),     # train dataset directory
    'test_path': os.path.join(work_dir, 'voxceleb', 'test'),       # test dataset directory
    'model_path': os.path.join(work_dir, 'model'),                 # model save paths

    # Preprocessing
    'sr': 16000,                                            # sample rate
    'nfft': 512,                                            # fft kernel size
    'window': 0.025,                                        # window length (ms)
    'hop': 0.01,                                            # hop size (ms)
    'max_frames': 180,                                      # number of max frames
    'min_frames': 140,                                      # number of min frames
    'mels':40,                                              # number of mel banks. That will be the first dim of the np file.

    # Model. For LSTM model here.
    'nb_hidden': 384,                                       # number of hidden units
    'nb_proj': 128,                                         # number of projection units
    'nb_layers': 3,                                         # number of LSTM_Projection layers
    'loss':'softmax',                                       # loss function to use. 'softmax' or 'contrast'

    # Data Buffer. Change based on memory
    'K1': 10,                                               # times of N that buffer reads in
    'K2': 3,                                                # times of M that buffer reads in
    'flush_thres': 25,                                      # Freq to flush the buffer (thres*K1*K2)
    # Session
    'mode': 'train',                                        # train or test
    'N': 40,                                                # number of speakers per batch (default 16)
    'M': 15,                                                 # number of utterances per speaker (default 7)
    'lr': 0.01,                                             # initial learning rate
    'decay': 10000,                                        # num of iterations that lr decay by half
    'optim': ['sgd',                                        # type of the optimizer ('sgd', 'adam', 'rmsprop')
              {'beta1': 0.5, 'beta2': 0.9}],                # additional parameters (for 'adam', 'rmsprop')
    'nb_iters': 1e5,                                        # max iterations
    'save_iters': 2000,                                    # iteration of saving checkpoint
    'show_loss': 100,                                       # iteration to show the loss.
    'verbose': True,                                        # print training detail

    # Debug
    'debug': False,                                          # turn on debug info output
}

config = Config()
config.__dict__.update(config_dict)
