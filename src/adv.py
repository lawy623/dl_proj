__author__ = 'Bingqing Wei'
import numpy as np
import math
import random
import gc
from config import *
from data import wav2spectro

class Buffer:
    def __init__(self):
        """
        Param flush_thres: should be greater than 1
        """
        if not config.mode == 'train':
            return

        self.flush_thres = config.flush_thres
        self.K_N = config.K1 * config.N
        self.K_M = config.K2 * config.M
        self.count_down = int( config.K1 * config.K2 * self.flush_thres )
        self.counter = 0

        if config.mode == 'train':
            self.data_path = os.path.join(config.train_path)
        else:
            self.data_path = os.path.join(config.test_path)

        if config.verbose:
            print("Buffer start loading from {}".format(self.data_path))

        self.buffer = None
        self.flush()

    def update(self, npy_list):
        """
        Update the list to load in Memory. Flush when the buffer is not large, or buffer is empty.
        Param: npy_list - list of .npy files that are in the corresponding dir.
        Return: whether to flush the buffer
        """
        self.K_N = min(self.K_N, len(npy_list)) # If no much data, we load them all.
        self.count_down = int( ((self.K_N/config.N) * (self.K_M/config.M)) * self.flush_thres )
        self.counter = 0
        if config.verbose:
            print("Flushing frequence: ", self.count_down)
        return self.K_N != len(npy_list) or self.buffer is None

    def flush(self):
        """
        Decide Whether to flush the data. If yes, flush the buffer by loading random files.
        """
        npy_list = os.listdir(self.data_path)
        # Flush or not
        do_flush = self.update(npy_list)
        if not do_flush: return

        if config.verbose: print('flushing buffer')

        del self.buffer
        gc.collect()
        self.buffer = []

        sel_speakers = random.sample(npy_list, self.K_N)  # Pick K_N random speakers
        for file in sel_speakers:
            utters = np.load(os.path.join(self.data_path, file))
            utter_index = np.random.randint(0, utters.shape[0], self.K_M)  # Pick K_M random utters,
            self.buffer.append(utters[utter_index])

        self.buffer = np.concatenate(self.buffer, axis=0)  ## The buffer contains K_N * K_M data in total.

    def sample(self, speaker_num=config.N, utter_num=config.M, sel_speakers=None, frames=None):
        """
        Sample from the buffer. After loading K_N*K_M batchs to the memory, it reduces time.
        """
        if sel_speakers is None:
            sel_speakers = random.sample(range(self.K_N), speaker_num)

        batch = []  ## Store the N*M input batch
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = np.random.randint(0, utters.shape[0], utter_num)
            batch.append(utters[utter_index])
        batch = np.concatenate(batch, axis=0)

        if config.mode == 'train': # Randomly sample [min_frames, max_frames] frames for each batch.
            if frames is None:
                frames = np.random.randint(config.min_frames, config.max_frames)
            batch = batch[:, :, :frames]
        else:  # For testing, avg [min_frames, max_frames] will be sampled.
            if frames is None:
                frames = int((config.min_frames + config.max_frames) / 2)
            batch = batch[:, :, :frames]

        # shape = (frames, N * M, n_mel)
        batch = np.transpose(batch, axes=(2, 0, 1))

        self.counter += 1
        if self.counter >= self.count_down:
            self.flush()

        return batch, sel_speakers

    def random_batch(self, speaker_num=config.N, utter_num=config.M, selected_files=None, frames=None):
        """
        Sample from the global buffer.
        Sample N speakers with M utterances each,
        from a total K_N speaker with K_M utterances buffer.
        """
        return self.sample(speaker_num, utter_num, sel_speakers=selected_files, frames=frames)
