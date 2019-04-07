__author__ = 'Bingqing Wei'
import numpy as np
import math
import random
import gc
from config import *
from data import wav2spectro

class Buffer:
    def __init__(self, flush_thres=1.5):
        """
        Param flush_thres: should be greater than 1
        """
        if not config.mode == 'train': flush_thres = 0.2

        self.flush_thres = flush_thres
        self.K_N = config.K1 * config.N
        self.K_M = config.K2 * config.M
        self.count_down = int(math.sqrt(self.K_N * self.K_M * flush_thres))
        self.counter = 0

        if config.mode == 'train':
            self.data_path = os.path.join(config.train_path)
        else:
            self.data_path = os.path.join(config.train_path)
        self.buffer = None
        self.flush()

    def update(self, npy_list):
        """
        Update the list to load in Memory.
        Param npy_list: list of .npy files that are in the corresponding dir.
        Return: whether to flush the buffer
        """
        self.K_N = min(self.K_N, len(npy_list))
        self.count_down = int(math.sqrt(self.K_N * self.K_M * self.flush_thres))
        self.counter = 0
        return self.K_N != len(npy_list) or self.buffer is None

    def flush(self):
        npy_list = os.listdir(self.data_path)
        do_flush = self.update(npy_list)
        if not do_flush: return

        if config.debug: print('flushing buffer')

        del self.buffer
        gc.collect()
        self.buffer = []

        sel_speakers = random.sample(npy_list, self.K_N)
        for file in sel_speakers:
            utters = np.load(os.path.join(self.data_path, file))
            utter_index = np.random.randint(0, utters.shape[0], self.K_M)
            self.buffer.append(utters[utter_index])

        self.buffer = np.concatenate(self.buffer, axis=0)

    def sample(self, speaker_num=config.N, utter_num=config.M, sel_speakers=None, frames=None):
        if sel_speakers is None:
            sel_speakers = random.sample(range(self.K_N), speaker_num)

        batch = []
        for i in sel_speakers:
            utters = self.buffer[i * self.K_M:(i + 1) * self.K_M, :]
            utter_index = np.random.randint(0, utters.shape[0], utter_num)
            batch.append(utters[utter_index])
        batch = np.concatenate(batch, axis=0)
        if config.mode == 'train':
            if frames is None:
                frames = np.random.randint(config.min_frames, config.max_frames)
            batch = batch[:, :, :frames]
        else:
            if frames is None:
                frames = int((config.min_frames + config.max_frames) / 2)
            batch = batch[:, :, :frames]

        # shape = (frames, N * M, 40)
        batch = np.transpose(batch, axes=(2, 0, 1))
        self.counter += 1
        if self.counter >= self.count_down:
            self.flush()

        return batch, sel_speakers

# The global buffer
buffer = Buffer()

def reset_buffer():
    """
    reset our global buffer
    """
    global buffer
    buffer = Buffer()

def random_batch(speaker_num=config.N, utter_num=config.M, selected_files=None, frames=None):
    """
    Sample from the global buffer.
    """
    return buffer.sample(speaker_num, utter_num, sel_speakers=selected_files, frames=frames)
