import tensorflow as tf
import numpy as np
import os
import math
import random
from config import *

def random_batch(speaker_num = config.N, utter_num = config.M, path = config.train_path, shuffle = True, utter_start = 0):
    """
    Get the batch from desired path.
    No use in this proj. Too much I/O. Just for reference.
    """
    np_file_list = os.listdir(path)
    total_speaker = len(np_file_list)

    if shuffle:
        selected_files = random.sample(np_file_list, speaker_num)  # select random N speakers
    else:  ## This is for testing. Actually I am not sure about the logic.
        selected_files = np_file_list[:speaker_num]                # select first N speakers

    # N speaker, each with M utter. Each Utter is with size (n_mel, frames)
    utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker
        if shuffle:
            utter_index = np.random.randint(0, utters.shape[0], utter_num)   # select M utterances per speaker
            utter_batch.append(utters[utter_index])       # each speakers utterance [M, n_mels, frames] is appended
        else:
            utter_batch.append(utters[utter_start: utter_start+utter_num]) ## Specify a start point

    utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]

    if config.mode == 'train':
        frame_slice = np.random.randint(140,181)          # for train session, random slicing of input batch
        utter_batch = utter_batch[:,:,:frame_slice]
    else:
        utter_batch = utter_batch[:,:,:160]               # for test session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, NM, n_mels]

    return utter_batch
