import tensorflow as tf
import numpy as np
import os
import math
import random
from config import *

def get_test_batch(speaker_num = config.N, utter_num = config.M, path = config.test_path, utter_start = 0):
    """
    Get the batch from desired path.
    No use in this proj. Too much I/O. Just for reference.
    """
    if config.verbose:
        print("Getting data from: ", path)

    np_file_list = os.listdir(path)
    selected_files = np_file_list[:speaker_num]                # select first N speakers

    # N speaker, each with M utter. Each Utter is with size (n_mel, frames)
    utter_batch = []
    for file in selected_files:
        utters = np.load(os.path.join(path, file))        # load utterance spectrogram of selected speaker
        utter_batch.append(utters[utter_start: utter_start+utter_num]) ## Specify a start point

    utter_batch = np.concatenate(utter_batch, axis=0)     # utterance batch [batch(NM), n_mels, frames]

    utter_batch = utter_batch[:,:,:160]               # for valid/test session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, NM, n_mels]
    
    return utter_batch
