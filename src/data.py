from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import librosa
from config import *

voxceleb_path = os.path.join(data_path, 'voxceleb')

avg_frames = int((config.max_frames + config.min_frames) / 2.0)
hop_frames = int(avg_frames / 2)
utter_min_len = (config.max_frames * config.hop + config.window) * config.sr

def wav2spectro(utter_path):
    """
    Process a wav(utter_path) into a
    """
    utterances_spec = []
    utter, sr = librosa.core.load(utter_path, config.sr)  ## Load the path into real utter file.
    intervals = librosa.effects.split(utter, top_db=30)   ## Split into non-slice intervals, those are breakers. top_db is bound for silence. Follow the open source work.
    for interval in intervals:
        if (interval[1] - interval[0]) > utter_min_len: ## Only long enough intervals.
            utter_part = utter[interval[0]:interval[1]]
            S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                  win_length=int(config.window * sr), hop_length=int(config.hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=config.mels)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)

            if config.mode != 'infer':
                '''
                NOTE: each interval in utterance only extracts 2 samples
                '''
                utterances_spec.append(S[:, :config.max_frames])
                utterances_spec.append(S[:, -config.max_frames:])
            else:
                max_steps = int((S.shape[1] - avg_frames) / hop_frames) + 1
                for i in range(max_steps):
                    utterances_spec.append(S[:, hop_frames * i : hop_frames * i + avg_frames])
    return utterances_spec

def save_spectrogram(speakers, train_path, valid_path, test_path, test_split, valid_split):
    """
    Save the speaker files in to preprocessed .npy files.
    :param speakers: [speaker_1, speaker_2, ..., speaker_n] -> each speaker:[utter_1, utter_2, ..., utter_m](these are paths)
    :param train_path: Save preprocessed train file location.
    :param test_path:  Save preprocessed train file location.
    :param test_path:  Save preprocessed train file location.
    :param test_split: ratio of test
    :param valid_split: ratio of valid
    """
    print("Train writes to: %s", train_path)
    print("Valid writes to: %s", valid_path)
    print("Test writes to: %s", test_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    total_speaker_num = len(speakers)

    test_speaker_num = int(total_speaker_num * test_split)
    valid_speaker_num = int(total_speaker_num * valid_split)
    train_speaker_num = int(total_speaker_num - test_speaker_num - valid_speaker_num)

    print('total speaker number : {}'.format(len(speakers)))
    nb_utters = np.array([len(x) for x in speakers])
    print('min nb_utterances: {}, max_nb_utterances: {}'.format(np.min(nb_utters), np.max(nb_utters)))
    print('Train : {}| Valid : {}| Test : {}'.format(train_speaker_num, valid_speaker_num, test_speaker_num))

    # Do some random permutation.
    permu = np.random.permutation(len(speakers))
    np.save(os.path.join(work_dir, 'permute.npy'), permu)

    for i, idx in enumerate(tqdm(permu)):
        files = speakers[idx]
        print(idx)
        utterances_spec = []
        for utter_path in files: # Make each wav file into a spec file.
            utterances_spec.extend(wav2spectro(utter_path))
        print("len is {}".format(len(utterances_spec)))

        if i < train_speaker_num:
            np.save(os.path.join(train_path, 'speaker_{}.npy'.format(i)), utterances_spec)
        elif i < train_speaker_num + valid_speaker_num:
            np.save(os.path.join(valid_path, 'speaker_{}.npy'.format(i - train_speaker_num)), utterances_spec)
        else:
            np.save(os.path.join(test_path, 'speaker_{}.npy'.format(i - train_speaker_num - valid_speaker_num)), utterances_spec)

def save_spectrogram_voxceleb(test_split=0.1, valid_split=0.1):
    """
    Preprocess the Voxcelev dataset. Should have it downloaded.
    Do not follow the paper. Only use training dataset and separate it.
    Models are compared using the same test set.
    """
    print('Processing Voxceleb dataset...')

    audio_path = os.path.join(voxceleb_path, 'wav')

    train_path = config.train_path
    valid_path = config.valid_path
    test_path = config.test_path

    speakers = [] # That is the utter for all the speakers.
    for folder in os.listdir(audio_path):
        speaker_path = os.path.join(audio_path, folder) # Each folder is for one speaker. Identify by idxxxxx.
        utters = [] # That is all the utters for one single speaker
        for sub_folder in os.listdir(speaker_path):
            sub_utter_path = os.path.join(speaker_path, sub_folder)
            for wav_fname in os.listdir(sub_utter_path):
                utters.append(os.path.join(sub_utter_path, wav_fname))
        speakers.append(utters)

    save_spectrogram(speakers, train_path, valid_path, test_path, test_split, valid_split)

if __name__ == '__main__':
    save_spectrogram_voxceleb()
