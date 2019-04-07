import tensorflow as tf
import os
import shutil
from model import *
from config import *
import argparse

# Reading args from user input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help="model type(train/test)")
parser.add_argument('--model_path', type=str, default=config.model_path, help="model path")
parser.add_argument('--iter', type=str, default=0, help="model checkpoint id")
args = parser.parse_args()

config.mode = args.mode
config.model_path = args.model_path

if __name__ == "__main__":
    tf.reset_default_graph()
    sess = tf.Session()
    model = Model()
    if config.mode == 'train':
        print("\nTraining Session")
        if os.path.exists(config.model_path):
            shutil.rmtree(config.model_path)
        os.makedirs(config.model_path)
        model.train(sess)
    elif config.mode == 'test':
        print("\nTest Session")
        model_name = 'model.ckpt-' + args.iter
        if os.path.isdir(config.model_path):
            model.test(sess, os.path.join(config.model_path, 'check_point', model_name))
        else:
            raise AssertionError("model path doesn't exist!")
    else:
        raise AssertionError("model path doesn't exist!")
