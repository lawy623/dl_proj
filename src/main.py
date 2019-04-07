import tensorflow as tf
import os
import shutil
from model import *
from config import *
import argparse

# Reading args from user input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help="model type(train/test/infer)")
parser.add_argument('--model_path', type=str, default=config.model_path, help="model path")
args = parser.parse_args()

config.mode = args.mode
config.model_path = args.model_path

if __name__ == "__main__":
    tf.reset_default_graph()
    sess = tf.Session()
    model = Model()
    if config.mode == 'train':
        print("\nTraining Session")
        os.makedirs(config.model_path)
        model.train(sess, config.model_path)
    elif config.mode == 'test':
        print("\nTest Session")
        if os.path.isdir(config.model_path):
            model.test(sess, os.path.join(config.model_path, 'check_point', 'model.ckpt-0'))
        else:
            raise AssertionError("model path doesn't exist!")
    else:
        print("\nInfer Session")
        model.infer(sess, path=os.path.join(config.model_path, 'check_point', 'model.ckpt-0'), thres=0.57)
