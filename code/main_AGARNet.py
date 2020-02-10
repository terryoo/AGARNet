import argparse
import h5py
from glob import glob
import numpy as np
import os
import tensorflow as tf
from test_AGARNet_EST_SP import deblocker
from utils import *
import scipy.misc as ms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint/reconstruction/', help='models are saved here')
parser.add_argument('--test_set', dest='test_set', default='./testset/', help='dataset for testing')
parser.add_argument('--result_dir', dest='result_dir', default='./results/estimation_SP/', help='test sample are saved here')
args = parser.parse_args()

def deblocker_test(denoiser):
    denoiser.test(args.test_set, ckpt_dir=args.ckpt_dir, save_folder= args.result_dir)

def main(_):

    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = deblocker(sess)
            deblocker_test(model)

    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = deblocker(sess)
            deblocker_test(model)

if __name__ == '__main__':
    tf.app.run()
