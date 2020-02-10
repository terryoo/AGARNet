import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow.contrib.slim as slim

def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
	tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
	tmp *= scale
	return x + tmp

def resBlock3D(x,channels=64,kernel_size=[3,3,3],scale=1):
    tmp = tf.layers.conv3d(x,channels,kernel_size,padding='same',name='conv1')
    tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv3d(tmp, channels, kernel_size, padding='same', name='conv2')
    tmp *= scale
    return x + tmp

def resBlock_generating2(x,generated_kernel,batch_size=16,channels=128,kernel_size=[3,3],scale=1):
    tmp = slim.conv2d(x, channels, kernel_size, activation_fn=None)
    tmp = tf.nn.relu(tmp)
    tmp = slim.conv2d(tmp, channels, kernel_size, activation_fn=None)
    with tf.variable_scope('Split_Convolution'):
        gated_results = tf.multiply(tmp,tf.nn.sigmoid(generated_kernel),name = 'multiply')
    tmp = scale * tf.nn.tanh(gated_results)
    return x + tmp

def resBlock3D_generating4(x,qp_generated_kernel,qp_table_generated_kernel, channels=64,kernel_size=[3,3,3],scale=1):
    with tf.variable_scope('resblock3D_generating4'):
        tmp = tf.layers.conv3d(x,channels,kernel_size,padding='same',name='conv1')
        tmp = tf.nn.relu(tmp)
        tmp = tf.layers.conv3d(tmp, channels, kernel_size, padding='same', name='conv2')
        with tf.variable_scope('Gating_Generating'):
            qp_gated = tf.nn.sigmoid(tf.multiply(qp_generated_kernel,x,name = 'qp_gated'))
            qp_gated = tf.layers.conv3d(qp_gated,channels,[1,1,1],padding='same',name='qp_gated_conv1')
            qp_table_gated = tf.nn.sigmoid(tf.multiply(qp_table_generated_kernel,x,name='qp_table_gated'))
            qp_table_gated = tf.layers.conv3d(qp_table_gated, channels, [1, 1, 1], padding='same', name='qp_table_gated_conv1')
            overall_qp = tf.nn.sigmoid(tf.multiply(qp_gated,qp_table_gated,name='multiply_qp_qptable'))
            overall_qp = tf.layers.conv3d(overall_qp, channels, [1, 1, 1], padding='same',
                                              name='overall_qp_conv1')
    return x + scale*tf.multiply(tmp,overall_qp)

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data
def load_images_2d(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(im.size[1], im.size[0])
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')

def save_images_color(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('RGB')
    im.save(filepath, 'png')

def cal_psnr(im1, im2):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def jpeg_qtable(quality,tnum=0):
    dct_order =  np.array([1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33,
             41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50,
             43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
             45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55,
             62, 63, 56, 64]) - 1;
    if quality < 0 or quality == 0:
        quality = 1
    elif quality > 100:
        quality = 100
    elif quality < 50:
        quality = 5000 / quality
    else:
        quality = 200 - quality * 2

    if tnum == 0:
        t =np.array([16,  11,  10,  16,  24,  40,  51,  61,
            12,  12,  14,  19,  26,  58,  60,  55,
            14,  13,  16,  24,  40,  57,  69,  56,
            14,  17,  22,  29,  51,  87,  80,  62,
            18,  22,  37,  56,  68, 109, 103,  77,
            24,  35,  55,  64,  81, 104, 113,  92,
            49,  64,  78,  87, 103, 121, 120, 101,
            72,  92,  95,  98, 112, 100, 103,  99])
    else:
        t = np.zeros((8,8))
    t = np.floor((t*quality+50)/100)
    t = np.clip(t,1,255)
    a_t = np.zeros((len(t)))
    for i in range(len(t)):
        a_t[i] = t[dct_order[i]]
    return a_t

def four_part(jpeg_image,clean_image,border_line):

    img_size = np.shape(clean_image)
    partial_h = np.int32(
        np.floor(img_size[1] / 2 + border_line) + (8 - np.mod(np.floor(img_size[1] / 2 + border_line), 8)))
    partial_w = np.int32(
        np.floor(img_size[2] / 2 + border_line) + (8 - np.mod(np.floor(img_size[2] / 2 + border_line), 8)))
    p_jpeg_image = np.zeros((4, partial_h, partial_w, 1))
    p_jpeg_image[0,:, :, :] = jpeg_image[:, 0:partial_h, 0:partial_w, :]
    p_jpeg_image[1,:, :, :] = jpeg_image[:, 0:partial_h, img_size[2] - partial_w:img_size[2], :]
    p_jpeg_image[2,:, :, :] = jpeg_image[:, img_size[1] - partial_h:img_size[1], 0:partial_w, :]
    p_jpeg_image[3,:, :, :] = jpeg_image[:, img_size[1] - partial_h:img_size[1],
                               img_size[2] - partial_w:img_size[2], :]

    return  p_jpeg_image


def inv_four_part(poutput_clean_image, clean_image, border_line):
    output_clean_image = np.zeros_like(clean_image)
    weight_clean_image = np.zeros_like(clean_image)

    img_size = np.shape(clean_image)
    partial_h = np.int32(
        np.floor(img_size[1] / 2 + border_line) + (8 - np.mod(np.floor(img_size[1] / 2 + border_line), 8)))
    partial_w = np.int32(
        np.floor(img_size[2] / 2 + border_line) + (8 - np.mod(np.floor(img_size[2] / 2 + border_line), 8)))

    output_clean_image[:, 0:partial_h, 0:partial_w, :] += poutput_clean_image[0,:, :, :]
    weight_clean_image[:, 0:partial_h, 0:partial_w, :] += np.ones_like(poutput_clean_image[0,:, :, :])

    output_clean_image[:, 0:partial_h, img_size[2] - partial_w:img_size[2], :] += poutput_clean_image[1,:, :, :]
    weight_clean_image[:, 0:partial_h, img_size[2] - partial_w:img_size[2], :] += np.ones_like(
        poutput_clean_image[1,:, :, :])

    output_clean_image[:, img_size[1] - partial_h:img_size[1], 0:partial_w, :] += poutput_clean_image[2,:, :, :]
    weight_clean_image[:, img_size[1] - partial_h:img_size[1], 0:partial_w, :] += np.ones_like(
        poutput_clean_image[2,:, :, :])

    output_clean_image[:, img_size[1] - partial_h:img_size[1], img_size[2] - partial_w:img_size[2],
    :] += poutput_clean_image[3,:, :, :]
    weight_clean_image[:, img_size[1] - partial_h:img_size[1], img_size[2] - partial_w:img_size[2],
    :] += np.ones_like(poutput_clean_image[3,:, :, :])

    output_clean_image = np.divide(output_clean_image, weight_clean_image)

    return output_clean_image

def four_part_est(jpeg_image, jpeg_batch_qp, clean_image,border_line):

    img_size = np.shape(clean_image)
    partial_h = np.int32(
        np.floor(img_size[1] / 2 + border_line) + (8 - np.mod(np.floor(img_size[1] / 2 + border_line), 8)))
    partial_w = np.int32(
        np.floor(img_size[2] / 2 + border_line) + (8 - np.mod(np.floor(img_size[2] / 2 + border_line), 8)))
    p_jpeg_image = np.zeros((4, partial_h, partial_w, 1))
    p_qp = np.zeros((4, partial_h, partial_w, 1))

    p_jpeg_image[0,:, :, :] = jpeg_image[:, 0:partial_h, 0:partial_w, :]
    p_jpeg_image[1,:, :, :] = jpeg_image[:, 0:partial_h, img_size[2] - partial_w:img_size[2], :]
    p_jpeg_image[2,:, :, :] = jpeg_image[:, img_size[1] - partial_h:img_size[1], 0:partial_w, :]
    p_jpeg_image[3,:, :, :] = jpeg_image[:, img_size[1] - partial_h:img_size[1],
                               img_size[2] - partial_w:img_size[2], :]

    p_qp[0,:, :, :] = jpeg_batch_qp[0:partial_h, 0:partial_w, :]
    p_qp[1,:, :, :] = jpeg_batch_qp[0:partial_h, img_size[2] - partial_w:img_size[2], :]
    p_qp[2,:, :, :] = jpeg_batch_qp[img_size[1] - partial_h:img_size[1], 0:partial_w, :]
    p_qp[3,:, :, :] = jpeg_batch_qp[img_size[1] - partial_h:img_size[1], img_size[2] - partial_w:img_size[2], :]

    return  p_qp, p_jpeg_image