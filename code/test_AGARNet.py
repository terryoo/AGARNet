
from six.moves import xrange
from utils import *
from model import *
from glob import glob

import numpy as np
import tensorflow as tf
import imageio

qp_table = np.load('QP_Table.npy')

class deblocker(object):

    def __init__(self, sess, batch_size = 8,input_c_dim = 1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.feature_size = 32
        self.batch_size = batch_size
        DCTmtx = np.load('DCTmtx.npy')
        DCTmtx = DCTmtx[:, :, None, :].astype(np.float32)
        # build model

        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='jpeg_image')
        self.gt = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='gt')
        self.qp = tf.placeholder(tf.float32, [None, None, None, 1],
                                 name='qp')
        self.QP_Table = tf.placeholder(tf.float32, [None, None, None, 64],
                                 name='QP_Table')

        self.Y,self.idct_recon = AGARNet(self.Y_,DCTmtx,self.qp, self.QP_Table,self.feature_size,batch_size = self.batch_size,output_channels=1)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")


    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_folder):
        """ Test Deblocker with Compressed Quality Factor"""

        # init variables
        print("[*] Testing...")
        test_folder = test_files + '/jpeg/'
        ori_folder = test_files + '/ori/'
        test_set = ['LIVE1', 'classic5']
        jpeg_qp_list = [10, 20, 30, 40, 50, 60, 70, 80]
        border_line = 80
        tf.initialize_all_variables().run()
        # assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for jpeg_idx in range(len(jpeg_qp_list)):
            jpeg_qp = jpeg_qp_list[jpeg_idx]
            jpeg_batch_qp = np.zeros((1, 1, 1, 1))
            jpeg_batch_qp[0, 0, 0, 0] = jpeg_qp / 100 # applied qp. It can be changed with user hyper parameter such as 10, 20 ...
            jpeg_batch_qp_table = qp_table[np.int32(jpeg_qp/10-1)] / 255.0
            jpeg_batch_qp_table = jpeg_batch_qp_table[None,None,None,:]
            for test_set_idx in range(len(test_set)):
                full_path = test_folder + str(jpeg_qp) + '/' + test_set[test_set_idx] + '/*.png'
                test_data_jpeg = glob(full_path)
                psnr_sum = 0
                mid_save_folder = save_folder + str(jpeg_qp) + '/' +  test_set[test_set_idx]
                if not os.path.exists(mid_save_folder):
                    os.makedirs(mid_save_folder)
                for idx in xrange(len(test_data_jpeg)):
                    file_name = test_data_jpeg[idx].split('/')[-1]
                    jpeg_image = load_images(test_data_jpeg[idx]).astype(np.float32) / 255.0
                    clean_path = ori_folder + test_set[test_set_idx] + '/' + file_name
                    clean_image = load_images(clean_path).astype(np.float32) / 255.0

                    p_jpeg_image = four_part(jpeg_image,clean_image,border_line) # patch process due to dimension matching (H % 8 = 0 and W % = 0)
                    poutput_clean_image = self.sess.run(
                        self.Y,
                        feed_dict={self.Y_: p_jpeg_image,
                                   self.qp: jpeg_batch_qp,
                                   self.QP_Table:jpeg_batch_qp_table})
                    output_clean_image = inv_four_part(poutput_clean_image, clean_image, border_line) # inverse patch process due to dimension matching

                    groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
                    outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
                    # calculate PSNR
                    psnr = cal_psnr(groundtruth, outputimage)
                    psnr_sum += psnr
                    full_ave_folder = mid_save_folder + '/' + file_name
                    imageio.imwrite(full_ave_folder, outputimage[0, :, :, 0])

                avg_psnr = psnr_sum / len(test_data_jpeg)
                print("--- Test ---- %3d Average PSNR %.2f  ---" % (jpeg_qp, avg_psnr))







