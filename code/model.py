import tensorflow as tf
import tensorflow.contrib.slim as slim
from six.moves import xrange
from utils import *

def DCTconv(input_, DCTmtx, name, stride=8, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse == True:
            scope.reuse_variables()
        w = tf.Variable(DCTmtx, trainable=False, name='weight')
        output = tf.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding="SAME")
        return output


def IDCTconv(input_, DCTmtx, output_shahpe, name, reuse=False):
    with tf.variable_scope(name) as scope:
        s = input_.get_shape().as_list()
        if reuse == True:
            scope.reuse_variables()
        input_ = tf.transpose(input_,[0,2,3,1])
        w = tf.Variable(DCTmtx, trainable=False, name='weight')
        output = tf.nn.conv2d_transpose(input_, w, output_shape=[output_shahpe[0], output_shahpe[1], output_shahpe[2], 1], strides=[1, 8, 8, 1],
                                        padding="VALID")
        return output

def AGARNet(input,DCTmtx, jpeg_qp,QPTable, feature_size,batch_size,output_channels=1):
    scaling_factor = 0.1
    input_shape = tf.unstack(tf.shape(input))
    DCTconv_input = DCTconv(input,DCTmtx,'dct_input')
    DCTconv_input = tf.transpose(DCTconv_input,[0,3,1,2])
    DCTconv_input = DCTconv_input[:,:,:,:,None]
    decoder_first = 5
    decoder_second = 10
    dct_feature_factor = 1
    dct_feature_size = 32 * dct_feature_factor

    with tf.variable_scope('Generating_CNN'):
        x = tf.layers.conv2d(jpeg_qp, 64, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense1')
        generated_kernel4 = tf.layers.conv2d(x, 4*feature_size, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense2')
        generated_kernel2 = tf.layers.conv2d(x, 2 * feature_size, [1, 1],
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                             padding='same', name='dense3')
        generated_kernel1 = tf.layers.conv2d(x,  feature_size, [1, 1],
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                             padding='same', name='dense4')
    with tf.variable_scope('Generating_DCT'):
        generated_dct2 = tf.layers.conv2d(jpeg_qp, 64, [1, 1],
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                          padding='same', name='dense5')
        generated_dct = tf.layers.conv2d(generated_dct2, 32, [1, 1],
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                         padding='same', name='dense6')
        generated_dct = generated_dct[:, None, :, :,  :]

    with tf.variable_scope('Generating_DCT_QPT'):
        generated_dct_qpt = tf.layers.conv2d(QPTable, 64, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense1')
        generated_dct_qpt = tf.nn.relu(generated_dct_qpt)
        generated_dct_qpt = tf.layers.conv2d(generated_dct_qpt,  64, [1, 1],
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                         padding='same', name='dense2')
        generated_dct_qpt = tf.transpose(generated_dct_qpt,[0,3,1,2])
        generated_dct_qpt =  generated_dct_qpt[:,:,:,:,None]
    with tf.variable_scope('Reconstruction'):
        with tf.variable_scope('DCT_Domain2'):
            with tf.variable_scope('block1'):
                x = tf.layers.conv3d(DCTconv_input,dct_feature_size,[3,3,3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name='conv1')
                conv_1 = x
            with tf.variable_scope('block2'):
                x = tf.layers.conv3d(x, dct_feature_size,[3, 3, 3], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same', name='conv1')

            for layers in xrange(3, 10+1):
                with tf.variable_scope('block%d' % layers):
                    x = resBlock3D_generating4(x,generated_dct,generated_dct_qpt,channels=dct_feature_size,kernel_size=[3,3,3],scale=1)

            x = tf.layers.conv3d(x, dct_feature_size, 1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv_1x1last_')
            x = tf.layers.conv3d(x, dct_feature_size, [3,3,3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same', name='conv_last')
            x = x + conv_1
            x = tf.layers.conv3d(x, output_channels, [3, 3, 3], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv_dct_last')
            idct_reconstruction = IDCTconv(x[:,:,:,:,0], DCTmtx,input_shape, 'idct')

        with tf.variable_scope('Pixel_Domain'):

            input_concat = tf.concat([idct_reconstruction,input],axis=3)

            with tf.variable_scope('Encoder_1x1'):
                x = tf.layers.conv2d(input, feature_size, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='conv1')
                _conv1 = x
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = x + _conv1
                conv_1 = x
                conv1_shape = tf.unstack(tf.shape(x))
                x = tf.nn.relu(x)
                x = tf.layers.average_pooling2d(x,[2,2],[2,2],padding='same',name='pooling')


            with tf.variable_scope('Encoder_2x2'):
                x = tf.layers.conv2d(x, 2*feature_size, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='conv1')
                _conv2 = x
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 2*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = x + _conv2
                conv_2 = x
                conv2_shape = tf.unstack(tf.shape(x))
                x = tf.nn.relu(x)
                x = tf.layers.average_pooling2d(x,[2,2],[2,2],'same',name='pooling')

            with tf.variable_scope('Encoder_4x4'):
                x = tf.layers.conv2d(x, 4*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv1')
                conv_3 = x
                for layers in xrange(0, 20):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel4,batch_size, 4*feature_size, scale=scaling_factor)
                x = slim.conv2d(x, 4*feature_size, [1, 1])
                x = x + conv_3

            with tf.variable_scope('Decoder_2x2'):
                x = tf.image.resize_bilinear(x,[conv2_shape[1],conv2_shape[2]],name='upsampled')
                x = tf.layers.conv2d(x, 4*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre1')
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 4*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = tf.concat([x,conv_2],axis=-1)
                x = tf.layers.conv2d(x, 2 * feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre2')
                for layers in xrange(decoder_first, decoder_second):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel2, batch_size, 2 * feature_size, scale=scaling_factor)

            with tf.variable_scope('Decoder_1x1'):
                x = tf.image.resize_bilinear(x, [conv1_shape[1], conv1_shape[2]], name='upsampled')
                x = tf.layers.conv2d(x, 2*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre1')
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 2*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = tf.concat([x, conv_1], axis=-1)
                x = tf.layers.conv2d(x, feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre2')
                for layers in xrange(decoder_first, decoder_second):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel1, batch_size, 1 * feature_size,
                                                 scale=scaling_factor)
                x = tf.layers.conv2d(x, output_channels, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_last')

            with tf.variable_scope('concat_images'):
                weighted_concat_inputs = tf.layers.conv2d(input_concat, output_channels, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), name='concat_image_weight', use_bias=False)
            x = x+weighted_concat_inputs

    return x,idct_reconstruction

def AGARNet_SP(input,DCTmtx, dct_qp, jpeg_qp, QPTable, feature_size,batch_size,output_channels=1):
    scaling_factor = 0.1
    input_shape = tf.unstack(tf.shape(input))
    DCTconv_input = DCTconv(input,DCTmtx,'dct_input')
    DCTconv_input = tf.transpose(DCTconv_input,[0,3,1,2])
    DCTconv_input = DCTconv_input[:,:,:,:,None]
    decoder_first = 5
    decoder_second = 10
    dct_feature_factor = 1
    dct_feature_size = 32 * dct_feature_factor

    with tf.variable_scope('Generating_CNN'):
        x = tf.layers.conv2d(jpeg_qp, 64, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense1')
        generated_kernel4 = tf.nn.avg_pool(tf.layers.conv2d(x, 4*feature_size, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense2'),[1,4,4,1],[1,4,4,1],padding='VALID')
        generated_kernel2 = tf.nn.avg_pool(tf.layers.conv2d(x, 2 * feature_size, [1, 1],
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                             padding='same', name='dense3'),[1,2,2,1],[1,2,2,1],padding='VALID')
        generated_kernel1 = tf.layers.conv2d(x,  feature_size, [1, 1],
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                             padding='same', name='dense4')
    with tf.variable_scope('Generating_DCT'):
        generated_dct2 = tf.layers.conv2d(dct_qp, 64, [1, 1],
                                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                          padding='same', name='dense5')
        generated_dct = tf.layers.conv2d(generated_dct2, 32, [1, 1],
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                         padding='same', name='dense6')
        generated_dct = generated_dct[:,None,  :, :, :]

    with tf.variable_scope('Generating_DCT_QPT'):
        generated_dct_qpt = tf.layers.conv2d(QPTable, 64, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='dense1')
        generated_dct_qpt = tf.nn.relu(generated_dct_qpt)
        generated_dct_qpt = tf.layers.conv2d(generated_dct_qpt,  64, [1, 1],
                                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                         padding='same', name='dense2')
        generated_dct_qpt = tf.transpose(generated_dct_qpt,[0,3,1,2])
        generated_dct_qpt =  generated_dct_qpt[:,:,:,:,None]
    with tf.variable_scope('Reconstruction'):
        with tf.variable_scope('DCT_Domain2'):
            with tf.variable_scope('block1'):
                x = tf.layers.conv3d(DCTconv_input,dct_feature_size,[3,3,3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name='conv1')
                conv_1 = x
            with tf.variable_scope('block2'):
                x = tf.layers.conv3d(x, dct_feature_size,[3, 3, 3], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same', name='conv1')

            for layers in xrange(3, 10+1):
                with tf.variable_scope('block%d' % layers):
                    x = resBlock3D_generating4(x,generated_dct,generated_dct_qpt,channels=dct_feature_size,kernel_size=[3,3,3],scale=1)

            x = tf.layers.conv3d(x, dct_feature_size, 1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv_1x1last_')
            x = tf.layers.conv3d(x, dct_feature_size, [3,3,3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same', name='conv_last')
            x = x + conv_1
            x = tf.layers.conv3d(x, output_channels, [3, 3, 3], kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),padding='same', name='conv_dct_last')
            idct_reconstruction = IDCTconv(x[:,:,:,:,0], DCTmtx,input_shape, 'idct')

        with tf.variable_scope('Pixel_Domain'):

            input_concat = tf.concat([idct_reconstruction,input],axis=3)

            with tf.variable_scope('Encoder_1x1'):
                x = tf.layers.conv2d(input, feature_size, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='conv1')
                _conv1 = x
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = x + _conv1
                conv_1 = x
                conv1_shape = tf.unstack(tf.shape(x))
                x = tf.nn.relu(x)
                x = tf.layers.average_pooling2d(x,[2,2],[2,2],padding='same',name='pooling')


            with tf.variable_scope('Encoder_2x2'):
                x = tf.layers.conv2d(x, 2*feature_size, [3, 3],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), padding='same',name ='conv1')
                _conv2 = x
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 2*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = x + _conv2
                conv_2 = x
                conv2_shape = tf.unstack(tf.shape(x))
                x = tf.nn.relu(x)
                x = tf.layers.average_pooling2d(x,[2,2],[2,2],'same',name='pooling')

            with tf.variable_scope('Encoder_4x4'):
                x = tf.layers.conv2d(x, 4*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv1')
                conv_3 = x
                for layers in xrange(0, 20):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel4,batch_size, 4*feature_size, scale=scaling_factor)
                x = slim.conv2d(x, 4*feature_size, [1, 1])
                x = x + conv_3

            with tf.variable_scope('Decoder_2x2'):
                x = tf.image.resize_bilinear(x,[conv2_shape[1],conv2_shape[2]],name='upsampled')
                x = tf.layers.conv2d(x, 4*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre1')
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 4*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = tf.concat([x,conv_2],axis=-1)
                x = tf.layers.conv2d(x, 2 * feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre2')
                for layers in xrange(decoder_first, decoder_second):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel2, batch_size, 2 * feature_size, scale=scaling_factor)

            with tf.variable_scope('Decoder_1x1'):
                x = tf.image.resize_bilinear(x, [conv1_shape[1], conv1_shape[2]], name='upsampled')
                x = tf.layers.conv2d(x, 2*feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre1')
                for layers in xrange(0, decoder_first):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock(x, 2*feature_size, kernel_size=[3, 3], scale=scaling_factor)
                x = tf.concat([x, conv_1], axis=-1)
                x = tf.layers.conv2d(x, feature_size, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_pre2')
                for layers in xrange(decoder_first, decoder_second):
                    with tf.variable_scope('block%d' % layers):
                        x = resBlock_generating2(x, generated_kernel1, batch_size, 1 * feature_size,
                                                 scale=scaling_factor)
                x = tf.layers.conv2d(x, output_channels, [3, 3],
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02),
                                     padding='same', name='conv_last')

            with tf.variable_scope('concat_images'):
                weighted_concat_inputs = tf.layers.conv2d(input_concat, output_channels, [1, 1],kernel_initializer=tf.contrib.layers.variance_scaling_initializer(0.02), name='concat_image_weight', use_bias=False)
            x = x+weighted_concat_inputs

    return x,idct_reconstruction

def Estimation_block2(input):
    # batch x 128 x 128 x 1
    with tf.variable_scope('Estimation_block2'):
        output = tf.layers.conv2d(input, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        output = tf.layers.conv2d(output, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        output = tf.nn.max_pool(output,[1,4,4,1],[1,4,4,1],padding='VALID',name='maxpool1') # batch x 96 x 96 x 32

        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv4')
        output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1],padding='VALID', name='maxpool2') # batch x 48 x 48 x 64

        output = tf.layers.conv2d(output, 96, 3, padding='same', activation=tf.nn.relu, name='conv5')
        output = tf.layers.conv2d(output, 96, 3, padding='same', activation=tf.nn.relu, name='conv6')
        output = tf.nn.max_pool(output, [1, 4, 4, 1], [1, 4, 4, 1], padding='VALID',name='maxpool3')  # batch x 96 x 96 x 32

        output = tf.layers.conv2d(output, 192, 3, padding='same', activation=tf.nn.relu, name='conv7')
        output = tf.layers.conv2d(output, 192, 3, padding='same', activation=tf.nn.relu, name='conv8')
        output = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='avgpool') # batch x 24 x 24 x 128

        output = tf.layers.conv2d(output, 1, 1, padding='same', activation=tf.nn.relu, name='conv_last')
        h_sigma = tf.reduce_mean(tf.reduce_mean(output,axis=1,keep_dims=True),axis=2,keep_dims=True,name='h_sigma')  # batch x 1 x 1 x 1

    return h_sigma

def Estimation_block2_SP(input):
    # batch x 128 x 128 x 1
    with tf.variable_scope('Estimation_block2'):
        output = tf.layers.conv2d(input, 32, 3, padding='same', activation=tf.nn.relu, name='conv1')
        output = tf.layers.conv2d(output, 32, 3, padding='same', activation=tf.nn.relu, name='conv2')
        output = tf.nn.max_pool(output,[1,4,4,1],[1,4,4,1],padding='VALID',name='maxpool1') # batch x 96 x 96 x 32

        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv3')
        output = tf.layers.conv2d(output, 64, 3, padding='same', activation=tf.nn.relu, name='conv4')
        output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1],padding='VALID', name='maxpool2') # batch x 48 x 48 x 64

        output = tf.layers.conv2d(output, 96, 3, padding='same', activation=tf.nn.relu, name='conv5')
        output = tf.layers.conv2d(output, 96, 3, padding='same', activation=tf.nn.relu, name='conv6')
        output = tf.nn.max_pool(output, [1, 4, 4, 1], [1, 4, 4, 1], padding='VALID',name='maxpool3')  # batch x 96 x 96 x 32

        output = tf.layers.conv2d(output, 192, 3, padding='same', activation=tf.nn.relu, name='conv7')
        output = tf.layers.conv2d(output, 192, 3, padding='same', activation=tf.nn.relu, name='conv8')
        output = tf.nn.avg_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='avgpool') # batch x 24 x 24 x 128

        output = tf.layers.conv2d(output, 1, 1, padding='same', activation=tf.nn.relu, name='conv_last')
        h_sigma = tf.reduce_mean(tf.reduce_mean(output,axis=1,keep_dims=True),axis=2,keep_dims=True,name='h_sigma')  # batch x 1 x 1 x 1

    return h_sigma,output