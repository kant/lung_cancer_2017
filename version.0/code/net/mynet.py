from net.blocks import *
from net.file import *


#dummy network for debuuging
def MyNet_0( input_shape=(1,1,1), output_shape = (1,1,1)):

    H,  W,  C         = input_shape
    oH, oW, num_class = output_shape
    assert(H==oH)
    assert(W==oW)


    input = tf.placeholder(shape=[None, H, W, C], dtype=tf.float32, name='input')

    #256x256
    with tf.variable_scope('block1') as scope:
        block1 = conv2d_bn_relu(input, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')
        block1 = maxpool(block1,kernel_size=(2,2),stride=[1,2,2,1], padding='SAME')

    #128x128
    with tf.variable_scope('block2') as scope:
        block2 = conv2d_bn_relu(block1, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')
        block2 = maxpool(block2, kernel_size=(2, 2), stride=[1, 2,2, 1], padding='SAME')

    #64x64
    with tf.variable_scope('block3') as scope:
        block3 = conv2d_bn_relu(block2, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')
        block3 = maxpool(block3, kernel_size=(2, 2), stride=[1, 2,2, 1], padding='SAME')

    # 32x32
    with tf.variable_scope('block4') as scope:
        block4 = conv2d_bn_relu(block3, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')
        block4 = maxpool(block4, kernel_size=(2, 2), stride=[1, 2,2, 1], padding='SAME')


    # 16x16
    with tf.variable_scope('block5') as scope:
        block5 = conv2d_bn_relu(block4, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')


    # 32x32
    with tf.variable_scope('block6') as scope:
        block6 = upsample2d(block5)
        block6 = conv2d_bn_relu(block6, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')

    #64x64
    with tf.variable_scope('block7') as scope:
        block7 = upsample2d(block6)
        block7 = conv2d_bn_relu(block7, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')

    #128x128
    with tf.variable_scope('block8') as scope:
        block8 = upsample2d(block7)
        block8 = conv2d_bn_relu(block8, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')

    #256x256
    with tf.variable_scope('block9') as scope:
        block9 = upsample2d(block8)
        block9 = conv2d_bn_relu(block9, num_kernels=8, kernel_size=(5, 5), stride=[1, 1, 1, 1], padding='SAME')

    logit = conv2d(block9, num_kernels=num_class, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='logit')
    return logit


#--------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print  ( 'running main function ...' )

    log = Logger()  # log file
    log.open('/root/share/out/kaggle/00/xxx_log.txt', mode='a')  #udacity  #kaggle

    out_dir = '/root/share/out/udacity/00/tf'
    empty(out_dir)

    num_class=1
    H, W, C = (256, 256, 1)
    logit = MyNet_0(input_shape=(H,W,C), output_shape=(H, W, num_class))
    data  = tf.get_default_graph().get_tensor_by_name('input:0')

    images = np.zeros(shape=(1,H,W,C), dtype=np.float32)

    #input   = tf.get_default_graph().get_tensor_by_name('input:0')
    #print(input)

    # draw graph to check connections
    with tf.Session()  as sess:
        sess.run(tf.global_variables_initializer(),feed_dict={IS_TRAIN_PHASE:True})
        sess.run(logit, feed_dict={data:images , IS_TRAIN_PHASE:True})

        summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
        # cd /opt/anaconda3/lib/python3.5/site-packages/tensorflow/tensorboard
        # python tensorboard.py --logdir /root/share/out/kaggle/00/tf


        #print_macs_to_file(log)
    print ('sucess!')

