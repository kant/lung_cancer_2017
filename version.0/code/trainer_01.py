import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#


from net.common import *
from net.blocks import *
from net.mynet import MyNet_0 as make_net

def pixelwise_cross_entropy(logit, label):
    N, H, W, num_class = label.get_shape().as_list()
    assert(num_class==1)
    flat_logit = tf.reshape(logit, [-1])
    flat_label = tf.reshape(label, [-1])
    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(flat_logit, flat_label))
    return loss

# <todo> : other version of dice loss
#https://www.kaggle.com/c/ultrasound-nerve-segmentation/discussion/21358
#https://github.com/jakeret/tf_unet/blob/master/tf_unet/unet.py
def pixelwise_dice_coefficient(prob, label, is_binarised=True):
    N, H, W, num_class = label.get_shape().as_list()
    assert(num_class==1)

    flat_label = tf.reshape(label, [-1,H* W*num_class])
    if is_binarised: # theshold to 0 and 1
        prob = tf.cast(tf.round(prob), tf.float32)
    flat_prob  = tf.reshape(prob,  [-1,H* W*num_class])

    intersection = tf.reduce_sum(tf.mul(flat_prob,flat_label), axis=1,keep_dims=True)
    label2 = tf.reduce_sum(tf.mul(flat_label, flat_label), axis=1,keep_dims=True)
    prob2  = tf.reduce_sum(tf.mul(flat_prob,  flat_prob ), axis=1,keep_dims=True)
    union  = label2+prob2
    loss   = tf.reduce_mean( 1 - 2*tf.div(intersection, union))
    return loss


def save_batch_results(batch_datas, batch_labels, batch_logits, batch_probs, batch_index, dir, is_show=None):
    makedirs(dir)

    batch_size = len(batch_datas)
    for n in range(batch_size):
        img = data_to_color_img(np.squeeze(batch_datas[n]), is_auto=True)
        l = data_to_gray_img(np.squeeze(batch_logits[n]), is_auto=True)
        y = data_to_gray_img(np.squeeze(batch_labels[n]))
        p = data_to_gray_img(np.squeeze(batch_probs[n]))
        i = batch_index[n]
        img = draw_segment_results(img, y, p)
        cv2.putText(img, '%d' % i, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if is_show is not None:
            imshow('img', img)
            imshow('logit', l)
            imshow('label', y)
            imshow('prob', p)
            cv2.waitKey(is_show)

        cv2.imwrite(dir + '/%05d.png' % i, img)



def load_data():

    #train_images = force_4d(np.load( '/root/share/data/LUNA2016/dummy/images.npy'))
    #train_labels = force_4d(np.load( '/root/share/data/LUNA2016/dummy/nodule_masks.npy'))
    #train_labels = train_labels.astype(np.float32)

    #example data ----------------------------------------------------------------

    train_images=np.zeros(shape=(28,512,512),dtype=np.float32)
    train_labels=np.zeros(shape=(28,512,512),dtype=np.float32)
    for j in range(0,28):
        i=j
        img  =cv2.imread('/root/share/data/LUNA2016/dummy1/images/%04d.png'%j,0)
        label=cv2.imread('/root/share/data/LUNA2016/dummy1/masks/%04d.png'%j,0)
        train_images[i]=img.astype(np.float32)
        train_labels[i]=label.astype(np.float32)



    valid_images=np.zeros(shape=(5,512,512),dtype=np.float32)
    valid_labels=np.zeros(shape=(5,512,512),dtype=np.float32)
    for j in range(28, 33):
        i=j-28
        img  =cv2.imread('/root/share/data/LUNA2016/dummy1/images/%04d.png'%j,0)
        label=cv2.imread('/root/share/data/LUNA2016/dummy1/masks/%04d.png'%j,0)
        valid_images[i]=img.astype(np.float32)
        valid_labels[i]=label.astype(np.float32)

    train_images=force_4d(train_images/255.)
    train_labels=force_4d(train_labels/255.)
    valid_images=force_4d(valid_images/255.)
    valid_labels=force_4d(valid_labels/255.)

    return train_images, train_labels,   valid_images, valid_labels



def test_net( datas, labels,  batch_size, data, label, loss, metric, sess):

    num = len(datas)
    all_loss = 0
    all_acc = 0
    all = 0
    for n in range(0, num, batch_size):
        #print('\r  evaluating .... %d/%d' % (n, num), end='', flush=True)
        start = n
        end = start+batch_size if start+batch_size<=num else num
        batch_datas  = datas  [start:end]
        batch_labels = labels [start:end]

        fd = {data: batch_datas, label: batch_labels, IS_TRAIN_PHASE : False}
        test_loss, test_acc = sess.run([loss, metric], feed_dict=fd)

        a = end-start
        all += a
        all_loss += a*test_loss
        all_acc  += a*test_acc

    assert(all==num)
    loss = all_loss/all
    acc  = all_acc/all

    return loss, acc


def test_net_and_save( datas, labels,  index, batch_size, data, label, loss, metric, prob, logit,  sess, out_dir ):

    num = len(datas)
    all_loss = 0
    all_acc = 0
    all = 0
    for n in range(0, num, batch_size):
        #print('\r  evaluating .... %d/%d' % (n, num), end='', flush=True)
        start = n
        end = start+batch_size if start+batch_size<=num else num
        batch_datas  = datas  [start:end]
        batch_labels = labels [start:end]
        batch_index  = index  [start:end]


        fd = {data: batch_datas, label: batch_labels, IS_TRAIN_PHASE : False}
        test_loss, test_acc, batch_probs, batch_logits= sess.run([loss, metric, prob, logit], feed_dict=fd)
        save_batch_results(batch_datas, batch_labels, batch_logits, batch_probs, batch_index, out_dir, is_show=None)

        a = end-start
        all += a
        all_loss += a*test_loss
        all_acc  += a*test_acc

    assert(all==num)
    loss = all_loss/all
    acc  = all_acc/all

    return loss, acc

#--------------------------------------------------------------------------------------------------------
def run_train():


    # output dir, etc
    out_dir = '/root/share/out/kaggle/02'
    makedirs(out_dir)
    makedirs(out_dir+'/check_points')


    log = Logger()  # log file
    log.open(out_dir+'/log.txt', mode='a')
    log.write('--- [START %s] %s\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    # load data ---------------
    log.write('** some data setting **\n')

    train_images, train_labels, valid_images, valid_labels = load_data()
    train_index = list(range(len(train_images)))
    valid_index = list(range(len(valid_images)))

    _, height, width, channel = train_images.shape
    _, _, _, num_class        = train_labels.shape
    num_train   = len(train_images)
    num_valid   = len(valid_images)

    log.write('height=%d, width=%d, channel=%d\n' % (height, width, channel))
    log.write('num_class=%d\n' % (num_class))
    log.write('num_train=%d\n' % (num_train))
    log.write('num_valid=%d\n' % (num_valid))
    log.write('\n')


    if 0: # <check>
        for n in range(min(100, num_train)):
            print (n)
            img   = data_to_color_img(np.squeeze(train_images[n]), is_auto=True)
            l = data_to_gray_img(np.squeeze(train_labels[n]))
            img = draw_contour(img, l)

            imshow('img',   img )
            imshow('label', l)
            cv2.waitKey(0)




    # load net ---------------
    log.write('** net settings **\n')
    log.write('\n')

    logit = make_net(input_shape =(height, width, channel), output_shape=(height, width, num_class))
    data  = tf.get_default_graph().get_tensor_by_name('input:0')
    label = tf.placeholder(dtype=tf.float32, shape=[None,height, width, num_class])
    prob  = tf.nn.sigmoid(logit)
    #prob  = tf.nn.sigmoid(tf.clip_by_value(logit, 1e-10, 10.0))

    # define loss
    loss = pixelwise_cross_entropy(logit, label)
    metric = pixelwise_dice_coefficient(prob, label)

    #  solver
    batch_size = 4
    max_run    = 800
    epoch_log  = 10
    learning_rate = tf.placeholder(tf.float32, shape=[])
    solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum= 0.9)
    solver_step = solver.minimize(loss)


    log.write('** some solver setting **\n')
    log.write('\tbatch_size = %d\n'%batch_size)
    log.write('\tmax_run = %d\n'%max_run)
    log.write('\tepoch_log = %d\n'%epoch_log)
    log.write('\n')

    # start training here ------------------------------------------------

    sess = tf.InteractiveSession()
    with sess.as_default():
        sess.run(tf.global_variables_initializer(), feed_dict={IS_TRAIN_PHASE: True})

        saver  = tf.train.Saver()
        writer = tf.summary.FileWriter(out_dir + '/tf', graph=tf.get_default_graph())

        log.write('\n')
        log.write(' run  epoch   iter    rate      |  train_loss    (err)     |  valid_loss    (err)     |     \n')
        log.write('--------------------------------------------------------------------------------------------\n')

        tic = timer()
        iter = 0
        for r in range(max_run):
            rate = 0.1

            argument_images, argument_labels, argument_index  = shuffle_data(train_images, train_labels, train_index)
            num_argument = len(argument_images)
            N = num_argument//batch_size
            iter_log = max(round(float(epoch_log * num_train) / float(batch_size)), 1)
            for n in range(N):
                iter = iter + 1
                run   = r + float(n) / float(N)
                epoch = float(iter * batch_size) / float(num_train)

                batch_datas, batch_labels, batch_index =  generate_train_batch_next(argument_images, argument_labels, argument_index, n, batch_size)
                fd = {data: batch_datas, label: batch_labels, learning_rate: rate, IS_TRAIN_PHASE: True}
                _, batch_loss, batch_acc, batch_probs, batch_logits = sess.run([solver_step, loss, metric, prob, logit], feed_dict=fd)

                log.write('\r')
                log.write('%6.1f  %5.1f   %05d   %f  |  %f    (%f)  ' %
                          (run, epoch, iter, rate, batch_loss, batch_acc), is_file=0)

                # do validation here!
                ##if iter%iter_log==0 or n == N-1:
                ##if n == N-1:
                if iter % iter_log == 0 :
                    save_batch_results(batch_datas, batch_labels, batch_logits, batch_probs, batch_index, out_dir+'/out/train', is_show=None)

                    toc = timer()
                    sec_pass = toc - tic
                    min_pass = sec_pass / 60.

                    val_loss, val_acc =  test_net(valid_images, valid_labels, batch_size, data, label, loss, metric, sess)
                    log.write('\r')
                    log.write('%6.1f  %5.1f   %05d   %f  |  %f    (%f)  |  %f    (%f)  | %4.1f min  \n' %
                          (run, epoch, iter, rate, batch_loss, batch_acc, val_loss, val_acc, min_pass ))

                pass


        #final test! ------------------------------------------
        # save final checkpoint
        saver.save(sess, out_dir + '/check_points/final.ckpt')

        log.write('\n')
        log.write('** evaluation on test set **\n' )
        test_loss, test_acc = test_net_and_save(valid_images, valid_labels, valid_index, batch_size, data, label, loss, metric, prob, logit, sess, out_dir+'/out/test')
        log.write('\rtest_loss=%f    (test_acc=%f)\n' % ( test_loss, test_acc))

        log.write('\n')
        log.write('sucess\n')

    pass




def run_test():

    # output dir, etc
    out_dir = '/root/share/out/kaggle/02'


    # data -------------------------------------------------------------------------
    print('read data:\n')
    train_images, train_labels, valid_images, valid_labels = load_data()
    #valid_images=train_images
    #valid_labels=train_labels
    valid_index = list(range(len(valid_images)))

    _, height, width, channel = train_images.shape
    _, _, _, num_class        = train_labels.shape
    num_train   = len(train_images)
    num_valid   = len(valid_images)

    # net  -----------------------------------------------
    logit = make_net(input_shape=(height, width, channel), output_shape=(height, width, num_class))
    data = tf.get_default_graph().get_tensor_by_name('input:0')
    label = tf.placeholder(dtype=tf.float32, shape=[None, height, width, num_class])
    prob = tf.nn.sigmoid(logit)
    # prob  = tf.nn.sigmoid(tf.clip_by_value(logit, 1e-10, 10.0))

    # define loss
    loss   = pixelwise_cross_entropy(logit, label)
    metric = pixelwise_dice_coefficient(prob, label)


    # start testing here ------------------------------------------------

    sess = tf.InteractiveSession()
    with sess.as_default():
        saver  = tf.train.Saver()
        #saver.restore(sess, out_dir + '/check_points/final.ckpt')
        saver.restore(sess, out_dir + '/check_points/final.ckpt')

        # shuffle and test
        print('** evaluation on test set **')

        # do random test of random batch size (ensure thaere is not bug!!!)
        for i in range(10):
            images,  labels, index = shuffle_data(valid_images, valid_labels)

            batch_size =  np.random.randint(1, 8)
            test_loss, test_acc = test_net(images, labels, batch_size, data, label, loss, metric, sess)
            print('  %d,   batch_size=%3d  : %f    (%f)' % ( i, batch_size, test_loss, test_acc))

        #save
        test_loss, test_acc = test_net_and_save(valid_images, valid_labels, valid_index, batch_size, data, label, loss, metric, prob, logit, sess, out_dir+'/out/test')
        print('')
        print('final : %f    (%f)' % (test_loss, test_acc))






## MAIN ##############################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_train()
    #run_test()




