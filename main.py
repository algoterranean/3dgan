import tensorflow as tf, numpy as np, cv2 as cv
import sys, random, argparse, os, uuid, pickle, h5py
from models import test
from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim



class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset[()]
        # self.dataset = cv.cvtColor(self.dataset, cv.COLOR_BGR2GRAY)
        self.current_pos = 0
        self.num_examples = len(dataset)

    def next_batch(self, batch_size):
        if self.current_pos + batch_size > self.num_examples:
            self.current_pos = 0
        x = self.dataset[self.current_pos : self.current_pos+batch_size]
        self.current_pos += batch_size
        # return (x, None)
        return (x, None)
        # return (np.reshape(x, (batch_size, 64*64*3)), None)

        
        # if self.current_pos + batch_size < self.num_examples:
        #     files = self.files[self.current_pos:self.current_pos+batch_size]
        #     self.current_pos += batch_size
        # else:
        #     remainder = self.current_pos + batch_size - self.num_examples
        #     files = self.files[self.current_pos:] + self.files[0:remainder]
        #     self.current_pos = remainder
        #     images = []
        # for f in files:
        #     i = cv.imread(f,cv.IMREAD_COLOR)
        #     images.append(cv.resize(i, (64, 64)))
        #     a = np.reshape(np.array(images), (batch_size, 64*64*3))
        # return (a, None)    

        
        
    # def __init__(self, text_file):
    #     self.files = []
    #     d = os.path.dirname(text_file)
    #     with open(text_file) as f:
    #         for line in f:
    #             full_path = os.path.join(d, line)
    #             self.files.append(full_path.rstrip())
    #     self.num_examples = len(self.files)
    #     self.current_pos = 0

    # def next_batch(self, batch_size):
    #     if self.current_pos + batch_size < self.num_examples:
    #         files = self.files[self.current_pos:self.current_pos+batch_size]
    #         self.current_pos += batch_size
    #     else:
    #         remainder = self.current_pos + batch_size - self.num_examples
    #         files = self.files[self.current_pos:] + self.files[0:remainder]
    #         self.current_pos = remainder
    #     images = []
    #     for f in files:
    #         i = cv.imread(f,cv.IMREAD_COLOR)
    #         images.append(cv.resize(i, (64, 64)))
    #     a = np.reshape(np.array(images), (batch_size, 64*64*3))
    #     return (a, None)
        
    

class Floorplans:
    def __init__(self, root_dir='/mnt/research/datasets/floorplans/'):
        # self.test = Dataset(os.path.join(root_dir, 'test_set.txt'))
        # self.train = Dataset(os.path.join(root_dir, 'train_set.txt'))
        # self.validation = Dataset(os.path.join(root_dir, 'validation_set.txt'))
        self.file = h5py.File("/mnt/research/projects/hem/datasets/floorplan_64_float32.hdf5", 'r')
        sys.stdout.write("Loading test...")
        sys.stdout.flush()
        self.test = Dataset(self.file['test/images'])
        sys.stdout.write("done!\n\rLoading train...")
        sys.stdout.flush()
        self.train = Dataset(self.file['train/images'])
        sys.stdout.write("done!\n\rLoading validation...")
        sys.stdout.flush()
        self.validation = Dataset(self.file['validation/images'])
        sys.stdout.write("done!\n\r")
        sys.stdout.flush()

        print(self.train, self.test, self.validation)
        

        
# helper functions
def generate(data, tensor, xs, include_actual=True):
    # print(xs.shape)
    examples = sess.run(tensor, feed_dict={x: xs})
    montage = None
    for i, pred in enumerate(examples):
        if include_actual:
            # v = np.vstack((np.reshape(data.test.images[i], (28, 28)) * 255.0,
            #                np.reshape(pred, (28, 28)) * 255.0))
            # v = np.vstack((np.reshape(data.test.dataset[i], (64, 64, 3)) * 255.0,
            #                np.reshape(pred, (64, 64, 3)) * 255.0))

            gray_img = cv.cvtColor(data.test.dataset[i], cv.COLOR_BGR2GRAY)
            v = np.vstack((gray_img * 255.0,
                           np.reshape(pred, (64, 64)) * 255.0)) 
            # v = np.vstack((np.reshape(data.test.dataset[i], (64, 64)) * 255.0,
            #                np.reshape(pred, (64, 64)) * 255.0))
        else:
            # v = np.reshape(pred, (28, 28)) * 255.0
            # v = np.reshape(pred, (64, 64, 3)) * 255.0
            v = np.reshape(pred, (64, 64)) * 255.0            
        montage = v if montage is None else np.hstack((montage, v))
    return montage


def print_progress(epoch, completed, total, loss):
    sys.stdout.write('\r')
    sys.stdout.write('Epoch {:03d}: {:05d}/{:05d}: {:.4f}'.format(epoch, completed, total, loss))
    sys.stdout.flush()


def get_dataset(name):
    if name == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        return input_data.read_data_sets("MNIST_data", one_hot=True)
    elif name == 'floorplan':
        return Floorplans()


def prep_workspace(dirname):
    subdirs = [os.path.join(dirname, "checkpoints"),
               os.path.join(dirname, "images"),
               os.path.join(dirname, "logs")]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for d in subdirs:
        if not os.path.exists(d):
            os.mkdir(d)
            
    return {'train_loss': open(os.path.join(dirname, "logs", "train_loss.csv"), 'a'),
            'validate_loss': open(os.path.join(dirname, "logs", "validate_loss.csv"), 'a'),
            'test_loss' : open(os.path.join(dirname, "logs", "test_loss.csv"), 'a')}

    



    
# comand-line execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--layers', type=int, nargs='+', default=(512, 256, 128))
    parser.add_argument('--seed', type=int, default=os.urandom(4))
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--dir', type=str, default='runs/{}'.format(uuid.uuid4()))
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    # for repeatability purposes
    random.seed(args.seed)

    # dataset
    data = get_dataset(args.dataset)
    
    # model
    sess = tf.Session()

    # x = tf.placeholder("float", [None, 64, 64, 3])
    x = tf.placeholder("float", [None, 64, 64, 3])
    y_hat = test.model(tf.reshape(tf.image.rgb_to_grayscale(x), (-1, 64*64)), args.layers)
    

    # loss_l1 = tf.reduce_mean(tf.abs(x - y_hat))
    loss_l2 = tf.reduce_mean(tf.pow(tf.reshape(tf.image.rgb_to_grayscale(x), (-1, 64*64)) - y_hat, 2))
    # loss_l2 = tf.reduce_mean(tf.pow(x - y_hat, 2))
    # loss_rmse = tf.sqrt(tf.reduce_mean(tf.pow(x - y_hat, 2)))
    # def l(actual, pred):
    #     x = tf.image.rgb_to_grayscale(tf.reshape(actual, (-1, 64, 64)))
    #     y_hat = tf.image.rgb_to_grayscale(tf.reshape(pred, (-1, 64, 64)))
    #     return tf.reduce_mean(tf.abs(x - y_hat))
    
    loss = loss_l2
    
    # loss = tf_ssim(tf.reshape(x, (-1, 64, 64, 3)), tf.reshape(y_hat, (-1, 64, 64, 3)))
    # tf_ssim
    # tf_ms_ssim

    optimizer = tf.train.RMSPropOptimizer(args.lr).minimize(loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_epoch = tf.Variable(1, name='global_epoch', trainable=False)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    montage = None

    if args.resume:
        #saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))
        print('Model restored. Global step:', sess.run(global_step))
        
    # workspace
    log_files = prep_workspace(args.dir)
    if not args.resume:
        pickle.dump(args, open(os.path.join(args.dir, 'settings'), 'wb'))
        tf.train.export_meta_graph(os.path.join(args.dir, 'model'))



    start_epoch = sess.run(global_epoch)
    for epoch in range(start_epoch, args.epochs+start_epoch):
        # perform training
        n_trbatches = int(data.train.num_examples/args.batchsize)
        completed = 0
        for i in range(n_trbatches):
            xs, ys = data.train.next_batch(args.batchsize)
            _, l = sess.run([optimizer, loss], feed_dict={x: xs})
            completed += args.batchsize
            sess.run(global_step.assign(completed + (epoch-1)*(n_trbatches*args.batchsize)))
            log_files['train_loss'].write('{:05d},{:.5f}\n'.format(completed + (epoch-1)*(n_trbatches*args.batchsize), l))
            print_progress(epoch, completed, data.train.num_examples, l)

        # perform validation
        n_valbatches = int(data.validation.num_examples/args.batchsize)
        vl = 0.0
        for i in range(n_valbatches):
            xs, ys = data.validation.next_batch(args.batchsize)
            vl += sess.run(loss, feed_dict={x: xs})
        log_files['validate_loss'].write('{:05d},{:.5f}\n'.format(completed + (epoch-1)*(n_trbatches*args.batchsize), vl/n_valbatches))	   
        sys.stdout.write(', validation: {:.4f}'.format(vl/n_valbatches))
        sys.stdout.write('\r\n')

        # montage
        sys.stdout.write('Generating examples to disk...')
        # TODO: should reshape this on the fly, and only if necessary
        examples = data.test.dataset[:args.examples]
        # tf.reshape(tf.image.rgb_to_grayscale(x), (-1, 64*64))
        
        # examples = np.reshape(examples, (args.examples, 64*64*3))
        # examples = np.reshape(examples, (args.examples, 64*64))
        # examples = tf.image
        row = generate(data, y_hat, examples, epoch==1)
        imgfile = os.path.join(args.dir, 'images', 'montage_{:03d}.png'.format(epoch))
        cv.imwrite(imgfile, row)
        montage = row if montage is None else np.vstack((montage, row))
        sys.stdout.write('complete!\r\n')
        sys.stdout.flush()

        sess.run(global_epoch.assign(epoch+1))

        # snapshot
        sys.stdout.write('Writing snapshot to disk...')
        chkfile = os.path.join(args.dir, 'checkpoints', 'epoch_{:03d}.ckpt'.format(epoch))
        saver.save(sess, chkfile, global_step=global_step)
        sys.stdout.write('complete!\r\n')
        sys.stdout.flush()



    cv.imwrite(os.path.join(args.dir, 'images', 'montage.png'), montage)
    
    # perform test
    n_tebatches = int(data.test.num_examples/args.batchsize)
    tel = 0.0
    completed = 0
    for i in range(n_tebatches):
        xs, ys = data.test.next_batch(args.batchsize)
        tel += sess.run(loss, feed_dict={x: xs})
        completed += args.batchsize
        sys.stdout.write('\r')
        sys.stdout.write('test: {:.4f}'.format(l))
        sys.stdout.flush()
    log_files['test_loss'].write('{:05d},{:.5f}\n'.format((epoch) * n_trbatches * args.batchsize, tel/n_tebatches))        
    sys.stdout.write('\r\n')
                   

