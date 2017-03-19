import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
import sys, random, argparse, os, uuid, pickle, h5py, cv2, time
# from models import test
from models import simple_fc, simple_cnn
from msssim import MultiScaleSSIM, tf_ssim, tf_ms_ssim
from data import Floorplans
from util import *




parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--examples', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--layers', type=int, nargs='+', default=(512, 256, 128))
parser.add_argument('--seed', type=int, default=os.urandom(4))
parser.add_argument('--dataset', type=lambda s: s.lower(), default='mnist')
parser.add_argument('--dir', type=str, default='workspace/{}'.format(uuid.uuid4()))
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--interactive', default=False, action='store_true')
parser.add_argument('--model', type=lambda s: s.lower(), default='fc')
parser.add_argument('--grayscale', default=False, action='store_true')
parser.add_argument('--loss', type=lambda s: s.lower(), default='l1')
parser.add_argument('--optimizer', type=lambda s: s.lower(), default='rmsprop')
parser.add_argument('--momentum', type=float, default=0.01)
parser.add_argument('--decay', type=float, default=0.9)
parser.add_argument('--centered', default=False, action='store_true')
args = parser.parse_args()

# TODO: should keep the args that were passed in this time
if args.resume:
    # copy passed-in values
    args_copy = argparse.Namespace(**vars(args))
    # load previous args
    args = pickle.load(open(os.path.join(args.dir, 'settings'), 'rb'))
    # adjust for new passed in values
    args.resume = True
    args.epochs = args_copy.epochs
    print('Args restored')
    
    

# seed
print('Setting seed to', args.seed)
random.seed(args.seed)

# session and global vars
sess = tf.Session()

# variables to track training progress. useful when resuming training from disk.
with tf.variable_scope('global_vars'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_epoch = tf.Variable(1, name='global_epoch', trainable=False)
    global_batchsize = tf.Variable(args.batchsize, name='global_batchsize', trainable=False)


# input tensor for dataset.
# TODO: move this to Dataset class or something. should be self-describing

if args.dataset == 'mnist':
    x_input = tf.placeholder("float", [None, 784])
    x = tf.reshape(x_input, [-1, 28, 28, 1])
elif args.dataset == 'floorplan':
    x_input = tf.placeholder("float", [None, 64, 64, 3])
    if args.grayscale:
        x = tf.image.rgb_to_grayscale(x_input)
    else:
        x = x_input


# model    
if args.model == 'fc':
    y_hat = simple_fc(x, args.layers)
elif args.model == 'cnn':
    y_hat = simple_cnn(x, args.layers)

# loss
with tf.variable_scope('loss_functions'):
    loss_functions = {'l1': tf.reduce_mean(tf.abs(x - y_hat)),
                      'l2': tf.reduce_mean(tf.pow(x - y_hat, 2)),
                      'rmse': tf.sqrt(tf.reduce_mean(tf.pow(x - y_hat, 2))),
                      'ssim': 1.0 - tf_ssim(tf.image.rgb_to_grayscale(x), tf.image.rgb_to_grayscale(y_hat)),
                      'crossentropy': -tf.reduce_sum(x * tf.log(y_hat))}
loss = loss_functions[args.loss]

# optimizer
with tf.variable_scope('optimizers'):
    optimizers = {'rmsprop': tf.train.RMSPropOptimizer(args.lr, args.decay, args.momentum, centered=args.centered),
                  'adadelta': tf.train.AdadeltaOptimizer(args.lr),
                  'gd': tf.train.GradientDescentOptimizer(args.lr),
                  'adagrad': tf.train.AdagradOptimizer(args.lr),
                  'momentum': tf.train.MomentumOptimizer(args.lr, args.momentum),
                  'adam': tf.train.AdamOptimizer(args.lr),
                  'ftrl': tf.train.FtrlOptimizer(args.lr),
                  'pgd': tf.train.ProximalGradientDescentOptimizer(args.lr),
                  'padagrad': tf.train.ProximalAdagradOptimizer(args.lr)}
optimizer = optimizers[args.optimizer]

# training step
train_step = optimizer.minimize(loss, global_step=global_step)


    
        
# workspace
log_files = prep_workspace(args.dir)


# session saving/loading
saver = tf.train.Saver()
if not args.resume:
    pickle.dump(args, open(os.path.join(args.dir, 'settings'), 'wb'))
    tf.train.export_meta_graph(os.path.join(args.dir, 'model'))
    sess.run(tf.global_variables_initializer())
else:
    #saver = tf.train.import_meta_graph(os.path.join(args.dir, 'model'))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(args.dir, 'checkpoints')))
    print('Model restored. Current step=', sess.run(global_step), ', epoch=', sess.run(global_epoch), ', batch size=', sess.run(global_batchsize))
    


# tensorboard
montage = None
tb_writer = tf.summary.FileWriter(os.path.join(args.dir, 'logs'), graph=tf.get_default_graph())
summary_node = tf.summary.merge_all()


# print out schematic/visualization of model
total_params = visualize_parameters()
print('Total params: {}'.format(total_params))




# dataset
data = get_dataset(args.dataset)


print('Starting training')
start_epoch = sess.run(global_epoch)
for epoch in range(start_epoch, args.epochs+start_epoch):
    epoch_start_time = time.time()
    training_start_time = time.time()
    
    # perform training
    n_trbatches = int(data.train.num_examples/args.batchsize)
    completed = 0
    total_train_loss = 0.0
    for i in range(n_trbatches):
        xs, ys = data.train.next_batch(args.batchsize)
        _, l = sess.run([train_step, loss], feed_dict={x_input: xs})
        total_train_loss += l
        completed += args.batchsize
        # sess.run(global_step.assign(completed + (epoch-1)*(n_trbatches*args.batchsize)))
        log_files['train_loss'].write('{:05d},{:.5f}\n'.format(completed + (epoch-1)*(n_trbatches*args.batchsize), l))
        if args.interactive:
            print_progress(epoch, completed, data.train.num_examples, l)
    training_end_time = time.time()
    if not args.interactive:
        print('Epoch {}: Train loss ({:.5f}), elapsed time {}'.format(epoch, total_train_loss/n_trbatches, training_end_time-training_start_time))

    validation_start_time = time.time()
    # perform validation
    n_valbatches = int(data.validation.num_examples/args.batchsize)
    vl = 0.0
    for i in range(n_valbatches):
        xs, ys = data.validation.next_batch(args.batchsize)
        vl += sess.run(loss, feed_dict={x_input: xs})
    validation_end_time = time.time()
    log_files['validate_loss'].write('{:05d},{:.5f}\n'.format(completed + (epoch-1)*(n_trbatches*args.batchsize), vl/n_valbatches))
    if args.interactive:
        sys.stdout.write(', validation: {:.4f}'.format(vl/n_valbatches))
        sys.stdout.write('\r\n')
    else:
        print('Epoch {}: Validation loss ({:.5f}), elapsed time {}'.format(epoch, vl/n_valbatches, validation_end_time - validation_start_time))

    # montage
    if args.interactive:
        sys.stdout.write('Generating examples to disk...')
    else:
        print('Generating examples to disk...')
    # TODO: should reshape this on the fly, and only if necessary
    examples = data.test.images[:args.examples]
    row = generate_example_row(data, y_hat, examples, epoch==1, sess, x_input, args)
    if montage is not None:
        print('row:', row.shape, 'montage:', montage.shape)
    imgfile = os.path.join(args.dir, 'images', 'montage_{:03d}.png'.format(epoch))
    cv2.imwrite(imgfile, row)
    montage = row if montage is None else np.vstack((montage, row))
    if args.interactive:
        sys.stdout.write('complete!\r\n')
        sys.stdout.flush()

    # keep track of current epoch
    sess.run(global_epoch.assign(epoch+1))

    # tensorboard
    if summary_node is not None:
        summary_result = sess.run(summary_node, feed_dict={x_input: xs})
        tb_writer.add_summary(summary_result, epoch)
        
    # snapshot
    if args.interactive:
        sys.stdout.write('Writing snapshot to disk...')
    else:
        print('Writing snapshot to disk...')
    chkfile = os.path.join(args.dir, 'checkpoints', 'epoch_{:03d}.ckpt'.format(epoch))
    saver.save(sess, chkfile, global_step=global_step)
    if args.interactive:
        sys.stdout.write('complete!\r\n')
        sys.stdout.flush()
    epoch_end_time = time.time()
    print('Total elapsed epoch time: {}'.format(epoch_end_time - epoch_start_time))

# save complete montage
cv2.imwrite(os.path.join(args.dir, 'images', 'montage.png'), montage)
    
# perform test
n_tebatches = int(data.test.num_examples/args.batchsize)
tel = 0.0
completed = 0
for i in range(n_tebatches):
    xs, ys = data.test.next_batch(args.batchsize)
    tel += sess.run(loss, feed_dict={x_input: xs})
    completed += args.batchsize
    if args.interactive:
        sys.stdout.write('\r')
        sys.stdout.write('test: {:.4f}'.format(l))
        sys.stdout.flush()
log_files['test_loss'].write('{:05d},{:.5f}\n'.format((epoch) * n_trbatches * args.batchsize, tel/n_tebatches))
if args.interactive:
    sys.stdout.write('\r\n')
else:
    print('Test loss: {:.5f}'.format(tel/n_tebatches))

# close down log files
for key in log_files:
    log_files[key].close()

# generate charts
train_loss = np.genfromtxt(os.path.join(args.dir, "logs", "train_loss.csv"), delimiter=',')
test_loss = np.genfromtxt(os.path.join(args.dir, "logs", "test_loss.csv"), delimiter=',')
validate_loss = np.genfromtxt(os.path.join(args.dir, "logs", "validate_loss.csv"), delimiter=',')
plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Palatino']})
for x in [(train_loss, {}), (validate_loss, {'color': 'firebrick'})]:
    data, plot_args = x
    iters = data[:,[0]]
    vals = data[:,[1]]
    plt.plot(iters, vals, **plot_args)
    plt.xlabel('Iteration')
    plt.ylabel(r'$\ell_1$ Loss')
plt.savefig(os.path.join(args.dir, "images", "loss.pdf"))
