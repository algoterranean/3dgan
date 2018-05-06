# provide syntax highlighting to tracebacks
import colored_traceback
colored_traceback.add_hook()
# only log errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import hem


# vargs = {'dir': 'hi',
#          'epochs': 100,
#          'batch_size': 512,
#          'epoch_size': -1,
#          'max_to_keep': 0,
#          'n_gpus': 2,
#          'n_threads': 6,
#          }

# TODO Use tf.tile to duplicate dataset into two branches, one for estimator and one for GAN


# TODO
# 1. argument parsing only parses one model and discards the remaining ones
# 2.

hem.message('Welcome to Hem')
hem.message('Initializing...')
args = hem.parse_args(display=True)
hem.init_working_dir(args)
vargs = vars(args)

hem.message('Initializing dataset...')
x, handle, iterators = hem.get_dataset_tensors(args)

hem.message('Initializing model...')
estimator_model = hem.get_model('mean_depth_estimator')(x, args)

vargs['g_arch'] = 'E2'
vargs['d_arch'] = 'E2'
sampler_model = hem.get_model('experimental_sampler')(x, estimator_model, args)

hem.message('Initializing supervisor...')

sv = hem.init_supervisor(args)

hem.message('Starting training...')
vargs['epochs'] = '30'
hem.train(estimator_model, iterators, handle, sv, args)

# with sv.sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     sess.run(tf.assign(sv.global_step, 0))
#     sess.run(tf.assign(sv.global_epoch, 0))
#
vargs['epochs'] = '300'
vargs['lr'] = 1e-4
hem.train(sampler_model, iterators, handle, sv, args) #, reset=True)


