"""Useful training functions."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import time
from tqdm import tqdm
import sys
import hem


def default_training(train_op):
    """Trainining function that just runs an op (or list of ops)."""
    losses = hem.collection_to_dict(tf.get_collection('losses'))

    def helper(sess, args, handle, handle_value):
        _, results = sess.run([train_op, losses], feed_dict={handle: handle_value})
        return results
    return helper


def average_gradients(tower_grads, check_numerics=False, name=None):
    """Average the gradients from all GPU towers.

    Args:
      tower_grads: List, tuples of (grad, var) values.

    Returns:
      A list containing the averaged gradients in (grad, var) form.
    """

    average_grads = []
    with tf.name_scope(name, 'average_gradients', values=tower_grads):
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            # print('GRAD', grad)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            if check_numerics:
                grad = tf.check_numerics(grad, 'GRADIENT ERROR on layer {}, grad {}'.format(v, grad))
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def init_optimizer(args):
    """Helper function to initialize an optimizer from given arguments.

    Args:
      args: Argparse structure.

    Returns:
      An initialized optimizer.
    """
    with tf.variable_scope('optimizers'):
        if args.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(args.lr,
                                             decay= args.decay,
                                             momentum=args.momentum,
                                             centered=args.centered)
        elif args.optimizer == 'adadelta':
            return tf.train.AdadeltaOptimizer(args.lr)
        elif args.optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(args.lr)
        elif args.optimizer == 'sgd':
            return tf.train.GradientDescentOptimizer(args.lr)
        elif args.optimizer == 'pgd':
            tf.train.ProximalGradientDescentOptimizer(args.lr)
        elif args.optimizer == 'padagrad':
            return tf.train.ProximalAdagradOptimizer(args.lr)
        elif args.optimizer == 'momentum':
            return tf.train.MomentumOptimizer(args.lr,
                                              args.momentum)
        elif args.optimizer == 'adam':
            return tf.train.AdamOptimizer(args.lr,
                                          args.beta1,
                                          args.beta2)
        elif args.optimizer == 'ftrl':
            return tf.train.FtrlOptimizer(args.lr)





def train(model, iterators, handle, sv, args, reset=False):

    try:
        checkpoint_path = os.path.join(args.dir, 'checkpoint')
        losses = hem.collection_to_dict(tf.get_collection('losses'))

        with sv.sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # summary_train_writer.add_graph(sess.graph, global_step=global_step)
            # initialize
            start_time = time.time()
            if reset:
                sess.run(sv.reset_global_step)
                sess.run(sv.reset_global_epoch)
            current_step = int(sess.run(sv.global_step))
            current_epoch = int(sess.run(sv.global_epoch))

            # set max epochs based on +n or n format
            max_epochs = current_epoch + int(args.epochs[1:]) if '+' in args.epochs else int(args.epochs)
            # initialize datasets
            for k, v in iterators.items():
                sess.run(iterators[k]['x'].initializer)
            # get handles for datasets
            training_handle = sess.run(iterators['train']['handle'])
            validation_handle = sess.run(iterators['validate']['handle'])
            if 'test' in iterators and iterators['test']['handle'] is not None:
                test_handle = sess.run(iterators['test']['handle'])

            # save model params before any training has been done
            if current_step == 0:
                hem.message('Generating baseline summaries and checkpoint...')
                sv.sv.saver.save(sess, save_path=checkpoint_path, global_step=sv.global_step)
                sv.summary_writers['train'].add_summary(sess.run(sv.summary_op, feed_dict={ handle: validation_handle }),
                                                        global_step=sess.run(sv.global_step))

            hem.message('Starting training...')
            for epoch in range(current_epoch, max_epochs):
                prog_bar = tqdm(range(iterators['train']['batches']),
                                desc='Epoch {:3d}'.format(epoch + 1),
                                unit='batch')
                running_total = None
                for i in prog_bar:
                    # train and display status
                    status = model.train(sess, args, { handle: training_handle })
                    hem.update_moving_average(status, running_total, prog_bar)
                    # record 10 extra summaries (per epoch) in the first 3 epochs
                    if epoch < 3 and i % int((iterators['train']['batches'] / 10)) == 0:
                        sv.summary_writers['train'].add_summary(
                            sess.run(sv.summary_op, feed_dict={ handle: training_handle }),
                            global_step=sess.run(sv.global_step))
                    elif epoch >= 3 and i % int((iterators['train']['batches'] / 3)) == 0:
                        sv.summary_writers['train'].add_summary(
                            sess.run(sv.summary_op, feed_dict={ handle: training_handle }),
                            global_step=sess.run(sv.global_step))
                    sess.run(sv.increment_global_step)
                    # print('global step:', sess.run(sv.global_step))

                # update epoch count
                sess.run(sv.increment_global_epoch)
                current_epoch = int(sess.run(sv.global_epoch))
                # generate end-of-epoch summaries
                sv.summary_writers['train'].add_summary(sess.run(sv.summary_op, feed_dict={ handle: training_handle }),
                                                     global_step=sess.run(sv.global_step))

                # save checkpoint
                sv.sv.saver.save(sess, save_path=checkpoint_path, global_step=sv.global_epoch)
                # perform validation
                hem.inference(sess, losses, sv.summary_op, iterators['validate']['batches'], handle, validation_handle,
                              'Validation', sv.summary_writers['validate'], sv.global_step)
                # perform testing, if asked
                if (epoch + 1) in args.test_epochs:
                    hem.inference(sess, losses, iterators['test']['batches'], handle, test_handle, 'Test',
                                  sv.summary_writers['test'], sv.global_step)

            hem.message('\nTraining complete! Elapsed time: {}s'.format(int(time.time() - start_time)))

    except Exception as e:
        print('Caught unexpected exception during training:', e, e.message)
        sys.exit(-1)
