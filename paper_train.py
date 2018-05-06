import colored_traceback
colored_traceback.add_hook()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'     # only log errors
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)   # only log errors
import sys
import time
from tqdm import tqdm
import cv2
import numpy as np
import hem

# usage:
# python paper_train.py examples/paper/baseline.config --dir /mnt/storage/thesis/baseline

def train(model, iterators, handle, sv, dataset_moments_op, mean_image_placeholder, args, reset=False):
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

            def calc_mean_var_image(n_batches, dataset_handle):
                mean_img, var_img = sess.run(dataset_moments_op, feed_dict={handle: dataset_handle})
                for i in range(n_batches - 1):
                    m, v = sess.run(dataset_moments_op, feed_dict={handle: dataset_handle})
                    mean_img += m
                    var_img += v
                    # mean_img += sess.run(mean_image_op, feed_dict={handle: dataset_handle})
                return mean_img / n_batches, var_img / n_batches
                # return mean_img / n_batches

            print('Calculating mean depth image for training...')
            mean_training_img, var_training_img = calc_mean_var_image(iterators['train']['batches'], training_handle)
            print('Calculating mean depth image for validation...')
            mean_validation_img, var_validation_img = calc_mean_var_image(iterators['validate']['batches'], validation_handle)
            cv2.imwrite(os.path.join(args.dir, 'mean_training_img.png'), np.transpose(mean_training_img, [1, 2, 0]) * 255.0)
            cv2.imwrite(os.path.join(args.dir, 'var_training_img.png'), np.transpose(var_training_img, [1, 2, 0]) * 255.0)
            cv2.imwrite(os.path.join(args.dir, 'mean_validation_img.png'), np.transpose(mean_validation_img, [1, 2, 0]) * 255.0)
            cv2.imwrite(os.path.join(args.dir, 'var_validation_img.png'), np.transpose(var_validation_img, [1, 2, 0]) * 255.0)

            # save model params before any training has been done
            if current_step == 0:
                hem.message('Generating baseline summaries and checkpoint...')
                sv.sv.saver.save(sess, save_path=checkpoint_path, global_step=sv.global_step)
                sv.summary_writers['train'].add_summary(sess.run(sv.summary_op, feed_dict={ handle: validation_handle,
                                                                                            mean_image_placeholder: mean_validation_img }),
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
                            sess.run(sv.summary_op, feed_dict={ handle: training_handle, mean_image_placeholder: mean_training_img }),
                            global_step=sess.run(sv.global_step))
                    elif epoch >= 3 and i % int((iterators['train']['batches'] / 3)) == 0:
                        sv.summary_writers['train'].add_summary(
                            sess.run(sv.summary_op, feed_dict={ handle: training_handle, mean_image_placeholder: mean_training_img }),
                            global_step=sess.run(sv.global_step))
                    sess.run(sv.increment_global_step)
                    # print('global step:', sess.run(sv.global_step))

                # update epoch count
                sess.run(sv.increment_global_epoch)
                current_epoch = int(sess.run(sv.global_epoch))
                # generate end-of-epoch summaries
                sv.summary_writers['train'].add_summary(sess.run(sv.summary_op, feed_dict={ handle: training_handle, mean_image_placeholder: mean_training_img }),
                                                     global_step=sess.run(sv.global_step))

                # save checkpoint
                sv.sv.saver.save(sess, save_path=checkpoint_path, global_step=sv.global_epoch)
                # perform validation
                hem.inference(sess, losses, sv.summary_op, iterators['validate']['batches'], handle, validation_handle, mean_image_placeholder, mean_validation_img,
                              'Validation', sv.summary_writers['validate'], sv.global_step)
                # perform testing, if asked
                if (epoch + 1) in args.test_epochs:
                    hem.inference(sess, losses, iterators['test']['batches'], handle, test_handle, 'Test',
                                  sv.summary_writers['test'], sv.global_step)

            hem.message('\nTraining complete! Elapsed time: {}s'.format(int(time.time() - start_time)))


    except Exception as e:
        print('Caught unexpected exception during training:', e, e.message)
        sys.exit(-1)


if __name__ == '__main__':
    hem.message('Parsing arguments...')
    args = hem.parse_args()

    hem.message('Initializing working dir...')
    hem.init_working_dir(args)

    hem.message('Initializing input pipeline...')
    x, handle, iterators = hem.get_dataset_tensors(args)

    hem.message('Initializing model...')
    model = hem.get_model(args.model)(x, args)

    y = hem.crop_to_bounding_box(x[1], 17, 17, 29, 29)
    # TODO is this correct?
    dataset_moments_op = tf.nn.moments(y, axes=0)

    hem.message('Initializing supervisor...')
    sv = hem.HemSupervisor(args)
    train(model, iterators, handle, sv, dataset_moments_op, model.mean_image_placeholder, args)

