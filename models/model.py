import tensorflow as tf
import time
from sys import stdout
from util import print_progress, fold




class Model:
    def __init__(self):
        self._sess = tf.get_default_session()
        self._encoder = None
        self._decoder = None
        self._latent = None
        self._latent_loss = None
        self._generated_loss = None
        self._loss = None
        self._train_op = None
        self._optimizer = None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def latent(self):
        return self._latent

    @property
    def loss(self):
        return self._loss

    @property
    def latent_loss(self):
        return self._latent_loss

    @property
    def generated_loss(self):
        return self._generated_loss


    def train(self, epoch, x_input, data, batch_size, tb_writer, summary_nodes, display=True):
        epoch_start_time = time.time()
        n_batches = int(data.train.num_examples/batch_size)
        for i in range(n_batches):
            xs, ys = data.train.next_batch(batch_size)
            _, l, summary = self._sess.run([self._train_op, self.loss, summary_nodes], feed_dict={x_input: xs})
            print_progress(epoch, batch_size*(i+1), data.train.num_examples, epoch_start_time, {'loss': l})
            tb_writer.add_summary(summary, epoch)

        # perform validation
        results = fold(self._sess, x_input, [self.loss], data.validation, batch_size, int(data.validation.num_examples/batch_size))
        stdout.write(', validation: {:.4f}\r\n'.format(results[0]))

        







            
