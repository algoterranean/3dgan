import numpy as np
import tensorflow as tf
import hem


class TestOpsLosses(tf.test.TestCase):
    def test_rmse(self):
        with self.test_session() as sess:
            x = tf.placeholder(tf.float32, (1, 64, 64, 3))
            x_hat = tf.placeholder(tf.float32, (1, 64, 64, 3))
            x_data = np.ones((1, 64, 64, 3))
            l = hem.rmse(x, x_hat)
            # 1 - 1 = 0
            x_hat_data_ones = np.ones((1, 64, 64, 3))
            results = sess.run(l, feed_dict={x: x_data, x_hat: x_hat_data_ones})
            self.assertAllClose(results, 0)
            # 1 - 0 = 1
            x_hat_data_zeros = np.zeros((1, 64, 64, 3))
            results = sess.run(l, feed_dict={x: x_data, x_hat: x_hat_data_zeros})
            self.assertAllClose(results, 1)
            # -1 - 1 = 2
            x_data = np.ones((1, 64, 64, 3)) * -1
            results = sess.run(l, feed_dict={x: x_data, x_hat: x_hat_data_ones})
            self.assertAllClose(results, 2)
            # 1 - -1 = 2
            results = sess.run(l, feed_dict={x: x_hat_data_ones, x_hat: x_data})
            self.assertAllClose(results, 2)

    # def test_rmse_batch(self):
    #     with self.test_session() as sess:
    #         x = tf.placeholder(tf.float32, (100, 64, 64, 3))
    #         x_hat = tf.placeholder(tf.float32, (100, 64, 64, 3))
    #         x_data = np.ones((100, 64, 64, 3))
    #         l = rmse(x, x_hat)
    #
    #         x_hat_data_ones = np.ones((100, 64, 64, 3))
    #         results = sess.run(l, feed_dict={x: x_data, x_hat: x_hat_data_ones})
    #         self.assertAllClose(results, 0)
    #
    #         x_hat_data_zeros = np.zeros((100, 64, 64, 3))
    #         results = sess.run(l, feed_dict={x: x_data, x_hat: x_hat_data_zeros})
    #         self.assertAllClose(results, 1)
