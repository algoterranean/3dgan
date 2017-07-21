import tensorflow as tf
from util.misc import collection_to_dict


class TestGetCollectionDict(tf.test.TestCase):

    def testd(self):
        with self.test_session():
            a = tf.constant(1, dtype=tf.int32, name='a')
            b = tf.constant([1, 2, 3], dtype=tf.int32, name='b')
            tf.add_to_collection('tests', a)
            tf.add_to_collection('tests', b)

            c = tf.get_collection('tests')
            d = collection_to_dict(c)
            assert d[u'a'] == a
            assert d[u'b'] == b
            print(d)

    def teste(self):
        print('hi')


# if __name__ == '__main__':
#     tf.test.main()