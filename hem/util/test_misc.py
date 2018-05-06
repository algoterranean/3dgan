import tensorflow as tf

from hem.util.misc import chunks, collection_to_dict


class TestUtilMisc(tf.test.TestCase):
    def test_collection_to_dict(self):
        with self.test_session():
            # add two variables to the collection
            a = tf.constant(1, dtype=tf.int32, name='a')
            b = tf.constant([1, 2, 3], dtype=tf.int32, name='b')
            tf.add_to_collection('tests', a)
            tf.add_to_collection('tests', b)
            # get collection
            c = tf.get_collection('tests')
            # convert to dict
            d = collection_to_dict(c)
            # ensure the items in the dict-collection match the originals
            assert d[u'a'] == a
            assert d[u'b'] == b

    def test_chunks(self):
        x = list(range(10))
        # split x into (two) lists of size 5
        y = list(chunks(x, 5))
        assert len(y) == 2
        assert len(y[0]) == 5
        assert y[0] == [0, 1, 2, 3, 4]
        assert len(y[1]) == 5
        assert y[1] == [5, 6, 7, 8, 9]



