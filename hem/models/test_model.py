import tensorflow as tf
import hem

class TestModel(hem.ModelPlugin):
    name = 'test'

    @staticmethod
    def arguments():
        args = { '--test_arg': { 'type': int,
                                 'default': 20,
                                 'help': """Example test model argument""" }}
        return args

    @staticmethod
    def create(x, args):
        print('creating model')


    @staticmethod
    def train(sess, args, feed_dict):
        print('training one step')




