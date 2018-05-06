import tensorflow as tf
import hem

def get_model(model_name):
    model_plugins = hem.search_for_plugins(plugin_dir='hem/models',
                                           plugin_module='hem.models',
                                           plugin_name='ModelPlugin')
    return model_plugins[model_name]


class ModelPlugin:
    def __init__(self):
        self.name = None

    @staticmethod
    def arguments():
        return {}

    # @staticmethod
    # def create(x, args):
    #     pass

    def train(self, sess, args, feed_dict):
        pass

    # @staticmethod
    # def inference():
    #     pass


if __name__ == '__main__':
    import hem
    loaded_plugins = hem.search_for_plugins(plugin_dir='hem/models',
                                            plugin_module='hem.models',
                                            plugin_name='ModelPlugin')
    model = loaded_plugins['test']
    print('loaded plugins:', loaded_plugins)
    print('model:', model)
