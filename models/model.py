import tensorflow as tf


class Model:
    def __init__(self):
        self._encoder_node = None
        self._decoder_node = None
        self._latent_node = None
        self._train_op = None
        self._latent_loss_node = None

    @property
    def encoder(self):
        return self._encoder_node

    @property
    def decoder(self):
        return self._decoder_node

    @property
    def latent(self):
        return self._latent_node

    @property
    def loss(self):
        return self._loss_node

    @property
    def latent_loss(self):
        return self._latent_loss_node

    @property
    def generated_loss(self):
        return self._generated_loss_node

    # @property
    # def train_op(self):
    #     return self._train_op




    

    
        
        
