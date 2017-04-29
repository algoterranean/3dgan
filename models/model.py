import tensorflow as tf


class Model:
    def __init__(self):
        self._encoder = None
        self._decoder = None
        self._latent = None
        self._latent_loss = None
        self._generated_loss = None
        self._loss = None
        self._train_op = None

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

    # @property
    # def train_op(self):
    #     return self._train_op




    

    
        
        
