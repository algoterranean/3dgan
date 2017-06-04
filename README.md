
# Autoencoders and GANs in Tensorflow

Example autoencoder and GAN implementations in Tensorflow. 

## Models

This code implements the following CNN models in Tensorflow:

- Vanilla Autoencoder
- Variational Bayes Autoencoder
- Vanilla GAN
- Wasserstein GAN
- Improved Wasserstein GAN

## Datasets

This code supports the following datasets out of the box:

- MNIST
- CIFAR-10
- Floorplans

Additional datasets can be added easily from the command line. The file format is assumed to be a `TFRecord`s file, 
where each entry has an `image_raw` key describing the image contents. The format for this image should be a
`width x height x depth` Numpy array that has been serialized to a string. See `data\cifar_tfrecords.py` for example
code on how to do this. 

## Features

In addition, this code supports the following features:

- Numerous command-line arguments for configuring the models and training episodes.
- Visualization of weights, activations, 'best-fit' (gradient descent in image space), and generative samples for already trained models.
- Detailed Tensorboard summaries for most of the above visualizations, plus layer statistics (in histogram and scalar format) like sparsity, gradients, etc.
- Handy abstractions for common ops, utility functions, model methods, etc.
- Support for multiple GPUs.
- Ability to resume training from disk.

## To Use

To run this code, either run `train.py` (to initialize and train a model) or `visualize.py` (for additional visualizations from a checkpointed model). 
Command line arguments and information on them are available via the `-h` flag. 

For example, to train the Improved WGAN model using ADAM on the CIFAR-10 dataset using 2 GPUs:

```
python train.py --model wgan \
                --data cifar \
                --optimizer adam \
                --beta1 0.5 \
                --beta2 0.9 \
                --lr 1e-4 \
                --batch_size 512 \
                --epochs 100 \
                --n_gpus 2 \
                --dir workspace/cifar_test
```

The `dir` argument points to the location where you wish to store the training checkpoints, Tensorboard summaries, and visualization outputs.

It is recommended that, during training, you run Tensorboard to monitor the progress (point it to same dir you passed to `train.py`):

```
tensorboard --logdir workspace/cifar_test
```

