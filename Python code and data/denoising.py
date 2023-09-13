import numpy as np
import pandas as pd
import os
import tensorflow as tf
from utils import include_file_extension, HiddenPrints
from config import paths, n2v_config
from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator


def suppress_tf_warnings():
    """
    Suppress annoying tensorflow warnings
    from https://stackoverflow.com/questions/55081911/tensorflow-2-0-0-alpha0-tf-logging-set-verbosity
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def limit_gpu_memory_tf():
    """
    Limit tensorflow GPU memory consumption
    from https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Limited memory growth for GPU {gpu}")


def train_n2v(images_file_name, model_name, train_epochs=100, N_images_for_train=None, random_seed=0):
    """
    Train a N2V model using input images and export it to a folder with name "model_name".
    The images filename rather than the images variable is required because N2V has to read the file itself.
    Comment 1: this is a slow process, do it on a computer with GPU and lots of RAM.
    Comment 2: see config.py for the N2V hyperparameters used

    Args:
        images_file_name (str): the name of the preprocessed images file ("tif" extension is not mandatory), this
        function looks for it in the paths['preprocessed'] folder (see config.py)
        model_name (str): the newly trained model will be saved with this name
        train_epochs (int): number of epochs for training
        N_images_for_train (int or None): how many images from the input preprocessed images stack should be used in
        practice for the training. If None then use the entire stack, which is ok for stacks with <300 images.
        random_seed (int): seed for randomization in the shuffling of images before training.
    """
    # tensorflow stuff
    suppress_tf_warnings()  # to prevent super pesky warnings
    limit_gpu_memory_tf()  # important so we won't use too much memory

    datagen = N2V_DataGenerator()

    # n2v dataloader requires the folder itself, not only the full path
    print("Loading preprocessed images for training...")
    imgs = datagen.load_imgs_from_directory(directory=paths['preprocessed'],
                                            filter=include_file_extension(images_file_name, "tif"), dims='TYX')[0]

    # Add a color axes because that's what N2V wants...
    if len(imgs.shape) <= 3:
        imgs = imgs[..., np.newaxis]

    N_images = imgs.shape[0]

    rng = np.random.default_rng(seed=random_seed)
    all_inds = rng.permutation(N_images)

    if N_images_for_train is None:
        N_images_for_train = N_images
    train_set_indices = all_inds[:N_images_for_train]
    # we use only one image for validation because it mostly slows the process and we judge the final results by eye
    val_set_indices = [all_inds[0]]

    patch_shape = n2v_config['n2v_patch_shape']

    # the n2v generate patches function prints too many stuff so we wrap it with something to suppress printing
    with HiddenPrints():
        X_train = datagen.generate_patches_from_list([imgs[train_set_indices, ...]],
                                                     shape=patch_shape)
        X_val = datagen.generate_patches_from_list([imgs[val_set_indices, ...]], shape=patch_shape)

    train_steps_per_epoch = int(X_train.shape[0] / n2v_config['train_batch_size'])
    config = N2VConfig(X_train, train_epochs=train_epochs,
                       train_steps_per_epoch=train_steps_per_epoch, **n2v_config)  # from the config file

    model = N2V(config, model_name, basedir=paths['n2v'])

    # train the model
    print("Starting to train the model...")
    training_history = model.train(X_train, X_val)

    # save the model
    model.export_TF(name=model_name,
                    description='This is the 2D Noise2Void example trained on miR data in python.',
                    authors=[""],
                    test_img=X_val[0, ..., 0], axes='YX',
                    patch_shape=patch_shape)
    save_training_history(training_history, model_name)  # also save the training history as a csv file
    print(f"Done, saved model {model_name}")


def save_training_history(training_history, model_name):
    """
    Save the n2v training history file as a csv
    Args:
        training_history (dict): output of the n2v model.train function
        model_name (str): name of the newly trained n2v model
    """
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(training_history.history)
    # save to csv:
    hist_csv_file = os.path.join(paths['n2v'], model_name, 'history.csv')
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def denoise_images(imgs_input, model_name):
    """
    Denoise the preprocessed images using the input n2v model
    Args:
        imgs_input (ndarray): N x height x width (preprocessed) images stack
        model_name (str): name of the model to use for denoising the images
    Returns:
        denoised_imgs (ndarray): denoised images in the same shape as imgs_input, and in np.uint16 pixels
    """
    limit_gpu_memory_tf()  # important so we won't use too much memory
    model = N2V(config=None, name=model_name, basedir=paths['n2v'])
    model.load_weights('weights_last.h5')

    with HiddenPrints():
        denoised_imgs = [model.predict(img, axes='YX', n_tiles=(2, 1))
                         for img in imgs_input]
    denoised_imgs = np.asarray(denoised_imgs)
    denoised_imgs = np.round(denoised_imgs)
    denoised_imgs = np.minimum(denoised_imgs, 2 ** 16 - 1)
    denoised_imgs = np.maximum(denoised_imgs, 0)
    denoised_imgs = denoised_imgs.astype(np.uint16)

    return denoised_imgs
