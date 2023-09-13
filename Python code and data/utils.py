import numpy as np
import tifffile
import os
import sys
import pickle
from config import paths


def read_tif(imgpath):
    """
    Read a tif image file

    Args:
        imgpath (str): The input path to be modified. Does not have to include the "tif" extension
    Returns:
        imagesArray (ndarray): The loaded images in either N x height x width shape (N is the number of image in the
        stack), or in height x width shape if it's a single image.
    """
    imagesArray = np.asarray(tifffile.imread(include_file_extension(imgpath, "tif")))  # read with tifffile
    return imagesArray


def save_tif(imgpath, img):
    """
    Read a tif image file

    Args:
        imgpath (str): The full path including the image name to save the image at. "tif" extension is not mandatory.
        img (ndarray): A N x height x width ndarray where N is the number of images in the stack.
    """
    tifffile.imwrite(include_file_extension(imgpath, "tif"), img)


def include_file_extension(path, extension):
    """
    Ensure a path ends with the desired extension.

    Args:
        path (str): The input path to be modified.
        extension (str): The desired extension (without the ".")
    Returns:
        path: The modified path with the desired extension.
    """
    base, ext = os.path.splitext(path)
    if ext[1:].lower() != extension.lower():
        path = base + "." + extension.lower()
    return path


def save_sklearn_model(model, model_name):
    """
    Dumps sklearn model to a pickle file in the paths['ml'] folder (see config.py)

    Args:
        model: typically a (sklearn.decomposition.PCA,sklearn.pipeline.Pipeline) tuple but this could be anything
        model_name: name of the model to save (including ".pkl" extension)
    """
    model_path = os.path.join(paths['ml'], model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=4)


def load_sklearn_model(model_name):
    """
    Reads a sklearn model from a pickle file in the paths['ml'] folder (see config.py)
    WARNING: make sure to read only files you trust (e.g. models you saved)

    Args:
        model_name: name of the model to load (including ".pkl" extension)
    """
    model_path = os.path.join(paths['ml'], model_name)

    with open(model_path, 'rb') as f:
        model = pickle.load(f, fix_imports=False)
    return model


class HiddenPrints:
    """
    This silences annoying prints in one of the N2V function
    """

    # from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
