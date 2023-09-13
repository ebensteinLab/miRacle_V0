import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay


def transform_X_for_pca(X, threshold_percentile=50, normalize=True, symmetrize=True):
    """
    This transforms the crops to the format entered to the PCA and keeps an array of k, the normalization factor
    Args:
        X (ndarray): all the crops, of shape [N_crops, height, width]
        threshold_percentile (float): percentile used to determine the background threshold
        normalize (bool): normalize crops by dividing according to the mean intensity of the top blob?
        symmetrize (bool): symmetrize crops by folding the right half on top of the left half?
    Returns:
        X_out (ndarray): output of shape [N_crops, height*width / d] where d=2 if symmetrize else d=1
        k_arr (ndarray): normalization factor k of each crop (array of 1 if not normalize)
        out_mask (ndarray): array of bools of the images to keep. the keeping criterion is relevant only if normalize
        and is if the top blob mean intensity minus the subtracted background is positive.
    """
    half_width = int(X.shape[-1] / 2)
    X_out = X.reshape(X.shape[0], -1).astype(float).copy()
    if symmetrize:
        X_out = X_out[:, :half_width]
    image_edges = np.hstack([X[:, 0, :], X[:, 1:, -1], X[:, -1, :-1], X[:, 1:-1, 0]])  # pixels at image edges
    threshold = np.percentile(image_edges, threshold_percentile, axis=1)  # threshold of background to subtract
    # subtract background and set negative pixels to 0. transpositions are done for broadcasting purposes
    out_images = np.maximum((X.T - threshold).T, 0.)

    # normalize
    if normalize:
        eps = 10 ** -10  # a small number
        k_arr = X[:, 6:8, 4:6].mean(axis=(1, 2))  # this is called k in the paper. this estimates the max intensity.
        k_arr = np.maximum(k_arr - threshold, eps)  # to make sure we don't divide by k<=0
        # divide by k. transpositions are done for broadcasting purposes
        out_images = (out_images.T / k_arr).T

    # symmetrize
    if symmetrize:
        for j in range(half_width):
            out_images[:, :, j] += out_images[:, :, -j]
        out_images = out_images[:, :, :half_width].reshape(X_out.shape[0], -1)
    else:
        out_images = out_images.reshape(X_out.shape[0], -1)

    if normalize:
        out_mask = k_arr > eps
        X_out = out_images[out_mask]
        k_arr = k_arr[out_mask]
    else:
        out_mask = np.ones(len(X_out), dtype=bool)
        k_arr = np.ones(len(X_out))
    return X_out, k_arr, out_mask


def augment_train_data(crops, df, noise_strength, random_seed, n_augmentations, augment_mir_types=(0, 1, 2)):
    """
    This augments the training data. This only augments the miR and not the noise according to augment_mir_tpes
    Args:
        crops (ndarray): all the crops, of shape [N_crops, height, width]
        df (pandas df): contains the information regarding each crop
        noise_strength (float): std of the zero-mean gaussian to be added to each crop
        augment_mir_types (array): which mir types to augment. we don't augment noise miRs
        random_seed (int): which random seed to use for the augmentation
        n_augmentations (int): how many augmentations to make to each crop of the appropriate mir type
    Returns:
        crops (ndarray): the new array of crops with augmentation
        df (pandas df): the new df including augmentation

    """
    df = df.copy()
    df["is_augmentation"] = False
    raw_df = df.copy()
    raw_crops = crops.copy()

    rs = np.random.RandomState(random_seed)

    for mir_type in augment_mir_types:
        isreallymir = raw_df.is_really_mir.values.astype(bool)
        mirtype = raw_df.mir_type.values
        for n in range(n_augmentations):
            if mir_type == "noise":
                extra_crops = raw_crops[~isreallymir].copy()
                extra_df = raw_df[~isreallymir].copy()
            else:
                extra_crops = raw_crops[(mirtype == mir_type) & isreallymir].copy()
                extra_df = raw_df[(mirtype == mir_type) & isreallymir].copy()
            extra_df["is_augmentation"] = True
            noise = rs.normal(size=extra_crops.shape, scale=noise_strength)

            extra_crops += np.abs(noise).astype(np.uint16) * (np.sign(noise) > 0)
            extra_crops -= np.abs(noise).astype(np.uint16) * (np.sign(noise) < 0)
            crops = np.vstack([crops, extra_crops])
            df = pd.concat([df, extra_df]).reset_index(drop=True)

    return crops, df


def train_model(crops, df, pca_components=20, C=.05, train_percent=90.,
                random_seed=None, noise_strength=10.,
                n_augmentations=6, normalize=True, symmetrize=True):
    """
    This trains a model given crops and a dataframe.
    crops is [N_crops,height,width] in shape
    df must contain columns "mir_type" and "is_really_mir".
    This augments the data, then transforms it for PCA then trains the PCA and SVM based on the labeled classes.

   Args:
        crops (ndarray): all the crops, of shape [N_crops, height, width]
        df (pandas df): contains the information regarding each crop
        pca_components (int): hyperparameter of how many pca components to learn
        C (float): hyperparameter which is inverse to the regularization strength of the SVM classifier
        train_percent (float): what percentage of the data to have as the training set (the rest is validation)
        random_seed (int or None): which random seed to use for the train/validation split and for the augmentation
            (randomized seed if None)
        noise_strength (float): std of the zero-mean gaussian to be added to each crop during augmentation
        n_augmentations (int): how many augmentations to make to each crop of the appropriate mir type
        normalize (bool): normalize crops by dividing according to the mean intensity of the top blob?
        symmetrize (bool): symmetrize crops by folding the right half on top of the left half?
    Returns:
        pca (sklearn.decomposition.PCA instance): learned pca model
        classifier (sklearn.Pipieline.pipeline instance): pipeline of learned StandardScaler + SVM model
    """
    N = len(df)
    if random_seed is not None:
        np.random.seed(random_seed)
    # inds_train = np.sort(np.random.choice(N, int(train_percent / 100 * N), replace=False))
    train_mask = np.zeros(N, dtype=bool)
    train_mask[:int(train_percent / 100 * N)] = True
    np.random.shuffle(train_mask)
    validation_mask = ~train_mask

    df_train = df.iloc[train_mask].reset_index(drop=True)
    df_validation = df.iloc[validation_mask].reset_index(drop=True)
    crops_train = crops[train_mask]
    crops_validation = crops[validation_mask]

    crops_train, df_train = augment_train_data(crops_train, df_train, noise_strength=noise_strength,
                                               random_seed=random_seed,
                                               n_augmentations=n_augmentations)

    X_train, k_arr_train, keep_mask_train = transform_X_for_pca(crops_train, normalize=normalize,
                                                                symmetrize=symmetrize)
    X_validation, k_arr_validation, keep_mask_validation = transform_X_for_pca(crops_validation, normalize=normalize,
                                                                               symmetrize=symmetrize)
    df_train = df_train.loc[keep_mask_train].reset_index(drop=True)
    df_validation = df_validation.loc[keep_mask_validation].reset_index(drop=True)

    k_arr_train = k_arr_train.reshape(-1, 1)
    k_arr_validation = k_arr_validation.reshape(-1, 1)

    # 0 is the noise class and 1+ 0/1/2 is a miR class
    y_train = (df_train.is_really_mir * (1 + df_train.mir_type)).values
    y_validation = (df_validation.is_really_mir * (1 + df_validation.mir_type)).values

    pca = decomposition.PCA(n_components=pca_components, whiten=False, svd_solver="full")
    train_X_transformed = pca.fit_transform(X_train)
    validation_X_transformed = pca.transform(X_validation)

    train_X_transformed = np.hstack([train_X_transformed, k_arr_train])
    validation_X_transformed = np.hstack([validation_X_transformed, k_arr_validation])

    classifier = make_pipeline(StandardScaler(), SVC(C=C, probability=True, decision_function_shape="ovo"))
    classifier.fit(train_X_transformed, y_train)

    disp_confusion_matrices(classifier, df_train, train_X_transformed, validation_X_transformed, y_train, y_validation)

    return pca, classifier


def disp_confusion_matrices(classifier, df_train, train_X_transformed, validation_X_transformed, y_train, y_validation):
    """
    This displays the confusion matrix for the train set and for the validation set
    Args:
        classifier (sklearn.Pipieline.pipeline instance): pipeline of learned StandardScaler + SVM model
        df_train (pandas df): all the data regarding training crops (including augmentation)
        train_X_transformed (ndarray): output of the pca for the training crops
        validation_X_transformed (ndarray): output of the pca for the validation crops
        y_train (ndarray): the classes label (0/1/2/3) of each training crop
        y_validation (ndarray): the classes label (0/1/2/3) of each validation crop
    """
    # display for the raw train set
    y_pred_train = classifier.predict(train_X_transformed)
    y_train = y_train[~df_train["is_augmentation"]]  # don't include augmentations
    y_pred_train = y_pred_train[~df_train["is_augmentation"]]
    disp = ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train)
    disp.ax_.set_title("Labeled train set (from V_TIMDER)")
    # display for the validation set
    y_pred_validation = classifier.predict(validation_X_transformed)
    disp = ConfusionMatrixDisplay.from_predictions(y_validation, y_pred_validation)
    disp.ax_.set_title("Labeled validation set (from V_TIMDER)")
