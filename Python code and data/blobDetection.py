import numpy as np
import pandas as pd
from config import paths
from utils import include_file_extension
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from sklearn.neighbors import BallTree


def convert_blobs_df(thunderstorm_output_path, filename, pixelsize=120., photons2adu=3.6, save_files=True):
    """
    Convert the output if ImageJ's ThunderSTORM output to a df with proper units. Also merge very close blobs.

    Args:
        thunderstorm_output_path (str): the full path to the csv output of ThunderSTORM
        filename (str): the name with which the output df is saved (in paths['blobs'] folder, see config.py)
        pixelsize (float): should correspond to ThunderSTORM's pixelsize camera calibartion parameter
        photons2adu (float): should correspond to ThunderSTORM's photons2adu camera calibartion parameter
        save_files (bool): whether to save output df in the folder.
    Returns:
        df (pandas df): df with all the blobs data
    """

    # for more about the ThunderSTORM outputs see:
    # https://www.neurocytolab.org/tscolumns/
    df = pd.read_csv(thunderstorm_output_path)
    df = df.rename(columns={"x [nm]": "x", "y [nm]": "y", "sigma [nm]": "sigma",
                            "intensity [photon]": "amplitude", "offset [photon]": "background",
                            "bkgstd [photon]": "background_std", "uncertainty [nm]": "uncertainty",
                            "chi2": "chi2"})
    df["id"] = df["id"].astype(int)
    df["frame"] = (df["frame"] - 1).astype(int)
    df[["y", "x", "sigma", "uncertainty"]] /= pixelsize
    df["sigma"] /= np.sqrt(2)  # now sigma is the estimate for both sigma_x and sigma_y (and not the total 2d std)
    df[["background", "amplitude", "background_std"]] /= photons2adu
    # blob = background + amplitude * normalized gaussian centered around x,y with variance sigma**2
    df['filename'] = filename

    # Also merge close blobs
    df = resolve_close_blobs(df)

    if save_files:
        df.to_csv(os.path.join(paths["blobs"], include_file_extension(filename, "csv")))  # save the dataframe
    return df


def resolve_close_blobs(blobs_df, dist_threshold=1.5, preference_pos="mean", parallelized=True):
    """
    Get a blob stack and for each very close blobs (in the same frame) and merge them to some effective blob.
    Args:
        blobs_df (pandas df): df with all blobs data (including x,y,id,frame)
        dist threshold (float): how close should the blobs be to be merged
        preference_pos (str): mean of blob position with "mean" or the top blob position with "top" for merged blobs
        parallelized (bool): run in this in parallel on different frames
    Returns:
        out_df (pandas df): df with all surviving blobs (merged or unmerged)
    """
    dfs_by_frame = [group for _, group in blobs_df.groupby("frame")]
    new_dfs_by_frame = np.empty(len(blobs_df.frame.unique()), dtype=object)

    if parallelized:
        with ProcessPoolExecutor() as executor:
            for n, new_frame_df in enumerate(
                    executor.map(resolve_close_blobs_in_a_frame, dfs_by_frame, repeat(dist_threshold),
                                 repeat(preference_pos))):
                new_dfs_by_frame[n] = new_frame_df
    else:
        for n, frame_df in enumerate(dfs_by_frame):
            new_frame_df = resolve_close_blobs_in_a_frame(frame_df, dist_threshold, preference_pos)
            new_dfs_by_frame[n] = new_frame_df
    out_df = pd.concat(new_dfs_by_frame).sort_values("id").reset_index(drop=True)
    return out_df


def resolve_close_blobs_in_a_frame(frame_df, dist_threshold, preference_pos):
    """
    Get a blob stack and for each very close blobs in a single frame, merge them to some effective blob.
    Args:
        frame_df (pandas df): df with all blobs data in a frame (including x,y,id)
        dist threshold (float): how close should the blobs be to be merged
        preference_pos (str): mean of blob position with "mean" or the top blob position with "top" for merged blobs
    Returns:
        out_frame_df (pandas df): frame_df with all surviving blobs (merged or unmerged)
    """

    frame_df["orig_blobs_id"] = ""
    XY = frame_df[["x", "y"]].values
    inds = frame_df.index
    neighbors = find_neighbors(XY, dist_threshold)
    inds_to_remove_mask = np.zeros(len(frame_df), dtype=bool)
    examined_groups = np.empty(len(frame_df), dtype=object)
    df_rows_to_add = np.empty(len(frame_df), dtype=object)
    for i, neighbors_of_i in enumerate(neighbors):
        if len(neighbors_of_i) > 1:
            blobs_to_merge = frame_df.loc[inds[neighbors_of_i]].sort_index()
            sorted_ids = np.sort(blobs_to_merge.id.values)

            # skip if we already saw this group of blobs
            continue_flag = False
            for j in range(i):
                if np.all(sorted_ids == examined_groups[j]):
                    continue_flag = True
                    break
            if continue_flag:
                continue

            new_blob = blobs_to_merge.loc[inds[i]:inds[i], :].copy()
            new_blob.x = blobs_to_merge.x.mean()
            if preference_pos == "mean":
                new_blob.y = blobs_to_merge.y.mean()
            elif preference_pos == "top":
                new_blob.y = blobs_to_merge.y.min()
            new_blob.sigma = np.sqrt((blobs_to_merge.sigma ** 2).sum())
            new_blob.amplitude = blobs_to_merge.amplitude.sum()
            new_blob.background = blobs_to_merge.background.mean()
            new_blob.background_std = np.sqrt((blobs_to_merge.background_std ** 2).sum())
            new_blob.chi2 = np.sqrt((blobs_to_merge.chi2 ** 2).sum())
            new_blob.uncertainty = blobs_to_merge.uncertainty.max()
            new_blob["orig_blobs_id"] = '_'.join(str(e) for e in blobs_to_merge.id.values)
            df_rows_to_add[i] = pd.DataFrame(new_blob)
            inds_to_remove_mask[neighbors_of_i] = True
            examined_groups[i] = sorted_ids

    isnone_arr = [(x is not None) for x in df_rows_to_add]
    if np.any(isnone_arr):
        df_rows_to_add = pd.concat(df_rows_to_add[isnone_arr])
    else:
        df_rows_to_add = pd.DataFrame([])
    filtered_frame_df = frame_df.copy()[np.logical_not(inds_to_remove_mask)]
    out_frame_df = pd.concat([filtered_frame_df, df_rows_to_add]).reset_index(drop=True)
    return out_frame_df


def find_neighbors(X, threshold):
    """
    Find neighbors close up to a threshold (L2 distance). This includes oneself as a neighbor.

    Args:
        X (ndarray): A Nx2 matrix of x,y coordinates.
        threshold (float): The threshold distance underneath which two points x1,y1 and x2,y2 are considered neighbors.
    Returns:
        neighbors_ind (ndarray): Length N ndarray where the nth element is a list of the indices (from 0 to N-1) of the
        neighbors of point xn,yn. n is always in neighbors_ind[n].
    """
    tree = BallTree(X)
    neighbors_ind = tree.query_radius(X, threshold, return_distance=False, count_only=False, sort_results=False)
    return neighbors_ind
