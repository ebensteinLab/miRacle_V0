import os

DATA_DIR = "D:/miRacle_data"  # enter the full path, not shortcuts like ./Data!

IMAGES_DIR = os.path.join(DATA_DIR, "Images")
MODELS_DIR = os.path.join(DATA_DIR, "Models")
V_TIMDER_DIR = os.path.join(DATA_DIR, "v_timder")

paths = {
    "raw": os.path.join(IMAGES_DIR, "Raw"),
    "preprocessed": os.path.join(IMAGES_DIR, "Preprocessed"),
    "denoised": os.path.join(IMAGES_DIR, "Denoised"),
    "blobs": os.path.join(DATA_DIR, "Blobs"),
    "crops": os.path.join(DATA_DIR, "Crops"),
    "n2v": os.path.join(MODELS_DIR, "n2v"),
    "ml": os.path.join(MODELS_DIR, "ml"),
    "v_timder_tables": os.path.join(V_TIMDER_DIR, "Tables"),
    "v_timder_reference": os.path.join(V_TIMDER_DIR, "Reference Crops"),
    "v_timder_noisy": os.path.join(V_TIMDER_DIR, "Noisy Crops"),
    "v_timder_denoised": os.path.join(V_TIMDER_DIR, "Denoised Crops")
}

# n2v training configurations
n2v_config = {
    "unet_kern_size": 3,
    "train_loss": 'mae',
    "batch_norm": True,
    "train_batch_size": 128,
    "n2v_perc_pix": 0.198,
    "n2v_patch_shape": (64, 64),
    "n2v_neighborhood_radius": 3,  # optimized for our miR data
    "unet_residual": False,  # this is n2v2
    "n2v_manipulator": "median",  # this is n2v2; use "uniform_withCP" for n2v1
    "blurpool": True,  # this is n2v2; use False for n2v 1
    "skip_skipone": False,  # this is n2v2; use "uniform_withCP" for n2v1
}

color_scheme = [
    (0.839, 0.152, 0.156),  # red
    (0.172, 0.627, 0.172),  # green
    (0.11, 0.42, 0.64),  # modified blue
]
