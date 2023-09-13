# miRacle_V0
Code for miRacle paper


## Setup:
1. Copy the folder "Data folders" wherever you wish to store the data.
2. Update DATA_DIR in `config.py` to the path to the above folder
3. Put your raw images in the Images/Raw folder (a sample file is provided)
4. Requirements:
   * For the Image_Processing notebook, install all the requirements in the [miRacle.yml] file (recommended with conda).
   * For the other notebooks, only standard Python libraries are required.

## Image processing:
Follow the Image_Processing notebook to denoise the raw images, then detect blobs using ThunderSTORM (requires using ImageJ externally). Afterwards the ThunderSTORM output can be converted to a more suitable format and then crops can be extracted from it for both V_TIMDER and for automatic classification. Comment: You can either train a new Noise2Void denoising model or use the provided one. In addition, example outputs of this notebook are provided.

## V_TIMDER:
Use the V_TIMDER notebook to quickly and conveniently classify crops to one of three miR categories.
Because of file sizes of the larger crops, the inputs for V_TIMDER are not provided but they can be easily generated using the Cropping notebook (whose inputs are provided). Comment: this works best on windows or linux

## Data analysis:
Follow the Data_Analysis notebook to train a PCA+SVM model or use an existing one (provided) to classify the crops to noise or one of the three miR types.
The inputs for this notebook are provided.
