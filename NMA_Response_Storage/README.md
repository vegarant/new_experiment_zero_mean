# Overview of data

----------------------

* `runners` - Contains the runner objects used to generate the worst-case perturbations for AUTOMAP. Runner 5 is used for the main paper, and Runner 12 is used for the figure in the supplementary information.
* `fastMRI_mask` - The sampling masks with 4 and 8 times acceleration, generated using code from the fastMRI challenge. 
* `HCP_mgh_1002_T2_subset_N_128.mat` Counting from 0, we use the following images in the figures
    - Figure 2: Image number 37, 50 and 76.
    - Extended Data Figure 1: 49, 116, 150.
* `dataset1.mat` - This file contains modified images from the HCP 1002 dataset. Image 49 is used for Extended Data Figure 1.
* `HCP_mgh_1004_T2_subset_N_128.mat` - This file contains two images. The first image is from the file with HCP nbr 1003, and the second image is from GE Healthcare. The second image is used for Extended Data Figure 3.
* `HCP_mgh_1033_T2_subset_N_128.mat` - This file contains the image used in Figure 1 and Extended Data Figure 2. 
* `k_mask_idx.mat` - The sampling pattern used for AUTOMAP in the format AUTOMAP requires.
* `k_mask.mat` - The sampling pattern used for AUTOMAP in the format LASSO requires.

----------------------


