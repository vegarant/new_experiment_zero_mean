# AUTOMAP Matters Arising Response

This repository hosts the source code related to SI Figure 3 in "Supplementary Information for the paper 'On non-robustness,
hallucinations and unpredictability in AI for imagingi'"

The codebase structure is derived from the Matters Arising repository at https://github.com/MattRosenLab/NMA_Response

## Setup

The original AUTOMAP network weights for the undersampled Cartesian model can be downloaded [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/cs_poisson_for_vegard.h5), and the augmented model weights can be downloaded [here](https://drive.google.com/file/d/1EdJlLaY2bvfPF1sSgMghiQhaZpZGncNx/view?usp=sharing).  Both models should be downloaded and placed in the same folder. Modify `adv_tools_PNAS/automap_config.py` to point to this folder.
The data genereated while running these experiments can be downloaded from [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/storage_automap_final.zip).

To run the LASSO code, please install the [UiO-CS/optimization](https://github.com/UiO-CS/optimization) and [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) packages.

Tensorflow version 1.14 is best for compatibility.

## File Overview and Script Ordering

The files should be run in this order

1. NMARESP_Fig5_Step1_gnoise_levels_exp{X}.py
2. NMARESP_Fig5_Step2_gNoise_LASSO_exp{X}.py
3. NMARESP_Fig5_Step3_and_Fig6_exp{X}.py

where X represent the experiment number.

* Experiment 1: Augmented AUTOMAP, left image
* Experiment 2: Original AUTOMAP, left image
* Experiment 3: Augmented AUTOMAP, right image
* Experiment 4: Original AUTOMAP, right image

