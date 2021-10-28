# AUTOMAP Matters Arising Response

This repository hosts the source code and data related to the authors' Response paper to the Matters Arising entitled 'Deep learning through domain-transform manifold learning for image reconstruction is not robust.' 

The codebase structure is derived from the Matters Arising repository at https://github.com/vegarant/automap_not_robust.

## Setup

The original AUTOMAP network weights for the undersampled Cartesian model can be downloaded [here](https://www.mn.uio.no/math/english/people/aca/vegarant/data/cs_poisson_for_vegard.h5), The retrained model weights can be downloaded [here](https://drive.google.com/file/d/1EdJlLaY2bvfPF1sSgMghiQhaZpZGncNx/view?usp=sharing).  Both models should be downloaded and placed in the `NMA_Response_Storage` folder. The data for the global and local robustness experiments for Figure 2 can be downloaded [here](https://drive.google.com/drive/folders/1uziWP5A2tWju3k67_tEapxFY-BSb7ZDh?usp=sharing) and also placed in the `NMA_Response_Storage` folder.

As stated in the Matters Arising repository, after downloading the data, modify the paths in the file `adv_tools_PNAS/automap_config.py` to link all relevant paths to the data, and please install the [UiO-CS/optimization](https://github.com/UiO-CS/optimization) and [tf-wavelets](https://github.com/UiO-CS/tf-wavelets) packages.

Tensorflow version 1.14 is best for compatibility.

## File Overview and Script Ordering

The scripts are structured in a manner such that it should be straightforward to reproduce the results in our paper, organized primarily by the Figure count.  The main scripts are prefixed with `NMARESP_Fig*` and there is often `Step[#]` to indicate the scripts should be run in stepwise order as dependent data is generated in sequence.
