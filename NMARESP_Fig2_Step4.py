import os

# use_gpu = False
# # compute_node = 1
# if use_gpu:
#     # os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
#     # print('Compute node: {}'.format(compute_node))
#     os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
# else: 
#     os.environ["CUDA_VISIBLE_DEVICES"]= "-1"


import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path
from PIL import Image

from adv_tools_PNAS.automap_config import src_weights, src_data;

from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import sys

from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

rmse_vec = np.load(join(src_data,'NMARESP_rmse_1m_images.npy'))

hist = np.histogram(rmse_vec,np.arange(0.015,0.02,0.00001))

plt.figure(figsize=(9,6))
plt.plot(hist[1][1:],hist[0])


plt.ylabel('Count')
plt.xlabel('RMSE')

plt.savefig('figure2a.png')

rmse_std = rmse_vec.std()

adv_noise_rmse = 0.03263265387668578

print((adv_noise_rmse-rmse_vec.mean())/rmse_std)

matplotlib.rcParams.update({'font.size': 16})
rmse_vec = np.load(join(src_data,'NMARESP_rmse_1m_images.npy'))

hist = np.histogram(rmse_vec,np.arange(0.00,0.04,0.00001))

plt.figure(figsize=(9*1.25,6*1.25))
plt.plot(hist[1][1:],hist[0])

plt.ylabel('Count')
plt.xlabel('RMSE')
plt.vlines(adv_noise_rmse,0,20000,'red','--')
plt.legend(['Gaussian Noise','Adversarial Noise'],loc='upper left', frameon=False)

plt.savefig('figure2b.png')

global_metric = np.load(join(src_data,'NMARESP_global_ratio.npy'))

local_metric = np.load(join(src_data,'NMARESP_local_ratio.npy'))

matplotlib.rcParams.update({'font.size': 22})

plt.figure(figsize=(6*1.5,6*1.5))
plt.plot(global_metric)

plt.ylabel('Global $L_\phi$')
plt.xlabel('Samples')

plt.savefig('figure2c.png')

matplotlib.rcParams.update({'font.size': 22})
plt.figure(figsize=(6*1.5,6*1.5))
plt.plot(local_metric)

plt.ylabel('Local $L_\phi$')
plt.xlabel('Samples')

plt.savefig('figure2d.png')


