"""
This script reads the worst-case perturbations produced by the script
'Demo_test_automap_stability.py' and 'Demo_test_automap_stability_knee.py',
samples these perturbations, and use the sampled perturbations as the mean for
randomly drawn gaussian noise vectors.  These noise vectors are then added to
the measurements of other images than the worst-case perturbations were
computed for.

Adjust the variables 'runner_id_automap' and 'pert_nbr', to test the knee image
"""
import os

use_gpu = False
# compute_node = 1
if use_gpu:
    # os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    # print('Compute node: {}'.format(compute_node))
    os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"


import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path
from PIL import Image

from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor, scale_to_01
from adv_tools_PNAS.Runner import Runner;
from adv_tools_PNAS.Automap_Runner import Automap_Runner;
from adv_tools_PNAS.RESP_automap_tools_natmod import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import sys

runner_id_automap = 5; # Set to 12, to test the knee image perturbations
pert_nbr = 2; # Set to 0 to test the knee image perturbations
print(f'runner id: {runner_id_automap}, pert nbr: {pert_nbr}')

dest_plots = 'plots_non_zero_mean';
dest_data = 'data_non_zero_mean';
src_noise = 'data_non_zero_mean';


if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);
if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

N = 128

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

runner = load_runner(runner_id_automap);
print(runner)
HCP_nbr = 1002



fname_data = f'automap_rID_{runner_id_automap}_random_pert.mat'
data_noise = scipy.io.loadmat(join(src_noise, fname_data))

HCP_nbr = 1002
data = loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
mri_data = data['im'];
im_nbrs = [37, 50, 76];

# HCP_nbr = 1033
# data = loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
# mri_data = data['im'];
# im_nbrs = [3];


batch_size = 1;
mri_data.shape[0];

sess = tf.compat.v1.Session()


raw_f, _ = compile_network(sess, batch_size)

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)


# ADV-ONLY VS ADV-GAUSS


fname_out_advonly = f'automap_output_advonly_orignet_rID_{runner_id_automap}_array.npy'
fname_out_advgauss = f'automap_output_adbgauss_orignet_rID_{runner_id_automap}_array.npy'
fname_out_onlygauss = f'automap_output_onlygauss_orignet_rID_{runner_id_automap}_array.npy'

amap_orig_advonly_arr = np.load(join(dest_data, fname_out_advonly))
amap_orig_advgauss_arr = np.load(join(dest_data, fname_out_advgauss))
amap_orig_onlygauss_arr = np.load(join(dest_data, fname_out_onlygauss))

# amap_orig_advonly_arr[4,2,:,:]



def rmse_comp(ref,inp):
    
    ref_norm = ref-ref.mean()
    
#     inp_mag = np.abs(inp_im)
    inp_norm = inp-inp.mean()
    inp_norm = inp_norm / (inp_norm.max()-inp_norm.min())

    rmse = np.sqrt(((inp_norm-ref_norm)**2).mean())
    
    return rmse

def ssim_comp(ref,inp):
    ref_norm = ref-ref.mean()
    
#     inp_mag = np.abs(inp_im)
    inp_norm = inp-inp.mean()
    inp_norm = inp_norm / (inp_norm.max()-inp_norm.min())

    ssim_out = ssim(ref_norm,inp_norm)

    return ssim_out



num_imgs = 3
offset = 2

fig, axs = plt.subplots(2, 3,figsize=(15*1.25,10*1.25))

r_vals=[2,3,4]


rmse_array = np.zeros((2,3))
ssim_array = np.zeros((2,3))


for i in range(num_imgs):

    refimg = mri_data[76,:,:]

    r_value = r_vals[i]
    # print(axs[1,i])
    axs[0,i].imshow(amap_orig_advonly_arr[r_value,2,:,:],cmap='gray')
    axs[1,i].imshow(amap_orig_advgauss_arr[r_value,2,:,:],cmap='gray')

    rmse_array[0,i] = rmse_comp(refimg,amap_orig_advonly_arr[r_value,2,:,:])
    rmse_array[1,i] = rmse_comp(refimg,amap_orig_advgauss_arr[r_value,2,:,:])

    ssim_array[0,i] = ssim_comp(refimg,amap_orig_advonly_arr[r_value,2,:,:])
    ssim_array[1,i] = ssim_comp(refimg,amap_orig_advgauss_arr[r_value,2,:,:])
    
    print('adv_only rmse:', rmse_array[0,i])
    print('adv_gauss rmse:', rmse_array[1,i])

    print('adv_only ssim:', ssim_array[0,i])
    print('adv_gauss ssim:', ssim_array[1,i])

    print('-----')

    axs[0,i].axis('off')
    axs[1,i].axis('off')

fig.savefig('figure4.png')


import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# plt.figure(figsize=(12,10))


fig, ax = plt.subplots(1, 1,figsize=(6*1.5,5*1.5))


x = np.arange(3)
width = 0.30
  
plt.bar([-1],0.026, width,color='green')
plt.bar(x-0.15, rmse_array[0,:], width)
plt.bar(x+0.15, rmse_array[1,:], width)


plt.ylim([0,0.20])

plt.ylabel('RMSE')
plt.legend(['Gaussian Noise only','Adversarial Noise only','Adversarial + Gaussian Noise'],loc='upper right', frameon=False)
ax.set_xticks([-1,0,1,2])
ax.set_xticklabels(['$r_0$','$r_1$','$r_2$','$r_3$'])

frame1 = plt.gca()
# frame1.axes.get_xaxis().set_visible(False)

ax.set_axisbelow(True)
ax.grid(color='0.75', linestyle='-', axis='y', linewidth=1)

fig.savefig('figureS1a.png')

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# plt.figure(figsize=(12,10))


fig, ax = plt.subplots(1, 1,figsize=(6*1.5,5*1.5))


x = np.arange(3)
width = 0.30
  
plt.bar([-1],0.92, width,color='green')
plt.bar(x-0.15, ssim_array[0,:], width)
plt.bar(x+0.15, ssim_array[1,:], width)


plt.ylim([0,1.0])

plt.ylabel('SSIM')
plt.legend(['Gaussian Noise only','Adversarial Noise only','Adversarial + Gaussian Noise'],loc='upper right', frameon=False)
ax.set_xticks([-1,0,1,2])
ax.set_xticklabels(['$r_0$','$r_1$','$r_2$','$r_3$'])

frame1 = plt.gca()
# frame1.axes.get_xaxis().set_visible(False)

ax.set_axisbelow(True)
ax.grid(color='0.75', linestyle='-', axis='y', linewidth=1)

fig.savefig('figureS1b.png')









