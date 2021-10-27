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
from adv_tools_PNAS.automap_tools import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
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

data = loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
mri_data = data['im'];
im_nbrs = [37, 50, 76];

# HCP_nbr = 1033
# data = loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
# mri_data = data['im'];
# im_nbrs = [2];


batch_size = 1;
mri_data.shape[0];

sess = tf.compat.v1.Session()


raw_f, _ = compile_network(sess, batch_size)

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)



automap_recon_advonly_arr = np.zeros(shape=(5,len(im_nbrs),128,128))
automap_recon_advgauss_arr = np.zeros(shape=(5,len(im_nbrs),128,128))

automap_recon_advonly_arr_input = np.zeros(shape=(5,len(im_nbrs),19710))
automap_recon_advgauss_arr_input = np.zeros(shape=(5,len(im_nbrs),19710))


input_scale = 1/1.0

# fID = open(join(dest_data, 'magnitude_e.txt'), 'w')
data_dict = {}
for r_value in range(0,5):

    # rr = runner.r[r_value];
    # rr = rr[pert_nbr, :, :];
    # rr = np.expand_dims(rr, 0)
    # e = sample(rr);
    # if r_value == 0:
    #     e_random = np.random.normal(loc=0, scale=0.01, size=e.shape)
    # else:
    #     e_random = np.random.normal(loc=e, scale=0.01, size=e.shape)
    # im_nbr= im_nbrs[0];
    # image = mri_data[im_nbr];
    # image = np.expand_dims(image, 0);
    # Ax = sample(image)
    # n_diff_e = l2_norm_of_tensor(e - e_random)
    # n_e = l2_norm_of_tensor(e);
    # n_y = l2_norm_of_tensor(Ax)
    # n_r = l2_norm_of_tensor(rr);
    # str1 = f"r: {r_value}, |r|: {n_r}  |e-e_rand|/|e|: {n_diff_e/n_e}, |e|: {n_e}";

    # data_dict[f"e{r_value}"] = e_random

    e_random = data_noise[f"e{r_value}"];
    
    # print(str1)
    # fID.write(str1 + "\n")

    for i in range(len(im_nbrs)):
        im_nbr= im_nbrs[i];
        image = mri_data[im_nbr];
        image = np.expand_dims(image, 0);

        Ax = sample(image)
        
        advonly_input = (Ax+e_random)*input_scale
        output_advonly = raw_f(advonly_input)

        advgauss_input = (Ax+e_random+np.random.normal(0,scale=0.01,size=Ax.shape))*input_scale
        output_advgauss = raw_f(advgauss_input)
        
        # print(output.shape)

        automap_recon_advonly_arr[r_value,i,:,:] = output_advonly
        automap_recon_advgauss_arr[r_value,i,:,:] = output_advgauss

        automap_recon_advonly_arr_input[r_value,i,:] = advonly_input
        automap_recon_advgauss_arr_input[r_value,i,:] = advgauss_input

        # im_rec = np.uint8(np.squeeze(255*scale_to_01(raw_f(Ax+e_random))));

        # pil_im_rec = Image.fromarray(im_rec);
        # pil_im_rec.save(join(dest_plots, f'im_rec_rID_{runner_id_automap}_automap_HCP_{HCP_nbr}_im_nbr_{im_nbr}_random_non_zero_mean_r_idx_{r_value}.png'))


fname_out_advonly = f'automap_output_advonly_orignet_rID_{runner_id_automap}_array.npy'
fname_out_advgauss = f'automap_output_adbgauss_orignet_rID_{runner_id_automap}_array.npy'

fname_out_advonly_input = f'automap_input_advonly_orignet_rID_{runner_id_automap}_array.npy'
fname_out_advgauss_input = f'automap_input_adbgauss_orignet_rID_{runner_id_automap}_array.npy'


np.save(join(dest_data, fname_out_advonly), automap_recon_advonly_arr)
np.save(join(dest_data, fname_out_advgauss), automap_recon_advgauss_arr)

np.save(join(dest_data, fname_out_advonly_input), automap_recon_advonly_arr_input)
np.save(join(dest_data, fname_out_advgauss_input), automap_recon_advgauss_arr_input)

# savemat(join(dest_data, f"automap_rID_{runner_id_automap}_random_pert.mat"), data_dict);
# fID.close()