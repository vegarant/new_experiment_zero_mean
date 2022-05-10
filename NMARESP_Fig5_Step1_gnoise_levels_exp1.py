import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path
from PIL import Image

from adv_tools_PNAS.automap_config import src_weights, src_data;
from adv_tools_PNAS.RESP_adversarial_tools import l2_norm_of_tensor, scale_to_01
from adv_tools_PNAS.RESP_Runner import Runner;
from adv_tools_PNAS.RESP_Automap_Runner import Automap_Runner;
from adv_tools_PNAS.RESP_automap_tools_natmod import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;


from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import sys
use_gpu = True
compute_node = 1
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

filename3 = join(src_data,'HCP_mgh_1033_T2_subset_N_128.mat')

data = loadmat(filename3)['im']

batch_size = 1;

gt_imgs = np.repeat(np.expand_dims(data[2,:,:],0),batch_size,axis=0)

sess = tf.compat.v1.Session()
raw_f, _ = compile_network(sess, batch_size) # Complies Augmented AUTOMAP

sample_im = lambda x: sample_image(x, k_mask_idx1, k_mask_idx2)
f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)


mri_data = sample_im(gt_imgs)

print(mri_data.shape)
print(gt_imgs.shape)

counter = 0
jj = 0

def rmse_comp(ref,inp):
    
    ref_norm = ref-ref.mean()
    
#     inp_mag = np.abs(inp_im)
    inp_norm = inp-inp.mean()
    inp_norm = inp_norm / (inp_norm.max()-inp_norm.min())
    rmse = np.sqrt(((inp_norm-ref_norm)**2).mean())
    
    return rmse

input_scale = 1.5


noise_levels = [0.001, 0.002, 0.004, 0.006, 0.008]

target_l2_ratios = [0.01, 0.02, 0.04,0.06,0.08]

num_times = len(noise_levels)
num_images = num_times*batch_size
rmse_gnoise_levels = np.zeros((num_images))


noise_gnoise_full = np.zeros((num_images,19710))
noisy_input_gnoise_full = np.zeros((num_images,19710))
automap_output_full = np.zeros((num_images,16384))


for ii in range(0,num_times):
    

    noise = np.random.normal(0,noise_levels[ii],(batch_size,19710))


    input_clean = mri_data[counter:counter+batch_size,:];
  
    output_clean = np.reshape(gt_imgs[counter:counter+batch_size,:,:],(batch_size,16384))
    
    
    
    noise_l2 = np.sqrt(np.sum(np.abs(noise[0,:].flatten())**2))
    input_l2 = np.sqrt(np.sum(np.abs(input_clean[0,:].flatten())**2))

    intermed_l2_ratio = noise_l2/input_l2
    
    noise = noise*target_l2_ratios[ii]/(intermed_l2_ratio)

    noise_l2 = np.sqrt(np.sum(np.abs(noise[0,:].flatten())**2))
    tuned_l2_ratio = noise_l2/input_l2

    print('|r|/|x|:',tuned_l2_ratio)

    input_noisy = (input_clean+noise)*input_scale 

    output_noisy = raw_f(input_noisy)
    

    output_noisy = np.squeeze(output_noisy)

    output_noisy =  np.reshape(output_noisy,(batch_size,16384))
    
    rmse_batch = np.zeros(batch_size)

    for kk in range(batch_size):
        rmse_batch[kk] = rmse_comp(output_clean[kk,:],output_noisy[kk,:])

    rmse_gnoise_levels[jj:jj+batch_size] = rmse_batch 
    noise_gnoise_full[jj:jj+batch_size,:] = noise
    noisy_input_gnoise_full[jj:jj+batch_size,:] = input_noisy
    automap_output_full[jj:jj+batch_size,:] = output_noisy

    print(ii)
    counter = counter+batch_size
    if counter >= mri_data.shape[0]:
        counter=0
    
    jj = jj+batch_size


np.save(join(src_data,'NMARESP_noisy_input_gnoise_levels_row1.npy'),noisy_input_gnoise_full)
np.save(join(src_data,'NMARESP_gnoise_levels_row1.npy'),noise_gnoise_full)
np.save(join(src_data,'NMARESP_automap_rmse_gnoise_levels_row1.npy'),rmse_gnoise_levels)
np.save(join(src_data,'NMARESP_automap_output_gnoise_levels_row1.npy'),automap_output_full)
