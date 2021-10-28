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
import mat73
use_gpu = True
compute_node = 1
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

filename = join(src_data,'test_fft_x_hcpt2wnz_LS2_poisson_notrunc_offset_v3_5_128.mat')

data = mat73.loadmat(filename);
mri_data = np.transpose(data['test_fft_x'])

batch_size = 100;
sess = tf.compat.v1.Session()
raw_f, _ = compile_network(sess, batch_size)

# image = mri_data[100,:] 
# noise = np.random.uniform(-.0125,.0125,(19710,))
# noise_l2 = np.sqrt(np.sum(noise.flatten()**2))
# signal_l2 = np.sqrt(np.sum(image.flatten()**2))
# n_over_s = noise_l2/signal_l2

num_times = 10000
counter = 0

num_images = num_times*batch_size

mae_output = np.zeros((num_images))
mae_input = np.zeros((num_images))

ratio_lc = np.zeros((num_images))
jj = 0

for ii in range(0,num_times):
    
    input_clean = mri_data[counter:counter+batch_size,:];
    # input_clean = np.expand_dims(input_clean, 0);    
    output_clean = raw_f(input_clean)
    output_clean = np.squeeze(output_clean)
    output_clean = output_clean[:,4:132,4:132]    
    
    noise = np.random.uniform(-.0125,.0125,(batch_size,19710))
    # noise = np.random.uniform(-.0125,.0125,(19710,))
    input_noisy = input_clean + noise        
    output_noisy = raw_f(input_noisy)
    output_noisy = np.squeeze(output_noisy)
    output_noisy = output_noisy[:,4:132,4:132] 
    
    output_clean = np.reshape(output_clean,(batch_size,16384))
    output_noisy = np.reshape(output_noisy,(batch_size,16384))
    
    mae_output[jj:jj+batch_size] = np.mean(abs(output_clean-output_noisy),axis=1) 
    mae_input[jj:jj+batch_size] = np.mean(abs(input_clean-input_noisy),axis=1) 
    
    # ratio_lc[ii] = np.amax(np.true_divide(mae_output,mae_input))

    print(ii)
    counter = counter+batch_size
    if counter == 10000:
        counter=0
    jj = jj + batch_size
    
# np.save('output_array_10k.npy',output_array)

div = np.zeros((num_images))
for ii in range(0,num_images):    
    div[ii] = np.true_divide(mae_output[ii],mae_input[ii])
    ratio_lc[ii] = np.amax(div)

# fig = plt.figure()
# plt.plot(ratio_lc)
# # plt.ylim(1,4.5)
# fig.suptitle('Local Robustness')
# plt.ylabel('L'r'$\phi$')
# plt.xlabel('Samples 100 batch')

np.save(join(src_data,'NMARESP_local_ratio.npy'),ratio_lc)







