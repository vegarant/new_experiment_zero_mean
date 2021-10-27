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
from adv_tools_PNAS.RESP_automap_tools import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;


from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import mat73
import sys
use_gpu = True
compute_node = 1
if use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    print('Compute node: {}'.format(compute_node))
else:
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"


k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

# filename1 = '/hdd3/matters_arising/storage_automap_not_robust/HCP_mgh_1002_T2_subset_N_128.mat'
# filename2 = '/hdd3/matters_arising/storage_automap_not_robust/HCP_mgh_1004_T2_subset_N_128.mat'
filename3 = join(src_data,'HCP_mgh_1033_T2_subset_N_128.mat')

# data1 = loadmat(filename1)
# data2 = loadmat(filename2)
data = loadmat(filename3)['im']

# gt_imgs = np.concatenate((data1['im'],data2['im'],data3['im']),axis=0)


# print(data3['im'].shape)

# sys.exit()

# mri_data = np.transpose(data['test_fft_x'])

# clean_data = mat73.loadmat(filename_x)
# clean_mri_data = np.transpose(clean_data['test_x'])

batch_size = 100;


gt_imgs = np.repeat(np.expand_dims(data[2,:,:],0),batch_size,axis=0)
# print(gt_imgs.shape)
# sys.exit()

sess = tf.compat.v1.Session()
raw_f, _ = compile_network(sess, batch_size,fname_weights='2021-06-26_te_009.h5')

sample_im = lambda x: sample_image(x, k_mask_idx1, k_mask_idx2)
f  = lambda x: hand_f(raw_f, x, k_mask_idx1, k_mask_idx2)


mri_data = sample_im(gt_imgs)

print(mri_data.shape)
print(gt_imgs.shape)
# sys.exit()

num_times = 10000
counter = 0
# mae_output = np.zeros((num_times))
# mae_input = np.zeros((num_times))

num_images = num_times*batch_size
rmse_1m = np.zeros((num_images))

jj = 0

def rmse_comp(ref,inp):
    
    ref_norm = ref-ref.mean()
    
#     inp_mag = np.abs(inp_im)
    inp_norm = inp-inp.mean()
    inp_norm = inp_norm / (inp_norm.max()-inp_norm.min())
    rmse = np.sqrt(((inp_norm-ref_norm)**2).mean())
    
    return rmse

input_scale = 1.5


for ii in range(0,num_times):
    
    input_clean = mri_data[counter:counter+batch_size,:];
    # input_clean = np.expand_dims(input_clean, 0);    
    # output_clean = raw_f(input_clean)
    # output_clean = np.squeeze(output_clean)
    # output_clean = output_clean[:,4:132,4:132]    
    output_clean = np.reshape(gt_imgs[counter:counter+batch_size,:,:],(batch_size,16384))
    
    noise = np.random.uniform(-.0125,.0125,(batch_size,19710))
    input_noisy = input_clean*input_scale + noise        
    output_noisy = raw_f(input_noisy)
    

    #Check noise level ratio
    # image_mean = np.mean(input_clean,axis=0)
    # noise = np.random.uniform(-0.0125,.0125,(19710,))
    # noise_l2 = np.sqrt(np.mean(noise.flatten()**2))
    # signal_l2 = np.sqrt(np.mean(image_mean.flatten()**2))
    # n_over_s = noise_l2/signal_l2
    # print(n_over_s)

    output_noisy = np.squeeze(output_noisy)
    # output_noisy = output_noisy[:,4:132,4:132]
    
    # output_noisy = np.rot90(np.flip(output_noisy,axis=2),axes=(1,2))

    output_noisy =  np.reshape(output_noisy,(batch_size,16384))
    
    rmse_batch = np.zeros(batch_size)

    for kk in range(batch_size):
        rmse_batch[kk] = rmse_comp(output_clean[kk,:],output_noisy[kk,:])

    rmse_1m[jj:jj+batch_size] = rmse_batch 

    print(ii)
    counter = counter+batch_size
    if counter >= mri_data.shape[0]:
        counter=0
    
    jj = jj+batch_size
# np.save('output_array_10k.npy',output_array)

# div = np.zeros((num_times))
# for ii in range(0,num_times):    
#     div[ii] = np.true_divide(mae_output[ii],mae_input[ii])
#     ratio_lc[ii] = np.amax(div)

# fig = plt.figure()
# plt.plot(ratio_lc)
# # plt.ylim(1,4.5)
# fig.suptitle('Local Robustness')
# plt.ylabel('L'r'$\phi$')
# plt.xlabel('Samples 100 batch')

np.save(join(src_data,'NMARESP_rmse_1m_images.npy'),rmse_1m)
# np.save('output_clean.npy',output_clean)
# np.save('output_noisy.npy',output_noisy)







