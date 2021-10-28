#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:19:48 2021

@author: lfi
"""
from adv_tools_PNAS.automap_config import src_weights, src_data;
from os.path import join 

import numpy as np
import matplotlib.pyplot as plt
import mat73

output_array = np.load(join(src_data,'NMARESP_global_ratio_output.npy'))

filename = join(src_data,'test_fft_x_hcpt2wnz_LS2_poisson_notrunc_offset_v3_5_128.mat')
data = mat73.loadmat(filename);
mri_data = np.transpose(data['test_fft_x'])

# image = output_array[500,:]
# image = np.reshape(image,(128,128))
# plt.imshow(image)

num_times = 1000000
counter = 0
mae_output = np.ones((num_times))
mae_input = np.ones((num_times))

ratio_gb = np.ones((num_times))

for ii in range(num_times):
    indx = np.random.randint(10000, size=(1, 2))
    if indx[0,0]==indx[0,1]:
        indx = np.random.randint(10000, size=(1, 2))
        
    recon1 = output_array[indx[0,0],:]
    recon2 = output_array[indx[0,1],:]
        
    input1 = mri_data[indx[0,0],:]
    input2 = mri_data[indx[0,1],:]
    
    mae_output[ii] = np.mean(abs(recon1-recon2)) 
    mae_input[ii] = np.mean(abs(input1-input2)) 
    
    ratio_gb[ii] = np.amax(mae_output/mae_input)
    
    print(ii)

fig = plt.figure()
plt.plot(ratio_gb)
plt.ylim(1,np.amax(ratio_gb)+.1)
fig.suptitle('Global Robustness')
plt.ylabel('L'r'$\phi$')
plt.xlabel('Samples')

np.save(join(src_data,'NMARESP_global_ratio.npy'),ratio_gb)

