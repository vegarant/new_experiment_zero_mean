import tensorflow as tf
import numpy as np
import os.path
from PIL import Image

from adv_tools_PNAS.automap_config import src_weights, src_data;
from os.path import join 

from adv_tools_PNAS.automap_tools import compile_network
import matplotlib.pyplot as plt
import mat73
from scipy.io import loadmat, savemat

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
 
output_array = np.zeros((10000,16384))

for ii in range(0,200,batch_size):
    
    # e_random = np.random.normal(loc=0, scale=noise_level[r_value], size=(1,19710))
    # e_random = 0

    image = mri_data[ii:ii+batch_size,:];
    # image = np.expand_dims(image, 0);    
    im_rec = raw_f(image)
    xx = np.squeeze(im_rec)
    xy = xx[:,4:132,4:132]    
    output_array[ii:ii+batch_size,:] = np.reshape(xy,(100,16384))
    
    
    print(ii)
            

np.save(join(src_data,'NMARESP_global_ratio_output.npy'),output_array)
