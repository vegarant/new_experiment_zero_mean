import sys
import time
import tensorflow as tf
import numpy as np
import h5py
import scipy.io
from os.path import join 
import os.path

from optimization.gpu.operators import MRIOperator
from optimization.gpu.proximal import WeightedL1Prox, SQLassoProx2
from optimization.gpu.algorithms import SquareRootLASSO
from optimization.utils import estimate_sparsity, generate_weight_matrix
from tfwavelets.dwtcoeffs import get_wavelet
from tfwavelets.nodes import idwt2d
from PIL import Image
import matplotlib.image as mpimg;
import scipy.io

from adv_tools_PNAS.automap_config import src_data;
from adv_tools_PNAS.adversarial_tools import l2_norm_of_tensor
from adv_tools_PNAS.automap_tools import read_automap_k_space_mask
from utils import convert_automap_samples_to_tf_samples_in_image_domain


from adv_tools_PNAS.automap_tools import load_runner, read_automap_k_space_mask, compile_network, hand_f, sample_image;


src_noise = 'data_non_zero_mean';

N = 128
wavname = 'db2'
levels = 3
use_gpu = True
compute_node = 2
dtype = tf.float64;
sdtype = 'float64';
scdtype = 'complex128';
cdtype = tf.complex128
wav = get_wavelet(wavname, dtype=dtype);

if use_gpu:
    # os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
    # print('Compute node: {}'.format(compute_node))
    os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
    
# if use_gpu:
#     os.environ["CUDA_VISIBLE_DEVICES"]= "%d" % (compute_node)
#     print('Compute node: {}'.format(compute_node))
# else: 
#     os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

# Turn on soft memory allocation
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.log_device_placement = False
sess = tf.compat.v1.Session(config=tf_config)

dest_data = 'data_non_zero_mean';
dest_plots = 'plots_non_zero_mean';

if not (os.path.isdir(dest_data)):
    os.mkdir(dest_data);

if not (os.path.isdir(dest_plots)):
    os.mkdir(dest_plots);

# Parameters for the CS-algorithm
# n_iter = 2000
tau = 0.6
sigma = 0.6
lam = 0.0001

############################################################################
###                     Build Tensorflow Graph                           ###
############################################################################

# Parameters for CS algorithm
pl_sigma = tf.compat.v1.placeholder(dtype, shape=(), name='sigma')
pl_tau   = tf.compat.v1.placeholder(dtype, shape=(), name='tau')
pl_lam   = tf.compat.v1.placeholder(dtype, shape=(), name='lambda')

# Build Primal-dual graph
tf_im = tf.compat.v1.placeholder(cdtype, shape=[N,N,1], name='image')
tf_samp_patt = tf.compat.v1.placeholder(tf.bool, shape=[N,N,1], name='sampling_pattern')

# For the weighted l^1-norm
pl_weights = tf.compat.v1.placeholder(dtype, shape=[N,N,1], name='weights')

tf_input = tf_im

op = MRIOperator(tf_samp_patt, wav, levels, dtype=dtype)
measurements = op.sample(tf_input)

tf_adjoint_coeffs = op(measurements, adjoint=True)
adj_real_idwt = idwt2d(tf.math.real(tf_adjoint_coeffs), wav, levels)
adj_imag_idwt = idwt2d(tf.math.imag(tf_adjoint_coeffs), wav, levels)
tf_adjoint = tf.complex(adj_real_idwt, adj_imag_idwt)

prox1 = WeightedL1Prox(pl_weights, pl_lam*pl_tau, dtype=dtype)
prox2 = SQLassoProx2(dtype=dtype)

alg = SquareRootLASSO(op, prox1, prox2, measurements, sigma=pl_sigma, tau=pl_tau, lam=pl_lam, dtype=dtype)

initial_x = op(measurements, adjoint=True)

result_coeffs = alg.run(initial_x)

real_idwt = idwt2d(tf.math.real(result_coeffs), wav, levels)
imag_idwt = idwt2d(tf.math.imag(result_coeffs), wav, levels)
tf_recovery = tf.complex(real_idwt, imag_idwt)

samp = np.swapaxes(np.fft.fftshift(np.array(h5py.File(join(src_data, 'k_mask.mat'), 'r')['k_mask']).astype(np.bool)), 0,1)
samp = np.expand_dims(samp, -1)

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

sample = lambda im: sample_image(im, k_mask_idx1, k_mask_idx2)

noisy_input_array = np.load(join(src_data,'NMARESP_noisy_input_gnoise_levels_row1.npy'))
print(noisy_input_array.shape)
# sys.exit()

n_iter_vec = [50,100,250,500,1000]

LASSO_gnoise_recons = np.zeros((noisy_input_array.shape[0],len(n_iter_vec),128,128))


with tf.compat.v1.Session() as sess:

    sess.run(tf.compat.v1.global_variables_initializer())
    weights = np.ones([128,128,1], dtype=sdtype);

    for im_number in range(0,noisy_input_array.shape[0],1):
        
        print('image number:',im_number)

        noisy_ksp = np.expand_dims(noisy_input_array[im_number,:],0)

        noisy_image = convert_automap_samples_to_tf_samples_in_image_domain(noisy_ksp/1.5, 
                                                                        k_mask_idx1,
                                                                        k_mask_idx2)
        
        for n_iter_ind in range(len(n_iter_vec)):

            n_iter = n_iter_vec[n_iter_ind]
            print('n_iter:',n_iter)
            _rec = sess.run(tf_recovery, feed_dict={ 'tau:0': tau,
                                                    'lambda:0': lam,
                                                    'sigma:0': sigma,
                                                    'weights:0': weights,
                                                    'n_iter:0': n_iter,
                                                    'image:0': noisy_image,
                                                    'sampling_pattern:0': samp})
            rec = np.abs(_rec[:,:,0]).astype(np.float64);

            LASSO_gnoise_recons[im_number,n_iter_ind,:,:] = rec

    np.save(join(src_data,'NMARESP_lasso_gnoise_recons_row1.npy'),LASSO_gnoise_recons)

