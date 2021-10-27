# 

"""

BASED ON RESP_Demo_test_automap_stability.py

This script searches for a perturbation to simulate worst-case effect for the
AUTOMAP network. The result is saved as a Runner object. Make sure you have
updated the automap_config.py file before running this script.
"""

import tensorflow as tf;
import scipy.io;
import h5py
from os.path import join;
import os;
import os.path;
import _2fc_2cnv_1dcv_L1sparse_64x64_tanhrelu_upg as arch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np;
from adv_tools_PNAS.automap_config import src_weights, src_data;

# from adv_tools_PNAS.automap_tools import read_automap_k_space_mask, compile_network, hand_f, hand_dQ, load_runner;
from adv_tools_PNAS.RESP_automap_tools import read_automap_k_space_mask, compile_network, RESP_hand_f, hand_dQ, load_runner;

from adv_tools_PNAS.adversarial_tools import scale_to_01
# from adv_tools_PNAS.Runner import Runner;
# from adv_tools_PNAS.Automap_Runner import Automap_Runner;

from adv_tools_PNAS.RESP_Runner import Runner;
from adv_tools_PNAS.RESP_Automap_Runner import Automap_Runner;


from PIL import Image

HCP_nbr = 1033
im_nbr = 2

# Parameters to the worst case perturbation algorithm. 
stab_eta = 0.001
stab_lambda = 0.1
stab_gamma = 0.9
stab_tau = 1e-5

k_mask_idx1, k_mask_idx2 = read_automap_k_space_mask();

data = scipy.io.loadmat(join(src_data, f'HCP_mgh_{HCP_nbr}_T2_subset_N_128.mat'));
mri_data = data['im'];

batch_size = mri_data.shape[0];

# Plot parameters
N = 128; # out image shape
bd = 5;  # Boundary between images
plot_dest = './plots_automap_stab';
data_dest = './RESP_data'
splits = 'splits';

if not(os.path.isdir(data_dest)):
    os.mkdir(data_dest)

if not (os.path.isdir(plot_dest)):
    os.mkdir(plot_dest);
    split_dest = join(plot_dest, splits);
    if not (os.path.isdir(split_dest)):
        os.mkdir(split_dest);



# Optimization parameters
max_itr = 8; # Update list below. This value is not relevant here
max_r_norm = float('Inf');
max_diff_norm = float('Inf');
max_r_over_x_norm = float('Inf');
la = stab_lambda;
warm_start = 'off';
warm_start_factor = 0.0;
perp_start = 'rand';
perp_start_factor = stab_tau;
reference = 'true';
momentum = stab_gamma;
learning_rate = stab_eta;
verbose=True; 

sess = tf.Session();

raw_f, raw_df = compile_network(sess, batch_size,fname_weights='2021-06-26_te_009.h5');

# f  = lambda x: hand_f( raw_f, x, k_mask_idx1, k_mask_idx2);
f = lambda x, noise: RESP_hand_f( raw_f, x, noise, k_mask_idx1, k_mask_idx2);

dQ = lambda x, r, label, la: hand_dQ(raw_df, x, r, label, la, 
                                          k_mask_idx1, k_mask_idx2); 

runner = Automap_Runner(max_itr, max_r_norm, max_diff_norm, max_r_over_x_norm, 
                         la=la, 
                         warm_start=warm_start,
                         warm_start_factor=warm_start_factor,
                         perp_start=perp_start,
                         perp_start_factor=perp_start_factor,
                         reference=reference,
                         momentum=momentum,
                         learning_rate= learning_rate,
                         verbose=verbose,
                         mask= [k_mask_idx1, k_mask_idx2]
                         );

# Update the number of iteration you would like to run
# max_itr_schedule = [12, 4, 4, 4];

max_r_over_x_norm_schedule = [0.02,0.04,0.06,0.08,0.10]


for i in range(len(max_r_over_x_norm_schedule)):
    max_r_over_x_norm = max_r_over_x_norm_schedule[i];
    runner.max_itr = 100;
    runner.max_r_over_x_norm = max_r_over_x_norm;
    runner.find_adversarial_perturbation(f, dQ, mri_data);

# for i in range(len(max_itr_schedule)):
#     max_itr = max_itr_schedule[i];
#     runner.max_itr = max_itr;
#     runner.find_adversarial_perturbation(f, dQ, mri_data);


runner_id = runner.save_runner(f);

print('Saving runner as nbr: %d' % runner_id);
runner1 = load_runner(runner_id);


input_scale = 1.5

fxr_array = np.zeros((len(runner1.r),128,128))
mri_data = runner1.x0[0];
for i in range(len(runner1.r)):
    rr = runner1.r[i];
    if i == 0:
        rr = np.zeros(rr.shape, dtype=rr.dtype);
    fxr = f(mri_data*input_scale,rr*input_scale);
    # fxr = f(mri_data*input_scale,rr*0);
    x = mri_data[im_nbr, :,:];
    r = rr[im_nbr, :,:];
    fxr = fxr[im_nbr, :,:];

    fxr_array[i,:,:] = fxr

np.save(join(src_data,'aug_automap_advpert_fxr_array.npy'),fxr_array)

    # im_left  = scale_to_01(abs(x+r));
    # im_right = scale_to_01(fxr);
    # im_out = np.ones([N, 2*N + bd]);
    # im_out[:,:N] = im_left;
    # im_out[:,N+bd:] = im_right;
    # fname_out = join(plot_dest, \
    #                  'rec_automap_runner_%d_r_idx_%d.png' % (runner_id, i));
    # plt.imsave(fname_out, im_out, cmap='gray');
    # fname_out_noisy = join(plot_dest, splits, \
    #                        'runner_%d_r_idx_%d_noisy.png' % (runner_id, i));
    # fname_out_noisy_rec = join(plot_dest, splits, \
    #                        'runner_%d_r_idx_%d_noisy_rec.png' % (runner_id, i));
    
    # Image_im_left = Image.fromarray(np.uint8(255*im_left));
    # Image_im_right = Image.fromarray(np.uint8(255*im_right));
    
    # Image_im_left.save(fname_out_noisy);
    # Image_im_right.save(fname_out_noisy_rec);

sess.close();



