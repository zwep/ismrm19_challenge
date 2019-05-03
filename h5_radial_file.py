#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo data for reproducible research study group initiative to reproduce [1]

[1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
Advances in sensitivity encoding with arbitrary k-space trajectories.
Magn Reson Med 46: 638-651 (2001)
"""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplot
import os
import pynufft


# ==========================================
# Inverse process functions
# ==========================================


def get_nufft_obj(trajectory, nimg, nfe):
    NufftObj = pynufft.NUFFT_cpu()

    n_size = np.prod(trajectory[0].shape)
    traj0_max = np.max(trajectory[0])
    traj1_max = np.max(trajectory[1])

    traj0 = np.reshape(trajectory[0], (n_size,)) / traj0_max * np.pi  # normalize to -pi..pi
    traj1 = np.reshape(trajectory[1], (n_size,)) / traj1_max * np.pi  # normalize to -pi..pi

    # Stack real and imag together
    om_traj = np.stack([traj0, traj1], axis=-1)
    NufftObj.plan(om_traj, (nimg, nimg), (nfe, nfe), (6, 6))

    return NufftObj


def get_recon_img(nufft_obj, rawdata, n_iter):
    n_size = np.prod(rawdata[0].shape[0:2])
    nCh = rawdata.shape[3]
    rendered_img = []
    for i in range(nCh):
        print('\t\t Reconstruction channel ', i)
        rawdata0 = np.reshape(rawdata[0, :, :, i], (n_size,))
        ypred = nufft_obj.solve(rawdata0, solver='cg', maxiter=n_iter)
        rendered_img.append(ypred)

    recon_img = np.sqrt(np.sum([np.square(np.abs(x)) for x in rendered_img], axis=0))

    return recon_img


# ==========================================
# Load data
# ==========================================

d_data = r'C:\Users\20184098\Documents\data\ismrm_challenge\rrsg_challenge\rrsg_challenge'
h5_dataset_brain = h5py.File(os.path.join(d_data, 'rawdata_brain_radial_96proj_12ch.h5'), 'r')
h5_dataset_heart = h5py.File(os.path.join(d_data, 'rawdata_heart_radial_55proj_34ch.h5'), 'r')

dir_image = r'C:\Users\20184098\PycharmProjects\ismrm19_challenge\result_images'

R_factors = [1, 2, 3, 4]
iter_range = list(range(1, 105, 5))

# Choose dataset
for name, h5_dataset in [('brain', h5_dataset_brain), ('heart', h5_dataset_heart)]:
    # pass

    print("Keys: %s" % h5_dataset.keys())
    h5_dataset_rawdata_name = list(h5_dataset.keys())[0]
    h5_dataset_trajectory_name = list(h5_dataset.keys())[1]

    trajectory = h5_dataset.get(h5_dataset_trajectory_name)[()]
    rawdata = h5_dataset.get(h5_dataset_rawdata_name)[()]

    [dummy, nFE, nSpokes, nCh] = rawdata.shape

    # ==========================================
    # Information
    # ==========================================

    print('Number of FE       ', nFE)
    print('Number of spokes   ', nSpokes)
    print('Number of channels ', nCh)

    nImg = int(nFE)
    print('Proposed img size  ', nImg)
    print('Rawdata size       ', rawdata.shape)

    # ==========================================
    # Recovery
    # ==========================================
    """
    1. 
    An implementation of the algorithm in Figure 1 of [1]. Any programming language, as well as third-party 
    software (e.g. implementations of the Non-Uniform Fourier transform or the Conjugate Gradient Algorithm) is fine, 
    as long as the source code can be shared and the computational results can be reproduced. 
    """

    nufft_obj = get_nufft_obj(trajectory, nImg, nFE)
    true_image = get_recon_img(nufft_obj, rawdata, 100)

    result_dict = {}
    for R in R_factors:
        print('Calculating reconstruction with R factor ', R)
        result_dict.setdefault(R, {})
        result_dict[R].setdefault('rel_error', [])
        result_dict[R].setdefault('recon_img', [])

        rel_error_list = []
        recon_img_list = []

        nufft_obj = get_nufft_obj(trajectory[:, :, 1::R], nImg, nFE)
        for i_iter in iter_range:
            print('\t Calculating reconstruction with maxiter ', i_iter)
            aprox_image = get_recon_img(nufft_obj, rawdata[:, :, 1::R, :], i_iter)


            rel_error = np.linalg.norm(aprox_image - true_image)/np.linalg.norm(true_image)
            rel_error_list.append(rel_error)

            if i_iter == 1:
                recon_img_list.append(aprox_image)
            elif i_iter == 11:
                recon_img_list.append(aprox_image)

        # As a last item.. add the single coil reconstruction
        single_coil = get_recon_img(nufft_obj, rawdata[:, :, 1::R, 0:1], i_iter)
        recon_img_list.append(single_coil)

        result_dict[R]['rel_error'].append(rel_error_list)
        result_dict[R]['recon_img'].append(recon_img_list)

    # ===
    # Store model results..
    # ===

    # ser_json = json.dumps(result_dict)
    # with open(os.path.join(dir_image, name + '_result_dict.json'), 'w') as f:
    #     f.write(ser_json)

    # ==========================================
    # Plot desired figures
    # ==========================================
    """
    Subsample the provided brain data by factors 2,3 and 4 (48, 32 and 24 projections), do a reconstruction, calculate 
    the error to the 96 projections reconstruction and plot the error and the number of iterations according to
    the results in Figure 4 in [1]. 
    """

    plt.figure(figsize=(20, 20))
    for R in R_factors:
        temp = result_dict[R]['rel_error'][0]
        plt.plot(iter_range, temp, label=R)

    plt.xlabel('number of iterations')
    plt.ylabel('relative error')
    plt.title('Overview of different R factors')
    plt.legend()
    name_fig = name + '_figure_4.png'
    plt.savefig(os.path.join(dir_image, name_fig))

    plt.close()

    """
    For the same data, show reconstruction results for different iteration numbers and a version from 
    a single receive coil according to the results in Figure 5. 
    """

    temp_recon_img_list = np.concatenate([np.array(result_dict[i]['recon_img']) for i in R_factors], axis=0)
    temp_name_list = ['R factor {}'.format(str(x)) for x in R_factors]

    hplot.plot_3d_list(temp_recon_img_list, name_list=temp_name_list, figsize=(20, 20))
    name_fig = name + '_figure_5.png'
    plt.savefig(os.path.join(dir_image, name_fig))

    plt.close()
