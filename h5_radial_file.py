#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo data for reproducible research study group initiative to reproduce [1]

[1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
Advances in sensitivity encoding with arbitrary k-space trajectories.
Magn Reson Med 46: 638-651 (2001)
"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplot
import os
import pynufft

# Load data
d_data = r'C:\Users\20184098\Documents\data\ismrm_challenge\rrsg_challenge\rrsg_challenge'
h5_dataset_brain = h5py.File(os.path.join(d_data, 'rawdata_brain_radial_96proj_12ch.h5'), 'r')
h5_dataset_heart = h5py.File(os.path.join(d_data, 'rawdata_heart_radial_55proj_34ch.h5'), 'r')

# Choose dataset
h5_dataset = h5_dataset_heart
# h5_dataset = h5_dataset_brain

with_subsample = True


print("Keys: %s" % h5_dataset.keys())
h5_dataset_rawdata_name = list(h5_dataset.keys())[0]
h5_dataset_trajectory_name = list(h5_dataset.keys())[1]

trajectory = h5_dataset.get(h5_dataset_trajectory_name)[()]
rawdata = h5_dataset.get(h5_dataset_rawdata_name)[()]

[dummy, nFE, nSpokes, nCh] = rawdata.shape

# Subsample
if with_subsample:
    R = 2  # 2, 3, 4
    print('Subsample factor', R)
    trajectory = trajectory[:, :, 1::R]
    # trajectory.shape
    rawdata = rawdata[:, :, 1::R, :]
    [dummy, nFE, nSpokes, nCh] = rawdata.shape

print('Number of FE       ', nFE)
print('Number of spokes   ', nSpokes)
print('Number of channels ', nCh)

nImg = int(nFE)
print('Proposed img size  ', nImg)
print('Rawdata size       ', rawdata.shape)

#  Demo: NUFFT reconstruction with BART
# inverse gridding

NufftObj = pynufft.NUFFT_cpu()

n_size = np.prod(trajectory[0].shape)
traj0_max = np.max(trajectory[0])
traj1_max = np.max(trajectory[1])

traj0 = np.reshape(trajectory[0], (n_size, )) / traj0_max * np.pi  # normalize to -pi..pi
traj1 = np.reshape(trajectory[1], (n_size, )) / traj1_max * np.pi  # normalize to -pi..pi

# As an example.. visualize the trajectories
plt.figure(1)
for i in np.arange(0, nSpokes, int(nSpokes * 0.10)):
    plt.scatter(trajectory[0, :, i], trajectory[1, :, i])
plt.title('trajectories subset')

# Stack real and imag together
om_traj = np.stack([traj0, traj1], axis=-1)
NufftObj.plan(om_traj, (nImg, nImg), (nFE, nFE), (6, 6))

renderd_img = []
for i in range(nCh):
    rawdata0 = np.reshape(rawdata[0, :, :, i], (n_size, ))
    ypred = NufftObj.solve(rawdata0, solver='cg', maxiter=25)
    renderd_img.append(ypred)

recon_img = np.sqrt(np.sum([np.square(np.abs(x)) for x in renderd_img], axis=0))

plt.figure(2)
plt.imshow(recon_img)
plt.title('Regridding SOS reconstruction')
