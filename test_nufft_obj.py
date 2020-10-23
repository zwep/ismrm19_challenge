import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import helper.plot_fun as hplotf
import os
import pynufft
from h5_radial_file import get_nufft_obj, get_recon_img

d_data = '/home/bugger/Documents/data/rrsg_challenge'
gpu_enabled = False
if gpu_enabled:
    cuda_devices = pynufft.helper.device_list()
    sel_index = 0
    sel_device = cuda_devices[sel_index]
else:
    sel_device = None

d_result = os.path.join(d_data, 'result_images')
if not os.path.isdir(d_result):
    os.mkdir(d_result)

h5_dataset_brain = h5py.File(os.path.join(d_data, 'rawdata_brain_radial_96proj_12ch.h5'), 'r')
h5_dataset_heart = h5py.File(os.path.join(d_data, 'rawdata_heart_radial_55proj_34ch.h5'), 'r')

name = 'brain'
h5_dataset = h5_dataset_brain

print("Keys: %s" % h5_dataset.keys())
h5_dataset_rawdata_name = list(h5_dataset.keys())[0]
h5_dataset_trajectory_name = list(h5_dataset.keys())[1]

trajectory = h5_dataset.get(h5_dataset_trajectory_name)[()]
rawdata = h5_dataset.get(h5_dataset_rawdata_name)[()]

[dummy, nImg, nSpokes, nCh] = rawdata.shape

nFE = int(nImg)

for oversampling_factor in [1, 2.5, 3]:
    temp_results = []
    n_iter = 5

    for i_kernel in range(1, 10):
        print(i_kernel)
        i_kernel = 6
        nufft_obj = get_nufft_obj(trajectory, nImg, int(oversampling_factor * nFE), n_kernel=i_kernel,
                                  device_index=sel_device)
        true_image, coil_images = get_recon_img(nufft_obj, rawdata, n_iter)
        temp_results.append(true_image)

    kernel_change = np.array(temp_results)[np.newaxis]
    fig_handle = hplotf.plot_3d_list(kernel_change, subtitle=[[str(x) for x in range(1, 10)]],
                        title=f'Oversampling factor {oversampling_factor}')
    fig_handle.savefig(os.path.join(d_result, f'result_oversampling_{oversampling_factor}.jpg'))
