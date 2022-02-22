import h5py
import numpy as np
import torch
np.set_printoptions(suppress=True)
f = h5py.File("/home/rylan/newh5s/MUSE_20190502_122146_25000.h5")
np_ecg = np.array(f["ECG"])
print(np_ecg)
ecg = torch.from_numpy(np_ecg)
print(ecg)
print(ecg.shape)
print(ecg.dtype)