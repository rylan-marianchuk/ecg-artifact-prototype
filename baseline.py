import plotly.graph_objs as go
import torch
from ecgDataset import ecgDataset
from torch.utils.data import DataLoader
import numpy as np
from transforms import PowerSpec
import pywt
from scipy.signal import firwin, lfilter, kaiserord
import emd

def baseline_wander(signal):
    return abs(torch.mean(signal[:130]).item() - torch.mean(signal[-130:]).item())


ds = ecgDataset("/home/rylan/PycharmProjects/newDL/ALL-max-amps.pt")

P = PowerSpec(normalize=True)


def CEEMD(signal):
    """
    Implementation of publication at  DOI:10.3390/s17122754, recommended by https://doi.org/10.1016/j.future.2020.10.024
    :param signal: (tensor) the ecg of shape=(N,)
    :return: (tensor) the denoised ecg of shape=(N,)
    """
    imfs, noise = emd.sift.complete_ensemble_sift(signal.numpy())
    fixed = False
    for i in range(1, imfs.shape[1] + 1):
        f = imfs[:, -i]
        zcr = ((f[:-1] * f[1:]) < 0).sum() / 2
        if zcr >= 1.5:
            scat = go.Figure(go.Scatter(y=f, mode='markers', marker=dict(color='green')))
            scat.show()
            signal -= torch.from_numpy(f)
            fixed = True
            break
    if not fixed:
        raise Exception("Did not find a imf to subtract")
    return signal

def DWT_Correct(signal):
    """
    Implementation of publication at DOI:10.1109/TBME.1985.325514, recommended by https://arxiv.org/pdf/1807.11359.pdf
    :param signal: (tensor) the ecg of shape=(N,)
    :return: (tensor) the denoised ecg of shape=(N,)
    """
    L = 9
    detail_coeffs = pywt.wavedec(signal, 'db8', level=L)
    detail_coeffs[-L] = np.zeros(detail_coeffs[-L].shape[0])
    build = pywt.waverec(detail_coeffs, 'db8')
    return build


def FIR_Correct(signal):
    """
    Implementation of publication at 10.1109/TBME.1985.325514
    :param signal: (tensor) the ecg of shape=(N,)
    :return: (tensor) the denoised ecg of shape=(N,)
    """
    numtaps, beta = kaiserord(67.0, 28.0/250.0)
    taps = firwin(numtaps, cutoff=67.0, window=('kaiser', beta), scale=False, nyq=250.0, pass_zero=False)
    filtered = lfilter(taps, 1.0, signal)

    #nyq_rate = 250.0
    #beta = 4.0
    #taps = firwin(51, 67.0, width=28.0, window=('kaiser', beta), fs=500.0, pass_zero=False)
    #filtered = lfilter(taps, 1.0, signal)
    return filtered

def collate_fn(batch):
    s = len(batch)
    ids = []
    X = torch.zeros(s, 8, 5000)
    bl = torch.zeros(s, 8)
    lowFreq = torch.zeros(s, 8)
    for i in range(s):
        X[i] = batch[i][0]
        bl[i] = torch.Tensor([baseline_wander(batch[i][0][x]) for x in range(8)])
        lowFreq[i] = torch.Tensor([P(batch[i][0][x])[:10].sum().item() for x in range(8)])
        ids.append(batch[i][2])
    return X, bl, lowFreq, ids


batch_size = 5000
ld = DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn)

for X, bl, lowFreq, ids in ld:
    ids = np.array(ids)
    fig = go.Figure(go.Scatter(x=bl[:,2], y=lowFreq[:,2], hovertext=ids[:,2], mode='markers', marker=dict(color="red", size=10.5)))
    fig.show()
    print()


