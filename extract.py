import torch
from waveformParameterExtract import weights
from decodeLeads import getLeads

def entropy_of_hist(signals):
    """

    :param signals: (tensor) shape=(M, 5000), each row a lead
    :return: (tensor) shape=(M,) the entropy of its binned histogram
    """
    res = torch.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        vals, bins = torch.histogram(signals[i], bins=40, density=True)
        #vals_NP, bins_NP = np.histogram(signals[i], bins=40, density=True)
        res[i] = -torch.sum(torch.log2(vals) * vals).item()
    return res

def autocorr(signals):
    """

    :param signals: (tensor) shape=(M, 5000), each row a lead
    :return: (tensor) shape=(M,) the area of its autocorrelation segments
    """
    res = torch.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        res[i] = weights(signals[i]).sum().item()
    return res


def curvelen(signals):
    """

    :param signals: (tensor) shape=(M, 5000), each row a lead
    :return: (tensor) shape=(M,) the curvelength
    """
    res = torch.zeros(signals.shape[0])
    for i in range(signals.shape[0]):
        lead = signals[i]
        L = 0
        for x in range(lead.shape[0] - 1):
            L += (1 + (lead[x + 1] - lead[x]) * (lead[x + 1] - lead[x])) ** 0.5
        res[i] = L
    return res


print(curvelen(torch.Tensor(torch.rand(1250000)).reshape(250, 5000)))
