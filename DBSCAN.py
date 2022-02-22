import numpy as np
import sqlite3
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from decodeLeads import getLeads

def distFunc(p, q):
    return np.linalg.norm(p - q)


def rangeQuery(DB, q, eps):
    N = DB.shape[0]
    neighbourSet = set()
    size = 0
    for p in range(N):
        if p == q: continue
        if distFunc(DB[p], DB[q]) <= eps:
            neighbourSet.add(p)
            size += 1
    return neighbourSet, size

def DBSCAN(DB, eps, minPts):
    """

    :param DB: (ndarray) shape=(N, 3) N batch size,
    :param eps:
    :param minPts:
    :return:
    """
    N = DB.shape[0]
    # -1 : Noise,   1: Core point
    # Initialize all to zero, undefined
    labels = np.zeros(N, dtype=np.int_)

    for p in range(N):
        if labels[p] != 0: continue
        neighbourSet, size = rangeQuery(DB, p, eps)
        if size < minPts:
            labels[p] = -1
            continue

        labels[p] = 1
        for q in neighbourSet:
            if labels[q] == -1:
                labels[q] = 1
            if labels[q] != 0: continue
            labels[q] = 1
            S, size = rangeQuery(DB, q, eps)
            if size >= minPts:
                neighbourSet = neighbourSet.union(S)
    return labels


def showColoredNoiseDetected3dScatter(path2sql, lead, eps, minPts):
    con = sqlite3.connect(path2sql)
    frame = pd.read_sql_query("SELECT * FROM wvfm_params WHERE LEAD=" + str(lead), con)
    #frame = pd.read_sql_query("SELECT * FROM wvfm_params", con)

    N = frame.shape[0]
    CL = np.array(frame["CURVELENGTH"])[:N]
    HE = np.array(frame["HISTENTROPY"])[:N]
    SAC = np.zeros(N)
    ids = np.array(frame["EUID"], dtype=np.str_)[:N]
    DB = np.zeros(shape=(N ,3))

    DB[:, 0] = (CL - CL.min()) / (CL.max() - CL.min())
    DB[:, 1] = (HE - HE.min()) / (HE.max() - HE.min())
    DB[:, 2] = SAC

    goodC = "#118ab2"
    artifactedC = "#ef476f"

    labeledArtifact = DBSCAN(DB, eps, minPts)
    col = [goodC if labeledArtifact[i] == 1 else artifactedC for i in range(N)]

    f = go.Figure(data=[go.Scatter3d(x=SAC, y=HE, z=CL,
                                     hovertext=ids, mode='markers', marker=dict(color=col, size=3.5))])
    f.update_layout(scene=dict(
        xaxis_title="Segment Autocorrelation Similarity",
        yaxis_title="Histogram Entropy",
        zaxis_title="Curve Length"
        )
    )
    f.show()

    noise_ind = np.argwhere(labeledArtifact == -1).flatten()
    for i in range(0, noise_ind.shape[0], 8):
        all8 = np.zeros(shape=(8, 5000))
        subplot_titles = []
        for j in range(0, 8):
            try:
                path = "/home/rylan/May_2019_XML/" + ids[noise_ind[i+j]].split(".")[0] + ".xml"
            except: continue
            subplot_titles.append(path.split('/')[-1])
            all8[j] = getLeads(path, 8)[lead]

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=subplot_titles)

        fig.add_trace(go.Scatter(y=all8[0], mode='markers', marker=dict(color='red')),
                      row=1, col=1)

        fig.add_trace(go.Scatter(y=all8[1], mode='markers', marker=dict(color='red')),
                      row=1, col=2)

        fig.add_trace(go.Scatter(y=all8[2], mode='markers', marker=dict(color='red')),
                      row=2, col=1)

        fig.add_trace(go.Scatter(y=all8[3], mode='markers', marker=dict(color='red')),
                      row=2, col=2)

        fig.add_trace(go.Scatter(y=all8[4], mode='markers', marker=dict(color='red')),
                      row=3, col=1)

        fig.add_trace(go.Scatter(y=all8[5], mode='markers', marker=dict(color='red')),
                      row=3, col=2)

        fig.add_trace(go.Scatter(y=all8[6], mode='markers', marker=dict(color='red')),
                      row=4, col=1)

        fig.add_trace(go.Scatter(y=all8[7], mode='markers', marker=dict(color='red')),
                      row=4, col=2)

        fig.update_layout(title_text="Detected as Artifact")
        fig.show()

showColoredNoiseDetected3dScatter("/home/rylan/CLionProjects/preprocess/wvfm_params.db", 0, 0.001, 16)

