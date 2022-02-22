import plotly.graph_objs as go
import torch
from ecgDataset import ecgDataset
from torch.utils.data import DataLoader
import numpy as np
from extract import *
import time
import cProfile


def main():
    minPts = 0
    e = 0
    batch_size = 500
    lead = 0

    ds = ecgDataset("/home/rylan/PycharmProjects/newDL/ALL-max-amps.pt")

    def collate_fn(batch):
        s = len(batch)
        ids = []
        X = torch.zeros(s, 8, 5000)
        for i in range(s):
            X[i] = batch[i][0]
            ids.append(batch[i][2])
        return X, ids


    ld = DataLoader(ds, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)

    for X, ids in ld:
        ids = np.array(ids)
        signals = X[:, lead, :]

        f = go.Figure(data=[go.Scatter3d(x=autocorr(signals), y=entropy_of_hist(signals), z=curvelen(signals),
                                         hovertext=ids[:,lead], mode='markers', marker=dict(color="orange", size=3.5))])
        f.update_layout(scene=dict(
            xaxis_title="Segment Autocorrelation Similarity",
            yaxis_title="Histogram Entropy",
            zaxis_title="Curve Length"
        )
        )
        f.show()
        break



start = time.time()
pr = cProfile.Profile()
pr.enable()
main()
pr.disable()
pr.print_stats(sort="time")
print("Finished in " + str(time.time() - start))
