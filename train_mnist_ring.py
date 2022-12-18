import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py import MPI

from neuralnet import MLP, Linear, cross_entropy


def bcast_weights():
    for layer in model.layers:
        if isinstance(layer, Linear):
            comm.Bcast(layer.w, size - 1)

def get_tag(rank, t):
    rank %= size
    return rank*10+t

def ring_gradient():
    for layer in model.layers:
        if not isinstance(layer, Linear):
            continue
        # Step 1 (Aggregation): each worker send one slice (M/N parameters) to the
        # next worker on the ring; repeat N times
        slice_len = len(layer.w) // size
        for t in range(size-1):
            i = (rank - t) % size
            slice = layer.dw[i*slice_len:(i+1)*slice_len]
            # print("before agg send from rank", rank)
            comm.Isend([slice, MPI.FLOAT], dest=(rank+1)%size, tag=get_tag(rank, t))
            recv_buf = np.empty_like(slice)
            comm.Recv([recv_buf, MPI.FLOAT], source=(rank-1)%size, tag=get_tag(rank-1, t))
            # print("agg recv done from rank", rank)
            j = (rank - 1 - t) % size
            layer.dw[j*slice_len:(j+1)*slice_len] += recv_buf
        # Step 2 (Broadcast): each worker send one slice of aggregated parameters
        # to the next worker; repeat N times
        for t in range(size - 1):
            i = (rank + 1 - t) % size
            slice = layer.dw[i*slice_len:(i+1)*slice_len]
            comm.Isend([slice, MPI.FLOAT], dest=(rank + 1) % size, tag=get_tag(rank, t))

            recv_buf = np.empty_like(slice)
            comm.Recv([recv_buf, MPI.FLOAT], source=(rank - 1) % size, tag=get_tag(rank-1, t))
            j = (rank - t) % size
            layer.dw[j * slice_len:(j + 1) * slice_len] = recv_buf



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

train_df = pd.read_csv("mnist/mnist_train.csv")
test_df = pd.read_csv("mnist/mnist_test.csv")


def proc(df):
    y = df["label"].values
    X = df.iloc[:, 1:].values / 255
    return X.astype(float), y.astype(int)


X_train, y_train = proc(train_df)
X_test, y_test = proc(test_df)

# print(rank, len(X_train))

n, d = X_train.shape

model = MLP(d, [128, 128, 128], 10, 0.1)

bs = 256 // size
ep = 5

bcast_weights()
for e in range(ep):
    if rank == size - 1:
        p = np.random.permutation(len(y_train))
    else:
        p = np.empty(len(y_train), dtype=int)
    comm.Bcast(p, root=size - 1)
    # print(rank, p[:10])
    interval = len(y_train) // size
    start_idx = rank * interval
    # end_idx = len(y_train) if rank == size-1 else (rank+1) * interval
    end_idx = (rank + 1) * interval
    X_train_local = X_train[p][start_idx:end_idx]
    y_train_local = y_train[p][start_idx:end_idx]
    print(rank, len(X_train_local), len(y_train_local))
    #
    # predicted = np.zeros(len(y_train), dtype=int)
    pbar = (
        tqdm(range(0, len(y_train_local), bs))
        if rank == size - 1
        else range(0, len(y_train_local), bs)
    )
    for i in pbar:
        x = X_train_local[i : i + bs]
        y = y_train_local[i : i + bs]
        y_hat = model(x)
        model.backward(y, y_hat)
        ring_gradient()
        model.step()
        if rank == size - 1:
            pbar.set_description(f"loss={cross_entropy(y, y_hat):.3f}")
        # print(rank, model.layers[0].w[0, :10])
        # break
    # break
