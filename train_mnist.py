import pandas as pd
import numpy as np
from neuralnet import NN, cross_entropy, Linear
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


train_df = pd.read_csv('mnist/mnist_train.csv')
test_df = pd.read_csv('mnist/mnist_test.csv')

def proc(df, size, rank):
    y = df['label'].values
    X = df.iloc[:, 1:].values / 255
    chunk_size = len(X) // (size-1)
    if rank == size - 1:
        return np.zeros((chunk_size, X.shape[1]), dtype=float), np.zeros((chunk_size), dtype=int)
    return X[rank*chunk_size: (rank+1)*chunk_size].astype(float), y[rank*chunk_size: (rank+1)*chunk_size].astype(int)

X_train, y_train = proc(train_df, size, rank)
X_test, y_test = proc(test_df, size, rank)

del train_df, test_df


# print(rank, len(X_train))

n, d = X_train.shape

model = NN(d, 10, [64], 0.001)

def bcast_weights():
    for layer in model.layers:
        if isinstance(layer, Linear):
            layer.w = comm.bcast(layer.w, size - 1)

def agg_gradient():
    for layer in model.layers:
        if isinstance(layer, Linear):
            dw = comm.gather(layer.dw, size - 1)
            if rank == size - 1:
                # print(len(dw))
                # print(dw[0])
                dw = dw[:-1]
                layer.dw = np.mean(dw)



bcast_weights()
# agg_gradient()
# print(rank, model.layers[-1].w)
# print(rank)
#
for i in range(100):
    # print(rank, i)
    if rank != size - 1:
        x = X_train[i]
        y = y_train[i]
        # print(rank)
        y_hat = model(x)
        model.backward(y, y_hat)
    agg_gradient()
    if rank == size - 1:
        model.step()
#     bcast_weights()
# #
if rank == 0:
    print(cross_entropy(y, y_hat))
