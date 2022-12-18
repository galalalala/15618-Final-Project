import pandas as pd
import numpy as np
from neuralnet import MLP, cross_entropy, Linear
from tqdm import tqdm


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

model = MLP(d, [512, 512, 512], 10, 0.1)

for l in model.layers:
    print(l)

print(X_train.shape, y_train.shape)

bs = 256
ep = 5

for e in range(ep):
    p = np.random.permutation(len(y_train))
    X_train = X_train[p]
    y_train = y_train[p]
    predicted = np.zeros(len(y_train), dtype=int)
    pbar = tqdm(range(0, n, bs))
    for i in pbar:
        x = X_train[i : i + bs]
        y = y_train[i : i + bs]
        y_hat = model(x)
        model.backward(y, y_hat)
        model.step()
        pbar.set_description(f"loss={cross_entropy(y, y_hat):.3f}")
        predicted[i : i + bs] = np.argmax(y_hat, axis=1)
    print(f"epoch={e}, acc={np.sum(predicted==y_train)/len(y_train):.3f}")
