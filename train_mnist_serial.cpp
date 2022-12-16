#include <random>
#include <iostream>
#include "mlp.h"
#include "utils.h"

int main() {
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    read_mnist(X, y);
    int n = X.size();
    int d = X[0].size();
    MLP model {d, {128, 128}, 10, 0.1};
    int bs = 256;
    for (int i = 0; i < n-bs; i += bs) {
        std::vector<std::vector<float>> X_batch;
        for (int j = 0; j < bs; ++j) {
            X_batch.push_back(X[i+j]);
        }
        std::vector<int> y_batch;
        for (int j = 0; j < bs; ++j) {
            y_batch.push_back(y[i+j]);
        }
        std::vector<std::vector<float>> y_hat = model(X_batch);
        model.backward(y_batch, y_hat);
        model.step();
        std::cout << "i=" << i << ",loss=" << cross_entropy(y_batch, y_hat) << std::endl;
    }
    return 0;
}