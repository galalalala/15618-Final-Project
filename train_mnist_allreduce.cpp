#include <random>
#include <iostream>
#include <mpi.h>
#include "mlp.h"
#include "utils.h"

std::vector<float> flatten(std::vector<std::vector<float>>mtx) {
    std::vector<float> res;
    for (auto const& vec : mtx) {
        res.insert(end(res), begin(vec), end(vec));
    }
    return res;
}

// connot directly send nested vector, so we first flatten it into continuous array
void bcast_weights(MLP* model, int word_rank, int word_size) {
    for (auto& l : model->linears) {
        int dim_in = l.w_.size();
        int dim_out = l.w_[0].size();
        int N = l.w_.size() * l.w_[0].size();
//        std::cout << "N = " << N << std::endl;
        std::vector<float> buf;
        if (word_rank == word_size - 1) {
            buf = flatten(l.w_);
        } else {
            buf.reserve(N);
        }
        MPI_Bcast(buf.data(), N, MPI_FLOAT, word_size - 1, MPI_COMM_WORLD);
//        std::cout << "rank = " << word_rank << ", bcast err = " << err  << " (success = " << MPI_SUCCESS << ")" << std::endl;
        if (word_rank != word_size - 1) {
            for (auto i = 0; i < dim_in; ++i) {
                for (auto j = 0; j < dim_out; ++j) {
                    l.w_[i][j] = buf[i*dim_out+j];
                }
            }
        }
    }
}

void allreduce_grad(MLP* model, int word_rank, int word_size) {
    for (auto& l : model->linears) {
        int dim_in = l.dw_.size();
        int dim_out = l.dw_[0].size();
        int N = l.dw_.size() * l.dw_[0].size();
//        std::cout << "N = " << N << std::endl;

        std::vector<float> sendbuf = flatten(l.dw_);
        std::vector<float> recvbuf(N);
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
//        std::cout << "rank = " << word_rank << ", reduce err = " << err  << " (success = " << MPI_SUCCESS << ")" << std::endl;
        for (auto i = 0; i < dim_in; ++i) {
            for (auto j = 0; j < dim_out; ++j) {
                l.dw_[i][j] = recvbuf[i * dim_out + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<std::vector<float>> X;
    std::vector<int> y;
    read_mnist(X, y);
    int n = X.size();
    int d = X[0].size();
    MLP model {d, {128, 128}, 10, 0.1};
    bcast_weights(&model, world_rank, world_size);
//    std::cout << "rank = " << world_rank << ", l[0][66][66] = " << model.linears[0].w_[66][66] << "\n";
//    std::cout << "rank = " << world_rank << ", l[1][123][123] = " << model.linears[1].w_[123][123] << "\n";
//    std::cout << "rank = " << world_rank << ", l[2][123][4] = " << model.linears[2].w_[123][4] << "\n";
    int bs = 256;
    int bs_total = bs * world_size;
    for (int i = 0; i <= n-bs_total; i += bs_total) {
        std::vector<std::vector<float>> X_batch;
        for (int j = world_rank; j < bs_total; j += world_size) {
            X_batch.push_back(X[i+j]);
        }
        std::vector<int> y_batch;
        for (int j = world_rank; j < bs_total; j += world_size) {
            y_batch.push_back(y[i+j]);
        }
        std::vector<std::vector<float>> y_hat = model(X_batch);
        model.backward(y_batch, y_hat);
        allreduce_grad(&model, world_rank, world_size);
        model.step();
        if (world_rank == world_size - 1) {
            std::cout << "i=" << i << ",loss=" << cross_entropy(y_batch, y_hat) << std::endl;
        }
    }
    MPI_Finalize();
}