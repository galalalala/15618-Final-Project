#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include <mpi.h>
#include "mlp.h"
#include "utils.h"
#include "argparse/argparse.h"
#include "pbar/pbar.h"

std::vector<float> flatten(std::vector <std::vector<float>> mtx) {
    std::vector<float> res;
    for (auto const &vec: mtx) {
        res.insert(end(res), begin(vec), end(vec));
    }
    return res;
}


int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("serial");

    program.add_argument("d")
            .help("dimension of hidden layers")
            .scan<'i', int>();
    program.add_argument("lr")
            .help("learning rate")
            .scan<'f', float>();
    program.add_argument("bs")
            .help("per-device batch size")
            .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == world_size - 1) {
        printf("n_hidden=%d, hidden_dim=%d, lr=%f, batch_size=%d\n", 1,
               program.get<int>("d"), program.get<float>("lr"), program.get<int>("bs"));
    }

    std::vector <std::vector<float>> X;
    std::vector<int> y;
    read_mnist(X, y);
    int n = X.size();
    int d = X[0].size();
    int per_device_hidden_dim = program.get<int>("d") / world_size;

    MLP_Mpara model{d, {per_device_hidden_dim}, 10, program.get<float>("lr")};

    int bs = program.get<int>("bs");
    progressbar bar(static_cast<int>(n / bs));
    std::vector<int> indices;
    std::vector<float> losses;
    Timer timer;
    for (int i = 0; i <= n - bs; i += bs) {
        std::vector <std::vector<float>> X_batch;
        for (int j = 0; j < bs; ++j) {
            X_batch.push_back(X[i + j]);
        }
        std::vector<int> y_batch;
        for (int j = 0; j < bs; ++j) {
            y_batch.push_back(y[i + j]);
        }
        std::vector <std::vector<float>> logits = model(X_batch);
        std::vector<float> sendbuf = flatten(logits);
        int N = 10 * bs;
        std::vector<float> recvbuf(N);
        MPI_Allreduce(sendbuf.data(), recvbuf.data(), N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
//        for (int j = 0; j < N; ++j) {
//            recvbuf[j] /= world_size;
//        }
        std::vector <std::vector<float>> logits_reduced;
        for (int j = 0; j < bs; ++j) {
            std::vector<float> row;
            for (int k = 0; k < 10; ++k) {
                row.push_back(recvbuf[10 * j + k]);
            }
            logits_reduced.push_back(row);
        }
        std::vector <std::vector<float>> y_hat = softmax(logits_reduced);
        model.backward(y_batch, y_hat);
        model.step();
        if (world_rank == world_size - 1) {
            bar.update();
            indices.push_back(i);
            losses.push_back(cross_entropy(y_batch, y_hat));
        }
    }
    if (world_rank == world_size - 1) {
        double t = timer.elapsed();
        std::cout << "t=" << t << std::endl;
        std::ostringstream oss;
        oss << "log/mpara-" << 1 << "-" << program.get<int>("d") << "-" << program.get<float>("lr") << "-"
            << program.get<int>("bs") << "-" << world_size << ".txt";
        std::string log_path = oss.str();
        save_log(indices, losses, t, log_path);
    }
    MPI_Finalize();
}