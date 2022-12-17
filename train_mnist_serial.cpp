#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include "mlp.h"
#include "utils.h"
#include "argparse/argparse.h"
#include "pbar/pbar.h"

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("serial");

    program.add_argument("n")
            .help("number of hidden layers")
    .scan<'i', int>();
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

    printf("n_hidden=%d, hidden_dim=%d, lr=%f, batch_size=%d\n", program.get<int>("n"),
           program.get<int>("d"), program.get<float>("lr"), program.get<int>("bs"));

    std::vector <std::vector<float>> X;
    std::vector<int> y;
    read_mnist(X, y);
    int n = X.size();
    int d = X[0].size();
    std::vector<int> hidden_sizes;
    for (auto i = 0; i < program.get<int>("n"); ++i) {
        hidden_sizes.push_back(program.get<int>("d"));
    }
    MLP model{d, hidden_sizes, 10, program.get<float>("lr")};
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
        std::vector <std::vector<float>> y_hat = model(X_batch);
        model.backward(y_batch, y_hat);
        model.step();
        bar.update();
//        std::cout << "i=" << i << ",loss=" << cross_entropy(y_batch, y_hat) << std::endl;
        indices.push_back(i);
        losses.push_back(cross_entropy(y_batch, y_hat));
    }
    double t = timer.elapsed();
    std::cout << "t=" << t << std::endl;
    std::ostringstream oss;
    oss << "log/serial-" << program.get<int>("n") << "-" << program.get<int>("d") << "-" << program.get<float>("lr") << "-" << program.get<int>("bs") << ".txt";
    std::string log_path = oss.str();
    save_log(indices, losses, t, log_path);
    return 0;
}