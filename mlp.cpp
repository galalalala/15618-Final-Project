#include <random>
#include <vector>
#include <cmath>
#include <array>
#include <utility>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <memory>
#include <variant>
#include <fstream>
#include <sstream>

std::vector <std::vector<float>> random_init(std::pair<int, int> shape) {
    int IN = shape.first;
    int OUT = shape.second;
    std::mt19937_64 gen(IN * OUT);
    std::uniform_real_distribution<float> dist(-0.1, 0.1);

    std::vector <std::vector<float>> w(IN, std::vector<float>(OUT));
    for (int i = 0; i < IN; ++i) {
        for (int j = 0; j < OUT; ++j) {
            w[i][j] = dist(gen);
        }
    }
    w[0] = std::vector<float>(OUT, 0.0);  // init bias as zero
    return w;
}

std::vector <std::vector<float>> softmax(const std::vector <std::vector<float>> &z) {
    int bs = z.size();
    int num_classes = z[0].size();

    std::vector <std::vector<float>> e(bs, std::vector<float>(num_classes));
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            e[i][j] = std::exp(z[i][j]);
        }
    }

    std::vector <std::vector<float>> res(bs, std::vector<float>(num_classes));
    for (int i = 0; i < bs; ++i) {
        float sum = std::accumulate(e[i].begin(), e[i].end(), 0.0f);
        for (int j = 0; j < num_classes; ++j) {
            res[i][j] = e[i][j] / sum;
        }
    }
    return res;
}

void test_softmax() {
    std::vector <std::vector<float>> z = {{1.0f, 2.0f},
                                          {2.0f, 1.0f}};
    std::vector <std::vector<float>> e = softmax(z);
    for (auto row: e) {
        for (auto elem: row) {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
}

float cross_entropy(const std::vector<int> &y, const std::vector <std::vector<float>> &y_hat) {
    int bs = y.size();
    float sum = 0.0f;
    for (int i = 0; i < bs; ++i) {
        sum += std::log(y_hat[i][y[i]]);
    }
    return -sum / bs;
}

void test_cross_entropy() {
    std::vector<int> y = {1, 2, 3, 4};
    std::vector <std::vector<float>> y_hat = {{0.1f, 0.2f, 0.2f, 0.4f, 0.1f},
                                              {0.4f, 0.3f, 0.1f, 0.1f, 0.1f},
                                              {0.2f, 0.1f, 0.3f, 0.3f, 0.1f},
                                              {0.2f, 0.1f, 0.4f, 0.2f, 0.1f}};
    std::cout << cross_entropy(y, y_hat) << std::endl;
}


std::vector<std::vector<float>> d_softmax_cross_entropy(const std::vector<int>& y, const std::vector<std::vector<float>>& y_hat) {
    int bs = y.size();
    int num_classes = y_hat[0].size();

    std::vector<std::vector<float>> y_one_hot(bs, std::vector<float>(num_classes, 0.0f));
    for (int i = 0; i < bs; ++i) {
        y_one_hot[i][y[i]] = 1.0f;
    }

    std::vector<std::vector<float>> res(bs, std::vector<float>(num_classes));
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            res[i][j] = (y_hat[i][j] - y_one_hot[i][j]) / bs;
        }
    }

    return res;
}

void test_d_softmax_cross_entropy() {
    std::vector<int> y = {1, 2, 3, 4};
    std::vector <std::vector<float>> logits = {{1, 2, 2, 4, 1},
                                               {4, 3, 1, 1, 1},
                                               {2, 1, 3, 3, 1},
                                               {2, 1, 4, 2, 1}};
    std::vector <std::vector<float>> y_hat = softmax(logits);
    std::vector <std::vector<float>> d = d_softmax_cross_entropy(y, y_hat);
    for (std::size_t i = 0; i < d.size(); ++i) {
        for (std::size_t j = 0; j < d[i].size(); ++j) {
            std::cout << d[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

class Sigmoid {
public:
    std::vector<std::vector<float>> cachez_;
    Sigmoid() {
        // Create cache to hold values for backward pass
        cachez_ = std::vector<std::vector<float>>();
    }

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& x) {
        int bs = x.size();
        int output_size = x[0].size();

        std::vector<std::vector<float>> z(bs, std::vector<float>(output_size));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < output_size; ++j) {
                z[i][j] = 1.0f / (1.0f + std::exp(-x[i][j]));
            }
        }
        cachez_ = z;
        return z;
    }

    std::vector<std::vector<float>> operator()(const std::vector<std::vector<float>>& x) {
        return forward(x);
    }

    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& dz) {
        int bs = dz.size();
        int output_size = dz[0].size();

        std::vector<std::vector<float>> res(bs, std::vector<float>(output_size));
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < output_size; ++j) {
                res[i][j] = dz[i][j] * (1 - cachez_[i][j]) * cachez_[i][j];
            }
        }
        return res;
    }
};

void test_d_sigmoid() {
    std::vector <std::vector<float>> x = {{1, 2, 2, 4, 1},
                                          {4, 3, 1, 1, 1},
                                          {2, 1, 3, 3, 1},
                                          {2, 1, 4, 2, 1}};
    int bs = x.size();
    int dim = x[0].size();
    Sigmoid sig;
    std::vector<std::vector<float>> z = sig(x);
    std::cout << "z =" << std::endl;
    for (std::size_t i = 0; i < z.size(); ++i) {
        for (std::size_t j = 0; j < z[i].size(); ++j) {
            std::cout << z[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<float>> one(bs, std::vector<float>(dim, 1.0f));
    std::vector<std::vector<float>> dx = sig.backward(one);
    std::cout << "dx =" << std::endl;
    for (std::size_t i = 0; i < dx.size(); ++i) {
        for (std::size_t j = 0; j < dx[i].size(); ++j) {
            std::cout << dx[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


class Linear {
public:
    int input_size_;
    int output_size_;
    float learning_rate_;
    std::vector <std::vector<float>> w_;
    std::vector <std::vector<float>> dw_;
    std::vector <std::vector<float>> cachex_;

    Linear(int input_size, int output_size, float learning_rate) :
            input_size_(input_size),
            output_size_(output_size),
            learning_rate_(learning_rate),
            w_(input_size + 1, std::vector<float>(output_size)),
            dw_(input_size + 1, std::vector<float>(output_size)) {
        // Initialize weights with random values using the random_init function
        w_ = random_init(std::pair(input_size + 1, output_size));
    }

    std::vector <std::vector<float>> forward(const std::vector <std::vector<float>> &x) {
        // Add bias to the input
        std::vector <std::vector<float>> x_padded(x.size(), std::vector<float>(x[0].size() + 1, 1.0));
        for (std::size_t i = 0; i < x.size(); ++i) {
            for (std::size_t j = 1; j < x[0].size() + 1; ++j) {
                x_padded[i][j] = x[i][j - 1];
            }
        }
        cachex_ = x_padded;

        // Perform matrix multiplication
        std::vector <std::vector<float>> z(x_padded.size(), std::vector<float>(output_size_));
        for (std::size_t i = 0; i < x_padded.size(); ++i) {
            for (int j = 0; j < output_size_; ++j) {
                for (int k = 0; k < input_size_ + 1; ++k) {
                    z[i][j] += x_padded[i][k] * w_[k][j];
                }
            }
        }
        return z;
    }

    std::vector <std::vector<float>> operator()(const std::vector <std::vector<float>> &x) {
        return forward(x);
    }

    std::vector <std::vector<float>> backward(const std::vector <std::vector<float>> &dz) {
        for (std::size_t i = 0; i < cachex_[0].size(); ++i) {
            for (std::size_t j = 0; j < dz[0].size(); ++j) {
                float sum = 0;
                for (std::size_t k = 0; k < dz.size(); ++k) {
                    sum += cachex_[k][i] * dz[k][j];
                }
                dw_[i][j] = sum;
            }
        }

        // Compute dx
        std::vector <std::vector<float>> dx(dz.size(), std::vector<float>(input_size_));
        for (std::size_t i = 0; i < dz.size(); ++i) {
            for (int j = 0; j < input_size_; ++j) {
                for (int k = 0; k < output_size_; ++k) {
                    dx[i][j] += dz[i][k] * w_[j + 1][k];
                }
            }
        }
        return dx;
    }

    void step() {
        for (std::size_t i = 0; i < w_.size(); ++i) {
            for (std::size_t j = 0; j < w_[i].size(); ++j) {
                w_[i][j] -= learning_rate_ * dw_[i][j];
            }
        }
    }
};


void test_d_linear() {
    std::vector <std::vector<float>> x = {{0, 1, 2},
                                          {3, 4, 5},
                                          {6, 7, 8},
                                          {9, 10, 11}};
    std::vector <std::vector<float>> w = {{0, 1},
                                          {2, 3},
                                          {4, 5},
                                          {6, 7}};
    int bs = x.size();
    int dim_in = x[0].size();
    int dim_out = w[0].size();
    Linear linear {dim_in, dim_out, 0.1f};
    std::cout << "init w =" << std::endl;
    for (std::size_t i = 0; i < linear.w_.size(); ++i) {
        for (std::size_t j = 0; j < linear.w_[i].size(); ++j) {
            std::cout << linear.w_[i][j] << " ";
        }
        std::cout << std::endl;
    }
    linear.w_ = w;
    std::vector<std::vector<float>> z = linear(x);
    std::cout << "z =" << std::endl;
    for (std::size_t i = 0; i < z.size(); ++i) {
        for (std::size_t j = 0; j < z[i].size(); ++j) {
            std::cout << z[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<float>> one(bs, std::vector<float>(dim_out, 1.0f));
    std::vector<std::vector<float>> dx = linear.backward(one);
    std::cout << "dx =" << std::endl;
    for (std::size_t i = 0; i < dx.size(); ++i) {
        for (std::size_t j = 0; j < dx[i].size(); ++j) {
            std::cout << dx[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "dw =" << std::endl;
    for (std::size_t i = 0; i < linear.dw_.size(); ++i) {
        for (std::size_t j = 0; j < linear.dw_[i].size(); ++j) {
            std::cout << linear.dw_[i][j] << " ";
        }
        std::cout << std::endl;
    }
    linear.step();
}


class MLP {
public:
    int input_size;
    std::vector<int> hidden_sizes;
    int output_size;
    int num_layers;
    float learning_rate;
    std::vector<Linear> linears;
    std::vector<Sigmoid> sigmoids;
    MLP(int input_size, std::vector<int> hidden_sizes, int output_size, float learning_rate)
            : input_size(input_size),
              hidden_sizes(hidden_sizes),
              output_size(output_size),
              learning_rate(learning_rate) {
        num_layers = hidden_sizes.size();
        linears.emplace_back(Linear(input_size, hidden_sizes[0], learning_rate));
        sigmoids.emplace_back(Sigmoid());
        for (int i = 1; i < num_layers; i++) {
            int hs = hidden_sizes[i];
            int hs_last = hidden_sizes[i - 1];
            linears.emplace_back(Linear(hs_last, hs, learning_rate));
            sigmoids.emplace_back(Sigmoid());
        }
        linears.emplace_back(Linear(hidden_sizes[num_layers - 1], output_size, learning_rate));
    }

    std::vector<std::vector<float>> forward(std::vector<std::vector<float>> x) {
        for (std::size_t i = 0; i < hidden_sizes.size(); ++i) {
            x = linears[i](x);
            x = sigmoids[i](x);
        }
        x = linears[hidden_sizes.size()](x);
        return softmax(x);
    }

    std::vector <std::vector<float>> operator()(std::vector <std::vector<float>> x) {
        return forward(x);
    }

    void backward(const std::vector<int>& y, const std::vector<std::vector<float>>& y_hat) {
        std::vector<std::vector<float>> g = d_softmax_cross_entropy(y, y_hat);
        for (int i = static_cast<int>(hidden_sizes.size()); i >= 0; --i) {
            g = linears[i].backward(g);
        }
    }

    void step() {
        for (auto& l : linears) {
            l.step();
        }
    }
};

void test_mlp() {
    MLP model {1024, {64}, 10, 0.1};
    std::vector <std::vector<float>> x(4, std::vector<float>(1024, 0.0f));
    std::vector<std::vector<float>> y_hat = model(x);
    for (auto row : y_hat) {
        for (auto p : row) {
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }
    for (auto l : model.linears) {
        std::cout << l.cachex_.size() << " ";
        std::cout << l.cachex_[0].size() << " ";
    }
    std::cout << std::endl;
    std::vector<int> y{1,2,3,4};
    model.backward(y, y_hat);
    model.step();
}

void read_mnist(std::vector<std::vector<float>>& X, std::vector<int>& y) {
    // Open the CSV file
    std::ifstream file("mnist/mnist_train.csv");

    // Read the first line and store it in a string
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        // Parse the line and store the values in a vector
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');
        y.push_back(std::stoi(cell));
        while (std::getline(ss, cell, ',')) {
            // Convert the cell string to a float and store it in the vector
            row.push_back(std::stof(cell) / 255.f);
        }

        // Add the row vector to the data matrix
        X.push_back(row);
    }
    // Close the file
    file.close();

//    for (int i = 0; i < 10; ++i) {
//        std::cout << y[i] << " ";
//    }
//    std::cout << std::endl;
//    float sum = 0;
//    for (std::size_t i = 0; i < X[0].size(); ++i) {
//        sum += X[1][i];
//    }
//    std::cout << sum;
}

int main() {
//    test_softmax();
//    test_cross_entropy();
//    test_d_softmax_cross_entropy();
//    test_d_sigmoid();
//    test_d_linear();
//    test_mlp();
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