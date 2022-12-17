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
#include <chrono>

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


void save_log(const std::vector<int>& indices,
                           const std::vector<float>& losses,
                           double t,
                           const std::string& filename) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    // File could not be opened for some reason.
    return;
  }

  for (std::size_t i = 0; i < indices.size(); ++i) {
    outFile << indices[i] << ' ' << losses[i] << '\n';
  }
  outFile << t << '\n';
}


class Timer {
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};
