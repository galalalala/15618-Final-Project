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