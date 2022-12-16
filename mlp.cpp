#include "mlp.h"

int main() {
    test_softmax();
    test_cross_entropy();
    test_d_softmax_cross_entropy();
    test_d_sigmoid();
    test_d_linear();
    test_mlp();
}