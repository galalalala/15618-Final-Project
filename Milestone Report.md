# Milestone Report

## Progress Summary
1. Implemented the baseline model, 2-layer MLP, supporting forward and backward passes on single machine.
2. Implemented the data parallellisim for the baseline 2-layer MLP model(without building the computational graph) on MNIST dataset supporting arbitray number of workers in MPI in Python interface (mpi4py). Currently, we calculate the gradients of each layer by hand, which is hard to scale up with the number of layers. 
3. Bechmarked the training and testing accuracies and runtime on a local PC.

## Preliminary Results
We are able to achieve data parallelism with a central parameter server. We benchmared the running time on 1 node and 4 nodes with 95% accuracy on MNIST. We observe 1.8x speed-up when using 4 nodes. The reason why we do not achieve a higher speed-up is due to frequent communication (every step) in our current implementation and the bottleneck of the central node.

## Updated Goals
1. Implement and benchmark the current design in C++ and compare the results with the Python implementation.
2. Modularize the calculation with backpropagation to support more layers without manually calculating the gradient of each layer.
3. Add a simple computational graph to support training residual layers.
4. Run experiments on the GHC cluster with more computational nodes and report the speedup with respect to number of workers.
5. Although we find it hard to implement model-parallalism in C++, we may try to implement ring data parallelism and model parallelism in Python if we have time.
