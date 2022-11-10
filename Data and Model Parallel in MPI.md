# Data and Model Parallel in MPI
## Proposal URL
https://github.com/galalalala/15618-Final-Project

## Summary
We will implement and experiment different parallel paradigms in deep learning. Specifically, data parallelsim and model parallelism will be implemented via MPI. The code suports forwad and backward pass runing on CPU in a cluster. We will analyze and compare the results of two parallel paradigms.
## Background
Training large scale machine learning models on single machine can be chanllanging due to the model complexity and the growth of the data. The training process could take hundreds of hours. Paralellizing the machine learning models is necessary to solve the the problems effciently at scale. For this project, we will implement data and model parallelism on a multi-layer MLP network.
## The Challenge
1. Efficient communications between the nodes in a cluster.
2. Consistency issues: how to update the gradient in each step.
3. Implementation of backpropagation can be challenging without build the computation graph.
 
## Resource
Our hardware resource consists of our PCs with Ubuntu installed and Intel multi-core x86-64 processors, the CMU GHC cluster, and the PSC cluster. We plan to start from our code written in the Machine Learning and Deep Learning Systems course. Although we are not going to use existing deep learning frameworks such as PyTorch, some documentaion of existing work might help us design our system.

## Goals and Deliverables

### Plan to Achieve (without computation graph)
1. Baseline multi-layer MLP with C++ running on single machine. 
2. Build data parallelism running on multiple machines. 
3. Build model parallelism running on multiple machines.
4. Compare and analyze the results of two parallel paradigms.

### Hope to Achieve (with computation graph)
1. Baseline ResNet with C++/CUDA running on single machine. 
2. Build data parallelism running on multiple machines.
3. Build model parallelism running on multiple machines.
4. Compare and analyze the results of two parallel paradigms.

## Platform Choice
We will use C++ as the main programming language. C++ is both performant and complete, making it suitable for computational heavy workloads such as deep learning training. Also, MPI has full support to C++, which makes it possible for us to implement our message passing ideas. In terms of the hardware, we will use our personal PC with Ubuntu and Intel processor to develop the single-node baseline and the GHC and PSC machines to test our multi-node implementation.

## Schedule
- Week of Nov 14. Implement the baseline multi-layer MLP with C++ running on single machine, including forward and backward pass.
- Week of Nov 21. Develop data parallelism across multiple machines using MPI. Evaluate the performance. Analyze the communication cost and bottleneck.
- Week of Nov 28. Develop model parallelism to distribute layers among machines. Evaluate the performance. Analyze the communication cost and bottleneck. Write the milstone report.
- Week of Dec 5. Explore the possibility of incorporating computational graph into our implementation to support more complex operators and model architecture such as ResNet.
- Week of Dec 12. Run experiments on the PSC cluster. Summarize and compare different algorithms. Finish the report and poster.