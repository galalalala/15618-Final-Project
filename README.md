# Data and Model Parallel in MPI
We implement data and model parallelism for MLP networks on CPU with Numpy and C++ back-ends. We only use built-in libraries for computation and the MPI protocol for message passing among nodes. For data parallelism, we implement various approaches, such as parameter server and AllReduce, and test them on different model configurations. Moreover, we implement the tensor model parallelism that splits the parameters into several devices. For both parallelism paradigms, we observe massive speedup on various model configurations.

Dependencies: recent version GCC and OpenMPI that supports C++20

Unzip `mnist/mnist.zip` and run with following commands:
- serial version: `make run_serial`
- parameter server: `make run_psrv`
- allreduce: `make run_allreduce`
- tensor model parallel `make rum_mpara`
