mlp:
	g++ -std=c++20 -Wall -g  mlp.cpp -o ./mlp

serial: mlp.h train_mnist_serial.cpp
	g++ -std=c++20 -Wall -g  train_mnist_serial.cpp -o ./serial

psrv: mlp.h train_mnist_param_srv.cpp
	mpic++ -std=c++20  -Wall -g  train_mnist_param_srv.cpp -o ./psrv

allreduce: mlp.h train_mnist_allreduce.cpp
	mpic++ -std=c++20  -Wall -g  train_mnist_allreduce.cpp -o ./allreduce

mpara: mlp.h train_mnist_mpara.cpp
	mpic++ -std=c++20  -Wall -g  train_mnist_mpara.cpp -o ./mpara

run_serial: serial
	./serial 2 128 0.1 32

run_psrv: psrv
	mpirun -n $(n) ./psrv 2 128 0.1 32

run_allreduce: allreduce
	mpirun -n $(n) ./allreduce 2 128 0.1 32

run_mpara: mpara
	mpirun -n $(n) ./mpara 128 0.1 256

.PHONY: run_serial run_psrv run_allreduce run_mpara