mlp:
	g++ -std=c++20 -Wall -g  mlp.cpp -o ./mlp

serial: mlp.h train_mnist_serial.cpp
	g++ -std=c++20 -Wall -g  train_mnist_serial.cpp -o ./serial

psrv: mlp.h train_mnist_param_srv.cpp
	mpic++ -std=c++20  -Wall -g  train_mnist_param_srv.cpp -o ./psrv

allreduce: mlp.h train_mnist_allreduce.cpp
	mpic++ -std=c++20  -Wall -g  train_mnist_allreduce.cpp -o ./allreduce

mpara: mlp.h train_mnist_mpara.cpp
	mpic++ -std=c++20  -Wall -g -O0 train_mnist_mpara.cpp -o ./mpara

run_serial: serial
	./serial 2 128 0.1 32

run_psrv: psrv
	mpirun -n $(n) ./psrv 2 128 0.1 32

run_allreduce: allreduce
	mpirun -n $(n) ./allreduce 2 128 0.1 32

run_mpara: mpara
	mpirun -n $(n) ./mpara 128 0.1 256

exp_serial: serial
# change number of layers
	./serial 1 32 0.1 32
	./serial 2 32 0.1 32
	./serial 3 32 0.1 32

# change hidden dim
	./serial 1 16 0.1 32
	./serial 1 32 0.1 32
	./serial 1 64 0.1 32
	./serial 1 128 0.1 32

exp_psrv: psrv
# change number of layers
	mpirun -n 1 ./psrv 1 32 0.1 32
	mpirun -n 2 ./psrv 1 32 0.1 32
	mpirun -n 4 ./psrv 1 32 0.1 32
	mpirun -n 8 ./psrv 1 32 0.1 32
	mpirun -n 16 ./psrv 1 32 0.1 32
	mpirun -n 32 ./psrv 1 32 0.1 32
	mpirun -n 64 ./psrv 1 32 0.1 32
	mpirun -n 128 ./psrv 1 32 0.1 32

	mpirun -n 1 ./psrv 2 32 0.1 32
	mpirun -n 2 ./psrv 2 32 0.1 32
	mpirun -n 4 ./psrv 2 32 0.1 32
	mpirun -n 8 ./psrv 2 32 0.1 32
	mpirun -n 16 ./psrv 2 32 0.1 32
	mpirun -n 32 ./psrv 2 32 0.1 32
	mpirun -n 64 ./psrv 2 32 0.1 32
	mpirun -n 128 ./psrv 2 32 0.1 32

	mpirun -n 1 ./psrv 3 32 0.1 32
	mpirun -n 2 ./psrv 3 32 0.1 32
	mpirun -n 4 ./psrv 3 32 0.1 32
	mpirun -n 8 ./psrv 3 32 0.1 32
	mpirun -n 16 ./psrv 3 32 0.1 32
	mpirun -n 32 ./psrv 3 32 0.1 32
	mpirun -n 64 ./psrv 3 32 0.1 32
	mpirun -n 128 ./psrv 3 32 0.1 32

# change hidden dim
	mpirun -n 1 ./psrv 1 16 0.1 32
	mpirun -n 2 ./psrv 1 16 0.1 32
	mpirun -n 4 ./psrv 1 16 0.1 32
	mpirun -n 8 ./psrv 1 16 0.1 32
	mpirun -n 16 ./psrv 1 16 0.1 32
	mpirun -n 32 ./psrv 1 16 0.1 32
	mpirun -n 64 ./psrv 1 16 0.1 32
	mpirun -n 128 ./psrv 1 16 0.1 32

	mpirun -n 1 ./psrv 1 32 0.1 32
	mpirun -n 2 ./psrv 1 32 0.1 32
	mpirun -n 4 ./psrv 1 32 0.1 32
	mpirun -n 8 ./psrv 1 32 0.1 32
	mpirun -n 16 ./psrv 1 32 0.1 32
	mpirun -n 32 ./psrv 1 32 0.1 32
	mpirun -n 64 ./psrv 1 32 0.1 32
	mpirun -n 128 ./psrv 1 32 0.1 32

	mpirun -n 1 ./psrv 1 64 0.1 32
	mpirun -n 2 ./psrv 1 64 0.1 32
	mpirun -n 4 ./psrv 1 64 0.1 32
	mpirun -n 8 ./psrv 1 64 0.1 32
	mpirun -n 16 ./psrv 1 64 0.1 32
	mpirun -n 32 ./psrv 1 64 0.1 32
	mpirun -n 64 ./psrv 1 64 0.1 32
	mpirun -n 128 ./psrv 1 64 0.1 32

	mpirun -n 1 ./psrv 1 128 0.1 32
	mpirun -n 2 ./psrv 1 128 0.1 32
	mpirun -n 4 ./psrv 1 128 0.1 32
	mpirun -n 8 ./psrv 1 128 0.1 32
	mpirun -n 16 ./psrv 1 128 0.1 32
	mpirun -n 32 ./psrv 1 128 0.1 32
	mpirun -n 64 ./psrv 1 128 0.1 32
	mpirun -n 128 ./psrv 1 128 0.1 32

exp_allreduce: allreduce
# change number all layers
	mpirun -n 1 ./allreduce 1 32 0.1 32
	mpirun -n 2 ./allreduce 1 32 0.1 32
	mpirun -n 4 ./allreduce 1 32 0.1 32
	mpirun -n 8 ./allreduce 1 32 0.1 32
	mpirun -n 16 ./allreduce 1 32 0.1 32
	mpirun -n 32 ./allreduce 1 32 0.1 32
	mpirun -n 64 ./allreduce 1 32 0.1 32
	mpirun -n 128 ./allreduce 1 32 0.1 32

	mpirun -n 1 ./allreduce 2 32 0.1 32
	mpirun -n 2 ./allreduce 2 32 0.1 32
	mpirun -n 4 ./allreduce 2 32 0.1 32
	mpirun -n 8 ./allreduce 2 32 0.1 32
	mpirun -n 16 ./allreduce 2 32 0.1 32
	mpirun -n 32 ./allreduce 2 32 0.1 32
	mpirun -n 64 ./allreduce 2 32 0.1 32
	mpirun -n 128 ./allreduce 2 32 0.1 32

	mpirun -n 1 ./allreduce 3 32 0.1 32
	mpirun -n 2 ./allreduce 3 32 0.1 32
	mpirun -n 4 ./allreduce 3 32 0.1 32
	mpirun -n 8 ./allreduce 3 32 0.1 32
	mpirun -n 16 ./allreduce 3 32 0.1 32
	mpirun -n 32 ./allreduce 3 32 0.1 32
	mpirun -n 64 ./allreduce 3 32 0.1 32
	mpirun -n 128 ./allreduce 3 32 0.1 32

# change hidden dim
	mpirun -n 1 ./allreduce 1 16 0.1 32
	mpirun -n 2 ./allreduce 1 16 0.1 32
	mpirun -n 4 ./allreduce 1 16 0.1 32
	mpirun -n 8 ./allreduce 1 16 0.1 32
	mpirun -n 16 ./allreduce 1 16 0.1 32
	mpirun -n 32 ./allreduce 1 16 0.1 32
	mpirun -n 64 ./allreduce 1 16 0.1 32
	mpirun -n 128 ./allreduce 1 16 0.1 32

	mpirun -n 1 ./allreduce 1 32 0.1 32
	mpirun -n 2 ./allreduce 1 32 0.1 32
	mpirun -n 4 ./allreduce 1 32 0.1 32
	mpirun -n 8 ./allreduce 1 32 0.1 32
	mpirun -n 16 ./allreduce 1 32 0.1 32
	mpirun -n 32 ./allreduce 1 32 0.1 32
	mpirun -n 64 ./allreduce 1 32 0.1 32
	mpirun -n 128 ./allreduce 1 32 0.1 32

	mpirun -n 1 ./allreduce 1 64 0.1 32
	mpirun -n 2 ./allreduce 1 64 0.1 32
	mpirun -n 4 ./allreduce 1 64 0.1 32
	mpirun -n 8 ./allreduce 1 64 0.1 32
	mpirun -n 16 ./allreduce 1 64 0.1 32
	mpirun -n 32 ./allreduce 1 64 0.1 32
	mpirun -n 64 ./allreduce 1 64 0.1 32
	mpirun -n 128 ./allreduce 1 64 0.1 32

	mpirun -n 1 ./allreduce 1 128 0.1 32
	mpirun -n 2 ./allreduce 1 128 0.1 32
	mpirun -n 4 ./allreduce 1 128 0.1 32
	mpirun -n 8 ./allreduce 1 128 0.1 32
	mpirun -n 16 ./allreduce 1 128 0.1 32
	mpirun -n 32 ./allreduce 1 128 0.1 32
	mpirun -n 64 ./allreduce 1 128 0.1 32
	mpirun -n 128 ./allreduce 1 128 0.1 32

exp_mpara: mpara
# change hidden dim
	mpirun -n 1 ./mpara 128 0.1 256
	mpirun -n 2 ./mpara 128 0.1 256
	mpirun -n 4 ./mpara 128 0.1 256
	mpirun -n 8 ./mpara 128 0.1 256
	mpirun -n 16 ./mpara 128 0.1 256
	mpirun -n 32 ./mpara 128 0.1 256
	mpirun -n 64 ./mpara 128 0.1 256
	mpirun -n 128 ./mpara 128 0.1 256

	mpirun -n 1 ./mpara 256 0.1 256
	mpirun -n 2 ./mpara 256 0.1 256
	mpirun -n 4 ./mpara 256 0.1 256
	mpirun -n 8 ./mpara 256 0.1 256
	mpirun -n 16 ./mpara 256 0.1 256
	mpirun -n 32 ./mpara 256 0.1 256
	mpirun -n 64 ./mpara 256 0.1 256
	mpirun -n 128 ./mpara 256 0.1 256

	mpirun -n 1 ./mpara 512 0.1 256
	mpirun -n 2 ./mpara 512 0.1 256
	mpirun -n 4 ./mpara 512 0.1 256
	mpirun -n 8 ./mpara 512 0.1 256
	mpirun -n 16 ./mpara 512 0.1 256
	mpirun -n 32 ./mpara 512 0.1 256
	mpirun -n 64 ./mpara 512 0.1 256
	mpirun -n 128 ./mpara 512 0.1 256


.PHONY: run_serial run_psrv run_allreduce run_mpara exp_serial exp_psrv exp_allreduce exp_mpara