mlp:
	g++ -std=c++20 -Wall -g  mlp.cpp -o ./mlp

serial: mlp.h train_mnist_serial.cpp
	g++ -std=c++20 -Wall -g  train_mnist_serial.cpp -o ./serial

psrv: mlp.h train_mnist_param_srv.cpp
	mpic++ -std=c++14  -Wall -g  train_mnist_param_srv.cpp -o ./psrv

allreduce: mlp.h train_mnist_allreduce.cpp
	mpic++ -std=c++14  -Wall -g  train_mnist_allreduce.cpp -o ./allreduce