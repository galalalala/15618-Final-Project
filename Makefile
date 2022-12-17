mlp:
	g++-12 -std=c++20 -Wall -g  mlp.cpp -o ./mlp

serial:
	g++-12 -std=c++20 -Wall -g  train_mnist_serial.cpp -o ./serial