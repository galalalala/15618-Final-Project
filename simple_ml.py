import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def parse_mnist(image_filename, label_filename):

    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    with gzip.open(image_filesname, 'rb') as f:
        
        _ = struct.unpack('>I', f.read(4))[0]
        n_img = struct.unpack('>I', f.read(4))[0]
        n_row = struct.unpack('>I', f.read(4))[0]
        n_col = struct.unpack('>I', f.read(4))[0]

        image_buffer = f.read()
        images = np.frombuffer(image_buffer, dtype=np.uint8)\
            .reshape((n_img, -1)).astype(np.float32) / 255.0 

    with gzip.open(label_filename, 'rb') as f:
    
        _ = struct.unpack('>I', f.read(4))[0]
        n_label = struct.unpack('>I', f.read(4))[0]

        label_buffer = f.read()
        labels = np.frombuffer(label_buffer, dtype=np.uint8) 
        
    return images, labels
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(len(Z)), y]) 
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples, num_classes = X.shape
    input_dim, num_classes = theta.shape
    
    # batch_idx = np.random.choice(num_examples, batch, replace=False)
    num_batches = num_examples // batch

    for i in range(1, num_batches + 1):

      X_batch = X[batch * (i-1):i * batch, :]
      y_batch = y[batch * (i-1):i * batch]
      
      one_hot_y = np.zeros((batch, num_classes))
      one_hot_y[np.arange(batch), y_batch] = 1

      assert X_batch.shape == (batch, input_dim)
      assert y_batch.shape == (batch,)
      assert one_hot_y.shape == (batch, num_classes)
      
      z = np.exp(X_batch @ theta)
      Z =  z / np.sum(z, axis=1).reshape((-1,1))
      
      assert Z.shape == (batch, num_classes)

      gradent = (X_batch.T @ (Z - one_hot_y)) / batch
      
      assert gradent.shape == (input_dim, num_classes)
      
      theta -= lr * gradent
  


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """



    ### BEGIN YOUR CODE
    def RELU(X):
      X[X<=0] = 0
      return X

    def one_hot(y, batch, num_classes):
      # y: 1 x num_examples
      one_hot_y = np.zeros((batch, num_classes))
      one_hot_y[np.arange(batch), y] = 1
      return one_hot_y

    def indicator_func(Z1):
      Z1_copy = Z1.copy()
      Z1_copy[Z1_copy > 0] = 1
      Z1_copy[Z1_copy <=0] = 0
      return Z1_copy


    num_examples, input_dim = X.shape
    hidden_dim, num_classes = W2.shape

    num_batches = num_examples // batch

    for i in range(1, num_batches + 1):
      
      X_batch = X[(i-1) * batch:i * batch, :]
      y_batch = y[(i-1) * batch:i * batch]

      one_hot_y = one_hot(y_batch, batch, num_classes)
      Z1 = RELU(X_batch @ W1)
      assert Z1.shape == (batch, hidden_dim)

      z1_w2 = np.exp(Z1 @ W2) 
      assert z1_w2.shape == (batch, num_classes)

      G2 = z1_w2 / np.sum(z1_w2, axis=1).reshape((-1,1)) - one_hot_y
      assert G2.shape == (batch, num_classes)

     
      G1 = np.multiply(G2 @ W2.T, indicator_func(Z1))
      assert G1.shape == (batch, hidden_dim)

      gradient_W1 = X_batch.T @ G1 / batch
      gradient_W2 = Z1.T @ G2 / batch
      assert gradient_W1.shape == (input_dim, hidden_dim)
      assert gradient_W2.shape == (hidden_dim, num_classes)

      W1 -= lr * gradient_W1
      W2 -= lr * gradient_W2 

    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

def nn_step(X_batch, y_batch, W1, W2, batch=20):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarrray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarrray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """



    ### BEGIN YOUR CODE
    def RELU(X):
      X[X<=0] = 0
      return X

    def one_hot(y, batch, num_classes):
      # y: 1 x num_examples
      one_hot_y = np.zeros((batch, num_classes))
      one_hot_y[np.arange(batch), y] = 1
      return one_hot_y

    def indicator_func(Z1):
      Z1_copy = Z1.copy()
      Z1_copy[Z1_copy > 0] = 1
      Z1_copy[Z1_copy <=0] = 0
      return Z1_copy


    hidden_dim, num_classes = W2.shape

    one_hot_y = one_hot(y_batch, batch, num_classes)
    Z1 = RELU(X_batch @ W1)
    assert Z1.shape == (batch, hidden_dim)

    z1_w2 = np.exp(Z1 @ W2) 
    assert z1_w2.shape == (batch, num_classes)

    G2 = z1_w2 / np.sum(z1_w2, axis=1).reshape((-1,1)) - one_hot_y
    assert G2.shape == (batch, num_classes)

    
    G1 = np.multiply(G2 @ W2.T, indicator_func(Z1))
    assert G1.shape == (batch, hidden_dim)

    gradient_W1 = X_batch.T @ G1 / batch
    gradient_W2 = Z1.T @ G2 / batch


    return gradient_W1, gradient_W2


if __name__ == "__main__":
    from mpi4py import MPI

    import numpy as np

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    X_batch, y_batch = None, None
    bs_node = 50
    bs = size * bs_node
    num_epoch = 20

    n, k = 784, 10
    hidden_dim = 500
    np.random.seed(0)

    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    lr = 0.1

    if rank == 0:
        X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
        X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")
                           
    for epoch in range(num_epoch):
        for b in range(0, 60000, bs):
            if rank == 0:
                
                X_batch = X_tr[b : b + bs]
                y_batch = y_tr[b : b + bs]
                # print(y_batch)
             
            X_buf = np.empty([bs_node, n], dtype=np.float32)
            y_buf = np.empty([bs_node], dtype=np.uint8)

            comm.Scatter(X_batch, X_buf, root=0)
            comm.Scatter(y_batch, y_buf, root=0)

                # print(X_buf.shape, X_buf[0, 300:305])
                # print(y_buf.shape, y_buf)
            
            d_W1, d_W2 = nn_step(X_buf, y_buf, W1, W2, batch=bs_node)

            # print("d_W1 shape: ", d_W1.shape, d_W1[300, 400:405])
            # print("d_W2 shape: ", d_W2.shape)
            d_W1s, d_W2s = None, None
            # print(d_W1.dtype)
            if rank == 0:
                d_W1s = np.zeros([size, n, hidden_dim], dtype=np.float64)
                d_W2s = np.zeros([size, hidden_dim, k], dtype=np.float64)
                # print(d_W1s.shape)
            # print(d_W1.shape)
            comm.Gather(d_W1, d_W1s, root=0)
            comm.Gather(d_W2, d_W2s, root=0)

            if rank == 0:
                # print(d_W1s.shape)  
                # print(d_W1s[0, 300, 400:405], d_W1s[1, 300, 400:405])   
                d_W1 = d_W1s.mean(axis=0)
                d_W2 = d_W2s.mean(axis=0)        
                # print(d_W1s.shape)
            comm.Bcast(d_W1, root=0)
            comm.Bcast(d_W2, root=0)

            W1 -= lr * d_W1
            W2 -= lr * d_W2 

            if rank == 0:
                train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
                test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
                print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
                    .format(epoch, train_loss, train_err, test_loss, test_err))




