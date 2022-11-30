#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): posize_ter to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): posize_ter to y data, of size m
     *     theta (foat *): posize_ter to theta data, of size n*k, stored in row
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (size_t): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
      size_t num_batches = m / batch;
      
      // batched X and y
      float *X_b = new float[batch * n];
      float *y_b = new float[batch];

      // X.T and one hot y
      float *X_t = new float[n * batch];
      float *y_one_hot = new float[batch*k];
      
      // X * theta
      float *X_theta = new float[batch * k];

      // Z - I
      float *Z_I = new float[batch * k];

      //gradient
      float *gradient = new float[n * k];

      for (size_t b = 1; b <= num_batches; b++) {
        // fill batch X and y
        for (size_t i = 0; i < batch; i++) {
          for (size_t j = 0; j < n; j++) {
            X_b[i * n + j] = X[(b-1) * batch * n  + i * n + j];
          }
        }

        for (size_t i = 0; i < batch; i++){
          y_b[i] = y[(b-1) * batch + i];
        }

        // X transpose
        for (size_t i = 0; i < batch; i++) {
          for (size_t j = 0; j < n; j++) {
            X_t[j * batch + i] = X_b[i * n + j];
          }
        }
        // one hot y
        for (size_t i = 0; i < batch; i++) {
          for (size_t j = 0; j < k; j++){
            if (j == y_b[i])
              y_one_hot[i * k + j] = 1;
            else
              y_one_hot[i * k + j] = 0;
          }
        }
       
        // X*theta m,n @ n,k
        for (size_t i = 0; i < batch; i++){
          for (size_t j = 0; j < k; j++) {
            float sum = 0;
            for (size_t inner = 0; inner < n; inner++) {
              sum += X_b[i * n + inner] * theta[inner * k + j];    
            }
            X_theta[i * k + j] = exp(sum);
          }
        }

        // Z-I
        for (size_t i = 0; i < batch; i++){
          float sum = 0;
          for (size_t j = 0; j < k; j++) {
            sum += X_theta[i * k + j];
          }
          for (size_t j = 0; j < k; j++) {
            Z_I[i * k + j] = X_theta[i * k + j] / sum - y_one_hot[i * k + j];
          }
         
        }  

        // gradient
        for (size_t i = 0; i < n; i++) {
          for (size_t j = 0; j < k; j++) {
            float sum = 0;
            for (size_t inner = 0; inner < batch; inner++) {
              sum += (X_t[i * batch + inner] * Z_I[inner * k + j]) / batch;
            }
            gradient[i * k + j] = sum;
          }
        }

        //update theta
        for (size_t i = 0; i < n; i++) {
          for (size_t j = 0; j < k; j++) {
            theta[i * k + j] = theta[i * k + j] - lr * gradient[i * k + j];  
          }
        }
      }

      delete[] X_b;
      delete[] y_b;
      delete[] X_t;
      delete[] y_one_hot;
      delete[] X_theta;
      delete[] Z_I;
      delete[] gradient;
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           size_t batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
