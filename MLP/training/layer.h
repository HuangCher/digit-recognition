#include <cmath>
#include <Eigen/Dense>
#include <iostream>

struct layer {
	Eigen::VectorXd bias;
	Eigen::MatrixXd weights;
	
	Eigen::VectorXd z; // before ReLU
	Eigen::VectorXd activations; // after ReLU
	
	Eigen::VectorXd in; // only needed for first layer backpropagation
	
	Eigen::VectorXd ReLUPrime() const;
	layer(int in, int out); // size of input layer, size of output layer
	void feedForward(Eigen::VectorXd& input); // ReLU(weights * input + bias)
};

