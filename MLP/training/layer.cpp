#include "layer.h"

Eigen::VectorXd ReLU(const Eigen::VectorXd& x) {
	return x.array().max(0.0);
}

Eigen::VectorXd layer::ReLUPrime() const {
	return (z.array() > 0).cast<double>();
}

layer::layer(int in, int out) {
	bias = Eigen::VectorXd::Zero(out);
	weights = sqrt(6.d / (in + out)) * Eigen::MatrixXd::Random(out, in);
}

void layer::feedForward(Eigen::VectorXd& input) {
	if (input.size() != weights.cols()) throw -2;
	in = input;
	
	z = weights * input + bias;
	
	input = ReLU(z);
	activations = input;
}