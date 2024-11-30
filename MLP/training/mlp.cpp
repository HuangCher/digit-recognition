#include "mlp.h"

void mlp::insertLayer(layer l) {
	layers.push_back(l);
}

Eigen::VectorXd mlp::fProp(std::vector<double> image) {
	Eigen::VectorXd input = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(image.data(), image.size());
	for (auto& l : layers) {
		l.feedForward(input);
	}
	
	return input;
}

void mlp::bProp(int expected) {
	double learningRate = 0.001;
	
	Eigen::VectorXd target = Eigen::VectorXd::Zero(layers.back().bias.size());
	target(expected) = 1;
	
	Eigen::VectorXd delta = layers.back().activations - target;
	
	
	for (int i = layers.size() - 1; i >= 0; --i) {
		layer& current = layers[i];
		
		Eigen::VectorXd dL = delta.array() * current.ReLUPrime().array();
		
		
		Eigen::VectorXd dLdB = dL;
		Eigen::MatrixXd A;
		if (i == 0) {A = current.in.transpose();}
		else {A = layers[i - 1].z.transpose();}
		Eigen::MatrixXd dLdW = dL * A;
		
		
		current.weights -= learningRate * dLdW;
		current.bias -= learningRate * dLdB;
		
		
		if (i > 0) {
			delta = current.weights.transpose() * dL;
		}
	}
}

void mlp::writeLayers(std::string filename) const {
	std::ofstream layerFile("backend/" + filename);
	if (!layerFile.is_open()) {
		std::cerr << "Error: Unable to write layers to file" << std::endl;
		return;
	}
	
	layerFile << layers.size() << std::endl;
	
	for (const auto& l : layers) {
		// write bias
		layerFile << l.bias.size() << std::endl;
		for (int i = 0; i < l.bias.size(); ++i) {
			layerFile << l.bias[i] << " ";
		}
		layerFile << std::endl;

		// write weights
		layerFile << l.weights.rows() << " " << l.weights.cols() << std::endl;
		for (int i = 0; i < l.weights.rows(); ++i) {
			for (int j = 0; j < l.weights.cols(); ++j) {
				layerFile << l.weights(i, j) << " ";
			}
			layerFile << std::endl;
		}
	}
}