#include "layer.h"
#include <vector>
#include <fstream>

class mlp {
	std::vector<layer> layers;
	
	public:
		void insertLayer(layer l);
		
		Eigen::VectorXd fProp(std::vector<double> image); // forward propagation
		void bProp(int target); // backward propagation
		
		void writeLayers(std::string filename) const; // write layers to disk
};