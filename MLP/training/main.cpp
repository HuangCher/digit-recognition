#include <iomanip>
#include <fstream>
#include <cstdint>
#include "mlp.h"

struct image {
	unsigned char label;
	std::vector<double> pixels; // 28x28 image (784 pixels)
	
	void print() {
		std::cout << static_cast<int>(label) << std::endl;
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(pixels[28 * j + i] * 255) << " ";
				// std::cout << pixels[28 * j + i]<< " ";
			}
			std::cout << std::endl;
		}
	}
};

bool readFiles(std::string labelFileName, std::string imageFileName, std::vector<image>& images) {
	// Open files
	std::ifstream labelFile(labelFileName, std::ios::in | std::ios::binary);
	if (!labelFile.is_open()) {
		std::cerr << "Error: Unable to open label file" << std::endl;
		return false;
	}
	std::ifstream imageFile(imageFileName, std::ios::in | std::ios::binary);
	if (!imageFile.is_open()) {
		std::cerr << "Error: Unable to open image file" << std::endl;
		return false;
	}
	
	// Verify data
	uint32_t magicNumberLabel, numLabels;
	labelFile.read(reinterpret_cast<char*>(&magicNumberLabel), sizeof(magicNumberLabel));
	labelFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
	
	magicNumberLabel = __builtin_bswap32(magicNumberLabel); // Convert from big-endian to little-endian
	numLabels = __builtin_bswap32(numLabels);
	
	uint32_t magicNumberImage, numImages, numRows, numCols;
	imageFile.read(reinterpret_cast<char*>(&magicNumberImage), sizeof(magicNumberImage));
	imageFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
	imageFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
	imageFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));
	
	magicNumberImage = __builtin_bswap32(magicNumberImage); // Convert from big-endian to little-endian
	numImages = __builtin_bswap32(numImages);
	numRows = __builtin_bswap32(numRows);
	numCols = __builtin_bswap32(numCols);
	
	if (magicNumberLabel != 0x00000801) {
		std::cerr << "Error: Invalid magic number in the label file" << std::endl;
		return false;
	}
	if (magicNumberImage != 0x00000803) {
		std::cerr << "Error: Invalid magic number in the image file" << std::endl;
		return false;
	}
	
	if (numRows != 28 || numCols != 28) {
		std::cerr << "Error: Incorrect image dimensions" << std::endl;
		return false;
	}
	
	if (numImages != numLabels) {
		std::cerr << "Error: different quantities of Images and Labels. Images: " << numImages << " Labels: " << numLabels << std::endl;
		return false;
	}
	
	// Read data
	uint32_t numItems = numLabels;
	images.resize(numItems);
	unsigned char c;
	for (int i = 0; i < numItems; i++) {
		labelFile.read(reinterpret_cast<char*>(&images[i].label), 1);
		images[i].pixels.resize(28 * 28);
		for (int j = 0; j < 28 * 28; j++) {
			imageFile.read(reinterpret_cast<char*>(&c), 1);
			images[i].pixels[j] = (double)c / 255;
		}
	}
	
	return true;
}


int getMax(Eigen::VectorXd vec) {
	int maxIndex = 0;
	int index = 0;
	double max = 0;
	for (auto i : vec) {
		if (i > max) {
			maxIndex = index;
			max = i;
		}
		index++;
	}
	return maxIndex;
}


int main() {
	std::string dataset = "digits";
	
	std::string labelFile = "images/emnist-" + dataset + "-train-labels-idx1-ubyte";
	std::string imageFile = "images/emnist-" + dataset + "-train-images-idx3-ubyte";
	
	std::string labelTest = "images/emnist-" + dataset + "-test-labels-idx1-ubyte";
	std::string imageTest = "images/emnist-" + dataset + "-test-images-idx3-ubyte";
	
	// Read data
	
	std::cout << "Loading images..." << std::endl;
	std::vector<image> images;
	if (!readFiles(labelFile, imageFile, images)) {
		return -1;
	}
	
	std::vector<image> testImages;
	if (!readFiles(labelTest, imageTest, testImages)) {
		return -1;
	}
	
	// Configure MLP
	
	mlp m;
	if (dataset == "digits") {	
		m.insertLayer(layer(784, 16));
		m.insertLayer(layer(16, 16));
		m.insertLayer(layer(16, 10));
	} else if (dataset == "balanced") {
		m.insertLayer(layer(784, 80));
		m.insertLayer(layer(80, 70));
		m.insertLayer(layer(70, 47));
	} else {return -1;}
	// Note: the layers need to be inserted like so: 
	// m.insert(layer(784, A)); m.insert(layer(A, B)); m.insert(layer(B, C)); ...; m.insert(layer(X, max output value + 1));
	
	// Train MLP
	
	std::cout << "Training..." << std::endl; 
	
	int index = 0;
	for (auto i : images) {
		m.fProp(i.pixels);
		m.bProp(i.label);
		
		float percent = (index / (float)images.size()) * 100;
		index++;
		std::cout << std::setprecision(4) << "\rProgress: " << percent << "%";
		std::flush(std::cout);
	}
	std::cout << std::endl;
	
	// Test MLP
	
	std::cout << "Testing..." << std::endl;
	int failures = 0;
	int successes = 0;
	index = 0;
	for(auto i : testImages) {
		if (getMax(m.fProp(i.pixels)) == i.label) {successes++;}
		else {failures++;}
		
		float percent = (index / (float)testImages.size()) * 100;
		index++;
		std::cout << std::setprecision(4) << "\rProgress: " << percent << "%";
		std::flush(std::cout);
	}
	std::cout << std::endl << "Successes: " << successes << std::endl;
	std::cout << "Failures: " << failures << std::endl;
	
	// Write layers to disk
	
	std::cout << "Writing layers to disk..." << std::endl;
	m.writeLayers("layers.txt");
	
	return 0;
}
