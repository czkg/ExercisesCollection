/*
 * MNISTClassifier.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */

#include "MNISTClassifier.h"
#include<fstream>

#define epsilon 0.0001


MNISTClassifier::MNISTClassifier(const MNISTDataset& trainDataset) {
	// TODO Auto-generated stub, complete

	//convert from vector to Mat
	int number_of_images = trainDataset.images.size();
	int number_of_features = trainDataset.images[0].size();
	cv::Mat imageMat = cv::Mat::ones(number_of_images, number_of_features, CV_32FC1);
	cv::Mat labelMat = cv::Mat::ones(number_of_images, 1, CV_32SC1);

	for (int i = 0; i < number_of_images; ++i)
	{
		for (int j = 0; j<number_of_features; ++j)
			imageMat.at<float>(i, j) = trainDataset.images.at(i).at(j);
	}
	for (int i = 0; i < number_of_images; ++i)
	{
		labelMat.at<int>(i) = trainDataset.labels.at(i);
	}

	//memcpy(imageMat.data, trainDataset.images.data(), trainDataset.images.size() * trainDataset.images[0].size() * sizeof(float));
	//memcpy(labelMat.data, trainDataset.labels.data(), trainDataset.labels.size()*sizeof(int));
	
	//set up svm parameters
	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::RBF);
	svm->setGamma(0.01);
	svm->setC(10.0);

	std::cout << "training..." << std::endl;
	svm->train(imageMat, cv::ml::ROW_SAMPLE, labelMat);
	std::cout << "training complete!" << std::endl;
}

MNISTClassifier::~MNISTClassifier() {
	// TODO Auto-generated stub, complete
}

int MNISTClassifier::classifyImage(const cv::Mat& sample) {
	// TODO Auto-generated stub, copmlete

	return -1;
}

int MNISTClassifier::reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void MNISTClassifier::loadDatasetFromFiles(
	const std::string& labelsFile, const std::string& imagesFile, MNISTDataset& data) {
	// TODO Auto-generated stub, complete

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	std::cout << "loading files..." << std::endl;

	//read images
	std::ifstream imageFile(imagesFile, std::ios::binary);
	if (imageFile.is_open())
	{
		imageFile.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		imageFile.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		imageFile.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		imageFile.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		data.images.resize(number_of_images, std::vector<float>(n_rows*n_cols));
		
		for (int i = 0; i < number_of_images; ++i)
		{
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					imageFile.read((char*)&temp, sizeof(temp));
					data.images[i][(n_rows*r) + c] = (float)temp / 255.0;
				}
			}
		}
	}

	magic_number = 0;
	number_of_images = 0;

	//read labels
	std::ifstream labelFile(labelsFile, std::ios::binary);
	if (labelFile.is_open())
	{
		labelFile.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		labelFile.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);

		data.labels.resize(number_of_images);

		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			labelFile.read((char*)&temp, sizeof(temp));
			data.labels[i] = (int)temp;
		}
	}

}

void MNISTClassifier::runTestDatasetAndPrintStats(
		const MNISTDataset& inputDataset) {
	// TODO Auto-generated stub, complete

	//convert from vector to Mat
	int number_of_images = inputDataset.images.size();
	int number_of_features = inputDataset.images[0].size();
	cv::Mat testImage = cv::Mat::ones(number_of_images, number_of_features, CV_32FC1);
	cv::Mat testLabel = cv::Mat::ones(number_of_images, 1, CV_32SC1);

	for (int i = 0; i < number_of_images; ++i)
	{
		for (int j = 0; j<number_of_features; ++j)
			testImage.at<float>(i, j) = inputDataset.images.at(i).at(j);
	}
	for (int i = 0; i < number_of_images; ++i)
	{
		testLabel.at<int>(i) = inputDataset.labels.at(i);
	}

	//count the result!
	std::cout << "running test cases..." << std::endl;
	int count = 0;
	for (int i = 0; i < number_of_images; ++i)
	{
		cv::Mat image = testImage.row(i);
		float res = svm->predict(image);
		count += std::abs(res - testLabel.at<int>(i, 0)) <= epsilon ? 1 : 0;
	}
	std::cout << "error rate is " << (number_of_images - count + 0.f) / number_of_images*100.f << "%" << std::endl;
}
