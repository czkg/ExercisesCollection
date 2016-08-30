/*
 * main.cpp
 *
 *  Created on: Aug 30, 2016
 *      Author: roy_shilkrot
 */

#include "MNISTClassifier.h"

#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	MNISTDataset trainDataset, testDataset;
	MNISTClassifier::loadDatasetFromFiles("train-labels.idx1-ubyte", "train-images.idx3-ubyte", trainDataset);
	MNISTClassifier::loadDatasetFromFiles("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", testDataset);

	MNISTClassifier classifier(trainDataset);
	classifier.runTestDatasetAndPrintStats(testDataset);
	system("pause");
}
