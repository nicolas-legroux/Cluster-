/*
 * kmeans_clusterer.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: nicolas
 */

#include <ClusterXX/clustering/clusterer_parameters.hpp>
#include <ClusterXX/clustering/kmeans_clusterer.hpp>
#include <memory>
#include <stdexcept>
#include <vector>
#include <limits>
#include <ClusterXX/utils/utils.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

ClusterXX::KMeans_Clusterer::KMeans_Clusterer(const Eigen::MatrixXd &_data,
		std::shared_ptr<ClustererParameters> _params) :
		data(_data), dataToCluster(_data.cols(), true), currentDistortion(
				std::numeric_limits<double>::max()) {
	clusters.resize(_data.cols());
	parameters = std::dynamic_pointer_cast<KMeansParameters>(_params);
	if (!parameters) {
		throw std::invalid_argument(
				"Could not convert ClustererParameters pointer into KMeansParameters pointer.");
	}

	//initialize data structures
	medoids.resize(data.rows(), parameters->getK());
}

ClusterXX::KMeans_Clusterer::KMeans_Clusterer(const Eigen::MatrixXd &_data,
		std::shared_ptr<ClustererParameters> _params,
		std::vector<int> *initialClusters) :
		data(_data), dataToCluster(_data.cols()), currentDistortion(
				std::numeric_limits<double>::max()) {
	clusters = *initialClusters;
	parameters = std::dynamic_pointer_cast<KMeansParameters>(_params);
	if (!parameters) {
		throw std::invalid_argument(
				"Could not convert ClustererParameters pointer into KMeansParameters pointer.");
	}

	if (!data.cols() == clusters.size()) {
		throw std::invalid_argument("Initial clusters size is invalid.");
	}

	//initialize data structures
	medoids.resize(data.rows(), parameters->getK());

	transform(clusters.cbegin(), clusters.cend(), dataToCluster.begin(),
			[](int cluster) {
				return (cluster >= 0);
			});
}

//TODO : implement better algorithm
void ClusterXX::KMeans_Clusterer::initializeClustersRandomly() {
	std::default_random_engine engine(std::random_device { }());
	std::uniform_int_distribution<int> distribution(0, data.cols() - 1);
	std::vector<VectorXd> initialMedoids(parameters->getK());
	for (unsigned int i = 0; i < parameters->getK(); ++i) {
		while (true) {
			int randomInt = distribution(engine);
			if (dataToCluster[randomInt]) {
				VectorXd randomPoint = data.col(randomInt);
				bool different = true;
				//TODO Change this and make it cleaner
				double EPSILON = 0.0001;
				for (unsigned int j = 0; j < i; ++j) {
					if (SquaredEuclideanDistance().compute(randomPoint,
							initialMedoids[j]) < EPSILON) {
						different = false;
					}
				}
				if (different) {
					initialMedoids[i] = randomPoint;
					break;
				}
			}
		}
	}

	for (unsigned int i = 0; i < parameters->getK(); ++i) {
		medoids.col(i) = initialMedoids[i];
	}
}

void ClusterXX::KMeans_Clusterer::recalculateMeans() {
	std::vector<int> clusterSize(parameters->getK(), 0);
	//Reset medoids to 0
	medoids *= 0;

	for (unsigned int i = 0; i != clusters.size(); ++i) {
		int cluster = clusters[i];
		if (dataToCluster[i]) {
			++clusterSize[cluster];
			medoids.col(cluster) += data.col(i);
		}
	}

	for (unsigned int i = 0; i < parameters->getK(); ++i) {
		medoids.col(i) /= static_cast<double>(clusterSize[i]);
	}
}

double ClusterXX::KMeans_Clusterer::kMeansIteration() {

	MatrixXd distances = SquaredEuclideanDistance().computeMatrix(medoids, data);
	double newDistortion = 0;
	for (unsigned int i; i != data.cols(); ++i) {
		if (dataToCluster[i]) {
			newDistortion += std::sqrt(distances.col(i).minCoeff(&clusters[i]))
					/ data.cols();
		}
	}

	// Recalculate means
	recalculateMeans();
	return newDistortion;
}

void ClusterXX::KMeans_Clusterer::compute() {

	initializeClustersRandomly();

	currentDistortion = kMeansIteration();
	unsigned int iterations = 1;
	bool continueToIterate = true;

	if (parameters->getVerbose()) {
		std::cout << "Current distortion : " << currentDistortion << "\r"
				<< std::flush;
	}

	while (iterations < parameters->getMaxIterations() && continueToIterate) {
		++iterations;
		double newDistortion = kMeansIteration();

		continueToIterate = (std::fabs(currentDistortion - newDistortion)
				> Utilities::EPSILON);
		currentDistortion = newDistortion;

		if (parameters->getVerbose()) {
			std::cout << "Current distortion : " << currentDistortion << "\r"
					<< std::flush;
		}
	}

	if (iterations == parameters->getMaxIterations()) {
		std::cout << "K-Means did not converge." << std::endl;
	}
}

MatrixXd ClusterXX::KMeans_Clusterer::getMedoids() {
	return medoids;
}

double ClusterXX::KMeans_Clusterer::getDistortion() {
	return currentDistortion;
}
