/*
 * k_means.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: nicolas
 */

#ifndef TESTS_K_MEANS_CPP_
#define TESTS_K_MEANS_CPP_

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>
#include "../src/algorithms/clusterer.hpp"
#include "../src/algorithms/kmeans_clusterer.hpp"
#include "../src/algorithms/hierarchical_clusterer.hpp"
#include "../src/algorithms/spectral_clusterer.hpp"
#include "../src/utils/utils.hpp"
#include "clustering.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void Test::small_kmeans_test() {
	unsigned int K = 2;
	int Nmax = 10;
	unsigned int dim = 2;
	unsigned int N = 8;
	MatrixXd dataToCluster(dim, N);
	dataToCluster << 0, 1, 0, 8, 8, -1, -1, 9, 0, 1, 1, 7, 8, 0, -1, 10;

	std::cout << "Clustering the following matrix : " << std::endl
			<< dataToCluster << std::endl;

	std::shared_ptr<ClustererParameters> params = std::make_shared<
			KMeansParameters>(K, Nmax, true);

	KMeans_Clusterer clusterer(dataToCluster, params);
	std::vector<int> clusters = clusterer.cluster();
	MatrixXd medoids = clusterer.getMedoids();
	std::cout << "The medoids are : " << std::endl << medoids;
	std::cout << std::endl << "The cluster asignments are : " << std::endl;
	Utilities::print_vector(clusters);
}

void Test::small_hierarchical_test() {
	unsigned int K = 2;
	int Nmax = 10;
	unsigned int dim = 2;
	unsigned int N = 8;
	MatrixXd dataToCluster(dim, N);
	dataToCluster << 0, 1, 0, 8, 8, -1, -1, 9, 0, 1, 1, 7, 8, 0, -1, 10;

	std::cout << "Clustering the following matrix : " << std::endl
			<< dataToCluster << std::endl;

	std::shared_ptr<ClustererParameters> params = std::make_shared<
			HierarchicalParameters>(K,
			std::make_shared<SquaredEuclideanDistance>(),
			HierarchicalParameters::COMPLETE, true);

	Hierarchical_Clusterer clusterer(dataToCluster, params);
	std::vector<int> clusters = clusterer.cluster();
	std::cout << "The distance matrix is : " << std::endl
			<< clusterer.getDistanceMatrix();
	std::cout << std::endl << std::endl << "The cluster asignments are : "
			<< std::endl;
	Utilities::print_vector(clusters);
}

void Test::small_spectral_test() {
	unsigned int K = 2;
	int Nmax = 10;
	unsigned int dim = 2;
	unsigned int N = 8;
	MatrixXd dataToCluster(dim, N);
	dataToCluster << 0.01, 1, 0, 8, 8, -1, -1, 9, 0.01, 1, 1, 7, 8, 0, -1, 10;

	std::cout << "Clustering the following matrix : " << std::endl
			<< dataToCluster << std::endl;

	std::shared_ptr<ClustererParameters> params =
			std::make_shared<SpectralParameters>(K,
					std::make_shared<SquaredEuclideanDistance>(),
					SpectralParameters::GraphTransformationMethod(
							SpectralParameters::GraphTransformationMethod::K_NEAREST_NEIGHBORS, 2),
					true);

	Spectral_Clusterer clusterer(dataToCluster, params);
	std::vector<int> clusters = clusterer.cluster();
	std::cout << std::endl << "The cluster asignments are : "
			<< std::endl;
	Utilities::print_vector(clusters);
}

#endif /* TESTS_K_MEANS_CPP_ */
