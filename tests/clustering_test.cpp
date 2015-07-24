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
#include <ClusterXX/utils/utils.hpp>
#include "clustering_test.hpp"
#include <ClusterXX/clustering/algorithms.hpp>

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

	std::shared_ptr<ClusterXX::ClustererParameters> params = std::make_shared<
			ClusterXX::KMeansParameters>(K, Nmax, true);

	ClusterXX::KMeans_Clusterer clusterer(dataToCluster, params);
	clusterer.compute();
	std::vector<int> clusters = clusterer.getClusters();
	MatrixXd medoids = clusterer.getMedoids();
	std::cout << "The medoids are : " << std::endl << medoids;
	std::cout << std::endl << "The cluster asignments are : " << std::endl;
	ClusterXX::Utilities::print_vector(clusters);
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

	std::shared_ptr<ClusterXX::ClustererParameters> params = std::make_shared<
			ClusterXX::HierarchicalParameters>(K,
			std::make_shared<ClusterXX::SquaredEuclideanDistance>(),
			ClusterXX::HierarchicalParameters::COMPLETE, true);

	ClusterXX::Hierarchical_Clusterer clusterer(dataToCluster, params);
	clusterer.compute();
	std::vector<int> clusters = clusterer.getClusters();
	std::cout << "The distance matrix is : " << std::endl
			<< clusterer.getDistanceMatrix();
	std::cout << std::endl << std::endl << "The cluster asignments are : "
			<< std::endl;
	ClusterXX::Utilities::print_vector(clusters);
}

void Test::small_spectral_test() {
	unsigned int K = 2;
	unsigned int dim = 2;
	unsigned int N = 8;
	MatrixXd dataToCluster(dim, N);
	dataToCluster << 0.01, 1, 0, 8, 8, -1, -1, 9, 0.01, 1, 1, 7, 8, 0, -1, 10;

	std::cout << "Clustering the following matrix : " << std::endl
			<< dataToCluster << std::endl << std::endl;

	std::shared_ptr<ClusterXX::ClustererParameters> params =
			std::make_shared<ClusterXX::SpectralParameters>(K,
					std::make_shared<ClusterXX::SquaredEuclideanDistance>(),
					ClusterXX::SpectralParameters::GraphTransformationMethod(
							ClusterXX::SpectralParameters::GraphTransformationMethod::K_NEAREST_NEIGHBORS,
							2), true);

	std::cout << "*** Unnormalized version *** " << std::endl;
	ClusterXX::UnnormalizedSpectralClustering clusterer(dataToCluster, params);
	clusterer.compute();
	std::vector<int> clusters = clusterer.getClusters();
	std::cout << "The cluster asignments are : " << std::endl;
	ClusterXX::Utilities::print_vector(clusters);
	std::cout << std::endl;

	std::cout << "*** Normalized version (2000) *** " << std::endl;
	ClusterXX::NormalizedSpectralClustering_RandomWalk clusterer2(dataToCluster, params);
	clusterer2.compute();
	std::vector<int> clusters2 = clusterer2.getClusters();
	std::cout << "The cluster asignments are : " << std::endl;
	ClusterXX::Utilities::print_vector(clusters2);
	std::cout << std::endl;

	std::cout << "*** Normalized version (2002) *** " << std::endl;
	ClusterXX::NormalizedSpectralClustering_Symmetric clusterer3(dataToCluster, params);
	clusterer3.compute();
	std::vector<int> clusters3 = clusterer3.getClusters();
	std::cout << "The cluster asignments are : " << std::endl;
	ClusterXX::Utilities::print_vector(clusters3);
	std::cout << std::endl;
}

#endif /* TESTS_K_MEANS_CPP_ */
