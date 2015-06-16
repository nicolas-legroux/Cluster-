/*
 * kmeans_clusterer.hpp
 *
 *  Created on: Jun 15, 2015
 *      Author: nicolas
 */

#ifndef SRC_ALGORITHMS_KMEANS_CLUSTERER_HPP_
#define SRC_ALGORITHMS_KMEANS_CLUSTERER_HPP_

#include "clusterer.hpp"

class KMeans_Clusterer : public Clusterer {
private:
	const Eigen::MatrixXd &data;
	std::shared_ptr<KMeansParameters> parameters;
	std::vector<int> clusters;

	Eigen::MatrixXd medoids;
	std::vector<bool> dataToCluster;

	double currentDistortion;

	//Utilities
	void initializeClustersRandomly();
	void recalculateMeans();
	double kMeansIteration();

public:
	KMeans_Clusterer(const Eigen::MatrixXd &_data,
			std::shared_ptr<ClustererParameters> _params);
	KMeans_Clusterer(const Eigen::MatrixXd &_data,
			std::shared_ptr<ClustererParameters> _params,
			std::vector<int> *initialClusters);

	std::vector<int> cluster();
	Eigen::MatrixXd getMedoids();
	double getDistortion();
};

#endif /* SRC_ALGORITHMS_KMEANS_CLUSTERER_HPP_ */
