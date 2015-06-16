/*
 * spectral_clusterer.hpp
 *
 *  Created on: Jun 15, 2015
 *      Author: nicolas
 */

#ifndef SRC_ALGORITHMS_SPECTRAL_CLUSTERER_HPP_
#define SRC_ALGORITHMS_SPECTRAL_CLUSTERER_HPP_

#include "clusterer.hpp"
#include <Eigen/Dense>

class Spectral_Clusterer {
private:
	const Eigen::MatrixXd &originalData;
	std::shared_ptr<SpectralParameters> parameters;
	Eigen::MatrixXd distanceMatrix;
	Eigen::MatrixXd similarityMatrix;
	Eigen::MatrixXd laplacianMatrix;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver;
	//Utility Functions
	void computeSimilarityMatrix();
	void computeLaplacianMatrix();
public:
	Spectral_Clusterer(const Eigen::MatrixXd &_data,
			const std::shared_ptr<ClustererParameters> &_params);
	std::vector<int> cluster();

	const Eigen::MatrixXd &getDistanceMatrix() {
		return distanceMatrix;
	}

	const Eigen::MatrixXd &getSimilarityMatrix(){
		return similarityMatrix;
	}

	const Eigen::MatrixXd &getLaplacianMatrix(){
		return laplacianMatrix;
	}

	const Eigen::MatrixXd &getEigenvectors(){
		return eigenSolver.eigenvectors();
	}

	const Eigen::VectorXd &getEigenvalues(){
		return eigenSolver.eigenvalues();
	}
};

#endif /* SRC_ALGORITHMS_SPECTRAL_CLUSTERER_HPP_ */
