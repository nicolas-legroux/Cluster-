/*
 * tests.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: nicolas
 */

#include <iostream>
#include <Eigen/Dense>
#include <cstdio>
#include <ctime>
#include <ClusterXX/metrics/metrics.hpp>

#include "metrics_test.hpp"

using std::cout;
using std::endl;
using Eigen::VectorXd;
using Eigen::MatrixXd;

void Test::testMetrics1() {

	std::clock_t start;
	double duration;

	Eigen::MatrixXd X = MatrixXd::Random(1, 2000);
	Eigen::MatrixXd Y = MatrixXd::Random(1, 2);

	//cout << "left: " << endl << left << endl << endl;
	//cout << "right: " << endl << right << endl << endl;

	start = std::clock();

	ClusterXX::SquaredEuclideanDistance ed;
	MatrixXd dist = ed.compute(X);

	//cout << dist << endl << endl;

	duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;

	std::cout << "Duration: " << duration << '\n';

}

