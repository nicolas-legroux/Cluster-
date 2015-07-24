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

	Eigen::MatrixXd X = MatrixXd::Random(20000, 1000);
	Eigen::MatrixXd Y = MatrixXd::Random(5, 5);

	//cout << "left: " << endl << left << endl << endl;
	//cout << "right: " << endl << right << endl << endl;

	auto start = std::time(NULL);

	ClusterXX::ManhattanDistance ed;

	MatrixXd dist = ed.computeMatrix(X);

	//cout << X << std::endl << dist << endl << endl;

	auto end = std::time(NULL);
	auto duration = end - start;

	std::cout << start << '\n' << end << '\n' << duration << std::flush;

}

