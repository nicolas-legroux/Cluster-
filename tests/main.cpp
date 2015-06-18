#include <memory>
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include "tests.hpp"
#include <Eigen/Dense>
#include <ClusterXX/metrics/metrics.hpp>

int main() {
	Eigen::VectorXd v(5);
	v << 1, 3, -5, 1, 0;
	std::cout << v << std::endl << std::endl;
	std::sort(v.data(), v.data()+v.size());
	std::cout << v;

}
