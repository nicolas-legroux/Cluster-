#include <memory>
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include "tests.hpp"
#include <ctime>
#include <Eigen/Dense>
#include <ClusterXX/metrics/metrics.hpp>

unsigned int N = 1000;
unsigned int DIM = 20000;

std::vector<double> X(N * DIM);
std::vector<double> D(N * N);

int main() {
	Test::small_spectral_test();
}
