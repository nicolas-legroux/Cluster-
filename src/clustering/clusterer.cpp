#include <ClusterXX/clustering/clusterer.hpp>
#include <cassert>
#include <algorithm>
#include <iostream>

std::vector<int> ClusterXX::Clusterer::getClusters() {
	return clusters;
}

double ClusterXX::Clusterer::computeRandIndex(
		const std::vector<int> &otherClustering) {
	assert(clusters.size() == otherClustering.size());
	unsigned int n = clusters.size();
	int count = 0;
	for (unsigned int i = 0; i < n - 1; ++i) {
		for (unsigned int j = i + 1; j < n; ++j) {
			if ((clusters[i] == clusters[j])
					&& otherClustering[i] == otherClustering[j]) {
				count++;
			} else if ((clusters[i] != clusters[j])
					&& otherClustering[i] != otherClustering[j]) {
				count++;
			}
		}
	}

	auto numberOfPairs = [](int N) -> int {return N*(N-1)/2;};
	return (double) count / ((double) numberOfPairs(n));
}

double ClusterXX::Clusterer::computeAdjustedRandIndex(
		const std::vector<int> &otherClustering) {
	assert(clusters.size() == otherClustering.size());
	unsigned int n = clusters.size();

	int K1 = *max_element(clusters.cbegin(), clusters.cend()) + 1;
	int K2 = *max_element(otherClustering.cbegin(), otherClustering.cend()) + 1;

	std::vector<int> contingencyTable(K1 * K2, 0);
	std::vector<int> count1(K1, 0);
	std::vector<int> count2(K2, 0);

	for (unsigned int i = 0; i != n; ++i) {
		int c1 = clusters[i];
		int c2 = otherClustering[i];
		++contingencyTable[c1 * K2 + c2];
		++count1[c1];
		++count2[c2];
	}

	auto numberOfPairs = [](int n) -> int {return n*(n-1)/2;};

	std::transform(contingencyTable.begin(), contingencyTable.end(),
			contingencyTable.begin(), numberOfPairs);
	std::transform(count1.begin(), count1.end(), count1.begin(), numberOfPairs);
	transform(count2.begin(), count2.end(), count2.begin(), numberOfPairs);

	double index = (double) accumulate(contingencyTable.cbegin(),
			contingencyTable.cend(), 0.0);
	double temp1 = (double) accumulate(count1.cbegin(), count1.cend(), 0.0);
	double temp2 = (double) accumulate(count2.cbegin(), count2.cend(), 0.0);
	double maxIndex = 0.5 * (temp1 + temp2);
	double expectedIndex = temp1 * temp2 / (double) numberOfPairs(n);

	return (index - expectedIndex) / (maxIndex - expectedIndex);
}

void ClusterXX::Clusterer::printClustering(
		const std::map<int, std::string> &labelsMap,
		const std::vector<int> &realClusters) {
	std::cout << std::endl << "****** Clustering results : ******"
			<< std::endl << std::endl;

	unsigned int realClustersN = labelsMap.size();
	unsigned int computedClustersN = *std::max_element(clusters.cbegin(),
			clusters.cend()) + 1;
	std::vector<int> clusteringGraph(realClustersN * computedClustersN, 0);

	assert(realClusters.size() == clusters.size());

	for (unsigned int i = 0; i < realClusters.size(); ++i) {
		int realCluster = realClusters[i];
		int computedCluster = clusters[i];
		clusteringGraph[realCluster * computedClustersN + computedCluster]++;
	}

	std::cout << "-----------\t";
	for (unsigned int j = 0; j < computedClustersN; ++j) {
		std::cout << "#" << j << "\t";
	}
	std::cout << "SUM" << std::endl;

	for (unsigned int i = 0; i < realClustersN; ++i) {
		std::cout << labelsMap.at(i) << "\t";
		unsigned int sum = 0;
		for (unsigned int j = 0; j < computedClustersN; ++j) {
			sum += clusteringGraph[i * computedClustersN + j];
			std::cout << clusteringGraph[i * computedClustersN + j] << "\t";
		}
		std::cout << sum << std::endl;
	}

	unsigned int global_sum = 0;
	std::cout << "SUM" << "\t\t";
	for (unsigned int j = 0; j < computedClustersN; ++j) {
		unsigned int sum = 0;

		for (unsigned int i = 0; i < realClustersN; ++i) {
			sum += clusteringGraph[i * computedClustersN + j];
		}
		global_sum += sum;
		std::cout << sum << "\t";
	}

	std::cout << global_sum << std::endl;
}
