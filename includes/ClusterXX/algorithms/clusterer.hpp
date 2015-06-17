/*
 * template.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: nicolas
 */

#ifndef SRC_ALGORITHMS_CLUSTERER_HPP_
#define SRC_ALGORITHMS_CLUSTERER_HPP_

#include <vector>
#include <map>

namespace ClusterXX{

class Clusterer {
public:
	virtual void compute() = 0;
	std::vector<int> getClusters();
	double computeRandIndex(const std::vector<int> &otherClustering);
	double computeAdjustedRandIndex(const std::vector<int> &otherClustering);
	void printClustering(const std::map<int, std::string> &labelsMap,
			const std::vector<int> &realClusters);
	virtual ~Clusterer() = default;
protected:
	std::vector<int> clusters;
};

} //End of namespace ClusterXX

#endif /* SRC_ALGORITHMS_CLUSTERER_HPP_ */
