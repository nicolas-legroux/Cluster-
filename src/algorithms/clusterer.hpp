/*
 * template.hpp
 *
 *  Created on: Jun 11, 2015
 *      Author: nicolas
 */

#ifndef SRC_ALGORITHMS_CLUSTERER_HPP_
#define SRC_ALGORITHMS_CLUSTERER_HPP_

#include <vector>
#include <Eigen/Dense>

#include "clusterer_parameters.hpp"

class Clusterer{
public:
	virtual std::vector<int> cluster() = 0;
	virtual ~Clusterer() = 0;
};

class KMeans_Clusterer{

};

#endif /* SRC_ALGORITHMS_CLUSTERER_HPP_ */
