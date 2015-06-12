#ifndef SRC_ALGORITHMS_CLUSTERER_PARAMETERS_HPP_
#define SRC_ALGORITHMS_CLUSTERER_PARAMETERS_HPP_

#include "../metrics/metrics.hpp"
#include <memory>

class ClustererParameters {
private:
	bool verbose;
public:
	ClustererParameters(bool b) :
			verbose(b) {
	}
	virtual ~ClustererParameters() = 0;
	bool getVerbose() {
		return verbose;
	}
};

class KMeansParameters: public ClustererParameters {
private:
	unsigned int K;
	unsigned int maxIterations;
public:
	KMeansParameters(unsigned int _K, unsigned int _maxIter, bool verbose =
			false) :
			ClustererParameters(verbose), K(_K), maxIterations(_maxIter) {
	}

	unsigned int getK() {
		return K;
	}

	unsigned int getMaxIterations() {
		return maxIterations;
	}
};

class HierarchicalParameters: public ClustererParameters {
	enum LinkageMethod {
		AVERAGE, SINGLE, COMPLETE
	};
private:
	unsigned int K;
	std::shared_ptr<Metric> metricPtr;
	LinkageMethod linkageMethod;
public:
	HierarchicalParameters(unsigned int _K,
			const std::shared_ptr<Metric> &_metricPtr,
			const LinkageMethod &_linkageMethod = LinkageMethod::COMPLETE,
			bool verbose = false) :
			ClustererParameters(verbose), K(_K), metricPtr(_metricPtr), linkageMethod(
					_linkageMethod) {

	}
	//Set K=2
	HierarchicalParameters(const std::shared_ptr<Metric> &_metricPtr,
			const LinkageMethod &_linkageMethod = LinkageMethod::COMPLETE,
			bool verbose = false) :
			ClustererParameters(verbose), K(2), metricPtr(_metricPtr), linkageMethod(
					_linkageMethod) {

	}

	unsigned int getK() {
		return K;
	}

	std::shared_ptr<Metric> getMetricPtr() {
		return metricPtr;
	}

	LinkageMethod getLinkageMethod() {
		return linkageMethod;
	}
};

class SpectralParameters: public ClustererParameters {

	enum GraphTransformationMethodName {
		NO_TRANSFORMATION, K_NEAREST_NEIGHBORS
	};

	class GraphTransformationMethod {
	private:
		unsigned int kNearestNeighbors;
		double guaussianModelStdDev;
		GraphTransformationMethodName method;
	public:
		GraphTransformationMethod() :
				kNearestNeighbors(0), guaussianModelStdDev(0.0), method(
						GraphTransformationMethodName::NO_TRANSFORMATION) {

		}

		GraphTransformationMethod(GraphTransformationMethodName _method) :
				kNearestNeighbors(0), guaussianModelStdDev(0.0), method(
						GraphTransformationMethodName::NO_TRANSFORMATION) {
			assert(_method == GraphTransformationMethodName::NO_TRANSFORMATION);
		}

		GraphTransformationMethod(GraphTransformationMethodName _method,
				double param) :
				kNearestNeighbors(0), guaussianModelStdDev(0.0), method(_method) {
			assert(method != GraphTransformationMethodName::NO_TRANSFORMATION);
			if (method == GraphTransformationMethodName::K_NEAREST_NEIGHBORS) {
				kNearestNeighbors = param;
			}
			//TODO Implement Gaussian Model
			else {
				assert(false);
			}
		}

		GraphTransformationMethodName getMethodName() const {
			return method;
		}

		unsigned int getKNearestNeighbors() const {
			return kNearestNeighbors;
		}

		double getGaussianModelStdDev() const {
			return guaussianModelStdDev;
		}
	};
private:
	unsigned int K;
	std::shared_ptr<Metric> metricPtr;
	GraphTransformationMethod transformationMethod;
public:
	SpectralParameters(unsigned int _K,
			const std::shared_ptr<Metric> &_metricPtr,
			GraphTransformationMethod _transformationMethod =
					GraphTransformationMethod(), bool verbose = false) :
			ClustererParameters(verbose), K(_K), metricPtr(_metricPtr), transformationMethod(
					_transformationMethod) {
	}

	SpectralParameters(unsigned int _K,
			const std::shared_ptr<Metric> &_metricPtr,
			bool verbose) :
			SpectralParameters(_K, _metricPtr, GraphTransformationMethod(), verbose) {
	}

	unsigned int getK() {
		return K;
	}

	std::shared_ptr<Metric> getMetricPtr() {
		return metricPtr;
	}
};

#endif /* SRC_ALGORITHMS_CLUSTERER_PARAMETERS_HPP_ */
