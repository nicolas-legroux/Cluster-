#ifndef SRC_METRICS_METRICS_HPP_
#define SRC_METRICS_METRICS_HPP_

#include <memory>
#include <iostream>
#include <Eigen/Dense>

namespace MetricType {
enum MetricType {
	DISTANCE, SIMILARITY
};
}

namespace MetricName {
enum MetricName {
	PEARSON_CORRELATION,
	PEARSON_ABSOLUTE_CORRELATION,
	PEARSON_DISTANCE,
	SPEARMAN_CORRELATION,
	SPEARMAN_ABSOLUTE_CORRELATION,
	SPEARMAN_DISTANCE,
	COSINE_SIMILARITY,
	COSINE_ABSOLUTE_SIMILARITY,
	COSINE_DISTANCE,
	EUCLIDEAN_DISTANCE,
	SQUARED_EUCLIDEAN_DISTANCE,
	MANHATTAN_DISTANCE
};
} //End of MetricName namespace

namespace SimilarityMetricName {
enum SimilarityMetricName {
	PEARSON_CORRELATION,
	PEARSON_ABSOLUTE_CORRELATION,
	SPEARMAN_CORRELATION,
	SPEARMAN_ABSOLUTE_CORRELATION,
	COSINE_SIMILARITY,
	COSINE_ABSOLUTE_SIMILARITY
};
} //End of SimilarityMetricName namespace

namespace DistanceMetricName {
enum DistanceMetricName {
	PEARSON_DISTANCE,
	SPEARMAN_DISTANCE,
	COSINE_DISTANCE,
	EUCLIDEAN_DISTANCE,
	SQUARED_EUCLIDEAN_DISTANCE,
	MANHATTAN_DISTANCE
};
} //End of DistanceMetricName namespace

/*
 *
 * Virtual Base class. All metrics must implement its methods
 *
 */

class Metric {
private:
	bool verbose = false;
public:
	void setVerbose(bool b) {
		verbose = b;
	}
	bool getVerbose() const {
		return verbose;
	}
	virtual double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const = 0;
	virtual Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const = 0;
	virtual Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const = 0;
	virtual bool isDistanceMetric() const = 0;
	virtual ~Metric() = default;
};

/*
 *
 * Metric implementations
 *
 */

class SquaredEuclideanDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
};

class EuclideanDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
	~EuclideanDistance();
};

class ManhattanDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
};

class CosineSimilarity: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class CosineAbsoluteSimilarity: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class CosineDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
};

class PearsonCorrelation: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class PearsonAbsoluteCorrelation: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class PearsonDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
};

class SpearmanCorrelation: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class SpearmanAbsoluteCorrelation: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return false;
	}
};

class SpearmanDistance: public Metric {
public:
	double compute(const Eigen::VectorXd &left,
			const Eigen::VectorXd &right) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X,
			const Eigen::MatrixXd &Y) const;
	Eigen::MatrixXd compute(const Eigen::MatrixXd &X) const;
	bool isDistanceMetric() const {
		return true;
	}
};

//utility functions

MetricType::MetricType getMetricType(MetricName::MetricName metricName);
std::shared_ptr<Metric> buildMetric(MetricName::MetricName metricName);

#endif /* SRC_METRICS_METRICS_HPP_ */
