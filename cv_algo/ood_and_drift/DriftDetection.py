from scipy.spatial.distance import euclidean, cosine
from scipy.stats import ks_2samp
from sklearn.metrics import pairwise_distances
from scipy.stats import wasserstein_distance
import numpy as np
from sklearn.model_selection import KFold
import ot


class DriftDetector:
    """
    A class to detect drift between reference and test data using various distance metrics.

    Sample usage 1:
        reference_data = np.random.rand(1000, 128)
        test_data = np.random.rand(1000, 128)
        detector = DriftDetector(reference_data)
        detector.set_thresholds_for_all_metrics()
        result = detector.is_drift(test_data, "euclidean_distance")
        print(result)


    Sample usage 2:
        reference_data = np.random.rand(1000, 128)
        test_data = np.random.rand(1000, 128)
        detector = DriftDetector(reference_data)
        detector.set_threshold("euclidean_distance") # need to run once reference data is changed
        result = detector.is_drift(test_data, "euclidean_distance")
        print(result)

    """

    def __init__(self, reference_data):
        """Initialize the DriftDetector with reference data."""
        self.reference_data = reference_data
        self.thresholds = {}

    # Euclidean Distance
    def euclidean_distance(self, test_data):
        """Compute the Euclidean distance between the mean of the reference and test data."""
        return euclidean(self.reference_data.mean(axis=0), test_data.mean(axis=0))

    # Cosine Distance
    def cosine_distance(self, test_data):
        """Compute the cosine distance between the mean of the reference and test data."""
        return cosine(self.reference_data.mean(axis=0), test_data.mean(axis=0))

    # Maximum Mean Discrepancy
    def maximum_mean_discrepancy(self, test_data):
        """Compute the Maximum Mean Discrepancy (MMD) between the reference and test data."""
        mmd = np.mean(pairwise_distances(self.reference_data, test_data))
        return mmd

    # Population Stability Index
    def population_stability_index(self, test_data, bins=10):
        """Compute the Population Stability Index (PSI) between the reference and test data."""
        ref_hist, _ = np.histogram(self.reference_data, bins=bins)
        test_hist, _ = np.histogram(test_data, bins=bins)
        psi = np.sum(
            (ref_hist / len(self.reference_data) - test_hist / len(test_data)) ** 2
            / (ref_hist / len(self.reference_data))
        )
        return psi

    # KS Test
    def ks_test(self, test_data):
        """Perform the Kolmogorov-Smirnov test between the reference and test data."""
        ks_statistic, _ = ks_2samp(self.reference_data.flatten(), test_data.flatten())
        return ks_statistic

    # Partial Wasserstein Distance
    def partial_wasserstein_distance(self, test_data, p=1):
        """Compute the Partial Wasserstein Distance between the reference and test data."""
        # Compute the histograms of the reference and test data
        ref_hist, ref_bins = np.histogram(self.reference_data.flatten(), bins=10)
        test_hist, test_bins = np.histogram(test_data.flatten(), bins=10)

        # Normalize the histograms
        ref_hist = ref_hist / ref_hist.sum()
        test_hist = test_hist / test_hist.sum()

        # Create a cost matrix
        n_bins = len(ref_bins) - 1
        M = ot.dist(
            ref_bins[:-1].reshape((n_bins, 1)), test_bins[:-1].reshape((n_bins, 1))
        )
        M /= M.max()

        # Compute the Wasserstein distance
        wdist = ot.emd2(ref_hist, test_hist, M, numItermax=1000000)

        return wdist

    # Benchmarking Different Metrics
    def benchmark(self, test_data):
        """Benchmark different metrics and return the results as a dictionary."""
        results = {
            "euclidean_distance": self.euclidean_distance(test_data),
            "cosine_distance": self.cosine_distance(test_data),
            "maximum_mean_discrepancy": self.maximum_mean_discrepancy(test_data),
            "population_stability_index": self.population_stability_index(test_data),
            "ks_test": self.ks_test(test_data),
            "partial_wasserstein_distance": self.partial_wasserstein_distance(
                test_data
            ),
        }
        return results

    # Setting Threshold Based on Cross-Validation
    def set_threshold(self, metric, folds=5):
        """Set the threshold for a specific metric using K-Fold cross-validation."""
        kf = KFold(n_splits=folds)
        thresholds = []

        for train_index, test_index in kf.split(self.reference_data):
            train_data, val_data = (
                self.reference_data[train_index],
                self.reference_data[test_index],
            )
            detector = DriftDetector(train_data)
            threshold = getattr(detector, metric)(val_data)
            thresholds.append(threshold)

        self.thresholds[metric] = np.mean(thresholds)
        return self.thresholds[metric]

    # Setting thresholds for all metrics based on cross-validation
    def set_thresholds_for_all_metrics(self, folds=5):
        """Set the thresholds for all metrics using cross-validation."""
        metrics = [
            "euclidean_distance",
            "cosine_distance",
            "maximum_mean_discrepancy",
            "population_stability_index",
            "ks_test",
            "partial_wasserstein_distance",
        ]
        for metric in metrics:
            self.set_threshold(metric, folds=folds)

    # Adaptive Threshold Updation
    def update_threshold(self, metric, test_data, alpha=0.1):
        """Update the threshold for a specific metric based on the new test data."""
        new_threshold = getattr(self, metric)(test_data)
        self.thresholds[metric] = (1 - alpha) * self.thresholds[
            metric
        ] + alpha * new_threshold

    # Bootstrapping for Uncertainty Estimation
    def bootstrap(self, metric, test_data, n_iterations=1000):
        """Bootstrap the specified metric for uncertainty estimation."""
        statistics = []

        for i in range(n_iterations):
            sampled_test_data = np.random.choice(
                test_data.flatten(), len(test_data.flatten())
            )
            stat = getattr(self, metric)(sampled_test_data)
            statistics.append(stat)

        return np.mean(statistics), np.std(statistics)

    # Method to check drift
    def is_drift(self, test_data, metric, threshold=None):
        """
        Check if drift has occurred based on the specified metric and threshold.

        Args:
            test_data (array-like): Test data to compare against the reference.
            metric (str): Metric to use for comparison. Must be one of the supported metrics.
            threshold (float, optional): Threshold for the metric. If None, the preset threshold is used.

        Returns:
            dict: A dictionary containing information about the drift detection.
        """
        # Check if the threshold is provided; if not, use the default threshold for the metric
        if threshold is None:
            threshold = self.thresholds.get(metric, None)
            if threshold is None:
                raise ValueError(
                    f"Threshold for {metric} not set. Please set it first or provide it as an argument."
                )

        # Calculate the distance metric for the test data
        distance_metrics = {
            "euclidean_distance": self.euclidean_distance,
            "cosine_distance": self.cosine_distance,
            "maximum_mean_discrepancy": self.maximum_mean_discrepancy,
            "population_stability_index": self.population_stability_index,
            "ks_test": self.ks_test,
            "partial_wasserstein_distance": self.partial_wasserstein_distance,
        }
        distance_metric_function = distance_metrics.get(metric, None)
        if distance_metric_function is None:
            raise ValueError(
                f"Unknown metric {metric}. Please use one of {list(distance_metrics.keys())}."
            )

        distance_value = distance_metric_function(test_data)

        # Determine if drift has occurred
        drift_occurred = distance_value > threshold

        # Return the result
        result = {
            "is_drift": drift_occurred,
            "distance_metric_value": distance_value,
            "threshold_value_used": threshold,
            "metric_used": metric,
        }
        return result
