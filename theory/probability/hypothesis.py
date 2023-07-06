import numpy as np
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class EstimationMetricsCont:
    """
    Estimation metrics for continuous data.
    n: sample size
    xbar: sample mean
    s: sample standard deviation
    """
    n: int
    xbar: float
    s: float

    def __repr__(self):
        return f"EstimationMetricsCont(n={self.n}, xbar={self.xbar:.3f}, s={self.s:.3f})"


@dataclass
class EstimationMetricsProp:
    """
    Estimation metrics for proportional data.
    n: sample size
    x: number of successes
    p: sample proportion
    """
    n: int
    x: int
    p: float

    def __repr__(self):
        return f"EstimationMetricsProp(n={self.n}, x={self.x}, p={self.p:.3f})"


class HypothesisTesting:
    def __init__(self, data_control, data_variation):
        self.data_control = data_control
        self.data_variation = data_variation

    def test_continuous(self, alpha=0.05):
        control_metrics = self.compute_continuous_metrics(self.data_control)
        variation_metrics = self.compute_continuous_metrics(self.data_variation)

        dof = self.degrees_of_freedom(control_metrics, variation_metrics)
        t_statistic = self.t_statistic_diff_means(control_metrics, variation_metrics)
        reject = self.reject_nh_t_statistic(t_statistic, dof, alpha)
        return reject
    
    def test_proportional(self, alpha=0.05):
        control_metrics = self.compute_proportional_metrics(self.data_control)
        variation_metrics = self.compute_proportional_metrics(self.data_variation)

        z_statistic = self.z_statistic_prop_diff(control_metrics, variation_metrics)
        p_value = self.p_value_z_statistic(z_statistic)
        reject = self.reject_nh_z_statistic(p_value, alpha)
        return reject


    @staticmethod
    def compute_continuous_metrics(data):
        metrics = EstimationMetricsCont(
            n=len(data),
            xbar=np.mean(data),
            s=np.std(data, ddof=1)
        )
        return metrics

    @staticmethod
    def degrees_of_freedom(control_metrics, variation_metrics):
        n1, s1 = control_metrics.n, control_metrics.s
        n2, s2 = variation_metrics.n, variation_metrics.s

        s1n1 = np.square(s1) / n1
        s2n2 = np.square(s2) / n2
        denom = (np.square(s1n1) / (n1 - 1)) + (np.square(s2n2) / (n2 - 1))
        dof = np.square(s1n1 + s2n2) / denom

        return dof

    @staticmethod
    def t_statistic_diff_means(control_metrics, variation_metrics):
        n1, xbar1, s1 = control_metrics.n, control_metrics.xbar, control_metrics.s
        n2, xbar2, s2 = variation_metrics.n, variation_metrics.xbar, variation_metrics.s

        s1n1 = np.square(s1) / n1
        s2n2 = np.square(s2) / n2
        t = (xbar1 - xbar2) / np.sqrt(s1n1 + s2n2)

        return t

    @staticmethod
    def reject_nh_t_statistic(t_statistic, dof, alpha=0.05):
        p_value = (1 - stats.t.cdf(abs(t_statistic), dof)) * 2
        reject = p_value < alpha
        return reject

    @staticmethod
    def compute_proportional_metrics(data):
        metrics = EstimationMetricsProp(
            n=len(data),
            x=sum(data),
            p=sum(data) / len(data)
        )
        return metrics

    @staticmethod
    def z_statistic_prop_diff(control_metrics, variation_metrics):
        n1, x1, p1 = control_metrics.n, control_metrics.x, control_metrics.p
        n2, x2, p2 = variation_metrics.n, variation_metrics.x, variation_metrics.p

        p = (x1 + x2) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(p * (1 - p) * (1/n1 + 1/n2))

        return z

    @staticmethod
    def p_value_z_statistic(z_statistic):
        p_value = (1 - stats.norm.cdf(abs(z_statistic))) * 2
        return p_value

    @staticmethod
    def reject_nh_z_statistic(p_value, alpha=0.05):
        reject = p_value < alpha
        return reject