from typing import Tuple
import numpy as np
import scipy.stats
from rliable import metrics as rli_metrics
from rliable import library as rli_library


def get_aggregate_stat_fn(aggregate_stat: str):
    if aggregate_stat == "iqm":
        return rli_metrics.aggregate_iqm
    elif aggregate_stat == "mean":
        return rli_metrics.aggregate_mean
    else:
        raise ValueError(f"Invalid aggregate stat: {aggregate_stat}")

def compute_aggregate_stat_and_ci(data: np.ndarray, agg_stat_name: str, return_interval_est: bool):
    '''Computes the aggregate statistic and the bootstrapped CI over the provided data.
    Returns a single point estimate and interval estimate for the entire data. 
    
    Args:
        data: The input NumPy ndarray of shape (num_runs, num_tasks).
        agg_stat_name: The name of the aggregate statistic to compute ('iqm' or 'mean').
        return_interval_est: Whether to return the bootstrapped CI.
    '''
    assert data.ndim == 2, "Data must be 2D."

    aggregate_stat_fn = get_aggregate_stat_fn(agg_stat_name)
    if return_interval_est:
        data_dict = {"data": data}
        point_est, interval_est = rli_library.get_interval_estimates(
            data_dict,
            func=lambda x: np.array([aggregate_stat_fn(x)]),
            reps=25000,
            confidence_interval_size=0.95
        )
        return point_est["data"].squeeze(), interval_est["data"].squeeze()
    else:
        return aggregate_stat_fn(data)

def compute_aggregate_stat_and_ci_per_task(data: np.ndarray, agg_stat_name: str, return_interval_est: bool):
    '''Computes the aggregate statistic and the bootstrapped CI for each task separately.
    Args:
        data: The input NumPy ndarray of shape (num_runs, num_tasks).
        agg_stat_name: The name of the aggregate statistic to compute ('iqm' or 'mean').
        return_interval_est: Whether to return the bootstrapped CI.
    '''
    assert data.ndim == 2, "Data must be 2D."
    num_runs, num_tasks = data.shape
    aggregate_stat_fn = get_aggregate_stat_fn(agg_stat_name)
    
    if return_interval_est:
        point_ests = []
        interval_ests = []
        for task_idx in range(num_tasks):
            data_dict = {"data": data[:, [task_idx]]}
            point_est, interval_est = rli_library.get_interval_estimates(
                data_dict,
                func=lambda x: np.array([aggregate_stat_fn(x)]),
                reps=25000,
                confidence_interval_size=0.95
            )
            point_ests.append(point_est["data"].squeeze())
            interval_ests.append(interval_est["data"].squeeze())
        point_ests = np.array(point_ests) # shape (num_tasks,)
        interval_ests = np.array(interval_ests) # shape (num_tasks, 2)
        
        return point_ests, interval_ests
    else: # return the aggregate statistic for each task
        point_ests = []
        for task_idx in range(num_tasks):
            point_ests.append(aggregate_stat_fn(data[:, [task_idx]]))
        point_ests = np.array(point_ests) # shape (num_tasks,)
        return point_ests
