from concurrent.futures import ProcessPoolExecutor
import os
from typing import Tuple
import numpy as np
import scipy.stats
from rliable import metrics as rli_metrics
from rliable import library as rli_library


def _bootstrap_one(args):
    """Worker: run bootstrap CI for a single column of data."""
    data_col, agg_stat_name, reps = args
    from rliable import metrics as rli_metrics, library as rli_library
    aggregate_stat_fn = (
        rli_metrics.aggregate_iqm if agg_stat_name == "iqm" else rli_metrics.aggregate_mean
    )
    point_est, interval_est = rli_library.get_interval_estimates(
        {"data": data_col},
        func=lambda x: np.array([aggregate_stat_fn(x)]),
        reps=reps,
        confidence_interval_size=0.95,
    )
    return point_est["data"].squeeze(), interval_est["data"].squeeze()


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
        args_list = [(data[:, [i]], agg_stat_name, 25000) for i in range(num_tasks)]
        n_workers = min(num_tasks, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_bootstrap_one, args_list))
        point_ests = np.array([r[0] for r in results])
        interval_ests = np.array([r[1] for r in results])
        return point_ests, interval_ests
    else: # return the aggregate statistic for each task
        point_ests = []
        for task_idx in range(num_tasks):
            point_ests.append(aggregate_stat_fn(data[:, [task_idx]]))
        point_ests = np.array(point_ests) # shape (num_tasks,)
        return point_ests
