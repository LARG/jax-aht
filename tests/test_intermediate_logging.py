import unittest

import numpy as np

from common.intermediate_logging import (
    compute_comedi_xp_return,
    compute_sp_xp_returns,
    extract_masked_episode_stats,
    log_brdiv_intermediate_metrics,
    log_ippo_intermediate_metrics,
)


class FakeLogger:
    def __init__(self):
        self.items = []
        self.commit_calls = 0

    def log_item(self, tag, val, train_step=None, commit=True, **kwargs):
        self.items.append((tag, val, train_step, commit, kwargs))

    def commit(self):
        self.commit_calls += 1


class IntermediateLoggingTests(unittest.TestCase):
    def test_extract_masked_episode_stats_uses_episode_end_mask(self):
        metric = {
            "returned_episode": np.array([[False, True], [True, False]]),
            "returned_episode_returns": np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32),
            "percent_eaten": np.array([[0.1, 0.5], [0.9, 0.0]], dtype=np.float32),
        }

        stats = extract_masked_episode_stats(metric, ("returned_episode_returns", "percent_eaten"))

        self.assertAlmostEqual(stats["returned_episode_returns"], 3.0)
        self.assertAlmostEqual(stats["percent_eaten"], 0.7)

    def test_compute_sp_xp_returns_splits_pair_returns(self):
        pair_returns = np.array(
            [
                [[1.0]],
                [[2.0]],
                [[3.0]],
                [[4.0]],
            ],
            dtype=np.float32,
        )

        sp_return, xp_return = compute_sp_xp_returns(pair_returns, pop_size=2)

        self.assertAlmostEqual(sp_return, 2.5)
        self.assertAlmostEqual(xp_return, 2.5)

    def test_compute_comedi_xp_return_uses_only_valid_population_slice(self):
        xp_returns = np.array(
            [
                [[1.0]],
                [[5.0]],
                [[100.0]],
            ],
            dtype=np.float32,
        )

        xp_return = compute_comedi_xp_return(xp_returns, valid_population_size=2)

        self.assertAlmostEqual(xp_return, 3.0)

    def test_log_ippo_intermediate_metrics_logs_contextual_tags(self):
        logger = FakeLogger()
        metric = {
            "returned_episode": np.array([[False, True], [True, False]]),
            "returned_episode_returns": np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32),
        }

        result = log_ippo_intermediate_metrics(
            logger,
            metric,
            ("returned_episode_returns",),
            update_step=7,
            seed_idx=1,
            partner_idx=3,
        )

        self.assertEqual(result, np.int32(0))
        self.assertEqual(logger.commit_calls, 1)
        self.assertEqual(
            logger.items[0][0],
            "Train/Intermediate/Seed_1/Partner_3/returned_episode_returns",
        )
        self.assertEqual(logger.items[0][2], 7)

    def test_log_brdiv_intermediate_metrics_logs_eval_and_loss_scalars(self):
        logger = FakeLogger()
        metric = {
            "update_steps": np.int32(4),
            "pg_loss_conf_agent": np.array([1.0, 3.0], dtype=np.float32),
            "pg_loss_br_agent": np.array([2.0, 4.0], dtype=np.float32),
            "value_loss_conf_agent": np.array([5.0], dtype=np.float32),
            "value_loss_br_agent": np.array([6.0], dtype=np.float32),
            "entropy_conf": np.array([7.0], dtype=np.float32),
            "entropy_br": np.array([8.0], dtype=np.float32),
            "eval_ep_last_info": {
                "returned_episode_returns": np.array(
                    [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
                    dtype=np.float32,
                )
            },
        }

        result = log_brdiv_intermediate_metrics(logger, metric, seed_idx=2)

        self.assertEqual(result, np.int32(0))
        self.assertEqual(logger.commit_calls, 2)
        logged_tags = [item[0] for item in logger.items]
        self.assertIn("Losses/Intermediate/Seed_2/ConfPGLoss", logged_tags)
        self.assertIn("Eval/Intermediate/Seed_2/AvgSPReturnCurve", logged_tags)


if __name__ == "__main__":
    unittest.main()
