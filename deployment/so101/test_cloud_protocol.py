"""Check cloud action shape, identities and timestamps before motor integration."""

import copy
import unittest

import numpy as np
from action_protocol import select_action, validate_action


class CloudProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(
            policy="pi05_cloud_first81", checkpoint_step=16000, model_sha256="expected", horizon=50, fps=30
        )
        self.message = dict(
            policy=self.config["policy"],
            checkpoint_step=16000,
            model_sha256="expected",
            control_mode="joint",
            joint_units="degrees",
            gripper_units="0_to_100",
            left_arm={"actions": [[n] * 6 for n in range(50)]},
            right_arm={"actions": [[n + 100] * 6 for n in range(50)]},
            **{"timestamp/ms": 100000, "delta/ms": 1000 / 30},
        )

    def test_fifty_frame_chunk_and_time_alignment(self):
        actions, delta = validate_action(self.message, self.config)
        self.assertEqual(actions.shape, (50, 12))
        self.assertAlmostEqual(delta, 1 / 30)
        np.testing.assert_equal(select_action(self.message, self.config, 100.15, 99), [4] * 6 + [104] * 6)

    def test_reject_wrong_model_units_and_horizon(self):
        for key, value in [
            ("policy", "pi05_so101_fold"),
            ("checkpoint_step", 68660),
            ("model_sha256", "other"),
            ("joint_units", "radians"),
            ("gripper_units", "0_to_1"),
            ("delta/ms", 50),
        ]:
            message = copy.deepcopy(self.message)
            message[key] = value
            with self.assertRaises(ValueError):
                validate_action(message, self.config)
        for n in (30, 49, 51):
            message = copy.deepcopy(self.message)
            message["left_arm"]["actions"] = [[0] * 6] * n
            with self.assertRaises(ValueError):
                validate_action(message, self.config)

    def test_reject_stale_future_warmup_and_nonfinite(self):
        for now, earliest in [(100.41, 99), (99.9, 99), (100.1, 100.05), (float("nan"), 99)]:
            with self.assertRaises(ValueError):
                select_action(self.message, self.config, now, earliest)
        self.message["right_arm"]["actions"][40][5] = float("nan")
        with self.assertRaises(ValueError):
            validate_action(self.message, self.config)


if __name__ == "__main__":
    unittest.main()
