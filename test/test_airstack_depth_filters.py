import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mononav_airstack import (
    filter_isolated_near_depth,
    filter_temporal_near_depth,
    reproject_depth_to_camera,
)


class AirStackDepthFilterTest(unittest.TestCase):
    def test_spatial_filter_replaces_only_isolated_near_pixel(self):
        depth = np.full((11, 11), 2000, dtype=np.uint16)
        depth[3:8, 3:8] = 1000
        depth[1, 1] = 800

        filtered, rejected = filter_isolated_near_depth(depth)

        self.assertEqual(filtered[1, 1], 2000)
        self.assertEqual(filtered[5, 5], 1000)
        self.assertEqual(rejected, 1)

    def test_identity_reprojection_preserves_depth(self):
        depth = np.full((5, 7), 2500, dtype=np.uint16)
        intrinsic = np.array(
            [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        )
        pose = np.eye(4)

        projected = reproject_depth_to_camera(depth, pose, pose, intrinsic)

        np.testing.assert_allclose(projected, depth, atol=1.0e-3)

    def test_temporal_filter_rejects_small_near_island_but_keeps_large_obstacle(self):
        reference = np.full((40, 40), 4000, dtype=np.float32)
        current = np.full((40, 40), 4000, dtype=np.uint16)
        current[2:4, 2:4] = 2000
        current[10:30, 10:30] = 2000

        filtered, rejected = filter_temporal_near_depth(
            current, reference, max_component_pixels=50
        )

        self.assertTrue(np.all(filtered[2:4, 2:4] == 4000))
        self.assertTrue(np.all(filtered[10:30, 10:30] == 2000))
        self.assertEqual(rejected, 4)


if __name__ == "__main__":
    unittest.main()
