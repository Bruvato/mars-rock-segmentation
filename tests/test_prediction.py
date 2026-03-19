from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deps import cv2, np
from prediction import grow_component_support, split_instances


class SplitInstancesTests(unittest.TestCase):
    def test_porous_single_rock_stays_single_instance(self) -> None:
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(mask, (120, 120), 60, 1, thickness=cv2.FILLED)
        for center in ((95, 108), (145, 110), (120, 145), (120, 88)):
            cv2.circle(mask, center, 10, 0, thickness=cv2.FILLED)

        labels = split_instances(mask > 0)

        self.assertEqual(int(labels.max()), 1)

    def test_touching_rocks_split_when_two_strong_peaks_exist(self) -> None:
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(mask, (90, 120), 40, 1, thickness=cv2.FILLED)
        cv2.circle(mask, (150, 120), 40, 1, thickness=cv2.FILLED)
        cv2.rectangle(mask, (110, 108), (130, 132), 1, thickness=cv2.FILLED)

        labels = split_instances(mask > 0)

        self.assertEqual(int(labels.max()), 2)


class GrowComponentSupportTests(unittest.TestCase):
    def test_growth_expands_connected_probability_support_only(self) -> None:
        seed = np.zeros((160, 160), dtype=bool)
        cv2.circle(seed.view(np.uint8), (60, 80), 10, 1, thickness=cv2.FILLED)

        probability = np.full((160, 160), 0.05, dtype=np.float32)
        score = np.full((160, 160), 0.05, dtype=np.float32)
        valid = np.ones((160, 160), dtype=bool)

        cv2.circle(probability, (60, 80), 30, 0.55, thickness=cv2.FILLED)
        cv2.circle(score, (60, 80), 30, 0.35, thickness=cv2.FILLED)
        cv2.circle(probability, (125, 80), 12, 0.60, thickness=cv2.FILLED)
        cv2.circle(score, (125, 80), 12, 0.40, thickness=cv2.FILLED)

        grown = grow_component_support(seed, probability, score, valid)

        self.assertGreater(int(grown.sum()), int(seed.sum()) * 3)
        self.assertTrue(grown[80, 80])
        self.assertFalse(grown[80, 125])


if __name__ == "__main__":
    unittest.main()
