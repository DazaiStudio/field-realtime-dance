"""Unit tests for the live morrisness score."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from culture_score import CultureScore

SYNTHETIC_MAP = {
    "features": ["energy", "sway"],
    "log_features": ["energy"],
    "mean": [1.0, 0.1],
    "std": [0.5, 0.05],
    "centroid_morris": [1.0, 1.0],
    "centroid_baye": [-1.0, -1.0],
}


def make_score(map_data=SYNTHETIC_MAP) -> CultureScore:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(map_data, f)
        path = Path(f.name)
    score = CultureScore(path)
    path.unlink()
    return score


def converge(score: CultureScore, metrics: dict, steps: int = 400) -> float:
    out = None
    for _ in range(steps):
        out = score.update(metrics)
    return out


class TestCultureScore(unittest.TestCase):
    def metrics_at(self, z_energy: float, z_sway: float) -> dict:
        # invert the transform: feature = mean + z*std; energy is log10(1+x)
        log_energy = SYNTHETIC_MAP["mean"][0] + z_energy * SYNTHETIC_MAP["std"][0]
        energy = 10 ** log_energy - 1.0
        sway = SYNTHETIC_MAP["mean"][1] + z_sway * SYNTHETIC_MAP["std"][1]
        return {"energy": energy, "sway": sway}

    def test_morris_centroid_scores_high(self):
        score = make_score()
        out = converge(score, self.metrics_at(1.0, 1.0))
        self.assertGreater(out, 0.95)

    def test_baye_centroid_scores_low(self):
        score = make_score()
        out = converge(score, self.metrics_at(-1.0, -1.0))
        self.assertLess(out, 0.05)

    def test_midpoint_scores_half(self):
        score = make_score()
        out = converge(score, self.metrics_at(0.0, 0.0))
        self.assertAlmostEqual(out, 0.5, places=2)

    def test_score_is_bounded(self):
        score = make_score()
        out = converge(score, self.metrics_at(10.0, -10.0))
        self.assertGreaterEqual(out, 0.0)
        self.assertLessEqual(out, 1.0)

    def test_reset_clears_history(self):
        score = make_score()
        converge(score, self.metrics_at(1.0, 1.0))
        score.reset()
        self.assertEqual(score._ema, {})

    def test_try_load_missing_file_returns_none(self):
        self.assertIsNone(CultureScore.try_load(Path("does_not_exist.json")))

    def test_shipped_map_loads_if_present(self):
        # morrisness ships disabled until the centroids are validated against
        # the live pipeline (lite model, live analysis rate) - not the 60fps
        # full-model offline data they currently come from.
        path = Path(__file__).resolve().parents[1] / "culture_map.json"
        if not path.exists():
            self.skipTest("culture_map.json not shipped (morrisness disabled)")
        score = CultureScore.try_load(path)
        self.assertIsNotNone(score)
        self.assertEqual(len(score.features), 9)
        self.assertEqual(len(score.centroid_morris), 9)


if __name__ == "__main__":
    unittest.main()
