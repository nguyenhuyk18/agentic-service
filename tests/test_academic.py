import json
import unittest
from pathlib import Path

import numpy as np

from ml.career.academic import COMBINATIONS, FAMILY_ACADEMICS, academic_fit, combo_totals, fuse, percentile
from ml.career.taxonomy import FAMILIES

QUANTILES = json.loads((Path(__file__).resolve().parent.parent / "ml" / "artifacts" / "combo_quantiles.json").read_text(encoding="utf-8"))
SCIENCE = {"toan": 9, "hoa_hoc": 9, "sinh_hoc": 9.25, "ngu_van": 5.5, "ngoai_ngu": 6}


class AcademicTests(unittest.TestCase):
    def test_every_family_has_known_combinations(self):
        self.assertEqual(set(FAMILY_ACADEMICS), set(FAMILIES))
        for info in FAMILY_ACADEMICS.values():
            self.assertTrue(set(info["combos"]) <= set(COMBINATIONS))

    def test_combination_totals_need_all_three_subjects(self):
        totals = combo_totals({"toan": 8, "ngu_van": 7, "ngoai_ngu": 6, "vat_li": 9})
        self.assertEqual(totals["D01"], 21.0)
        self.assertEqual(totals["A01"], 23.0)
        self.assertNotIn("A00", totals)

    def test_percentiles_follow_the_2024_distribution(self):
        self.assertEqual(percentile("A00", 0, QUANTILES), 0.0)
        self.assertEqual(percentile("A00", 30, QUANTILES), 1.0)
        self.assertAlmostEqual(percentile("D01", 19.65, QUANTILES), 0.5, delta=0.02)
        self.assertLess(percentile("A00", 20, QUANTILES), percentile("A00", 25, QUANTILES))

    def test_core_subjects_separate_families_with_the_same_combination(self):
        fits = academic_fit(SCIENCE, QUANTILES, "thi_thu")
        self.assertEqual(fits["health_medicine"]["combo"], "B00")
        self.assertEqual(fits["psychology_counseling"]["combo"], "B00")
        self.assertGreater(fits["health_medicine"]["fit"], fits["psychology_counseling"]["fit"])

    def test_fusion_reorders_tied_interest_and_halves_aptitude_weight(self):
        classes = list(FAMILIES)
        fits = academic_fit(SCIENCE, QUANTILES, "thi_thu")
        fit = np.array([fits[c]["fit"] for c in classes])
        interest = np.full(len(classes), 1 / len(classes))
        fused = fuse(interest, fit, 1.5, np.array([bool(FAMILY_ACADEMICS[c].get("aptitude")) for c in classes]))
        self.assertAlmostEqual(float(fused.sum()), 1.0, places=6)
        self.assertGreater(fused[classes.index("health_medicine")], fused[classes.index("law_public")])
        # γ = 0 leaves the interest distribution unchanged.
        np.testing.assert_allclose(fuse(interest, fit, 0.0), interest)


if __name__ == "__main__":
    unittest.main()
