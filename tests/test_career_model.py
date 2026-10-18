import os
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from apps.routers.CareerModelRouter import InterestAnswers, NextScaleRequest, next_scale, score
from ml.career.features import scale_features
from ml.career.inference import CareerFamilyPredictor, mean_bucket
from ml.career.taxonomy import FAMILY_OF_PROFILE, map_major

TOKEN = {"AGENTIC_INTERNAL_TOKEN": "test-token"}
INVESTIGATIVE = {"R": [2, 2], "I": [5, 5, 5], "A": [2, 1], "S": [2, 2], "E": [1, 2], "C": [3, 3]}


class TaxonomyTests(unittest.TestCase):
    def test_maps_free_text_majors_including_misspellings(self):
        self.assertEqual(map_major("Computer Science"), "it_software")
        self.assertEqual(map_major("psycology"), "psychology_counseling")
        self.assertEqual(map_major("Business Administration"), "business_management")
        self.assertEqual(map_major("public administration"), "law_public")
        self.assertIsNone(map_major("undecided"))
        self.assertIsNone(map_major(None))

    def test_every_app_profile_has_a_family(self):
        self.assertEqual(len(FAMILY_OF_PROFILE), 40)


class FeatureTests(unittest.TestCase):
    def test_missing_scale_is_imputed_with_person_average(self):
        features = scale_features(np.array([[1.0, np.nan, 0.0, 0.5, 0.5, 0.5]]))
        self.assertAlmostEqual(features[0, 1], 0.5)
        self.assertFalse(np.isnan(features).any())

    def test_running_mean_bucket(self):
        self.assertEqual(mean_bucket([]), 9)
        self.assertEqual(mean_bucket([1]), 0)
        self.assertEqual(mean_bucket([5, 5]), 8)
        self.assertEqual(mean_bucket([3, 4]), 5)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CareerFamilyPredictor.load(
            os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts"))

    def test_probabilities_are_a_distribution(self):
        p = self.predictor.proba(self.predictor.means(INVESTIGATIVE))[0]
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertEqual(len(p), len(self.predictor.classes))

    def test_profiles_differ_in_the_expected_direction(self):
        """Holland theory sanity check: strong Investigative vs strong Social/Enterprising."""
        classes = self.predictor.classes
        sci = self.predictor.proba(self.predictor.means(INVESTIGATIVE))[0]
        social = self.predictor.proba(self.predictor.means(
            {"R": [1, 1], "I": [2, 2], "A": [3, 3], "S": [5, 5, 5], "E": [4, 4], "C": [2, 2]}))[0]
        science = [classes.index(f) for f in ("physical_sciences_math", "life_sciences", "it_software")]
        self.assertGreater(sci[science].sum(), social[science].sum())
        self.assertGreater(social[classes.index("psychology_counseling")], sci[classes.index("psychology_counseling")])

    def test_next_scale_ranks_only_allowed_scales(self):
        ranking = self.predictor.next_scale(INVESTIGATIVE, ["R", "S"])
        self.assertEqual({item["scale"] for item in ranking}, {"R", "S"})
        self.assertGreaterEqual(ranking[0]["expected_information_gain"], ranking[1]["expected_information_gain"])


class RouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_missing_or_wrong_internal_token(self):
        with patch.dict(os.environ, TOKEN):
            with self.assertRaises(HTTPException) as error:
                await score(InterestAnswers(answers=INVESTIGATIVE), x_internal_token="wrong")
        self.assertEqual(error.exception.status_code, 401)

    def test_rejects_out_of_range_answers(self):
        with self.assertRaises(ValidationError):
            InterestAnswers(answers={"I": [6]})
        with self.assertRaises(ValidationError):
            InterestAnswers(answers={"X": [3]})

    async def test_score_returns_families_profiles_and_explanations(self):
        with patch.dict(os.environ, TOKEN):
            result = await score(InterestAnswers(answers=INVESTIGATIVE), x_internal_token="test-token")
        self.assertEqual(len(result["profiles"]), 40)
        self.assertEqual(max(result["profiles"].values()), 1.0)
        self.assertEqual(len(result["explanations"]), 3)
        self.assertIn("I", result["explanations"][0]["scale_contributions"])

    async def test_next_scale_endpoint(self):
        with patch.dict(os.environ, TOKEN):
            result = await next_scale(NextScaleRequest(answers=INVESTIGATIVE, allowed=["A", "E"]), x_internal_token="test-token")
        self.assertIn(result["ranking"][0]["scale"], {"A", "E"})


if __name__ == "__main__":
    unittest.main()
