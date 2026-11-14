"""Basic smoke tests verifying the CancerFold package scaffold."""

import unittest

import cancerfold
from cancerfold.pipeline import describe_pipeline


class SmokeTest(unittest.TestCase):
    def test_version_and_pipeline_description(self):
        self.assertEqual(cancerfold.__version__, "0.1.0")
        self.assertIn("CancerFold", describe_pipeline())


if __name__ == "__main__":
    unittest.main()
