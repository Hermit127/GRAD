import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_simple_table2 import latency_of


class LatencyTests(unittest.TestCase):
    def test_grad_not_duplicated_and_includes_build(self):
        self.assertEqual(latency_of({'grad_latency_ms': [10], 'decisions': [
            {'latency_ms': 10, 'build_latency_ms': 2}]}), [12])

    def test_sanitizer(self):
        self.assertEqual(latency_of({'defense_latency_ms': [12, 20]}), [12, 20])

    def test_legacy(self):
        self.assertEqual(latency_of({'grad_latency_ms': [10]}), [10])


if __name__ == '__main__':
    unittest.main()
