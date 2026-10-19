"""Offline tests for the frozen Table 2 retry/dedup policy (zero API)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code/repro/final_table2'))

from retry_policy import (classify_error, resolve_cell, resolve_cells)  # noqa: E402


def rec(error=None, origin='a.json', **kw):
    base = {'suite': 'banking', 'user_task': 'user_task_0',
            'condition': 'attack', 'arm': 'grad',
            'injection_task': 'injection_task_0', 'utility': True,
            'security': False, 'error': error, 'origin': origin,
            'runner_contract': 'v1', 'benchmark_version': 'v1.2',
            'model': 'deepseek-chat', 'agentdojo_version': '0.1.35'}
    base.update(kw)
    return base


class ClassificationTests(unittest.TestCase):
    def test_ok_and_retryable_and_terminal(self):
        self.assertEqual(classify_error(None), 'ok')
        self.assertEqual(classify_error(''), 'ok')
        self.assertEqual(classify_error('RateLimitError: 429'), 'retryable')
        self.assertEqual(classify_error('APIConnectionError: boom'), 'retryable')
        self.assertEqual(classify_error('JSONDecodeError: bad json'), 'terminal')
        self.assertEqual(classify_error('KeyError: bug'), 'terminal')
        self.assertEqual(classify_error('AssertionError: logic'), 'terminal')


class ResolutionTests(unittest.TestCase):
    def test_last_success_wins_not_first(self):
        res = resolve_cell([rec(origin='old.json'),
                            rec(origin='new.json')])
        self.assertEqual(res['status'], 'ok')
        self.assertEqual(res['chosen']['origin'], 'new.json')

    def test_retryable_then_success_is_recovered(self):
        res = resolve_cell([rec(error='RateLimitError: 429'),
                            rec(origin='retry.json')])
        self.assertEqual(res['status'], 'ok')
        self.assertTrue(res['recovered'])
        self.assertEqual(res['n_attempts'], 2)
        self.assertEqual(res['n_retryable_errors'], 1)

    def test_only_retryable_is_final_retry_failure(self):
        res = resolve_cell([rec(error='APITimeoutError: t')])
        self.assertEqual(res['status'], 'retry_failure')
        self.assertIsNone(res['chosen'])

    def test_terminal_failure_is_not_replaced(self):
        res = resolve_cell([rec(error='KeyError: bug'),
                            rec(origin='retry.json')])
        self.assertEqual(res['status'], 'terminal_failure')
        self.assertEqual(res['n_terminal_errors'], 1)
        self.assertFalse(res['recovered'])
        self.assertIsNone(res['chosen'])

    def test_only_terminal_is_terminal_failure(self):
        res = resolve_cell([rec(error='KeyError: bug')])
        self.assertEqual(res['status'], 'terminal_failure')

    def test_protocol_mismatch_flagged(self):
        res = resolve_cell([rec(), rec(runner_contract='v2')])
        self.assertTrue(res['protocol_mismatch'])
        self.assertIsNone(res['chosen'])

    def test_same_protocol_no_mismatch(self):
        res = resolve_cell([rec(), rec()])
        self.assertFalse(res['protocol_mismatch'])

    def test_all_attempts_kept_and_marked(self):
        resolved, log = resolve_cells([
            rec(error='RateLimitError: 429', origin='a.json'),
            rec(origin='b.json'),
            rec(error='KeyError: bug', origin='c.json')])
        self.assertEqual(len(log), 3)
        self.assertEqual(sum(1 for e in log if e['chosen']), 0)
        self.assertEqual({e['classification'] for e in log},
                         {'retryable', 'ok', 'terminal'})
        self.assertEqual(len(resolved), 1)

    def test_separate_cells_not_merged(self):
        resolved, _ = resolve_cells([
            rec(injection_task='injection_task_0'),
            rec(injection_task='injection_task_1')])
        self.assertEqual(len(resolved), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
