"""Zero-API budget client tests (fake provider, no network)."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))
from budget_client import (BudgetStop, CallFailed, DEEPSEEK_FLASH, GLM_FLASH,
                           VendorBudgetClient)  # noqa: E402


def make_response(prompt=1000, completion=100, cached=0):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(
        usage=usage, model='fake-returned',
        choices=[SimpleNamespace(
            finish_reason='stop',
            message=SimpleNamespace(content='buy[X]', reasoning_content=''))])


class FakeOpenAI:
    def __init__(self, behavior):
        self.behavior = behavior
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create))
        self.kwargs_seen = []

    def create(self, **kwargs):
        self.kwargs_seen.append(kwargs)
        return self.behavior(kwargs)


class TestBudgetClient(unittest.TestCase):
    def _client(self, behavior, cap=1.0, spec=DEEPSEEK_FLASH, tmp=None):
        c = VendorBudgetClient(tmp, spec, cap)
        c.client = FakeOpenAI(behavior)
        return c

    def test_success_accounting_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(lambda kw: make_response(1000, 100, cached=400),
                             cap=1.0, tmp=tmp)
            r = c.chat.completions.create(
                messages=[{'role': 'user', 'content': 'hi'}], max_tokens=64)
            self.assertEqual(r.model, 'fake-returned')
            self.assertEqual(r.cached_tokens, 400)
            expected = (600 * 2.2 + 400 * 0.0429 + 100 * 8.6) / 1e6
            conservative = (1000 * 2.2 + 100 * 8.6) / 1e6
            self.assertAlmostEqual(r.cost_rmb, expected, places=10)
            self.assertGreater(c.state['charged_upper_rmb'],
                               c.state['charged_actual_rmb'])
            self.assertAlmostEqual(c.state['charged_upper_rmb'],
                                   conservative, places=10)
            self.assertEqual(c.state['calls'], 1)
            lines = (Path(tmp) / 'calls_deepseek-flash.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry['status'], 'ok')
            self.assertEqual(entry['model_returned'], 'fake-returned')
            self.assertIn('reserved_rmb', entry)

    def test_no_retries_and_failure_keeps_reservation(self):
        def boom(kw):
            raise RuntimeError('network down')
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(boom, cap=1.0, tmp=tmp)
            with self.assertRaises(CallFailed):
                c.chat.completions.create(
                    messages=[{'role': 'user', 'content': 'hi'}],
                    max_tokens=64)
            self.assertGreater(c.state['charged_upper_rmb'], 0.0)
            self.assertEqual(c.state['failed_calls'], 1)
            self.assertEqual(len(c.client.kwargs_seen), 1, 'no automatic retry')
            lines = (Path(tmp) / 'calls_deepseek-flash.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(lines[0])['status'], 'failed')

    def test_cap_checked_before_call(self):
        calls = {'n': 0}

        def behavior(kw):
            calls['n'] += 1
            return make_response(1000, 100)
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(behavior, cap=0.000001, tmp=tmp)
            with self.assertRaises(BudgetStop):
                c.chat.completions.create(
                    messages=[{'role': 'user', 'content': 'x' * 1000}],
                    max_tokens=1024)
            self.assertEqual(calls['n'], 0)

    def test_two_vendor_specs_prices(self):
        self.assertEqual(DEEPSEEK_FLASH.price_in, 2.2)
        self.assertEqual(DEEPSEEK_FLASH.price_out, 8.6)
        self.assertEqual(GLM_FLASH.price_in, 0.8)
        self.assertEqual(GLM_FLASH.price_out, 2.8)
        self.assertEqual(GLM_FLASH.extra_body['reasoning_effort'], 'low')
        self.assertTrue(DEEPSEEK_FLASH.key_path.exists())
        self.assertTrue(GLM_FLASH.key_path.exists(), 'GLM key must be stored '
                        'locally before paid runs')


if __name__ == '__main__':
    unittest.main(verbosity=2)
