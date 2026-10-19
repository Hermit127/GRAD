"""Offline tests for the RMB-capped Flash budget client (zero API)."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import flash_budget  # noqa: E402
from flash_budget import BudgetStop, FlashClient  # noqa: E402


class FakeUsage:
    def __init__(self, prompt=1000, completion=200):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_cache_hit_tokens = 100
        self.prompt_cache_miss_tokens = 900

    def model_dump(self):
        return {'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'prompt_cache_hit_tokens': 100,
                'prompt_cache_miss_tokens': 900}


class FakeCompletions:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError('simulated timeout')
        return types.SimpleNamespace(usage=FakeUsage(), model='deepseek-flash',
                                     choices=[])


class FakeOpenAI:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeOpenAI.last_kwargs = kwargs
        self.chat = types.SimpleNamespace(
            completions=FakeCompletions(fail=kwargs.pop('_fail', False)))


class BudgetClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.key = Path(self.tmp.name) / 'key.txt'
        self.key.write_text('test-key')
        self._saved = sys.modules.get('openai')
        fake = types.SimpleNamespace(OpenAI=FakeOpenAI)
        sys.modules['openai'] = fake
        self._fake = fake

    def tearDown(self):
        if self._saved is not None:
            sys.modules['openai'] = self._saved
        else:
            sys.modules.pop('openai', None)
        self.tmp.cleanup()

    def client(self, cap=2.9, fail=False):
        orig = self._fake.OpenAI

        class Wrapped(orig):
            def __init__(self, **kw):
                kw['_fail'] = fail
                super().__init__(**kw)

        self._fake.OpenAI = Wrapped
        return FlashClient(self.tmp.name, self.key, cap=cap)

    def test_pre_call_reservation_blocks_before_network(self):
        c = self.client(cap=0.0001)
        with self.assertRaises(BudgetStop):
            c.create(messages=[{'role': 'user', 'content': 'x' * 500}],
                     max_tokens=8192)
        self.assertEqual(c.client.chat.completions.calls, [])
        self.assertFalse(c.path.exists())

    def test_official_peak_prices_used(self):
        c = self.client()
        c.create(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=100)
        expected = (1000 * 2.2 + 200 * 8.6) / 1e6
        state = json.loads(c.path.read_text())
        self.assertAlmostEqual(state['calls'][0]['cost_upper_rmb'], expected,
                               places=9)
        self.assertAlmostEqual(state['charged_upper_rmb'], expected, places=9)
        self.assertIn('prompt_cache_hit_tokens', state['calls'][0]['usage'])

    def test_no_auto_retry_and_bounded_output(self):
        c = self.client()
        c.create(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=99999)
        self.assertEqual(FakeOpenAI.last_kwargs['max_retries'], 0)
        self.assertEqual(c.client.chat.completions.calls[0]['max_tokens'], 8192)
        self.assertEqual(c.client.chat.completions.calls[0]['model'], 'deepseek-flash')
        self.assertEqual(c.client.chat.completions.calls[0]['temperature'], 0.0)
        self.assertEqual(c.client.chat.completions.calls[0]['extra_body'],
                         {'thinking': {'type': 'disabled'}})

    def test_uncertain_call_retains_reservation(self):
        c = self.client(fail=True)
        with self.assertRaises(TimeoutError):
            c.create(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=100)
        state = json.loads(c.path.read_text())
        self.assertEqual(state['calls'][0]['status'], 'uncertain')
        self.assertGreater(state['charged_upper_rmb'], 0)

    def test_batch_ledger_is_separate_from_shared_ledger(self):
        c = self.client()
        c.create(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=100)
        self.assertTrue((Path(self.tmp.name) / 'budget.json').exists())
        shared = Path('./LOGS/sota_runtime_ledger.json')
        self.assertNotEqual(c.path, shared)

    def test_state_reload_keeps_same_cap(self):
        c = self.client()
        c.create(messages=[{'role': 'user', 'content': 'hi'}], max_tokens=100)
        c.lock.close()
        c2 = self.client()
        self.assertEqual(c2.state['charged_upper_rmb'],
                         c.state['charged_upper_rmb'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
