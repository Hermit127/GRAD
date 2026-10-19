"""Dedicated RMB-5 batch budget; conservative peak tariff, no automatic retries.

Reserves a byte-based input upper estimate + bounded output before transmission.
Uncertain requests retain their reservation. This batch never rewrites the legacy
shared ledger (other agents still use non-locking read/modify/write clients).
"""
import json
import time
import fcntl
from pathlib import Path
from types import SimpleNamespace


class BudgetStop(RuntimeError):
    pass


class FlashClient:
    def __init__(self, directory, key_path, cap=2.90):
        import openai
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.directory / 'budget.lock', 'a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.directory / 'budget.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'cap_rmb': cap, 'charged_upper_rmb': 0.0, 'calls': [],
            'price_input_per_million': 2.2, 'price_output_per_million': 8.6,
            'price_basis': 'Flash peak USD 0.3/1.2 per M at 7.15 RMB/USD, all input cache-miss (conservative upper bound)',
        }
        assert self.state['cap_rmb'] == cap
        self.client = openai.OpenAI(api_key=Path(key_path).read_text().strip(),
            base_url='https://api.deepseek.com', timeout=120, max_retries=0)
        self.chat = SimpleNamespace(completions=self)
        self.tag = ''

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=2))

    def create(self, **kwargs):
        kwargs['model'] = 'deepseek-flash'
        kwargs['temperature'] = 0.0
        kwargs['max_tokens'] = min(int(kwargs.get('max_tokens') or 8192), 8192)
        kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        # UTF-8 bytes over the whole serialized request, plus generous framing.
        bound_input = len(json.dumps(kwargs, ensure_ascii=False, default=str).encode()) + 8192
        reserve = (bound_input * 2.2 + kwargs['max_tokens'] * 8.6) / 1e6
        if self.state['charged_upper_rmb'] + reserve > self.state['cap_rmb']:
            raise BudgetStop('pre-call reservation would exceed RMB 5 cap')
        call = {'tag': self.tag, 'time': time.time(), 'reserved_rmb': reserve,
                'status': 'in_flight'}
        self.state['calls'].append(call)
        self.state['charged_upper_rmb'] += reserve
        self.save()
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
            u = response.usage
            if u is None:
                raise RuntimeError('missing usage; reservation retained')
            cost = (u.prompt_tokens * 2.2 + u.completion_tokens * 8.6) / 1e6
            usage = json.loads(json.dumps(u.model_dump(), default=str))
            call.update(status='ok', usage=usage,
                        model=str(response.model), cost_upper_rmb=cost)
            self.state['charged_upper_rmb'] += cost - reserve
            if cost > reserve:
                raise BudgetStop('reservation bound exceeded; stop for audit')
            return response
        except Exception as exc:
            call.update(status='uncertain', error=type(exc).__name__)
            raise
        finally:
            call['latency_ms'] = (time.perf_counter() - t0) * 1000
            self.save()
