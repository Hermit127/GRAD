"""Two-vendor budget client for Table 4 (DeepSeek Flash + GLM-5.3-Flash).

Rules frozen for this batch (PI 2026-09-18):
  * pre-call reservation from a conservative peak / all-cache-miss bound;
  * no automatic retries (``max_retries=0``); a failed call keeps its
    reservation and is recorded as failed;
  * every call logs requested model, returned model ID, usage, cached tokens,
    latency, reservation, actual cost and conservative cost;
  * the hard cap is checked against the reservation upper bound before each
    call; exhaustion raises :class:`BudgetStop` and the runner stops.

Pricing sources (checked 2026-09-18):
  * DeepSeek ``deepseek-flash`` (DeepSeek-V4.1-Flash): peak $0.3/M input
    (cache miss), $0.006/M cache hit, $1.2/M output at 7.15 RMB/USD
    -> 2.2 / 0.0429 / 8.6 RMB per M (upper bound).
  * GLM ``glm-5.3-flash``: 0.8 RMB/M input, 0.23 RMB/M cache hit,
    2.8 RMB/M output (open.bigmodel.cn pricing page).
"""
from __future__ import annotations

import os
import fcntl
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]


class BudgetStop(RuntimeError):
    pass


def _usage_to_dict(obj) -> dict:
    """Normalize openai usage objects / SimpleNamespace / dicts to a dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    try:
        raw = vars(obj)
    except TypeError:
        return {}
    return {k: (v if isinstance(v, (int, float, str, bool, type(None), dict))
                else _usage_to_dict(v)) for k, v in raw.items()}


class CallFailed(RuntimeError):
    pass


@dataclass
class VendorSpec:
    name: str
    model: str
    base_url: str
    key_path: Path
    price_in: float
    price_out: float
    price_cache: float
    extra_body: dict = field(default_factory=dict)
    price_basis: str = ''


DEEPSEEK_FLASH = VendorSpec(
    name='deepseek-flash', model='deepseek-flash',
    base_url='https://api.deepseek.com',
    key_path=REPO / 'ref/MINJA/rap/OpenAI_api_key.txt',
    price_in=2.2, price_out=8.6, price_cache=0.0429,
    extra_body={'thinking': {'type': 'disabled'}},
    price_basis=('DeepSeek peak all-cache-miss upper bound: $0.3/$1.2 per M '
                 'at 7.15 RMB/USD; cache hit $0.006/M.'),
)

GLM_FLASH = VendorSpec(
    name='glm-5.3-flash', model='glm-5.3-flash',
    base_url='https://open.bigmodel.cn/api/paas/v4',
    key_path=REPO / 'ref/.secrets/glm_api_key.txt',
    price_in=0.8, price_out=2.8, price_cache=0.23,
    # GLM-5.3-Flash cannot disable thinking (API rejects thinking.type
    # "disabled"); minimum reasoning effort is used and disclosed.
    extra_body={'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'},
    price_basis=('GLM-5.3-Flash list price: 0.8 RMB/M input, 0.23 RMB/M '
                 'cache hit, 2.8 RMB/M output.'),
)


class VendorBudgetClient:
    def __init__(self, directory: str | Path, spec: VendorSpec, cap: float,
                 tag: str = ''):
        import openai
        self.spec = spec
        self.tag = tag
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock_fh = open(self.directory / f'{spec.name}.lock', 'a')
        fcntl.flock(self.lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.directory / f'budget_{spec.name}.json'
        self.calls_path = self.directory / f'calls_{spec.name}.jsonl'
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            assert self.state['cap_rmb'] == cap, 'cap change requires new dir'
        else:
            self.state = {
                'vendor': spec.name, 'model': spec.model, 'cap_rmb': cap,
                'charged_upper_rmb': 0.0, 'charged_actual_rmb': 0.0,
                'calls': 0, 'failed_calls': 0,
                'price_basis': spec.price_basis,
                'reservation_rule': ('min(json_bytes+8192, ...) token upper '
                                     'bound * cache-miss price + max_tokens * '
                                     'output price'),
            }
        self.client = openai.OpenAI(
            api_key=(os.environ.get("DEEPSEEK_API_KEY" if spec.name.startswith("deepseek") else "GLM_API_KEY")
                     or spec.key_path.read_text().strip()),
            base_url=spec.base_url, timeout=180, max_retries=0)
        self.chat = SimpleNamespace(completions=self)

    # -- bookkeeping --------------------------------------------------------
    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _log_call(self, entry: dict):
        with open(self.calls_path, 'a') as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')

    # -- chat ---------------------------------------------------------------
    def create(self, **kwargs):
        call_tag = str(kwargs.pop('_tag', '') or self.tag)
        kwargs['model'] = self.spec.model
        kwargs.setdefault('temperature', 0.0)
        kwargs['max_tokens'] = int(kwargs.get('max_tokens') or 1024)
        if self.spec.extra_body:
            kwargs['extra_body'] = dict(self.spec.extra_body)
        bound_input = len(json.dumps(
            kwargs, ensure_ascii=False, default=str).encode()) + 8192
        reserve = (bound_input * self.spec.price_in +
                   kwargs['max_tokens'] * self.spec.price_out) / 1e6
        if self.state['charged_upper_rmb'] + reserve > self.state['cap_rmb']:
            raise BudgetStop(
                f"{self.spec.name}: reservation {reserve:.6f} would exceed cap "
                f"{self.state['cap_rmb']} (upper bound so far "
                f"{self.state['charged_upper_rmb']:.6f})")
        entry = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 'tag': call_tag, 'vendor': self.spec.name,
                 'model_requested': self.spec.model, 'status': 'in_flight',
                 'reserved_rmb': round(reserve, 6),
                 'body_bytes': bound_input - 8192}
        self.state['charged_upper_rmb'] += reserve
        self.save()
        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - reservation retained
            entry.update(status='failed', error=f'{type(exc).__name__}: '
                                                f'{str(exc)[:300]}',
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            self.state['failed_calls'] += 1
            self.state['calls'] += 1
            self._log_call(entry)
            self.save()
            raise CallFailed(entry['error']) from exc
        usage = response.usage
        if usage is None:
            entry.update(status='failed',
                         error='missing usage; reservation retained',
                         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            self.state['failed_calls'] += 1
            self.state['calls'] += 1
            self._log_call(entry)
            self.save()
            raise CallFailed(entry['error'])
        u = _usage_to_dict(usage)
        details = u.get('prompt_tokens_details') or {}
        if not isinstance(details, dict):
            details = _usage_to_dict(details)
        cached = int(details.get('cached_tokens') or 0)
        pt = int(u.get('prompt_tokens') or 0)
        ct = int(u.get('completion_tokens') or 0)
        miss = max(pt - cached, 0)
        cost = (miss * self.spec.price_in + cached * self.spec.price_cache +
                ct * self.spec.price_out) / 1e6
        conservative = (pt * self.spec.price_in +
                        ct * self.spec.price_out) / 1e6
        # Reconcile the reservation down to the conservative (all-cache-miss)
        # cost of this completed call. charged_upper_rmb therefore stays an
        # upper bound on money actually spent while not blocking the batch on
        # released reservations.
        self.state['charged_upper_rmb'] += conservative - reserve
        self.state['charged_actual_rmb'] += cost
        self.state['calls'] += 1
        message = response.choices[0].message
        entry.update(status='ok', model_returned=str(response.model),
                     finish_reason=response.choices[0].finish_reason,
                     usage=u, cached_tokens=cached,
                     cost_rmb=round(cost, 6),
                     conservative_rmb=round(conservative, 6),
                     latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        self._log_call(entry)
        self.save()
        return SimpleNamespace(
            content=str(getattr(message, 'content', '') or ''),
            reasoning=str(getattr(message, 'reasoning_content', '') or ''),
            model=str(response.model),
            finish_reason=response.choices[0].finish_reason,
            usage=u, cached_tokens=cached,
            cost_rmb=cost, elapsed_ms=entry['latency_ms'])


def make_clients(budget_dir: str | Path, cap_deepseek: float, cap_glm: float,
                 tag: str = '') -> dict:
    return {
        DEEPSEEK_FLASH.name: VendorBudgetClient(
            budget_dir, DEEPSEEK_FLASH, cap_deepseek, tag=tag),
        GLM_FLASH.name: VendorBudgetClient(
            budget_dir, GLM_FLASH, cap_glm, tag=tag),
    }
