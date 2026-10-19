"""New Flash matched panel; keeps every result and never imports legacy numbers."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))
import grad_agentdojo_hook as grad
import agentdojo_sanitizer_20260913 as san
from flash_budget import FlashClient, BudgetStop

OUT = ROOT / 'LOGS/final_table2/flash_matched_20260917'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--manifest', default=None)
    args = ap.parse_args()
    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not args.run:
        print(json.dumps(manifest, indent=2))
        return
    from agentdojo.task_suite.load_suites import get_suite
    from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
    assert importlib.metadata.version('agentdojo') == '0.1.35'
    grad.patch_deepseek_roles()
    grad.MODEL = san.MODEL = 'deepseek-flash'
    client = FlashClient(out, grad.API_KEY)
    san.CappedClient = lambda *a, **kw: client
    llm = OpenAILLM(client, 'deepseek-flash', temperature=0.0)
    gate = grad.GradGateClient(url='http://127.0.0.1:8779',
        threshold=manifest['grad_threshold'], fail_mode='closed', timeout=30)
    assert gate.probe()['status'] == 'ok', 'local gate unavailable'
    result_path = out / 'results.json'
    report = json.loads(result_path.read_text()) if result_path.exists() else {
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'runs': []}
    assert report['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    done = {r['case_id'] for r in report['runs']}
    new = 0
    errors = 0
    stop = False
    for case in manifest['cases']:
        for arm in manifest['arms']:
            cid = '|'.join([case['suite'], case['user_task'], str(case['injection_task']), arm])
            if cid in done:
                continue
            stats = {'calls': 0, 'was_safe_true': 0, 'was_safe_false': 0, 'errors': 0}
            gate.decisions.clear()
            if arm == 'none':
                pipeline = grad.build_pipeline(llm, ToolsExecutor())
            elif arm == 'grad':
                pipeline = grad.build_pipeline(llm, grad.make_grad_executor(gate))
            elif arm == 'sanitizer':
                pipeline = san.build_pipeline(llm, None, stats, True)
            else:
                raise ValueError(arm)
            suite = get_suite('v1.2', case['suite'])
            ut = suite.get_user_task_by_id(case['user_task'])
            inj = suite.get_injection_task_by_id(case['injection_task']) if case['injection_task'] else None
            if manifest.get('attack_style') == 'direct':
                from agentdojo.attacks.baseline_attacks import DirectAttack
                attack = DirectAttack(suite, pipeline)
            else:
                attack = grad.make_attack_class()(suite, pipeline)
            injections = attack.attack(ut, inj) if inj else {}
            client.tag = cid
            t0 = time.perf_counter()
            rec = dict(case, arm=arm, case_id=cid, error=None)
            stop = False
            before = client.state['charged_upper_rmb']
            try:
                rec.update(grad.run_one(suite, ut, inj, injections, pipeline))
                errors = 0
            except Exception as exc:
                rec.update(error=f'{type(exc).__name__}: {str(exc)[:200]}',
                           utility=None, security=None)
                errors += 1
                stop = isinstance(exc, BudgetStop) or errors >= 3
            rec['wall_ms'] = (time.perf_counter() - t0) * 1000
            rec['decisions'] = list(gate.decisions) if arm == 'grad' else []
            rec['sanitizer_stats'] = stats if arm == 'sanitizer' else None
            # Incremental defense modules, summed per complete trajectory.
            # Agent regeneration/replanning time is NOT isolated by these hooks.
            rec['defense_module_ms'] = (sum(d['latency_ms'] + d.get('build_latency_ms', 0)
                for d in gate.decisions) if arm == 'grad' else
                sum(stats.get('sanitizer_latency_ms', [])) if arm == 'sanitizer' else 0)
            rec['cost_upper_rmb'] = client.state['charged_upper_rmb'] - before
            report['runs'].append(rec)
            result_path.write_text(json.dumps(report, indent=2))
            new += 1
            print(json.dumps({k:rec.get(k) for k in ('case_id','utility','security','error','cost_upper_rmb')}) +
                  f' total_upper={client.state["charged_upper_rmb"]:.4f}', flush=True)
            if stop or (args.limit and new >= args.limit):
                break
        if stop or (args.limit and new >= args.limit):
            break


if __name__ == '__main__':
    main()
