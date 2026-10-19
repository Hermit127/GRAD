"""Contract tests for the 100-family confirmation environment.

Covers: counts, policy semantics (verify/harm/complete), scripted-mock anchor
parsing for every family, and argument disjointness from the development env.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import importlib.util

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import envlib  # noqa: E402
import envlib_conf  # noqa: E402

spec = importlib.util.spec_from_file_location(
    't4_runner_conf', REPO / 'code/repro/final_table4/runner.py')
t4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t4)

MOCK = t4.MockClient('test')


class TestConfEnv(unittest.TestCase):
    def test_counts(self):
        fams = envlib_conf.all_families()
        self.assertEqual(len(fams), 100)
        per_env = {}
        for f in fams:
            per_env.setdefault(f['env'], []).append(f['family'])
        for env in ('shop', 'workspace', 'finance', 'cloud'):
            self.assertEqual(len(per_env[env]), 25, env)

    def test_policy_semantics(self):
        for env_name, env in envlib_conf.ENVS.items():
            for fam_name, fam in env['families'].items():
                good, bad = fam['good_arg'], fam['bad_arg']
                self.assertEqual(env['verify'](good, fam, {})[0], 'allow',
                                 f'{env_name}:{fam_name} good verify')
                self.assertEqual(env['verify'](bad, fam, {})[0], 'block',
                                 f'{env_name}:{fam_name} bad verify')
                self.assertFalse(env['harm'](good, fam))
                self.assertTrue(env['harm'](bad, fam))
                self.assertTrue(env['complete'](good, fam))
                self.assertFalse(env['complete'](bad, fam))

    def test_mock_note_parsing(self):
        domain_of = {'shop': 'shop', 'workspace': 'workspace',
                     'finance': 'finance', 'cloud': 'cloud'}
        for f in envlib_conf.all_families():
            dom = domain_of[f['env']]
            got_bad = MOCK._arg_from_note(dom, f['note_bad'])
            got_good = MOCK._arg_from_note(dom, f['note_good'])
            if dom == 'shop':
                self.assertEqual(str(got_bad).upper(), f['bad_arg'].upper())
                self.assertEqual(str(got_good).upper(), f['good_arg'].upper())
            else:
                self.assertEqual(got_bad, f['bad_arg'], f['note_bad'][:60])
                self.assertEqual(got_good, f['good_arg'],
                                 f['note_good'][:60])

    def test_mock_s1_anchors(self):
        import re
        for f in envlib_conf.all_families():
            s1 = f['s1']
            if f['env'] == 'shop':
                m = re.search(r'looking for (\w+)', s1)
                self.assertEqual(m and m.group(1), f['family'])
            elif f['env'] == 'workspace':
                m = re.search(r'repo (\w+)', s1)
                self.assertEqual(m and m.group(1), f['family'])
            elif f['env'] == 'finance':
                m = re.search(r'invoice (\S+)', s1)
                self.assertEqual(m and m.group(1), f['family'])
            else:
                m = re.search(r'share (\S+)', s1)
                self.assertEqual(m and m.group(1), f['family'])

    def test_args_disjoint_from_dev(self):
        dev = set()
        for env in envlib.ENVS.values():
            for fam in env['families'].values():
                dev.add(str(fam['bad_arg']))
                dev.add(str(fam['good_arg']))
        for f in envlib_conf.all_families():
            for key in ('bad_arg', 'good_arg'):
                self.assertNotIn(str(f[key]), dev,
                                 f'{f["env"]}:{f["family"]} {key} reused')

    def test_suite_envs_are_built(self):
        for env in envlib_conf.ENVS.values():
            self.assertTrue(env['families'])
            self.assertIn('observation', env)
            self.assertIn('harm', env)


if __name__ == '__main__':
    unittest.main()
