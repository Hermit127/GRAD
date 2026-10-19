"""Pure-logic tests for the Prudentia skeleton (zero API calls)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prudentia.hitl import (  # noqa: E402
    EndorsementMode, FrozenApprovalOracle, HITLKind, HITLRequest,
    InterventionLedger, OracleMode,
)
from prudentia.ifc_labels import ContextLabel, Integrity, Readers, join_labels  # noqa: E402
from prudentia.metrics import (  # noqa: E402
    InterventionView, RunRecord, hitl_load, summarize, tcr_at_k, tcr_curve,
)
from prudentia.policies import (  # noqa: E402
    PolicyKind, ToolPolicy, check_tool_call,
)
from prudentia.variables import VariableStore  # noqa: E402


def label(integrity: Integrity, users=None) -> ContextLabel:
    readers = None if users is None else Readers(frozenset(users))
    return ContextLabel(integrity=integrity, readers=readers)


class IFCAndPolicyTests(unittest.TestCase):
    def test_join_integrity_takes_min(self):
        joined = join_labels([label(Integrity.TRUSTED), label(Integrity.UNTRUSTED)])
        self.assertIs(joined.integrity, Integrity.UNTRUSTED)

    def test_join_readers_intersects(self):
        a = label(Integrity.TRUSTED, {"A", "B", "C"})
        b = label(Integrity.TRUSTED, {"B", "C", "D"})
        joined = a.join(b)
        self.assertEqual(joined.readers.users, frozenset({"B", "C"}))

    def test_trusted_action_blocks_untrusted_context(self):
        policy = ToolPolicy(kind=PolicyKind.P_T, trusted_args=frozenset({"to"}))
        check = check_tool_call(policy, label(Integrity.UNTRUSTED),
                                {"to": label(Integrity.TRUSTED)})
        self.assertFalse(check.allowed)

    def test_trusted_action_allows_trusted_context(self):
        policy = ToolPolicy(kind=PolicyKind.P_T, trusted_args=frozenset({"to"}))
        check = check_tool_call(policy, label(Integrity.TRUSTED),
                                {"to": label(Integrity.TRUSTED)})
        self.assertTrue(check.allowed)

    def test_permitted_flow_requires_recipients_authorized(self):
        policy = ToolPolicy(kind=PolicyKind.P_F,
                            egress_args=frozenset({"body"}),
                            recipients_from="to")
        data = {"body": label(Integrity.TRUSTED, {"A", "B"})}
        ok = check_tool_call(policy, label(Integrity.TRUSTED), data,
                             recipients=frozenset({"A"}))
        self.assertTrue(ok.allowed)
        bad = check_tool_call(policy, label(Integrity.TRUSTED), data,
                              recipients=frozenset({"A", "Z"}))
        self.assertFalse(bad.allowed)

    def test_pt_or_pf_robust_declassification(self):
        policy = ToolPolicy(kind=PolicyKind.P_T_OR_P_F,
                            trusted_args=frozenset({"body"}),
                            egress_args=frozenset({"body"}))
        trusted = check_tool_call(policy, label(Integrity.TRUSTED),
                                  {"body": label(Integrity.TRUSTED)})
        self.assertTrue(trusted.allowed)
        untrusted_ctx = check_tool_call(
            policy, label(Integrity.UNTRUSTED),
            {"body": label(Integrity.UNTRUSTED, {"A"})},
            recipients=frozenset({"A"}))
        self.assertTrue(untrusted_ctx.allowed)


class VariableTests(unittest.TestCase):
    def test_naming_and_expansion_semantics(self):
        store = VariableStore()
        created = store.wrap_tool_result(
            "read_email", 0, {"body": "secret plans"},
            label(Integrity.UNTRUSTED, {"Emma"}))
        name = "#read_email-result-0.body#"
        self.assertIn(name, created)
        outcome = store.expand([name], endorse=False)
        self.assertTrue(outcome.context_tainted)
        self.assertTrue(outcome.all_variables_expanded)
        self.assertEqual(outcome.expanded_names, (name,))

    def test_endorsement_relabels_trusted(self):
        store = VariableStore()
        store.wrap_tool_result("read_email", 0, {"body": "todo"},
                               label(Integrity.UNTRUSTED))
        name = "#read_email-result-0.body#"
        outcome = store.expand([name], endorse=True)
        self.assertFalse(outcome.context_tainted)
        self.assertIs(store.label_of(name).integrity, Integrity.TRUSTED)
        self.assertTrue(store.get(name).endorsed)


class HITLTests(unittest.TestCase):
    def _request(self, kind=HITLKind.APPROVAL) -> HITLRequest:
        return HITLRequest(task_key="banking|user_task_0|benign|prudentia|0",
                           turn=3, kind=kind, tool="send_email",
                           policy_reason="P-T failed")

    def test_oracle_modes_frozen(self):
        request = self._request()
        optimistic = FrozenApprovalOracle(OracleMode.APPROVE_ALL,
                                          EndorsementMode.ENDORSE_ALL)
        pessimistic = FrozenApprovalOracle(OracleMode.REJECT_ALL,
                                           EndorsementMode.DENY_ALL)
        self.assertTrue(optimistic.decide(request))
        self.assertFalse(pessimistic.decide(request))

    def test_ledger_counts_kinds_separately(self):
        oracle = FrozenApprovalOracle()
        ledger = InterventionLedger(task_key="t")
        ledger.record(self._request(HITLKind.APPROVAL), True, oracle)
        ledger.record(self._request(HITLKind.ENDORSEMENT), True, oracle)
        self.assertEqual(ledger.approvals, 1)
        self.assertEqual(ledger.endorsements, 1)
        self.assertEqual(ledger.count(), 2)
        records = ledger.to_records()
        self.assertEqual([r["intervention_index"] for r in records], [1, 2])


class MetricsTests(unittest.TestCase):
    def _run(self, key, utility, violations, interventions):
        return RunRecord(task_key=key, suite="banking", condition="benign",
                         arm="prudentia", seed=0, utility=utility,
                         interventions=tuple(interventions),
                         policy_violations=violations)

    def test_hitl_load_only_successful_primary_and_formal(self):
        runs = [
            self._run("a", True, 2, [InterventionView("endorsement"),
                                     InterventionView("approval")]),
            self._run("b", False, 5, [InterventionView("approval")]),
        ]
        self.assertEqual(hitl_load(runs, "primary"), 2)
        self.assertEqual(hitl_load(runs, "formal"), 2)

    def test_tcr_at_k(self):
        runs = [
            self._run("a", True, 0, []),
            self._run("b", True, 2, [InterventionView("approval"),
                                     InterventionView("approval")]),
            self._run("c", False, 1, [InterventionView("approval")]),
        ]
        self.assertAlmostEqual(tcr_at_k(runs, 0, "primary"), 1 / 3)
        self.assertAlmostEqual(tcr_at_k(runs, 2, "primary"), 2 / 3)
        self.assertAlmostEqual(tcr_at_k(runs, None, "primary"), 2 / 3)
        self.assertAlmostEqual(tcr_curve(runs, (0, 2))["inf"], 2 / 3)

    def test_errors_excluded_and_from_dict_roundtrip(self):
        raw = {
            "task_key": "k", "suite": "travel", "condition": "benign",
            "arm": "prudentia", "seed": 1, "utility": True,
            "interventions": [{"kind": "endorsement", "approved": True}],
            "policy_violations": 0, "error": None,
        }
        good = RunRecord.from_dict(raw)
        broken = RunRecord.from_dict({**raw, "task_key": "k2",
                                      "error": "APIError"})
        self.assertEqual(good.endorsements, 1)
        self.assertEqual(hitl_load([good, broken], "primary"), 1)
        summary = summarize([good, broken])
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["utility"], 0.5)


if __name__ == "__main__":
    unittest.main()
