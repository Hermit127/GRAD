#!/usr/bin/env python
"""Deterministic generator for ``tool_policies.json`` (Prudentia, AgentDojo v1.2).

Run inside the frozen agentdojo venv (zero API calls)::

    python \
        code/repro/final_five_tables/prudentia/gen_tool_policies.py [--check]

Provenance / audit rules (plan section 9, assumption A3):
  - Labels are derived ONLY from the installed agentdojo v1.2 suite tools
    (name, parameters, description/docstring). No test results, no attack
    labels, no injection information are used.
  - The rule tables below encode the FIDES/Prudentia policy principles:
      * consequential action        -> P-T
      * egress of content           -> P-T or P-F
      * non-consequential read/query-> trivial (always allowed)
      * external-resource fetches   -> P-T on the resource-selecting argument
    Each rule carries a one-line cue (verb/argument that triggers it) so the
    mapping is reproducible and auditable.
  - The generator verifies that every classified tool exists in the installed
    suites and that every referenced argument exists in the tool signature.
  - ``--check`` compares the generated file with the on-disk JSON and exits
    non-zero on drift (CI/audit use).
  - For provenance only (never used for derivation), the script cross-checks
    the generated labels against the published FIDES 2025 per-tool policy
    table (arXiv:2505.23643, Table "policies") and prints any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BENCHMARK = "v1.2"
SUITES = ("banking", "slack", "travel", "workspace")
OUT_PATH = Path(__file__).resolve().parent / "tool_policies.json"

# ---------------------------------------------------------------------------
# Rule tables (name -> cue). All labels below are derived from the installed
# tool signatures/descriptions; cues are quoted verbatim fragments.
# ---------------------------------------------------------------------------

# Egress: content is transferred to recipients/readers outside the agent.
EGRESS_RULES: dict[str, dict] = {
    "send_email": {
        "egress_args": ["subject", "body", "attachments"],
        "recipients_from": "recipients",
        "cue": "doc: 'Sends an email with the given body to the given address' (content egress)",
    },
    "send_direct_message": {
        "egress_args": ["body"],
        "recipients_from": "recipient",
        "cue": "doc: 'Send a direct message from author to recipient with the given content'",
    },
    "send_channel_message": {
        "egress_args": ["body"],
        "recipients_from": "channel",
        "cue": "doc: 'Send a channel message from author to channel with the given content'",
    },
    "create_calendar_event": {
        "egress_args": ["title", "description", "location"],
        "recipients_from": "participants",
        "cue": "params: participants distribution of title/description/location",
    },
    "append_to_file": {
        "egress_args": ["content"],
        "recipients_from": None,
        "cue": "doc: 'Append content to a file in the cloud drive' (content egress to file readers)",
    },
}

# Consequential (P-T) actions that do not egress content to third parties.
PT_RULES: dict[str, str] = {
    "send_money": "doc: 'Sends a transaction to the recipient' (irreversible transfer)",
    "schedule_transaction": "doc: 'Schedule a transaction'",
    "update_scheduled_transaction": "doc: 'Update a scheduled transaction'",
    "update_password": "doc: 'Update the user password'",
    "update_user_info": "doc: 'Update the user information'",
    "delete_email": "doc: 'Deletes the email with the given email_id'",
    "create_file": "doc: 'Create a new file in the cloud drive'",
    "delete_file": "doc: 'Delete a file from the cloud drive by its filename'",
    "share_file": "doc: 'Share a file with a user'",
    "cancel_calendar_event": "doc: 'Cancels the event with the given event_id'",
    "reschedule_calendar_event": "doc: 'Reschedules the event with the given event_id'",
    "add_calendar_event_participants": "doc: 'Adds the given participants to the event'",
    "add_user_to_channel": "doc: 'Add a user to a given channel'",
    "invite_user_to_slack": "doc: 'Invites a user to the Slack workspace'",
    "remove_user_from_slack": "doc: 'Remove a user from the Slack workspace'",
    "post_webpage": "doc: 'Posts a webpage at a given URL with the given content'",
    "reserve_hotel": "doc: 'Makes a reservation for a hotel with the provided details'",
    "reserve_car_rental": "doc: 'Makes a reservation for a car rental with the provided details'",
    "reserve_restaurant": "doc: 'Makes a reservation for a restaurant with the provided details'",
}

# Fetches of external resources selected by an argument: the planner's decision
# and the selecting argument must be trusted (P-T with trusted_args).
EXTERNAL_FETCH_RULES: dict[str, list[str]] = {
    "get_webpage": ["url"],
}

# Tool results that must be encapsulated in UNTRUSTED variables: third-party
# authored text, private user data, or content that injection vectors can
# reach (emails, messages, webpages, files, reviews, calendar descriptions).
UNTRUSTED_RESULT_RULES: dict[str, str] = {
    # banking
    "get_iban": "private account identifier (confidentiality: user only)",
    "get_balance": "private account state",
    "get_most_recent_transactions": "transaction subjects are third-party authored text",
    "get_scheduled_transactions": "transaction subjects are third-party authored text",
    "read_file": "file contents can carry injected instructions",
    "get_user_info": "profile fields can be rewritten by injected instructions",
    # slack
    "read_channel_messages": "channel messages are third-party authored text",
    "read_inbox": "inbox messages are third-party authored text",
    "get_webpage": "web page content is attacker-influenceable",
    # travel
    "get_user_information": "profile fields can be rewritten by injected instructions",
    "get_rating_reviews_for_hotels": "reviews are third-party authored text",
    "get_rating_reviews_for_restaurants": "reviews are third-party authored text",
    "get_rating_reviews_for_car_rental": "reviews are third-party authored text",
    "search_calendar_events": "event titles/descriptions are user-authored text",
    "get_day_calendar_events": "event titles/descriptions are user-authored text",
    # workspace
    "get_unread_emails": "email bodies are attacker-influenceable text",
    "get_sent_emails": "email bodies are third-party authored text",
    "get_received_emails": "email bodies are attacker-influenceable text",
    "get_draft_emails": "draft bodies are user-authored text",
    "search_emails": "email bodies are attacker-influenceable text",
    "get_file_by_id": "file contents can carry injected instructions",
    "search_files": "file contents can carry injected instructions",
    "search_files_by_filename": "file contents can carry injected instructions",
    "list_files": "file contents can carry injected instructions",
}

# Published FIDES 2025 per-tool policy table (arXiv:2505.23643, tabs/policies)
# used as a *provenance cross-check only* (never as a derivation input).
FIDES2025_PUBLISHED: dict[str, str] = {
    "send_email": "P-F or P-T",
    "create_calendar_event": "P-F or P-T",
    "append_to_file": "P-F or P-T",
    "send_direct_message": "P-F or P-T",
    "send_channel_message": "P-F or P-T",
    "delete_email": "P-T",
    "reschedule_calendar_event": "P-T",
    "cancel_calendar_event": "P-T",
    "create_file": "P-T",
    "delete_file": "P-T",
    "share_file": "P-T",
    "reserve_hotel": "P-T",
    "reserve_restaurant": "P-T",
    "reserve_car_rental": "P-T",
    "send_money": "P-T",
    "schedule_transaction": "P-T",
    "update_scheduled_transaction": "P-T",
    "update_password": "P-T",
    "update_user_info": "P-T",
    "add_user_to_channel": "P-T",
    "invite_user_to_slack": "P-T",
    "remove_user_from_slack": "P-T",
    "get_webpage": "P-T",
    "post_webpage": "P-T",
    "add_calendar_event_participants": "P-T",
}


def _kind_for(name: str, params: list[str]) -> tuple[str, dict, str]:
    """Return (kind, extra_fields, rationale) for one tool."""

    if name in EGRESS_RULES:
        rule = EGRESS_RULES[name]
        return ("P-T or P-F", rule, rule["cue"])
    if name in EXTERNAL_FETCH_RULES:
        return ("P-T", {"trusted_args": list(EXTERNAL_FETCH_RULES[name])},
                "external resource selected by argument: decision+arg must be trusted")
    if name in PT_RULES:
        return ("P-T", {"trusted_args": list(params)}, PT_RULES[name])
    return ("trivial", {},
            "read/query tool with no consequential or egress verb in name/description")


def build_map() -> dict:
    from agentdojo.task_suite.load_suites import get_suite

    tools_seen: dict[str, dict] = {}
    signatures: dict[str, str] = {}
    for suite_name in SUITES:
        suite = get_suite(BENCHMARK, suite_name)
        for function in suite.tools:
            name = function.name
            if name in tools_seen:
                signatures[name] = f"{signatures[name]};{suite_name}"
                continue
            params = list(function.parameters.model_fields)
            kind, extra, rationale = _kind_for(name, params)

            if kind == "P-T or P-F":
                egress_args = list(extra["egress_args"])
                recipients_from = extra["recipients_from"]
                trusted_args = list(egress_args)
            elif kind == "P-T":
                egress_args = []
                recipients_from = None
                trusted_args = list(extra.get("trusted_args", params))
            else:
                egress_args = []
                recipients_from = None
                trusted_args = []

            # Signature validation: every referenced argument must exist.
            for arg in [*trusted_args, *egress_args]:
                if arg not in params:
                    raise SystemExit(
                        f"rule for {name}: argument {arg!r} not in signature "
                        f"{params}")
            if recipients_from is not None and recipients_from not in params:
                raise SystemExit(
                    f"rule for {name}: recipients_from {recipients_from!r} "
                    f"not in signature {params}")

            result_integrity = "U" if name in UNTRUSTED_RESULT_RULES else "T"
            reason = rationale
            if result_integrity == "U":
                reason += f"; result boxed UNTRUSTED ({UNTRUSTED_RESULT_RULES[name]})"

            tools_seen[name] = {
                "suite": signatures.get(name, suite_name),
                "consequential": kind in ("P-T", "P-T or P-F", "P-F"),
                "egress": kind in ("P-F", "P-T or P-F"),
                "always_allowed": kind == "trivial",
                "argument_source_requirements": {
                    arg: ("trusted" if arg in trusted_args else "any")
                    for arg in params
                },
                "kind": kind,
                "trusted_args": trusted_args,
                "egress_args": egress_args,
                "recipients_from": recipients_from,
                "result_integrity": result_integrity,
                "rationale": reason,
            }
            signatures[name] = suite_name

    # Coverage audit: every installed tool classified; rules not silently lost.
    used_rules = set(EGRESS_RULES) | set(PT_RULES) | set(EXTERNAL_FETCH_RULES) \
        | set(UNTRUSTED_RESULT_RULES)
    missing = sorted(rule for rule in used_rules if rule not in tools_seen)
    if missing:
        raise SystemExit(f"rule targets not found in installed suites: {missing}")

    payload = json.dumps(tools_seen, sort_keys=True, ensure_ascii=False,
                         indent=2)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    doc = {
        "_meta": {
            "generated_by": "gen_tool_policies.py",
            "benchmark_version": BENCHMARK,
            "suites": list(SUITES),
            "n_tools": len(tools_seen),
            "agentdojo_info": _agentdojo_info(),
            "derivation": ("rule tables over installed tool names, parameters "
                           "and descriptions (no test results, no attack labels)"),
            "tools_sha256": sha,
        },
        "tools": tools_seen,
    }
    return doc


def _agentdojo_info() -> dict:
    try:
        import agentdojo  # noqa: F401
        from importlib.metadata import version

        return {"package_version": version("agentdojo")}
    except Exception:  # pragma: no cover
        return {"package_version": "unknown"}


def cross_check_fides(tools: dict) -> list[str]:
    mismatches = []
    for name, published in FIDES2025_PUBLISHED.items():
        if name not in tools:
            mismatches.append(f"{name}: published by FIDES but not in v1.2 suites")
            continue
        ours = {part.strip() for part in tools[name]["kind"].split(" or ")}
        theirs = {part.strip() for part in published.split(" or ")}
        if ours != theirs:
            mismatches.append(
                f"{name}: generated {tools[name]['kind']!r} != FIDES table {published!r}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the on-disk JSON matches the generated map")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    doc = build_map()
    tools = doc["tools"]
    payload = json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n"

    mismatches = cross_check_fides(tools)
    print(f"tools classified: {len(tools)}")
    print(f"tools_sha256: {doc['_meta']['tools_sha256']}")
    if mismatches:
        print("FIDES table cross-check mismatches:")
        for item in mismatches:
            print(f"  - {item}")
    else:
        print("FIDES table cross-check: all overlapping tools match")

    out_path = Path(args.out)
    if args.check:
        if not out_path.exists():
            print(f"CHECK FAILED: {out_path} missing", file=sys.stderr)
            return 1
        current = out_path.read_text()
        generated = json.loads(payload)
        current_doc = json.loads(current)
        if json.dumps(current_doc.get("tools"), sort_keys=True) != \
                json.dumps(generated.get("tools"), sort_keys=True):
            print("CHECK FAILED: tool_policies.json drifted from rules",
                  file=sys.stderr)
            return 1
        file_sha = hashlib.sha256(current.encode()).hexdigest()
        print(f"CHECK OK: {out_path} matches generated map (file sha256 {file_sha})")
        return 0

    out_path.write_text(payload)
    file_sha = hashlib.sha256(payload.encode()).hexdigest()
    print(f"wrote {out_path}")
    print(f"file sha256: {file_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
