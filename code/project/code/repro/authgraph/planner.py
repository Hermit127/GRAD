"""Planner Agent: authorization graph in a clean context (paper 3.3).

Input isolation (Property 1) is enforced structurally and with a runtime
guard: ``plan`` accepts only the user prompt; the tool catalog is fixed at
construction. If the caller passes trace-derived strings via ``forbidden``,
the guard raises before any LLM call when one of them appears in the rendered
messages.
"""
from __future__ import annotations

import json

from prompts import (PLANNER_REPLAN_SYSTEM, PLANNER_SYSTEM, planner_replan_user,
                     planner_user)
from schema import validate_ag, validate_ag_subgraph


class PlannerError(RuntimeError):
    pass


class IsolationError(RuntimeError):
    pass


class Planner:
    def __init__(self, llm, tool_catalog: list[dict]):
        self.llm = llm
        self.tool_catalog = tool_catalog

    def _guard(self, system: str, user: str, forbidden: list[str] | None) -> None:
        blob = system + "\n" + user
        for s in forbidden or []:
            if s and len(str(s)) >= 16 and str(s) in blob:
                raise IsolationError(
                    "Planner input isolation violated: trace-derived string "
                    "found in the Planner messages")

    def plan(self, user_prompt: str, forbidden: list[str] | None = None) -> dict:
        system = PLANNER_SYSTEM
        user = planner_user(user_prompt, self.tool_catalog)
        self._guard(system, user, forbidden)
        ag = self.llm.chat_json(system, user, name="planner")
        errs = validate_ag(ag, self.tool_catalog)
        if errs:
            user2 = (user + "\n\n## Your previous output was rejected\n"
                     + "\n".join(f"- {e}" for e in errs)
                     + "\nReturn the corrected JSON object only.")
            self._guard(system, user2, forbidden)
            ag = self.llm.chat_json(system, user2, name="planner_retry")
            errs = validate_ag(ag, self.tool_catalog)
            if errs:
                raise PlannerError("authorization graph failed validation: "
                                   + "; ".join(errs))
        return ag

    def replan(self, user_prompt: str, observation: str,
               allowed_tools: list[str], next_step_index: int,
               forbidden: list[str] | None = None,
               trigger_tool: str | None = None) -> dict:
        system = (PLANNER_REPLAN_SYSTEM
                  .replace("{allowed_tools}", json.dumps(allowed_tools))
                  .replace("{next_step_index}", str(next_step_index)))
        user = planner_replan_user(user_prompt, observation, allowed_tools,
                                   next_step_index)
        self._guard(system, user, forbidden)
        ag = self.llm.chat_json(system, user, name="planner_replan")
        errs = validate_ag_subgraph(ag, self.tool_catalog, allowed_tools,
                                    trigger_tool=trigger_tool)
        if errs:
            user2 = (user + "\n\n## Your previous output was rejected\n"
                     + "\n".join(f"- {e}" for e in errs)
                     + "\nReturn the corrected JSON object only.")
            self._guard(system, user2, forbidden)
            ag = self.llm.chat_json(system, user2, name="planner_replan_retry")
            errs = validate_ag_subgraph(ag, self.tool_catalog, allowed_tools,
                                        trigger_tool=trigger_tool)
            if errs:
                raise PlannerError("replan sub-graph failed validation: "
                                   + "; ".join(errs))
        return ag
