"""Variable store implementing the Dual LLM / variable-hiding primitives.

Paper anchors:
  - Prudentia section 2 "Dual LLM and IFC" and section 4 "Endorsement vs.
    approval": variables hide untrusted tool results; expanding without
    endorsement taints the context permanently; Prudentia's tweak expands
    *all* variables when `ask_endorsement=False`.
  - Appendix D.3 system prompt (verbatim in prompts.py): naming conventions
    `#tool-result-i.field#`, endorsement relabels data to trusted without
    tainting the context, quarantined LLM outputs must never be expanded.

Pure bookkeeping only; no model calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional

from .ifc_labels import ContextLabel, Integrity

_NAME_RE = re.compile(r"^#(?P<body>[A-Za-z0-9_.\-]+)#$")


def render_value(value: Any) -> str:
    """Render a variable value for inclusion in a tool message (context)."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class VariableRef:
    name: str
    value: Any
    label: ContextLabel
    root: bool = True
    endorsed: bool = False


@dataclass(frozen=True)
class ExpandOutcome:
    expanded_names: tuple[str, ...]
    context_tainted: bool
    all_variables_expanded: bool
    endorsed: bool


class VariableStore:
    def __init__(self) -> None:
        self._vars: dict[str, VariableRef] = {}

    def wrap_tool_result(self, tool_name: str, call_index: int,
                         fields: Mapping[str, Any],
                         label: ContextLabel) -> dict[str, VariableRef]:
        """Store one tool result as per-field variables (root variables)."""

        created: dict[str, VariableRef] = {}
        for field_name, value in fields.items():
            name = f"#{tool_name}-result-{call_index}.{field_name}#"
            ref = VariableRef(name=name, value=value, label=label, root=True)
            self._vars[name] = ref
            created[name] = ref
        return created

    def wrap_result(self, tool_name: str, call_index: int, result: Any,
                    label: ContextLabel, *, root: bool = True
                    ) -> dict[str, VariableRef]:
        """Wrap an arbitrary tool result following the D.3 naming conventions.

        Naming (Prudentia appendix D.3):
          - basic scalar:          ``#tool-result-i#``
          - dict/model field:      ``#tool-result-i.key#``
          - list item scalar:      ``#tool-result-i-j#``
          - list item dict field:  ``#tool-result-i-j.key#``

        All created values are plain copies of the tool output; hiding the
        actual content from the planner context is the caller's job.
        """

        created: dict[str, VariableRef] = {}
        for name, value in _iter_result_variables(tool_name, call_index,
                                                  result):
            ref = VariableRef(name=name, value=value, label=label, root=root)
            self._vars[name] = ref
            created[name] = ref
        return created

    def add(self, ref: VariableRef) -> None:
        self._vars[ref.name] = ref

    def get(self, name: str) -> VariableRef:
        if name not in self._vars:
            raise KeyError(f"unknown variable {name!r}")
        return self._vars[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._vars)

    def root_names(self) -> tuple[str, ...]:
        return tuple(n for n, v in self._vars.items() if v.root)

    def resolve(self, names: Iterable[str]) -> tuple[Any, ...]:
        return tuple(self.get(n).value for n in names)

    def endorse(self, names: Iterable[str]) -> tuple[str, ...]:
        """Relabel root variables as trusted (Prudentia section 4)."""

        done = []
        for name in names:
            ref = self.get(name)
            self._vars[name] = replace(
                ref, label=replace(ref.label, integrity=Integrity.TRUSTED),
                endorsed=True)
            done.append(name)
        return tuple(done)

    def expand(self, names: Iterable[str], *, endorse: bool) -> ExpandOutcome:
        """Expand named variables.

        endorse=True: only the named variables are expanded and (if the human
        oracle approved the endorsement) they are relabelled trusted, keeping
        the context label intact.
        endorse=False: ALL *root* variables in memory are expanded and the
        context is marked permanently tainted (paper section 4 tweak).
        Quarantined-LLM outputs are never expanded (appendix D.3).
        """

        if endorse:
            expanded = tuple(names)
            self.endorse(expanded)
            return ExpandOutcome(expanded, False, False, True)
        expanded = self.root_names()
        for name in expanded:
            ref = self._vars[name]
            self._vars[name] = replace(ref, label=replace(
                ref.label, integrity=Integrity.UNTRUSTED))
        return ExpandOutcome(expanded, True, True, False)

    def label_of(self, name: str) -> ContextLabel:
        return self.get(name).label

    def tainted_label(self) -> ContextLabel:
        """Context label after an endorsement-free expansion."""

        labels = [v.label for v in self._vars.values()]
        result = ContextLabel()
        for label in labels:
            result = result.join(label)
        return replace(result, integrity=Integrity.UNTRUSTED)


def parse_variable_name(name: str) -> Optional[str]:
    match = _NAME_RE.match(name)
    return match.group("body") if match else None


def _model_dump(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def _iter_result_variables(tool_name: str, call_index: int,
                           result: Any):
    """Yield ``(name, value)`` pairs for one tool result (D.3 naming)."""

    base = f"#{tool_name}-result-{call_index}"
    result = _model_dump(result)

    if isinstance(result, Mapping):
        for key, value in result.items():
            yield f"{base}.{key}#", _model_dump(value)
        return

    if isinstance(result, (list, tuple)):
        for index, item in enumerate(result):
            item = _model_dump(item)
            if isinstance(item, Mapping):
                for key, value in item.items():
                    yield f"{base}-{index}.{key}#", _model_dump(value)
            else:
                yield f"{base}-{index}#", item
        return

    yield f"{base}#", result
