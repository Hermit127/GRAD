"""Authorization semantics for the memory-chain defense (v3, zero-API).

An authorization is a record over (subject user, session, action, object ASIN,
issued_seq, valid_until_seq, revoked_seq, source message). Natural-language
parsing is deterministic and can return unknown; unknown never becomes an
authorization and never forces a blanket block (blocks are still scoped to a
consumed untrusted record matching the proposed argument).

Structured form: AUTH{buy B0002} / AUTH{buy B0002 valid_until=42}.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ASIN_RE = re.compile(r'\bB[0-9][0-9A-Z]{3,11}\b')
STRUCT_RE = re.compile(r'AUTH\{([^}]*)\}', re.I)
NEG_CUES = ('do not', "don't", 'never', 'must not', 'cannot', "can't",
            'avoid', 'not buy', 'no ', 'forbid', 'prohibit', '不要', '不可',
            '严禁', '禁止')
POS_CUES = ('authorize', 'authorized', 'authorise', 'authorised', 'approved',
            'permission', 'you may', 'go ahead', 'buy')
DISCUSS_CUES = ('?', 'quote', 'quoted', 'review said', 'said to', 'said ',
                'discussed', 'mentioned', 'what about', 'should i', 'asking',
                '如果', '请问')
REVOKE_CUES = ('revoke', 'revoked', 'cancel', 'withdraw', '取消', '撤销')


@dataclass
class AuthRecord:
    asin: str
    state: str
    source_message_id: str
    session: int
    user_id: str
    issued_seq: int
    valid_until_seq: int | None = None
    revoked_seq: int | None = None
    reason: str = ''
    effective: bool = False
    structured: bool = False
    field_notes: list = field(default_factory=list)


def _window(text: str, start: int, width: int = 48) -> str:
    return text[max(0, start - width):start].lower()


def _sentence(text: str, start: int, stop: int) -> str:
    lo = max(text.rfind(c, 0, start) for c in '.!?;\n') + 1
    hi = min([x for x in (text.find(c, stop) for c in '.!?;\n') if x != -1]
             or [len(text)])
    return text[lo:hi]


def parse_authorizations(trace: dict, current_user_id: str) -> list[AuthRecord]:
    """Parse every task/user message into AuthRecord entries (all states)."""
    out = []
    for msg in trace.get('tasks', []):
        text = str(msg['text'])
        uid = msg.get('user_id', current_user_id)
        structured_spans = [(m.start(), m.end(), m.group(1))
                            for m in STRUCT_RE.finditer(text)]
        for asin in ASIN_RE.findall(text.upper()):
            spans = [text.upper().find(asin)]
            structured = next(((a, b, body) for a, b, body in structured_spans
                               if asin in body.upper()), None)
            state, reason = 'unknown', 'no_directive'
            valid_until = None
            is_struct = False
            if structured:
                is_struct = True
                body = structured[2].lower()
                state = 'authorized' if 'buy' in body or 'authorize' in body \
                    else 'unknown'
                reason = 'structured_authorization' if state == 'authorized' \
                    else 'structured_unclear'
                m = re.search(r'valid_until\s*=\s*(\d+)', body)
                valid_until = int(m.group(1)) if m else None
            else:
                for pos in spans:
                    sent = text
                    pre = text[:pos].lower()
                    neg = any(c in pre for c in NEG_CUES) or \
                        any(c in sent.lower() for c in NEG_CUES)
                    poscue = any(c in pre for c in POS_CUES)
                    discuss = any(c in sent.lower() for c in DISCUSS_CUES)
                    if neg:
                        state, reason = 'denied', 'negation_cue'
                        break
                    if poscue and not discuss:
                        state, reason = 'authorized', 'directive_cue'
                        break
                    if poscue and discuss:
                        state, reason = 'unknown', 'quoted_or_discussed'
                    else:
                        state, reason = 'unknown', 'mention_only'
            out.append(AuthRecord(asin=asin, state=state,
                                  source_message_id=msg['message_id'],
                                  session=int(msg.get('session', 1)),
                                  user_id=str(uid),
                                  issued_seq=int(msg['issued_seq']),
                                  valid_until_seq=valid_until,
                                  reason=reason, structured=is_struct,
                                  field_notes=[s[2] for s in structured_spans]))
    for msg in trace.get('tasks', []):
        text = str(msg['text']).lower()
        if not any(c in text for c in REVOKE_CUES):
            continue
        for rec in out:
            if rec.asin.lower() in text and \
                    msg['issued_seq'] >= rec.issued_seq and \
                    msg.get('user_id', current_user_id) == rec.user_id:
                rec.revoked_seq = min(
                    rec.revoked_seq or msg['issued_seq'], msg['issued_seq'])
    return out


def authorization_state(trace: dict, current_user_id: str, asin: str,
                        now_seq: int) -> dict:
    """Effective state for one ASIN at now_seq, with unknown reasons."""
    records = [r for r in parse_authorizations(trace, current_user_id)
               if r.asin == asin.upper() and r.issued_seq <= now_seq]
    if not records:
        return {'state': 'unknown', 'reason': 'not_mentioned', 'asin': asin,
                'source_message_ids': []}
    others = [r for r in records if r.user_id != current_user_id]
    mine = [r for r in records if r.user_id == current_user_id]
    if not mine and others:
        return {'state': 'unknown', 'reason': 'other_user',
                'asin': asin,
                'source_message_ids': [r.source_message_id for r in others]}
    if any(r.state == 'denied' and not r.revoked_seq for r in mine):
        return {'state': 'denied', 'reason': 'explicit_denial', 'asin': asin,
                'source_message_ids': [r.source_message_id for r in mine
                                       if r.state == 'denied']}
    for r in mine:
        if r.state != 'authorized':
            continue
        if r.revoked_seq is not None and r.revoked_seq <= now_seq:
            return {'state': 'unknown', 'reason': 'revoked', 'asin': asin,
                    'source_message_ids': [r.source_message_id]}
        if r.valid_until_seq is not None and now_seq > r.valid_until_seq:
            return {'state': 'unknown', 'reason': 'expired', 'asin': asin,
                    'source_message_ids': [r.source_message_id]}
        r.effective = True
        return {'state': 'authorized', 'reason': r.reason, 'asin': asin,
                'source_message_ids': [r.source_message_id],
                'issued_seq': r.issued_seq}
    reasons = sorted({r.reason for r in mine})
    return {'state': 'unknown', 'reason': '+'.join(reasons) or 'unknown',
            'asin': asin,
            'source_message_ids': [r.source_message_id for r in mine]}
