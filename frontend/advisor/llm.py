"""The only AI step: a local Ollama model explains the evidence packet. Code checks everything it says."""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import ollama

from . import config
from .schemas import ActionNote, Reason, ReportDraft, RunStats

SYSTEM = """You write short field reports for farmers and agronomists. You receive one JSON "packet" with measured
facts (evidence), reasons (drivers) and candidate actions that were all calculated by code.

Rules:
1. Use only facts in the packet. Never invent numbers, dates, places or products.
2. Every reason and every action explanation must cite evidence ids such as E3, copied exactly from the packet.
3. Pick actions only by their id from candidate_actions. Do not add new actions. Explain why each one matters in one
   or two short sentences, without repeating its full text.
4. Never name pesticide, fungicide, herbicide or fertilizer products and never give doses or rates.
5. Write in plain, short sentences. No jargon.
6. Never write ids (E3, A2, D1) inside sentences. Put evidence ids only in the evidence_ids lists.
7. Reply with JSON only, matching the requested schema."""

TASK = """Write the report for this packet.
- summary: two or three sentences. State the verdict and the confidence, and the single most important reason.
- reasons: three to five items. Cover every driver whose severity is 2. Cite evidence ids.
- actions: include every candidate action with must_include true, plus the others that matter (at most eight).
- open_questions: adapt the packet's open_questions (at most four).
- limitations: one or two sentences on what this report cannot tell the reader.
Only use numbers that appear in the packet."""

BANNED_WORDS = ["glyphosate", "tebuconazole", "prothioconazole", "azoxystrobin", "propiconazole", "epoxiconazole",
                "fluxapyroxad", "bixafen", "metconazole", "chlormequat", "paraquat", "mcpa", "2,4-d", "mancozeb",
                "chlorothalonil", "trifloxystrobin", "pyraclostrobin", "carbendazim", "imidacloprid"]
DOSE_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(kg|g|l|ml|litres?|liters?|t|tonnes?|lb|lbs)\s*(?:/|per)\s*(?:ha|hectare|acre|m2|m²)", re.I)
YIELD_WORDS = re.compile(r"yield|forecast|harvest|production|crop", re.I)
ID_PATTERN = re.compile(r"\b[EAD]\d+\b")
NUM_PATTERN = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")


# ----------------------------------------------------------------------------- server status
def client() -> ollama.Client:
    return ollama.Client(host=config.OLLAMA_HOST, timeout=240)


def status() -> dict:
    """Is the Ollama server up, and is the chosen model installed?"""
    out = {"server": False, "model": config.OLLAMA_MODEL, "installed": False, "version": None, "gpu_pct": None}
    try:
        c = client()
        names = {m.model for m in c.list().models}
        out["server"] = True
        out["installed"] = config.OLLAMA_MODEL in names or f"{config.OLLAMA_MODEL}:latest" in names
        for p in c.ps().models:
            if p.model == config.OLLAMA_MODEL and p.size:
                out["gpu_pct"] = round(100 * (p.size_vram or 0) / p.size)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:160]
    return out


# ----------------------------------------------------------------------------- calling the model
def _compact(packet: dict) -> str:
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def call_model(packet: dict, previous_errors: Optional[list[str]] = None) -> tuple[Optional[ReportDraft], dict]:
    """One attempt. Returns (draft or None, info). The reply is constrained to the ReportDraft JSON schema."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{TASK}\n\npacket:\n{_compact(packet)}"}]
    if previous_errors:
        messages.append({"role": "user", "content": "Your last reply failed these checks. Fix every one and reply again "
                         "with the full JSON:\n- " + "\n- ".join(previous_errors[:8])})
    prompt_chars = sum(len(m["content"]) for m in messages)
    # roughly 3 characters per token; leave room for the reply, and grow the window only when a packet needs it
    num_ctx = config.OLLAMA_NUM_CTX if prompt_chars / 3 + 2200 < config.OLLAMA_NUM_CTX else 8192
    kwargs = dict(model=config.OLLAMA_MODEL, messages=messages, format=ReportDraft.model_json_schema(),
                  options={"temperature": 0.2, "num_ctx": num_ctx, "num_predict": 2200},
                  keep_alive=config.OLLAMA_KEEP_ALIVE)
    t = time.time()
    info = {"seconds": 0.0, "prompt_tokens": 0, "output_tokens": 0, "error": None}
    try:
        try:
            resp = client().chat(think=False, **kwargs)        # skip the long hidden reasoning for structured replies
        except ollama.ResponseError as e:
            if "think" not in str(e).lower():
                raise
            resp = client().chat(**kwargs)                     # a model without thinking support
        info.update(seconds=time.time() - t, prompt_tokens=resp.prompt_eval_count or 0, output_tokens=resp.eval_count or 0)
        return ReportDraft.model_validate_json(resp.message.content), info
    except Exception as e:  # noqa: BLE001
        info.update(seconds=time.time() - t, error=f"{type(e).__name__}: {e}"[:300])
        return None, info


# ----------------------------------------------------------------------------- validator
def _numbers_in(text: str) -> list[tuple[float, int]]:
    out = []
    for tok in NUM_PATTERN.findall(ID_PATTERN.sub(" ", text)):
        tok = tok.replace(",", ".")
        out.append((float(tok), len(tok.split(".")[1]) if "." in tok else 0))
    return out


def allowed_numbers(packet: dict) -> list[float]:
    body = ID_PATTERN.sub(" ", json.dumps({k: v for k, v in packet.items() if k != "candidate_actions"}, ensure_ascii=False))
    nums = [float(t.replace(",", ".")) for t in NUM_PATTERN.findall(body)]
    for a in packet.get("candidate_actions", []):            # numbers inside action text (e.g. none today) are fair game
        nums += [float(t.replace(",", ".")) for t in NUM_PATTERN.findall(a["text"])]
    return nums


def validate(draft: ReportDraft, packet: dict) -> list[str]:
    """Return a list of problems. An empty list means the draft may be shown to the user."""
    errors: list[str] = []
    ev_ids = {e["id"] for e in packet["evidence"]}
    act = {a["id"]: a for a in packet["candidate_actions"]}

    texts = [("summary", draft.summary), ("limitations", draft.limitations)]
    texts += [(f"reason {i + 1}", r.text) for i, r in enumerate(draft.reasons)]
    texts += [(f"action {a.action_id}", a.explanation) for a in draft.actions]
    texts += [(f"question {i + 1}", q) for i, q in enumerate(draft.open_questions)]

    if not draft.summary.strip():
        errors.append("summary is empty")
    if not draft.reasons:
        errors.append("give at least one reason")
    if len(draft.reasons) > 6 or len(draft.actions) > 10 or len(draft.open_questions) > 5:
        errors.append("too many reasons, actions or questions")

    # ids must exist
    for i, r in enumerate(draft.reasons):
        bad = [x for x in r.evidence_ids if x not in ev_ids]
        if bad:
            errors.append(f"reason {i + 1} cites unknown evidence ids {bad}")
        if not r.evidence_ids:
            errors.append(f"reason {i + 1} cites no evidence")
    for a in draft.actions:
        if a.action_id not in act:
            errors.append(f"unknown action id {a.action_id}; use only ids from candidate_actions")
        bad = [x for x in a.evidence_ids if x not in ev_ids]
        if bad:
            errors.append(f"action {a.action_id} cites unknown evidence ids {bad}")

    # coverage
    chosen = {a.action_id for a in draft.actions}
    for aid, a in act.items():
        if a["must_include"] and aid not in chosen:
            errors.append(f"action {aid} is required but missing")
    cited = {x for r in draft.reasons for x in r.evidence_ids}
    for d in packet["drivers"]:
        if d["severity"] >= 2 and d["evidence"] and not (set(d["evidence"]) & cited):
            errors.append(f"no reason covers the major driver '{d['label']}' (cite {d['evidence']})")

    # ids belong in the id lists, not in the sentences
    for where, text in texts:
        if ID_PATTERN.search(text):
            errors.append(f"{where} contains a raw id like {ID_PATTERN.search(text).group(0)}; write plain words and keep ids in evidence_ids")

    # numbers, products and doses
    allowed = allowed_numbers(packet)
    for where, text in texts:
        low = text.lower()
        for w in BANNED_WORDS:
            if w in low:
                errors.append(f"{where} names a product or active ingredient ({w}); remove it")
        for m in DOSE_PATTERN.finditer(text):
            num, unit = float(m.group(1).replace(",", ".")), m.group(2).lower()
            # tonnes per hectare is fine when it restates a yield figure from the packet; any other rate is a dose
            is_yield = (unit in ("t", "tonne", "tonnes") and YIELD_WORDS.search(text)
                        and any(abs(num - a) <= 0.05 for a in allowed))
            if not is_yield:
                errors.append(f"{where} gives an application rate ({m.group(0)}); remove it")
        for x, d in _numbers_in(text):
            if not any(abs(x - a) <= 0.5 * 10 ** (-d) + 1e-9 for a in allowed):
                errors.append(f"{where} uses the number {x:g}, which is not in the packet")
    return list(dict.fromkeys(errors))


# ----------------------------------------------------------------------------- safe fallback
def fallback_draft(packet: dict) -> ReportDraft:
    """A plain report built only from the packet, used when the model fails its checks."""
    v = packet["verdict"]
    ev = {e["id"]: e for e in packet["evidence"]}
    sentence = {"Good": "The field looks healthy on the evidence available.",
                "Watch": "The field needs attention on the evidence available.",
                "Poor": "The field is in poor condition on the evidence available.",
                "Not enough information": "There is not enough information to judge the field yet."}[v["label"]]
    summary = f"{sentence} Confidence in this report is {v['confidence'].lower()}."
    y = next((e for e in packet["evidence"] if e["label"].startswith("Yield forecast")), None)
    if y:
        summary += f" The central yield forecast is {y['value']} t/ha ({y['note']})."
    drivers = sorted(packet["drivers"], key=lambda d: -d["severity"])
    reasons = [Reason(text=d["text"], evidence_ids=d["evidence"]) for d in drivers if d["severity"] >= 1 and d["evidence"]]
    if not reasons:
        reasons = [Reason(text=d["text"], evidence_ids=d["evidence"]) for d in drivers[:3] if d["evidence"]]
    actions = []
    for a in packet["candidate_actions"]:
        names = [ev[i]["label"] for i in a["evidence"] if i in ev]
        why = ("Based on: " + ", ".join(names) + ".") if names else "A routine step for every field."
        actions.append(ActionNote(action_id=a["id"], explanation=why, evidence_ids=a["evidence"]))
    return ReportDraft(
        summary=summary, reasons=reasons[:6], actions=actions[:10], open_questions=packet["open_questions"][:4],
        limitations=("This report comes from photos, weather and the details entered. It is decision support, "
                     "so confirm any fertilizer or spray decision with a local agronomist."))
