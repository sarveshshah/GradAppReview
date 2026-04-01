"""
AppReview — Multi-Agent Graduate Application Review System

A LangGraph-based pipeline that simulates a diverse admissions committee using
multiple LLM providers. Features 6 specialized judge agents, a cross-document
fact-checker, multi-model deliberation panel, and iterative feedback loops.

Configurable for any graduate program via files/program_context.txt.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime
from typing import Annotated, TypedDict

import litellm
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from markdownify import markdownify as md_convert

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Model pools — by mode (test = cheap/fast, full = premium)
# ---------------------------------------------------------------------------
MODEL_PROFILES = {
    "test": {
        "judge": [
            "anthropic/claude-haiku-4-5-20251001",  # Anthropic Haiku 4.5
            "gpt-4.1-nano",                          # OpenAI GPT-4.1 Nano
            "gemini/gemini-2.5-flash-lite",          # Google Gemini 2.0 Flash Lite
        ],
        "deliberation": [
            "anthropic/claude-sonnet-4-6",           # Sonnet for test deliberation
            "gpt-4.1-mini",                          # GPT-4.1 Mini
            "gemini/gemini-2.5-flash",               # Gemini 2.5 Flash
        ],
    },
    "full": {
        "judge": [
            "anthropic/claude-sonnet-4-6",           # Anthropic Sonnet 4.6
            "gpt-4.1",                               # OpenAI GPT-4.1
            "gemini/gemini-2.5-flash",               # Google Gemini 2.5 Flash
        ],
        "deliberation": [
            "anthropic/claude-opus-4-6",             # Anthropic Opus 4.6
            "gpt-5.4",                               # OpenAI GPT-5.4
            "gemini/gemini-3.1-pro-preview",         # Google Gemini 3.1 Pro Preview
        ],
    },
    "pro": {
        "judge": [
            "anthropic/claude-opus-4-6",             # Anthropic Opus 4.6
            "gpt-5.4",                               # OpenAI GPT-5.4
            "gemini/gemini-3.1-pro-preview",         # Google Gemini 3.1 Pro Preview
        ],
        "deliberation": [
            "anthropic/claude-opus-4-6",             # Anthropic Opus 4.6
            "gpt-5.4",                               # OpenAI GPT-5.4
            "gemini/gemini-3.1-pro-preview",         # Google Gemini 3.1 Pro Preview
        ],
    },
}

# Active pools — set at runtime based on --mode
JUDGE_MODELS: list[str] = []
DELIBERATION_MODELS: list[str] = []

KEY_MAP = {
    "anthropic/claude-haiku-4-5-20251001": "ANTHROPIC_API_KEY",
    "anthropic/claude-sonnet-4-6": "ANTHROPIC_API_KEY",
    "anthropic/claude-opus-4-6": "ANTHROPIC_API_KEY",
    "gpt-4.1-nano": "OPENAI_API_KEY",
    "gpt-4.1-mini": "OPENAI_API_KEY",
    "gpt-4.1": "OPENAI_API_KEY",
    "gpt-5.4": "OPENAI_API_KEY",
    "gemini/gemini-2.5-flash-lite": "GOOGLE_API_KEY",
    "gemini/gemini-2.5-flash": "GOOGLE_API_KEY",
    "gemini/gemini-3.1-pro-preview": "GOOGLE_API_KEY",
}

MAX_ITERATIONS = 3
HISTORY_PATH = os.path.join("outputs", "run_history.json")

# ---------------------------------------------------------------------------
# Program context & LOR loading
# ---------------------------------------------------------------------------
DEFAULT_PROGRAM = "Graduate Program"  # overridden at startup from program_context.txt
_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")


def _load_config_file(filename: str, fallback: str = "") -> str:
    """Load a config text file from the files/ directory."""
    path = os.path.join(_FILES_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return fallback


def _load_program_context() -> tuple[str, str]:
    """Load program context from files/program_context.txt.
    Returns (program_name, full_context). Parses name from first PROGRAM: line."""
    content = _load_config_file("program_context.txt")
    if content:
        # Extract program name from "PROGRAM: ..." line
        for line in content.splitlines():
            if line.strip().upper().startswith("PROGRAM:"):
                name = line.split(":", 1)[1].strip()
                return name, content
        return "Graduate Program", content
    return "Graduate Program", (
        "No program context file found. "
        "Evaluate the applicant based on typical graduate admissions criteria: "
        "academic record, relevant experience, statement quality, program fit, "
        "and letters of recommendation."
    )


def _load_lors() -> str:
    """Load LOR descriptions from files/lors.txt, or return a generic note."""
    content = _load_config_file("lors.txt")
    if content:
        return content
    return "No letter of recommendation information provided."


# ---------------------------------------------------------------------------
# Extracted text disclaimer
# ---------------------------------------------------------------------------
EXTRACTED_TEXT_NOTICE = (
    "IMPORTANT: Documents below are extracted/pasted text. Ignore formatting issues, "
    "layout problems, or visual artifacts. Focus ONLY on substantive content.\n\n"
    "DATE CONTEXT: Today is {date}. All dates in documents are real past/present "
    "dates — none are future. Do NOT flag any date as inconsistent.\n\n"
).format(date=datetime.now().strftime("%B %d, %Y"))

# ---------------------------------------------------------------------------
# Document categorization
# ---------------------------------------------------------------------------
DOC_CATEGORIES = {
    "transcript": "Academic Transcript",
    "sop": "Statement of Purpose",
    "statement": "Statement of Purpose",
    "purpose": "Statement of Purpose",
    "essay": "Personal Essay",
    "resume": "Resume/CV",
    "cv": "Resume/CV",
    "linkedin": "LinkedIn Profile",
    "website": "Personal Website/Portfolio",
    "portfolio": "Personal Website/Portfolio",
}


def _categorize_doc(name: str) -> str:
    lower = name.lower()
    for prefix, category in DOC_CATEGORIES.items():
        if prefix in lower:
            return category
    return "Application Material"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_models(tier: str = "judge") -> list[str]:
    pool = JUDGE_MODELS if tier == "judge" else DELIBERATION_MODELS
    available = [m for m in pool if os.getenv(KEY_MAP.get(m, ""), "")]
    if not available:
        sys.exit(f"No API keys found for {tier} tier models. Check your .env file.")
    return available


def _short(model: str) -> str:
    return model.split("/")[-1]


def _call_llm(model: str, system: str, user: str) -> str:
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    return resp.choices[0].message.content


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Attempt repairs for common LLM JSON issues:
    # 1. Truncated output — try closing open strings/arrays/objects
    repaired = raw
    # Close any unterminated string (odd number of unescaped quotes)
    if repaired.count('"') % 2 != 0:
        repaired += '"'
    # Balance brackets
    open_b = repaired.count("[") - repaired.count("]")
    repaired += "]" * max(open_b, 0)
    open_c = repaired.count("{") - repaired.count("}")
    repaired += "}" * max(open_c, 0)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # 2. Extract first JSON object with regex (skip preamble text)
    m = re.search(r"\{", raw)
    if m:
        candidate = raw[m.start():]
        # Try progressively trimming from the end
        for trim in ("", '"]}', '"]}'[:2], '"]}'[:1]):
            try:
                return json.loads(candidate + trim)
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError("Could not parse LLM response as JSON", raw, 0)


def _scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AppReview/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = md_convert(str(soup), strip=["img"])
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# Run history — LRU model rotation for judges
# ---------------------------------------------------------------------------


def _load_history() -> list[dict]:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def _save_history(history: list[dict]):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _assign_judges_lru(available: list[str]) -> dict[str, str]:
    """
    Distribute models across judges so each run uses a mix of providers.
    Models rotate via LRU: the starting offset shifts each run so that
    over multiple runs every judge sees every model.
    """
    history = _load_history()
    judge_names = list(JUDGES.keys())
    n = len(available)

    # Determine rotation offset: count how many past runs exist and shift
    offset = len(history) % n

    assignments = {}
    for idx, judge in enumerate(judge_names):
        assignments[judge] = available[(idx + offset) % n]

    return assignments


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_check() -> tuple[list[str], list[str]]:
    """Ping all models. Returns (passed_judges, passed_deliberation)."""
    print("\n" + "=" * 70)
    print("  PREFLIGHT CHECK")
    print("=" * 70)

    def check(pool: list[str], label: str) -> list[str]:
        passed = []
        for model in pool:
            key_env = KEY_MAP.get(model, "")
            if not os.getenv(key_env, ""):
                print(f"  SKIP  {_short(model):<35} {key_env} not set")
                continue
            try:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": "Reply OK."}],
                    max_tokens=5, temperature=0,
                )
                print(f"  OK    {_short(model):<35} {label}")
                passed.append(model)
            except Exception as e:
                reason = str(e).splitlines()[0][:80]
                print(f"  FAIL  {_short(model):<35} {reason}")
        return passed

    j = check(JUDGE_MODELS, "judge")
    d = check(DELIBERATION_MODELS, "deliberation")

    if not j:
        sys.exit("\n  No judge models available. Check API keys.")
    if not d:
        sys.exit("\n  No deliberation models available. Check API keys.")

    print(f"\n  Judges:       {len(j)}/{len(JUDGE_MODELS)} ready")
    print(f"  Deliberation: {len(d)}/{len(DELIBERATION_MODELS)} ready")
    print("=" * 70)
    return j, d


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _replace(existing, new):
    return new if new else existing


class ReviewState(TypedDict):
    documents: dict[str, str]
    doc_categories: dict[str, str]
    program: str
    fact_check: str  # cross-document consistency report
    # judge_name -> review dict (one model per judge now)
    reviews: Annotated[dict[str, dict], _replace]
    assignments: dict[str, str]   # judge_name -> model used
    deliberations: Annotated[list[dict], _replace]
    final_report: str
    decision: str
    feedback: str
    iteration: int


# ---------------------------------------------------------------------------
# Judge definitions
# ---------------------------------------------------------------------------

JUDGES = {
    "Academic Reviewer": (
        "You are a university professor evaluating a masters program application.\n"
        "Focus on: academic preparedness, GPA context, research experience, "
        "relevant coursework, achievements, intellectual curiosity.\n"
        "WEIGHT MOST: Academic Transcripts, Statement of Purpose.\n"
        "ALSO CONSIDER: Resume/CV for research or academic projects."
    ),
    "Industry Professional": (
        "You are a senior AI/ML industry professional.\n"
        "Focus on: work experience relevance, practical skills, career trajectory, "
        "industry knowledge, leadership, degree fit with career goals.\n"
        "WEIGHT MOST: Resume/CV, LinkedIn Profile.\n"
        "ALSO CONSIDER: Website/Portfolio, Statement of Purpose for goals."
    ),
    "Admissions Officer": (
        "You are an experienced graduate admissions officer.\n"
        "Focus on: application completeness, program fit, goal alignment, "
        "realistic expectations, comparison to typical successful applicants.\n"
        "WEIGHT ALL DOCUMENTS EQUALLY — assess the holistic package.\n"
        "Flag gaps, inconsistencies, or red flags."
    ),
    "Diversity & Fit Evaluator": (
        "You are a holistic admissions evaluator.\n"
        "Focus on: unique perspectives, community contributions, leadership, "
        "overcoming challenges, cultural fit, potential to enrich the cohort.\n"
        "WEIGHT MOST: Statement of Purpose, LinkedIn, Website/Portfolio.\n"
        "ALSO CONSIDER: Resume/CV for non-traditional experiences."
    ),
    "Writing Quality Analyst": (
        "You are a writing and communications expert.\n"
        "Focus on: clarity, logical structure, persuasiveness, grammar, "
        "authenticity, whether writing conveys the candidate's story.\n"
        "WEIGHT MOST: Statement of Purpose (primary writing sample).\n"
        "Documents are extracted text — ignore formatting issues."
    ),
    "Devil's Advocate": (
        "You are a skeptical contrarian reviewer whose JOB is to find reasons to REJECT.\n"
        "Assume other reviewers are being too generous. Your role is to stress-test the application.\n"
        "Focus on: weaknesses others might gloss over, unsubstantiated claims, gaps in evidence, "
        "red flags, overreliance on narrative vs. hard credentials, grade inflation, "
        "whether stated goals are realistic, and how this candidate compares to a strong rejected applicant.\n"
        "You MUST score lower than you normally would — a 7 from you means genuinely strong.\n"
        "WEIGHT ALL DOCUMENTS — look for what's MISSING, not what's present.\n"
        "If the application is truly exceptional, say so, but justify every point with evidence."
    ),
}

JUDGE_OUTPUT_FORMAT = """
Respond in this exact JSON format (no markdown fences):
{
    "score": <1-10>,
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...],
    "recommendation": "Accept" | "Revise" | "Reject",
    "reasoning": "2-3 sentence summary of your assessment"
}
"""

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def load_documents(state: ReviewState) -> dict:
    return {"iteration": state.get("iteration", 0)}


def run_fact_check(state: ReviewState) -> dict:
    """Cross-reference all documents for inconsistencies and flag issues."""
    available = _available_models("judge")
    # Use a random available model for the fact check
    model = random.choice(available)
    program = state.get("program", DEFAULT_PROGRAM)
    doc_cats = state.get("doc_categories", {})

    docs_parts = []
    for name, text in state["documents"].items():
        cat = doc_cats.get(name, "Application Material")
        docs_parts.append(f"## {cat}: {name}\n{text}")
    docs_text = "\n\n---\n\n".join(docs_parts)

    system = (
        f"{EXTRACTED_TEXT_NOTICE}"
        f"You are a meticulous fact-checker and consistency auditor for {program} applications.\n\n"
        "Your ONLY job is to cross-reference ALL documents against each other and flag:\n"
        "1. DATE INCONSISTENCIES: Do resume dates match transcript dates? Do employment timelines overlap logically?\n"
        "2. CLAIM VERIFICATION: Are SOP claims backed by evidence in resume/LinkedIn/portfolio?\n"
        "3. CONTRADICTIONS: Does any document contradict another?\n"
        "4. EXAGGERATIONS: Are quantified achievements (e.g., cost savings, metrics) plausible given the role/tenure?\n"
        "5. GAPS: Missing information that other documents suggest should exist.\n\n"
        "Do NOT evaluate the application's merit — only check factual consistency.\n"
        "If everything checks out, say so explicitly."
    )

    user_msg = (
        f"Cross-reference these application documents for {program}:\n\n"
        f"{docs_text}\n\n"
        "Respond in plain text with a structured consistency report. "
        "Use headers: DATE CHECK, CLAIM VERIFICATION, CONTRADICTIONS, EXAGGERATIONS, GAPS. "
        "Under each, list findings or state 'No issues found.'"
    )

    print(f"\n  [Fact Check] Running consistency audit ({_short(model)})...")
    try:
        report = _call_llm(model, system, user_msg)
    except Exception as e:
        report = f"Fact check failed: {e}"
    print(f"  [Fact Check] Complete.")
    return {"fact_check": report}


def run_judges(state: ReviewState) -> dict:
    """Run judges, each with one LRU-assigned standard model."""
    available = _available_models("judge")
    assignments = _assign_judges_lru(available)
    program = state.get("program", DEFAULT_PROGRAM)
    _, program_info = _load_program_context()
    lors = _load_lors()
    doc_cats = state.get("doc_categories", {})

    docs_parts = []
    for name, text in state["documents"].items():
        cat = doc_cats.get(name, "Application Material")
        docs_parts.append(f"## {cat}: {name}\n{text}")
    docs_text = "\n\n---\n\n".join(docs_parts)

    feedback_ctx = ""
    if state.get("feedback"):
        feedback_ctx = (
            f"\n\n## Previous Feedback (Iteration {state['iteration']})\n"
            f"{state['feedback']}"
        )

    print(f"\n  Judge assignments this run:")
    for j, m in assignments.items():
        print(f"    {j:<25} → {_short(m)}")
    print()

    fact_check = state.get("fact_check", "")
    fact_check_ctx = ""
    if fact_check:
        fact_check_ctx = (
            f"\n\nFACT-CHECK REPORT (cross-document consistency audit):\n"
            f"{fact_check}\n"
            f"Use the above to inform your review — penalize verified inconsistencies."
        )

    n_judges = len(JUDGES)
    reviews: dict[str, dict] = {}
    for i, (judge_name, system_prompt) in enumerate(JUDGES.items(), 1):
        model = assignments[judge_name]
        print(f"  [{i}/{n_judges}] {judge_name} ({_short(model)})...")

        full_system = (
            f"{EXTRACTED_TEXT_NOTICE}"
            f"TARGET PROGRAM: {program}\n\n"
            f"PROGRAM DETAILS:\n{program_info}\n\n"
            f"LETTERS OF RECOMMENDATION:\n{lors}\n\n"
            f"YOUR ROLE:\n{system_prompt}\n\nBe rigorous but fair."
        )
        user_msg = (
            f"Review the following application materials for {program}.\n\n"
            f"Documents provided:\n"
            + "\n".join(f"  - {doc_cats.get(n, 'Material')}: {n}" for n in state["documents"])
            + f"\n\n{docs_text}{feedback_ctx}{fact_check_ctx}\n\n{JUDGE_OUTPUT_FORMAT}"
        )

        try:
            raw = _call_llm(model, full_system, user_msg)
            review = _parse_json(raw)
            review["judge"] = judge_name
            review["model"] = model
        except json.JSONDecodeError:
            # Retry once — ask model to fix its own output
            try:
                print(f"    ↳ Parse error, retrying {judge_name}...")
                raw = _call_llm(model, full_system, user_msg)
                review = _parse_json(raw)
                review["judge"] = judge_name
                review["model"] = model
            except (json.JSONDecodeError, Exception) as e2:
                review = {
                    "judge": judge_name, "model": model,
                    "score": 5, "strengths": [], "weaknesses": [f"Parse error: {e2}"],
                    "recommendation": "Revise", "reasoning": str(e2),
                }
        except Exception as e:
            review = {
                "judge": judge_name, "model": model,
                "score": 5, "strengths": [], "weaknesses": [f"Error: {e}"],
                "recommendation": "Revise", "reasoning": str(e),
            }
        reviews[judge_name] = review

    # Print compact summary
    print(f"\n  {'Judge':<25} {'Model':<20} {'Score':>5}  Rec")
    print(f"  {'-'*25} {'-'*20} {'-'*5}  {'-'*6}")
    for j, r in reviews.items():
        print(f"  {j:<25} {_short(r['model']):<20} {r.get('score','?'):>5}  {r.get('recommendation','?')}")

    return {"reviews": reviews, "assignments": assignments}


def run_deliberation(state: ReviewState) -> dict:
    """All premium models independently deliberate on the 5 judge reviews."""
    available = _available_models("deliberation")
    program = state.get("program", DEFAULT_PROGRAM)
    lors = _load_lors()

    reviews_text = json.dumps(list(state["reviews"].values()), indent=2)

    system = (
        f"{EXTRACTED_TEXT_NOTICE}"
        f"You are a senior admissions committee member for {program}, "
        f"synthesizing 5 judge reviews into a final assessment.\n\n"
        f"LETTERS OF RECOMMENDATION:\n{lors}\n\n"
        "Analyze agreements and disagreements between judge roles.\n"
        "Weight your assessment carefully across all perspectives."
    )

    delib_format = """
Respond in this exact JSON format (no markdown fences):
{
    "overall_score": <1-10>,
    "verdict": "Accept" | "Revise" | "Reject",
    "consensus_strengths": ["str1", ...],
    "consensus_weaknesses": ["weak1", ...],
    "disagreements": ["where judges disagreed", ...],
    "reasoning": "2-3 sentence synthesis"
}
"""

    user_msg = (
        f"Reviews from 5 judges:\n\n{reviews_text}\n\n"
        f"Synthesize into a final assessment for {program}.{delib_format}"
    )

    deliberations = []
    for i, model in enumerate(available, 1):
        print(f"  [Deliberation {i}/{len(available)}] {_short(model)}...")
        try:
            raw = _call_llm(model, system, user_msg)
            delib = _parse_json(raw)
            delib["model"] = model
        except json.JSONDecodeError:
            try:
                print(f"    ↳ Parse error, retrying {_short(model)}...")
                raw = _call_llm(model, system, user_msg)
                delib = _parse_json(raw)
                delib["model"] = model
            except (json.JSONDecodeError, Exception) as e2:
                delib = {"model": model, "overall_score": 5, "verdict": "Revise", "reasoning": str(e2)}
        except Exception as e:
            delib = {"model": model, "overall_score": 5, "verdict": "Revise", "reasoning": str(e)}
        deliberations.append(delib)

    return {"deliberations": deliberations}


def meta_deliberation(state: ReviewState) -> dict:
    """Majority vote + compact terminal report."""
    deliberations = state["deliberations"]
    program = state.get("program", DEFAULT_PROGRAM)

    verdicts = [d.get("verdict", "Revise") for d in deliberations]
    scores = [d.get("overall_score", 5) for d in deliberations]
    vote_counts = Counter(verdicts)
    majority_verdict, majority_count = vote_counts.most_common(1)[0]

    if majority_count >= 2:
        decision = majority_verdict
        confidence = "HIGH" if majority_count == len(deliberations) else "MODERATE"
    else:
        decision = "Revise"
        confidence = "LOW (split — forcing revision)"

    avg_score = sum(scores) / len(scores) if scores else 5

    # Compact terminal output
    print(f"\n{'=' * 70}")
    print(f"  VERDICT: {decision}  |  Score: {avg_score:.1f}/10  |  Confidence: {confidence}")
    print(f"  Votes: {dict(vote_counts)}")
    print(f"{'=' * 70}")
    for d in deliberations:
        print(f"  {_short(d['model']):<25} {d.get('verdict','?'):<8} {d.get('overall_score','?')}/10")
    print(f"{'=' * 70}")

    return {
        "final_report": f"{decision} ({avg_score:.1f}/10, {confidence})",
        "decision": decision,
        "iteration": state["iteration"] + 1,
    }


def generate_feedback(state: ReviewState) -> dict:
    available = _available_models("deliberation")
    model = random.choice(available)
    program = state.get("program", DEFAULT_PROGRAM)
    lors = _load_lors()
    doc_cats = state.get("doc_categories", {})

    reviews_text = json.dumps(list(state["reviews"].values()), indent=2)
    docs_parts = []
    for name, text in state["documents"].items():
        cat = doc_cats.get(name, "Application Material")
        docs_parts.append(f"## {cat}: {name}\n{text}")
    docs_text = "\n\n---\n\n".join(docs_parts)

    system = (
        f"{EXTRACTED_TEXT_NOTICE}"
        f"You are an expert admissions consultant for {program}.\n\n"
        f"LETTERS OF RECOMMENDATION:\n{lors}\n\n"
        "Provide specific, actionable feedback. Focus on weaknesses multiple judges agree on."
    )
    user_msg = (
        f"The application received 'Revise'. Documents:\n{docs_text}\n\n"
        f"Judge reviews:\n{reviews_text}\n\n"
        "Provide prioritized feedback:\n"
        "1. TOP PRIORITY (flagged by most judges)\n"
        "2. MODERATE (flagged by some)\n"
        "3. MINOR (flagged by few)\n"
        "For each, cite which judge(s) flagged it."
    )

    print(f"\n  [Feedback] {_short(model)}...")
    feedback = _call_llm(model, system, user_msg)
    print(f"  Feedback generated ({len(feedback)} chars)")

    return {"feedback": feedback}


def should_continue(state: ReviewState) -> str:
    if state["decision"] == "Revise" and state["iteration"] < MAX_ITERATIONS:
        return "feedback"
    return "done"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    g = StateGraph(ReviewState)
    g.add_node("load_documents", load_documents)
    g.add_node("run_fact_check", run_fact_check)
    g.add_node("run_judges", run_judges)
    g.add_node("run_deliberation", run_deliberation)
    g.add_node("meta_deliberation", meta_deliberation)
    g.add_node("generate_feedback", generate_feedback)

    g.set_entry_point("load_documents")
    g.add_edge("load_documents", "run_fact_check")
    g.add_edge("run_fact_check", "run_judges")
    g.add_edge("run_judges", "run_deliberation")
    g.add_edge("run_deliberation", "meta_deliberation")
    g.add_conditional_edges(
        "meta_deliberation", should_continue,
        {"feedback": "generate_feedback", "done": END},
    )
    g.add_edge("generate_feedback", "run_judges")
    return g.compile()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def save_report(state: ReviewState) -> str:
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join("outputs", f"review_{ts}.md")

    program = state.get("program", DEFAULT_PROGRAM)
    reviews = state.get("reviews", {})
    deliberations = state.get("deliberations", [])
    assignments = state.get("assignments", {})
    decision = state.get("decision", "N/A")
    feedback = state.get("feedback", "")
    iterations = state.get("iteration", 0)

    verdicts = [d.get("verdict", "?") for d in deliberations]
    scores = [d.get("overall_score", 0) for d in deliberations]
    avg = sum(scores) / len(scores) if scores else 0
    vote_counts = Counter(verdicts)

    L = []  # lines
    L.append(f"# Application Review: {program}\n")
    L.append(f"**Date:** {datetime.now().strftime('%B %d, %Y')}  ")
    L.append(f"**Iterations:** {iterations}  ")
    L.append("")

    # Verdict
    emoji = {"Accept": "ACCEPTED", "Reject": "REJECTED", "Revise": "REVISE"}.get(decision, decision)
    L.append(f"## Result: {emoji}\n")
    L.append(f"**Average score:** {avg:.1f}/10  ")
    L.append(f"**Vote breakdown:** {dict(vote_counts)}\n")

    # Deliberation votes table
    L.append("| Deliberation Model | Verdict | Score |")
    L.append("|---|---|---|")
    for d in deliberations:
        L.append(f"| {_short(d['model'])} | {d.get('verdict','?')} | {d.get('overall_score','?')}/10 |")
    L.append("")

    # Consensus from deliberations
    strengths: set[str] = set()
    weaknesses: set[str] = set()
    disagreements: set[str] = set()
    for d in deliberations:
        strengths.update(d.get("consensus_strengths", []))
        weaknesses.update(d.get("consensus_weaknesses", []))
        disagreements.update(d.get("disagreements", []))

    if strengths:
        L.append("### Strengths (consensus)\n")
        for s in sorted(strengths):
            L.append(f"- {s}")
        L.append("")

    if weaknesses:
        L.append("### Weaknesses (consensus)\n")
        for w in sorted(weaknesses):
            L.append(f"- {w}")
        L.append("")

    if disagreements:
        L.append("### Disagreements Between Judges\n")
        for x in sorted(disagreements):
            L.append(f"- {x}")
        L.append("")

    # Fact check report
    fact_check = state.get("fact_check", "")
    if fact_check:
        L.append("---\n")
        L.append("## Fact-Check & Consistency Audit\n")
        L.append(fact_check)
        L.append("")

    # Deliberation reasoning
    L.append("---\n")
    L.append("## Deliberation Reasoning\n")
    for d in deliberations:
        L.append(f"### {_short(d['model'])} — {d.get('verdict','?')} ({d.get('overall_score','?')}/10)\n")
        L.append(d.get("reasoning", "N/A"))
        L.append("")

    # Judge reviews
    L.append("---\n")
    L.append("## Judge Reviews\n")
    L.append("| Judge | Model | Score | Recommendation |")
    L.append("|---|---|---|---|")
    for j, r in reviews.items():
        L.append(f"| {j} | {_short(r.get('model','?'))} | {r.get('score','?')}/10 | {r.get('recommendation','?')} |")
    L.append("")

    for judge_name, review in reviews.items():
        L.append(f"### {judge_name} ({_short(review.get('model','?'))}) — {review.get('score','?')}/10\n")
        L.append(f"**Recommendation:** {review.get('recommendation','?')}  ")
        L.append(f"**Reasoning:** {review.get('reasoning', 'N/A')}\n")

        s = review.get("strengths", [])
        w = review.get("weaknesses", [])
        if s:
            L.append("**Strengths:**")
            for item in s:
                L.append(f"- {item}")
            L.append("")
        if w:
            L.append("**Weaknesses:**")
            for item in w:
                L.append(f"- {item}")
            L.append("")

    # Feedback
    if feedback:
        L.append("---\n")
        L.append("## Improvement Feedback\n")
        L.append(feedback)
        L.append("")

    # Model assignments log
    L.append("---\n")
    L.append("## Run Configuration\n")
    L.append("| Judge | Assigned Model |")
    L.append("|---|---|")
    for j, m in assignments.items():
        L.append(f"| {j} | {_short(m)} |")
    L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md", ".text"}


def load_from_files(paths: list[str]) -> dict[str, str]:
    docs = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  Warning: {p} not found, skipping.")
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            docs[os.path.basename(p)] = f.read()
    if not docs:
        sys.exit("No valid documents found.")
    return docs


def load_from_dir(directory: str) -> dict[str, str]:
    if not os.path.isdir(directory):
        sys.exit(f"Directory not found: {directory}")
    docs = {}
    for fname in sorted(os.listdir(directory)):
        _, ext = os.path.splitext(fname)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            docs[fname] = f.read()
        print(f"  Loaded '{fname}' ({len(docs[fname])} chars)")
    if not docs:
        sys.exit(f"No supported files in '{directory}'. See files/NAMING.md for help.")
    return docs


def load_from_urls(urls: list[str]) -> dict[str, str]:
    from urllib.parse import urlparse
    docs = {}
    for url in urls:
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        print(f"  Scraping {url}...")
        try:
            text = _scrape_url(url)
            parsed = urlparse(url)
            path_slug = parsed.path.strip("/").replace("/", "_") or "home"
            doc_name = f"website_{parsed.netloc}_{path_slug}"
            docs[doc_name] = text
            print(f"    -> '{doc_name}' ({len(text)} chars)")
        except Exception as e:
            print(f"    -> Failed: {e}")
    return docs


def load_from_urls_file(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        sys.exit(f"URLs file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if not urls:
        sys.exit(f"No URLs found in {path}")
    print(f"  Found {len(urls)} URL(s) in {path}")
    return load_from_urls(urls)


def load_interactive() -> dict[str, str]:
    docs = {}
    print("Enter documents. Type 'done' to finish each, 'quit' to stop.\n")
    while True:
        name = input("Document name (e.g., 'resume', 'sop'): ").strip()
        if name.lower() == "quit":
            break
        print(f"Paste {name} (type 'done' when finished):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "done":
                break
            lines.append(line)
        docs[name] = "\n".join(lines)
        print(f"  -> Saved '{name}' ({len(docs[name])} chars)\n")
    if not docs:
        sys.exit("No documents provided.")
    return docs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Masters Program Eligibility Review — Multi-Agent Judge Panel"
    )
    parser.add_argument("--dir", default=None, const="files", nargs="?",
                        help="Load all files from directory (default: ./files/)")
    parser.add_argument("--files", nargs="+", help="Explicit file paths")
    parser.add_argument("--urls", nargs="+", help="URLs to scrape")
    parser.add_argument("--urls-file", default=None, metavar="FILE",
                        help="Text file with URLs (auto-detects files/urls.txt)")
    parser.add_argument("--interactive", action="store_true", help="Paste interactively")
    parser.add_argument("--mode", choices=["test", "full", "pro"], default="test",
                        help="test = cheap models for debugging, full = premium models for real assessment")
    args = parser.parse_args()

    if not args.urls and not args.urls_file and os.path.exists("files/urls.txt"):
        args.urls_file = "files/urls.txt"

    if not args.dir and not args.files and not args.urls and not args.urls_file and not args.interactive:
        parser.print_help()
        print("\nExamples:")
        print("  uv run python main.py --dir")
        print("  uv run python main.py --dir --urls-file files/urls.txt")
        print("  uv run python main.py --files sop.txt resume.txt")
        sys.exit(1)

    # Load documents
    documents: dict[str, str] = {}
    if args.dir:
        documents.update(load_from_dir(args.dir))
    if args.files:
        documents.update(load_from_files(args.files))
    if args.urls:
        documents.update(load_from_urls(args.urls))
    if args.urls_file:
        documents.update(load_from_urls_file(args.urls_file))
    if args.interactive:
        documents.update(load_interactive())

    if not documents:
        sys.exit("No documents loaded.")

    doc_categories = {name: _categorize_doc(name) for name in documents}

    # Load program name from config
    program_name, _ = _load_program_context()

    # Set model pools based on --mode
    profile = MODEL_PROFILES[args.mode]
    JUDGE_MODELS[:] = profile["judge"]
    DELIBERATION_MODELS[:] = profile["deliberation"]

    # Preflight
    verified_judges, verified_delib = preflight_check()
    JUDGE_MODELS[:] = verified_judges
    DELIBERATION_MODELS[:] = verified_delib

    # Summary
    mode_labels = {"test": "TEST (cheap models)", "full": "FULL (premium models)", "pro": "PRO (all flagship models)"}
    mode_label = mode_labels[args.mode]
    print(f"\n{'=' * 70}")
    print(f"  REVIEW: {program_name}  [{mode_label}]")
    print(f"{'=' * 70}")
    print(f"  Documents: {len(documents)}")
    for name, cat in doc_categories.items():
        print(f"    {cat}: {name} ({len(documents[name])} chars)")
    print(f"  Judge models:  {', '.join(_short(m) for m in verified_judges)}")
    print(f"  Delib models:  {', '.join(_short(m) for m in verified_delib)}")
    n_judges = len(JUDGES)
    print(f"  Calls/iter:    {n_judges} judge + 1 fact-check + {len(verified_delib)} deliberation = {n_judges + 1 + len(verified_delib)}")
    print(f"  Max iters:     {MAX_ITERATIONS}")
    print(f"{'=' * 70}")

    graph = build_graph()
    initial_state: ReviewState = {
        "documents": documents,
        "doc_categories": doc_categories,
        "program": program_name,
        "fact_check": "",
        "reviews": {},
        "assignments": {},
        "deliberations": [],
        "final_report": "",
        "decision": "",
        "feedback": "",
        "iteration": 0,
    }

    final_state = graph.invoke(initial_state)

    # Save run history
    history = _load_history()
    history.append({
        "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "assignments": final_state.get("assignments", {}),
        "verdict": final_state.get("decision", "?"),
        "avg_score": (
            sum(d.get("overall_score", 0) for d in final_state.get("deliberations", []))
            / max(len(final_state.get("deliberations", [])), 1)
        ),
        "iteration": final_state.get("iteration", 0),
    })
    _save_history(history)

    # Save markdown report
    report_path = save_report(final_state)

    print(f"\n{'=' * 70}")
    print(f"  FINAL: {final_state['decision']}  |  Iterations: {final_state['iteration']}")
    print(f"  Report: {report_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
