# GradAppReview

A multi-agent system that simulates a graduate admissions committee using LangGraph and multi-provider LLM orchestration. Built to explore agentic AI design patterns — specifically multi-model deliberation, role-based agent specialization, and iterative feedback loops.

## What It Does

Given a set of application documents (transcripts, SOP, resume, LinkedIn export, portfolio URLs), the system runs them through a pipeline that mirrors how a real admissions committee operates:

1. **Fact-Check Audit** — Cross-references all documents for date inconsistencies, unsubstantiated claims, contradictions, and gaps
2. **Judge Panel** (6 agents) — Each with a distinct evaluation lens and assigned a different LLM via round-robin rotation:
   - Academic Reviewer — GPA, coursework, research
   - Industry Professional — work experience, career trajectory
   - Admissions Officer — holistic application assessment
   - Diversity & Fit Evaluator — unique perspectives, cohort contribution
   - Writing Quality Analyst — SOP clarity, persuasiveness, authenticity
   - Devil's Advocate — stress-tests the application, finds reasons to reject
3. **Multi-Model Deliberation** — 3 premium-tier models independently synthesize all judge reviews into a verdict
4. **Meta-Deliberation** — Majority vote across deliberation models determines the final decision
5. **Feedback Loop** — If the verdict is "Revise," the system generates improvement feedback and re-runs judges (up to 3 iterations)

## Architecture

```
load_documents → fact_check → run_judges → deliberation → meta_deliberation
                                  ↑                              |
                                  └── generate_feedback ←────────┘
                                        (if verdict = Revise, max 3 iterations)
```

### Design Decisions

- **Multi-model by design** — Judges and deliberation panels use models from different providers (Anthropic, OpenAI, Google) to avoid single-model bias. LRU rotation ensures different combinations each run.
- **Tiered model strategy** — Standard models for judges (cost-efficient), premium models for deliberation (better reasoning). A `pro` mode uses flagship models everywhere.
- **LiteLLM** — Unified interface for all providers via a single `completion()` call.
- **Preflight checks** — Every model is pinged before the full run to catch API/billing issues early.
- **Resilient JSON parsing** — Auto-repairs common LLM JSON issues (truncated strings, unclosed brackets) and retries on failure.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone https://github.com/sarveshshah/GradAppReview.git
cd GradAppReview
uv sync

# Configure API keys (at least one provider required)
cp .env.example .env
# Edit .env with your keys
```

## Usage

Place application documents as `.txt` files in the `files/` directory. See [`files/NAMING.md`](files/NAMING.md) for naming conventions.

```bash
# Set up config files from templates
cp files/program_context.example.txt files/program_context.txt
cp files/lors.example.txt files/lors.txt
```

Edit `program_context.txt` with your target program's curriculum, admissions criteria, and any applicant-specific context. Edit `lors.txt` with your recommender details.

For website/portfolio content, add URLs to `files/urls.txt` (one per line) — they'll be scraped automatically.

```bash
# Test mode — cheap models, good for debugging
uv run python main.py --mode test

# Full mode — mid-tier judges, premium deliberation
uv run python main.py --mode full

# Pro mode — flagship models for everything
uv run python main.py --mode pro
```

### Modes

| Mode | Judge Models | Deliberation Models | Use Case |
|------|-------------|-------------------|----------|
| `test` | Haiku, GPT-4.1 Nano, Flash Lite | Sonnet, GPT-4.1 Mini, Flash | Development, debugging |
| `full` | Sonnet, GPT-4.1, Flash | Opus, GPT-5.4, Gemini 3.1 Pro | Standard assessment |
| `pro` | Opus, GPT-5.4, Gemini 3.1 Pro | Opus, GPT-5.4, Gemini 3.1 Pro | Maximum quality |

## Output

Reports are saved as structured Markdown in `outputs/review_TIMESTAMP.md` with:
- Fact-check audit results
- Individual judge reviews with scores, strengths, weaknesses
- Deliberation reasoning from each model
- Final verdict with confidence level
- Model assignment log for reproducibility

Run history is tracked in `outputs/run_history.json` to drive LRU model rotation.

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Stateful workflow orchestration with conditional edges
- **[LiteLLM](https://github.com/BerriAI/litellm)** — Unified multi-provider LLM interface
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)** + **[Markdownify](https://github.com/matthewwithanm/python-markdownify)** — Web scraping and HTML-to-text conversion

## Project Structure

```
.
├── main.py              # Entire application (single-file architecture)
├── files/               # Drop application documents here (.gitignored)
│   ├── NAMING.md        # Document naming conventions
│   ├── program_context.example.txt  # Template: target program details
│   ├── lors.example.txt             # Template: recommender info
│   └── urls.txt         # URLs to scrape for portfolio content
├── outputs/             # Generated reports and run history (.gitignored)
├── .env.example         # API key template
└── pyproject.toml       # Dependencies
```
