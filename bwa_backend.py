from __future__ import annotations

import base64
import json
import operator
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()


# ============================================================
# Configuration
# ============================================================

# Gemini 3.5 Flash-Lite
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Cloudflare Workers AI image model
CLOUDFLARE_IMAGE_MODEL = os.getenv(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-1-schnell",
)

# Anchored to this file so the frontend (same folder) always finds the output.
OUTPUT_ROOT = Path(__file__).resolve().parent / "blog_output"

MIN_SECTION_WORDS = 150
MAX_SECTION_WORDS = 400
TARGET_SECTION_WORDS = 275

MIN_SECTIONS = 4
MAX_SECTIONS = 6
MAX_IMAGES = 3
MAX_EVIDENCE = 25


# ============================================================
# 1. Schemas
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should understand.",
    )
    bullets: list[str] = Field(
        default_factory=list,
        description="3-6 concrete points the section must cover.",
    )
    target_words: int = TARGET_SECTION_WORDS
    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = True
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(
        ...,
        description=f"{MIN_SECTIONS}-{MAX_SECTIONS} ordered section tasks.",
    )


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: list[str] = Field(default_factory=list)
    max_results_per_query: int = 5


class ImageLabel(BaseModel):
    text: str = Field(..., description="Exact short technical label to render. 2-5 words.")
    anchor_x: float = Field(0.5, ge=0.08, le=0.92)
    anchor_y: float = Field(0.5, ge=0.12, le=0.88)
    side: Literal["left", "right", "top", "bottom"] = "right"


class ImageSpec(BaseModel):
    placeholder: str = Field(default="", description="Ignored; placeholders are assigned by the program.")
    section_title: str = Field(..., description="Exact title of the ## section whose first paragraph the image should follow.")
    filename: str = Field(..., description="Short descriptive image filename ending in .png.")
    alt: str
    caption: str
    prompt: str = Field(..., description="Visual-only prompt for FLUX. Never request text.")
    title: str = Field(default="", description="Short technical title, at most 7 words.")
    labels: list[ImageLabel] = Field(default_factory=list, description="Up to 6 exact labels to overlay around the visual.")


class GlobalImagePlan(BaseModel):
    images: list[ImageSpec] = Field(default_factory=list)


class State(TypedDict, total=False):
    # Inputs supplied by the frontend
    topic: str
    blog_style: str
    as_of: str
    audience: str
    tone: str
    include_images: bool

    # Routing / research
    mode: str
    needs_research: bool
    queries: list[str]
    evidence: list[EvidenceItem]
    recency_days: int

    # Planning
    plan: Optional[Plan]

    # Parallel workers append (task_id, section_markdown) pairs.
    sections: Annotated[list[tuple[int, str]], operator.add]

    # Reducer stages
    merged_md: str
    md_with_placeholders: str
    image_specs: list[dict]

    # Outputs
    final: str
    markdown_path: str
    output_dir: str
    logs: Annotated[list[str], operator.add]


# ============================================================
# 2. Gemini text model
# ============================================================

google_api_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not google_api_key:
    raise RuntimeError(
        "Google API key is missing. Add GOOGLE_API_KEY or GEMINI_API_KEY "
        "to your .env file."
    )

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=google_api_key,
    temperature=0.3,
    max_retries=2,
)


# ============================================================
# 3. General helpers
# ============================================================

FENCE_RE = re.compile(r"(```.*?```)", re.DOTALL)


def transform_outside_code(text: str, func) -> str:
    """Apply `func` only to the parts of `text` that are not fenced code."""
    parts = FENCE_RE.split(text)
    return "".join(
        part if index % 2 else func(part)
        for index, part in enumerate(parts)
    )


def safe_slug(title: str) -> str:
    """Convert a title into a filesystem-friendly folder name."""
    slug = title.strip().lower()
    slug = re.sub(r"[^a-z0-9 _-]+", "", slug)
    slug = re.sub(r"\s+", "_", slug).strip("_")
    return slug[:80] or "blog"


def count_words(text: str) -> int:
    """
    Count words in Markdown.

    Citation links and link URLs are removed first so they do not inflate
    the count. Words in code blocks are included.
    """
    text = re.sub(r"\[\d{1,3}\]\(https?://[^)]*\)", " ", text)
    text = re.sub(r"\]\(https?://[^)]*\)", "]", text)

    return len(
        re.findall(
            r"\b[\w]+(?:['’-][\w]+)*\b",
            text,
            flags=re.UNICODE,
        )
    )


def clean_section_markdown(markdown: str, title: str) -> str:
    """
    Ensure a section starts with its assigned level-two heading and that
    any inner headings are level three or deeper.
    """
    markdown = markdown.strip()

    # Remove an outer ```markdown fence if the model wrapped its answer.
    fenced = re.match(
        r"^```(?:markdown|md)?\s*\n(.*)\n```$",
        markdown,
        flags=re.DOTALL,
    )
    if fenced:
        markdown = fenced.group(1).strip()

    if not markdown:
        return f"## {title}\n\n"

    lines = markdown.splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and re.match(r"^#{1,6}\s+", lines[0]):
        lines[0] = f"## {title}"
    else:
        lines.insert(0, f"## {title}")

    cleaned = [lines[0]]
    in_code = False

    for line in lines[1:]:
        if line.strip().startswith("```"):
            in_code = not in_code

        if not in_code:
            match = re.match(r"^(#{1,2})\s+(.*)$", line)
            if match:
                line = f"### {match.group(2)}"

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def get_body_without_heading(markdown: str) -> str:
    """Remove the first Markdown heading before checking word count."""
    lines = markdown.strip().splitlines()

    if lines and re.match(r"^#{1,6}\s+", lines[0].strip()):
        return "\n".join(lines[1:]).strip()

    return markdown.strip()


def iso_to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def extract_text(response) -> str:
    """Handle string and list-based LangChain message content."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))

        return "\n".join(text_parts).strip()

    return str(content).strip()


def evidence_to_dict(item) -> dict:
    """Convert an EvidenceItem or dictionary into a plain dictionary."""
    if hasattr(item, "model_dump"):
        return item.model_dump()

    if isinstance(item, dict):
        return item

    return {}


def get_evidence_text(evidence: list[EvidenceItem]) -> str:
    """
    Format research evidence for the writing model.

    The number in brackets is the citation number the writer must use.
    """
    lines = []

    for index, item in enumerate(evidence[:MAX_EVIDENCE], start=1):
        lines.append(
            f"[{index}] Title: {item.title}\n"
            f"    URL: {item.url}\n"
            f"    Date: {item.published_at or 'unknown'}\n"
            f"    Summary: {item.snippet or ''}"
        )

    return "\n".join(lines) or "No research evidence was returned."


def escape_link_text(text: str) -> str:
    """Make text safe to use as Markdown link text."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.replace("[", "(").replace("]", ")")


# ============================================================
# 4. Citation handling
# ============================================================
# Canonical inline citation:   [3](https://actual-source.example/page)
# Several citations are separate links:   [1](url-1) [2](url-2)
#
# Every citation is checked against the real source list. Malformed or
# grouped citations produced by the model are repaired, and anything that
# cannot be matched to a real source is removed. URLs are never invented.

CITE_RE = re.compile(r"(?<![!\[])\[(\d{1,3})\]\((https?://[^\s)\]]+)\)")

# A URL that may contain balanced parentheses and never ends in punctuation.
_URL = (
    r"https?://(?:[^\s()>\]]|\([^\s()>\]]*\))*"
    r"(?:[^\s()>\],.;:!?'\"]|\([^\s()>\]]*\))"
)

# Any numeric citation that has a URL part, including malformed forms:
#   [[9]](url, [[1]](url2)      [1, 2](url)      [[3]](url1, url2)
URL_TOKEN_RE = re.compile(
    r"(?<!!)\[{1,2}\s*(?P<nums>\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\s*\]{1,2}"
    r"\(\s*<?(?P<urls>" + _URL + r"(?:[\s,;]+" + _URL + r")*)>?[\s,;]*\)?"
)

# Numeric citation with no URL at all:  [3]  or  [[3]]
BARE_RE = re.compile(
    r"(?<![!\w`])\[{1,2}(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]{1,2}(?![\(\[:])"
)

# Ordinary labelled link:  [label](url)
GENERIC_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\((https?://[^\s)]+)\)")

SOURCES_HEADING_RE = re.compile(r"(?im)^##\s+Sources\s*$")
SOURCE_LINE_RE = re.compile(
    r"^\s*(\d+)[.)]\s+\[(?P<title>.*?)\]\((?P<url>https?://[^\s)]+)\)"
)


def normalize_url(url: str) -> str:
    url = (url or "").strip().replace("%28", "(").replace("%29", ")")
    return url.rstrip("/.,;:").lower()


def url_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def md_url(url: str) -> str:
    """Make a URL safe as a Markdown link destination without changing it."""
    return (
        (url or "")
        .strip()
        .replace(" ", "%20")
        .replace("(", "%28")
        .replace(")", "%29")
    )


def display_title(item: EvidenceItem) -> str:
    """Readable source title; generic titles get the domain appended."""
    title = re.sub(r"\s+", " ", item.title or "").strip()
    domain = url_domain(item.url)

    if not title or title == item.url:
        return domain or item.url

    if len(title) < 12 and domain:
        return f"{title} ({domain})"

    return title


def _mark(number: int) -> str:
    return f"\u27e6C{number}\u27e7"


def _split_numbers(text: str) -> list[int]:
    return [int(part) for part in re.split(r"\s*[,;]\s*", text.strip())]


def sanitize_citations(markdown: str, sources: dict[int, str]) -> str:
    """
    Normalize every citation to `[n](url)`, where `n` and `url` come from
    `sources` (citation number -> real URL).

    - Malformed / grouped citations are repaired, each source becoming its
      own link.
    - The URL decides the number; a wrong label is corrected.
    - Citations whose URL matches no real source are removed.
    - Other links are reduced to plain text unless they point at a real
      source (then a numeric citation is added after the text).
    """
    norm_to_number = {
        normalize_url(url): number
        for number, url in sources.items()
        if url
    }

    def match_source(url: str, hinted: list[int]) -> Optional[int]:
        key = normalize_url(url)

        if key in norm_to_number:
            return norm_to_number[key]

        # A truncated URL (for example cut off at a comma) on the same site.
        candidates = [
            number
            for known, number in norm_to_number.items()
            if key
            and known.startswith(key)
            and url_domain(known) == url_domain(key)
        ]

        if len(candidates) == 1:
            return candidates[0]

        for number in hinted:
            if number in sources and url_domain(sources[number]) == url_domain(url):
                return number

        return None

    def url_token(match: re.Match) -> str:
        hinted = _split_numbers(match.group("nums"))
        urls = re.findall(_URL, match.group("urls"))
        resolved: list[int] = []

        for url in urls:
            number = match_source(url, hinted)

            if number and number not in resolved:
                resolved.append(number)

        return " ".join(_mark(number) for number in resolved)

    def generic(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        number = norm_to_number.get(normalize_url(url))
        return f"{label} {_mark(number)}" if number else label

    def bare(match: re.Match) -> str:
        valid = [n for n in _split_numbers(match.group(1)) if n in sources]
        return " ".join(_mark(n) for n in valid) if valid else match.group(0)

    def process(text: str) -> str:
        text = URL_TOKEN_RE.sub(url_token, text)
        text = GENERIC_LINK_RE.sub(generic, text)

        if sources:
            text = BARE_RE.sub(bare, text)

        # Keep adjacent citations as separate, space-separated links.
        return text.replace("\u27e7\u27e6", "\u27e7 \u27e6")

    processed = transform_outside_code(markdown, process)

    return re.sub(
        r"\u27e6C(\d+)\u27e7",
        lambda m: f"[{int(m.group(1))}]({md_url(sources[int(m.group(1))])})",
        processed,
    )


def renumber_citations(
    body: str,
    sources: dict[int, str],
) -> tuple[str, list[int]]:
    """
    Renumber citations in order of first appearance so the article reads
    [1], [2], [3]... Returns the new body and the original source numbers
    in their new order.
    """
    mapping: dict[int, int] = {}
    order: list[int] = []

    def repl(match: re.Match) -> str:
        old = int(match.group(1))

        if old not in sources:
            return ""

        if old not in mapping:
            mapping[old] = len(mapping) + 1
            order.append(old)

        return f"[{mapping[old]}]({md_url(sources[old])})"

    renumbered = transform_outside_code(
        body,
        lambda text: CITE_RE.sub(repl, text),
    )

    return renumbered, order


def split_sources(markdown: str) -> tuple[str, dict[int, str], dict[int, str], str]:
    """
    Split an article into (body, {number: url}, {number: title}, tail),
    where `tail` starts at the '## Sources' heading (empty if absent).
    """
    matches = list(SOURCES_HEADING_RE.finditer(markdown))

    if not matches:
        return markdown, {}, {}, ""

    index = matches[-1].start()
    body, tail = markdown[:index], markdown[index:]

    urls: dict[int, str] = {}
    titles: dict[int, str] = {}

    for line in tail.splitlines()[1:]:
        match = SOURCE_LINE_RE.match(line)

        if match:
            number = int(match.group(1))
            urls[number] = match.group("url").replace("%28", "(").replace("%29", ")")
            titles[number] = match.group("title")

    return body, urls, titles, tail


def repair_markdown_citations(markdown: str) -> str:
    """
    Repair malformed inline citations in a finished article (including
    blogs saved by older versions) using its own Sources list as the only
    allowed set of URLs. Returns the markdown unchanged if the article has
    no Sources section.
    """
    body, sources, _, tail = split_sources(markdown)

    if not sources:
        return markdown

    return sanitize_citations(body, sources) + tail


def validate_citation_markdown(markdown: str) -> list[str]:
    """
    Return a list of problems with the article's citations (empty if none):
    malformed syntax, numbers that do not match the Sources list, and
    citation URLs that are missing from Sources.
    """
    body, sources, _, _ = split_sources(markdown)
    problems: list[str] = []

    def mask(text: str) -> str:
        text = FENCE_RE.sub(lambda m: " " * len(m.group(0)), text)
        return re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)

    scan = mask(body)
    cited_numbers: set[int] = set()

    for match in CITE_RE.finditer(scan):
        number, url = int(match.group(1)), match.group(2)
        cited_numbers.add(number)

        if number not in sources:
            problems.append(f"Citation [{number}] is not in the Sources list.")
        elif normalize_url(url) != normalize_url(sources[number]):
            problems.append(
                f"Citation [{number}] links to {url} but Sources #{number} "
                f"is {sources[number]}."
            )

    residual = CITE_RE.sub(" ", scan)
    residual = re.sub(r"\[[^\]\n]*\]\(https?://[^\s)]+\)", " ", residual)

    for match in re.finditer(r"\]\(\s*https?://[^\n]{0,60}", residual):
        problems.append(f"Malformed link near: {match.group(0)[:70]!r}")

    for match in re.finditer(r"\[\[\d+\]\]|\[\d{1,3}\](?!\()", residual):
        problems.append(f"Citation without a link: {match.group(0)}")

    if cited_numbers and not sources:
        problems.append("Article has citations but no Sources section.")

    return problems


# ------------------------------------------------------------
# Claim/source verification
# ------------------------------------------------------------

class CitationVerdict(BaseModel):
    id: int
    supported: bool
    reason: str = ""


class CitationVerdicts(BaseModel):
    verdicts: list[CitationVerdict] = Field(default_factory=list)


VERIFY_SYSTEM = """
You check whether each cited source plausibly supports the claim it is
attached to.

For every item you get a claim, and the cited source's title, URL and a
short excerpt. Mark supported=false ONLY when the source clearly cannot
support the claim, for example:
- the source is about something unrelated to the claim (a dictionary entry
  or slang definition cited for a technical or enterprise claim, a product
  directory or homepage cited for an architectural detail),
- the excerpt contradicts the claim.

Mark supported=true when the title or excerpt covers the claim's topic, even
if the excerpt is too short to confirm every detail. Return one verdict per
item id.
"""


def collect_citation_claims(markdown: str) -> tuple[list[str], list[dict]]:
    """
    Find every canonical citation and the claim sentence it follows.
    Returns (lines, items); each item has line/start/end/number/url/claim.
    """
    lines = markdown.split("\n")
    items: list[dict] = []
    in_code = False

    for line_index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            continue

        for match in CITE_RE.finditer(line):
            before = CITE_RE.sub("", line[: match.start()])
            before = re.sub(
                r"^\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*|#+\s+)",
                "",
                before,
            )
            before = re.sub(r"[*_`]", "", before).strip()

            if not before and line_index > 0 and lines[line_index - 1].strip():
                before = re.sub(
                    r"[*_`]",
                    "",
                    CITE_RE.sub("", lines[line_index - 1]),
                ).strip()

            sentences = [s for s in re.split(r"(?<=[.!?])\s+", before) if s]
            claim = sentences[-1] if sentences else before

            if len(claim) < 40 and len(sentences) > 1:
                claim = f"{sentences[-2]} {claim}"

            items.append(
                {
                    "line": line_index,
                    "start": match.start(),
                    "end": match.end(),
                    "number": int(match.group(1)),
                    "url": match.group(2),
                    "claim": claim[:600],
                }
            )

    return lines, items


def verify_citations(
    body: str,
    evidence: list[EvidenceItem],
) -> tuple[str, list[str]]:
    """
    Ask the model whether each cited source can support its claim and remove
    citations that clearly cannot. Fails safe: on any error, or if most
    citations are flagged (which suggests a bad verdict), nothing is removed.
    """
    lines, items = collect_citation_claims(body)

    if not items:
        return body, []

    by_url = {normalize_url(item.url): item for item in evidence}
    checks: list[dict] = []

    for index, item in enumerate(items):
        source = by_url.get(normalize_url(item["url"]))

        if source is None or not item["claim"]:
            continue

        item["id"] = index
        checks.append(
            {
                "id": index,
                "claim": item["claim"],
                "title": source.title,
                "url": source.url,
                "excerpt": (source.snippet or "")[:900],
            }
        )

    if not checks:
        return body, []

    verifier = llm.with_structured_output(CitationVerdicts)
    unsupported: dict[int, str] = {}
    logs: list[str] = []

    try:
        for offset in range(0, len(checks), 25):
            chunk = checks[offset:offset + 25]
            text = "\n\n".join(
                f"ID {c['id']}\n"
                f"Claim: {c['claim']}\n"
                f"Source title: {c['title']}\n"
                f"Source URL: {c['url']}\n"
                f"Source excerpt: {c['excerpt'] or '(none)'}"
                for c in chunk
            )

            result = verifier.invoke(
                [
                    SystemMessage(content=VERIFY_SYSTEM),
                    HumanMessage(content=text),
                ]
            )

            valid_ids = {c["id"] for c in chunk}

            for verdict in result.verdicts:
                if verdict.id in valid_ids and not verdict.supported:
                    unsupported[verdict.id] = verdict.reason
    except Exception as exc:
        return body, [f"Citation verification skipped ({exc})."]

    if not unsupported:
        return body, [f"Citation check: all {len(checks)} citations passed."]

    if len(unsupported) > 0.5 * len(checks):
        return body, [
            "Citation check skipped removal: too many citations were "
            "flagged to trust the verdicts."
        ]

    spans_by_line: dict[int, list[tuple[int, int]]] = {}

    for item in items:
        if item.get("id") in unsupported:
            spans_by_line.setdefault(item["line"], []).append(
                (item["start"], item["end"])
            )
            logs.append(
                f"Removed citation [{item['number']}] "
                f"({item['url']}) from: {item['claim'][:90]!r} — "
                f"{unsupported[item['id']]}"
            )

    for line_index, spans in spans_by_line.items():
        line = lines[line_index]

        for start, end in sorted(spans, reverse=True):
            line = line[:start] + line[end:]

        line = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", line)
        line = re.sub(r"[ \t]+([.,;:!?])", r"\1", line)
        lines[line_index] = line.rstrip()

    logs.append(
        f"Citation check: removed {len(unsupported)} unsupported "
        f"citation(s) out of {len(checks)}."
    )

    return "\n".join(lines), logs


# ============================================================
# 5. Router
# ============================================================

ROUTER_SYSTEM = """
You are the routing module for a blog-writing agent.

Determine whether web research is needed before planning.

Modes:
- closed_book: evergreen concepts that can be explained from established
  knowledge without current facts.
- hybrid: evergreen topics that benefit from current examples, tools,
  products, or recent developments.
- open_book: current events, latest developments, recent news, prices,
  policies, or rapidly changing information.

Rules:
- If research is needed, provide 3-6 focused search queries.
- Queries must be specific to the topic.
- Use the supplied as-of date for current topics.
- Do not invent current events.
- Prefer research when factual claims need citations.
"""


def router_node(state: State) -> dict:
    as_of = state.get("as_of") or date.today().isoformat()
    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state.get('topic', '')}\n"
                    f"Requested style: {state.get('blog_style', 'Auto')}\n"
                    f"As-of date: {as_of}"
                )
            ),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    needs_research = bool(decision.needs_research) and (
        decision.mode != "closed_book"
    )

    queries = [q.strip() for q in decision.queries if q and q.strip()][:6]

    if needs_research and not queries:
        queries = [state.get("topic", "").strip()]

    return {
        "as_of": as_of,
        "needs_research": needs_research,
        "mode": decision.mode,
        "queries": queries,
        "recency_days": recency_days,
        "logs": [
            f"Router: mode={decision.mode}, "
            f"needs_research={needs_research}. {decision.reason}"
        ],
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


# ============================================================
# 6. Research using the Tavily REST API
# ============================================================

def tavily_search(
    query: str,
    max_results: int = 5,
    mode: str = "hybrid",
    recency_days: int = 45,
) -> tuple[list[dict], Optional[str]]:
    """
    Run one Tavily search.

    Returns (results, error_message). Never raises.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return [], "TAVILY_API_KEY is not set."

    payload: dict = {
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }

    if mode == "open_book":
        payload["topic"] = "news"
        payload["days"] = max(1, int(recency_days))

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return [], str(exc)

    output = []

    for result in data.get("results") or []:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url") or "").strip()

        if not url.startswith(("http://", "https://")):
            continue

        output.append(
            {
                "title": result.get("title") or url,
                "url": url,
                "snippet": (result.get("content") or "")[:1200],
                "published_at": (
                    result.get("published_date")
                    or result.get("published_at")
                ),
                "source": url_domain(url) or None,
            }
        )

    return output, None


def research_node(state: State) -> dict:
    """Collect Tavily evidence directly, without an extra LLM call."""
    queries = (state.get("queries") or [])[:6]
    mode = state.get("mode", "hybrid")
    recency_days = int(state.get("recency_days", 45))

    logs: list[str] = []
    per_query: list[list[dict]] = []

    for query in queries:
        results, error = tavily_search(
            query,
            max_results=5,
            mode=mode,
            recency_days=recency_days,
        )

        if error:
            logs.append(f"Search failed for {query!r}: {error}")

        per_query.append(results)

    # Round-robin across queries so every query is represented.
    interleaved: list[dict] = []
    longest = max((len(results) for results in per_query), default=0)

    for position in range(longest):
        for results in per_query:
            if position < len(results):
                interleaved.append(results[position])

    seen: set[str] = set()
    evidence: list[EvidenceItem] = []

    for item in interleaved:
        url = str(item.get("url") or "").strip()
        key = normalize_url(url)

        if not url or key in seen:
            continue

        seen.add(key)
        published = item.get("published_at")

        evidence.append(
            EvidenceItem(
                title=str(item.get("title") or url),
                url=url,
                published_at=str(published) if published else None,
                snippet=item.get("snippet"),
                source=item.get("source"),
            )
        )

    # For current topics, discard sources dated outside the requested
    # window. Undated results are retained.
    if mode == "open_book":
        as_of = iso_to_date(state.get("as_of")) or date.today()
        cutoff = as_of - timedelta(days=recency_days)

        evidence = [
            item
            for item in evidence
            if (
                (published := iso_to_date(item.published_at)) is None
                or cutoff <= published <= as_of
            )
        ]

    evidence = evidence[:MAX_EVIDENCE]

    if not evidence:
        logs.append(
            "No research evidence was collected. The blog will be written "
            "without citations."
        )
    else:
        logs.append(f"Research: collected {len(evidence)} unique sources.")

    return {"evidence": evidence, "logs": logs}


# ============================================================
# 7. Orchestrator: create the blog plan
# ============================================================

ORCHESTRATOR_SYSTEM = f"""
You are an expert blog editor and content planner.

Create a clear, useful outline for the requested blog.

Plan requirements:
- Create {MIN_SECTIONS}-{MAX_SECTIONS} sections.
- Each section must target {MIN_SECTION_WORDS}-{MAX_SECTION_WORDS} words.
- Aim for approximately {TARGET_SECTION_WORDS} words per section.
- Each task must contain a goal and 3-6 concrete bullets.
- Number task ids 1, 2, 3... in reading order.
- Avoid repetitive sections.
- Use a logical progression from introduction to conclusion.
- Make each section directly relevant to the topic.
- Explain concepts clearly and accurately.
- Do not create a section called "Sources" or "References"; a Sources
  list is added automatically.

Style:
- Auto: choose a suitable style for the topic and audience.
- Technical: use accurate technical terminology and deeper explanations.
- Non-technical: use simple language, define jargon, and use relatable examples.

Audience and tone:
- If the person supplied an audience or tone, use it in the plan.
- Otherwise choose an appropriate audience and tone.

Code:
- Set requires_code=true only when code genuinely helps explain the section.
- Do not force code into conceptual or non-technical sections.
- If code is included, it must be minimal, correct, and explained.

Citations:
- Prefer research evidence for current or externally verifiable claims.
- Do not plan citations to sources that are not in the evidence.
- Current claims must be supported by research evidence.

Research modes:
- closed_book: explain established concepts.
- hybrid: combine established concepts with researched examples.
- open_book: focus on requested current events or developments.
- Do not invent news if evidence is weak.

Return content matching the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    style = state.get("blog_style", "Auto")
    audience = (state.get("audience") or "").strip()
    tone = (state.get("tone") or "").strip()

    forced_kind = "news_roundup" if mode == "open_book" else None

    evidence_data = [
        {
            "title": item.title,
            "url": item.url,
            "published_at": item.published_at,
            "snippet": item.snippet,
        }
        for item in evidence
    ]

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(
            content=(
                f"Topic: {state.get('topic', '')}\n"
                f"Requested style: {style}\n"
                f"Requested audience: {audience or 'not specified'}\n"
                f"Requested tone: {tone or 'not specified'}\n"
                f"Research mode: {mode}\n"
                f"As-of date: {state.get('as_of', '')}\n"
                f"Recency days: {state.get('recency_days', 7)}\n"
                f"Required blog kind: {forced_kind or 'choose appropriately'}\n\n"
                f"Available evidence:\n"
                f"{json.dumps(evidence_data, ensure_ascii=False, default=str)}"
            )
        ),
    ]

    plan: Optional[Plan] = None
    last_error: Optional[Exception] = None

    for _ in range(2):
        try:
            plan = planner.invoke(messages)
            if plan and plan.tasks:
                break
        except Exception as exc:  # validation or API error; retry once
            last_error = exc

    if plan is None or not plan.tasks:
        raise RuntimeError(
            f"The planner did not return a valid blog plan: {last_error}"
        )

    if forced_kind:
        plan.blog_kind = forced_kind

    if audience:
        plan.audience = audience

    if tone:
        plan.tone = tone

    plan.tasks = plan.tasks[:MAX_SECTIONS]

    # Normalize ids and keep every target inside the required range.
    for index, task in enumerate(plan.tasks, start=1):
        task.id = index
        task.bullets = task.bullets[:6] or [task.goal]
        task.target_words = max(
            MIN_SECTION_WORDS,
            min(MAX_SECTION_WORDS, task.target_words),
        )

    return {
        "plan": plan,
        "logs": [
            f"Plan: '{plan.blog_title}' with {len(plan.tasks)} sections."
        ],
    }


# ============================================================
# 8. Fan out to section workers
# ============================================================

def fanout(state: State):
    plan = state.get("plan")

    if plan is None:
        raise ValueError(
            "Cannot create sections because the blog plan is missing."
        )

    evidence = [
        evidence_to_dict(item)
        for item in state.get("evidence", [])
    ]

    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state.get("topic", ""),
                "blog_style": state.get("blog_style", "Auto"),
                "mode": state.get("mode", "closed_book"),
                "as_of": state.get("as_of", ""),
                "recency_days": state.get("recency_days", 7),
                "plan": plan.model_dump(),
                "evidence": evidence,
            },
        )
        for task in plan.tasks
    ]


# ============================================================
# 9. Section worker
# ============================================================

WORKER_SYSTEM = f"""
You are an expert writer producing ONE section of a blog.

Write clear, informative, accurate Markdown for the supplied section.

MANDATORY WORD COUNT:
- The section body must contain at least {MIN_SECTION_WORDS} words.
- The section body must contain at most {MAX_SECTION_WORDS} words.
- Aim for approximately {TARGET_SECTION_WORDS} words.
- The heading does not count toward the word target.
- Do not use filler or repeat ideas to reach the minimum.
- Cover every supplied bullet meaningfully.

STRUCTURE:
- Start with exactly one level-two heading: ## Section Title
- Use paragraphs and, where useful, level-three subheadings (###).
- Never use level-one or level-two headings other than the first line.
- Explain concepts step by step.
- Include relevant examples when useful.
- Keep every paragraph related to the section goal.
- Avoid vague claims and unsupported statistics.

STYLE:
- Auto: choose an appropriate style for the topic and audience.
- Technical: explain mechanisms, methods, assumptions, and terminology.
- Non-technical: use plain language, define technical terms, and give examples.

CODE:
- Include code only if requires_code=true and code genuinely helps.
- Code must be relevant to this section.
- Keep code minimal and syntactically plausible.
- Put code in fenced blocks with a language tag.
- Explain what the code does.
- Never insert unrelated code just to include a code block.

CITATIONS (very important):
- The evidence list is numbered: [1], [2], [3]...
- Cite a source with EXACTLY this Markdown link: [n](url), where n is the
  evidence number and url is that same evidence item's URL, copied exactly.
  Example: Retrieval grounds answers in documents [2](https://example.com/page).
- Put each citation immediately after the sentence or claim it supports,
  inside the paragraph or bullet. Do NOT collect citations at the end of the
  section or article.
- Each source is its own link. For several sources, write separate links
  separated by a space:  [1](url-1) [3](url-3)
- NEVER put several URLs inside one link, NEVER separate URLs with commas
  inside parentheses, NEVER use nested brackets like [[1]], and always close
  every link with ")".
- Cite a source only if its title or summary directly supports the specific
  claim. If no source supports a claim, write the claim without a citation.
  Do not cite dictionary definitions, tool directories, or generic
  homepages for technical or factual claims they do not cover.
- Use only numbers and URLs from the evidence list.
- Do not fabricate sources, URLs, quotes, statistics, or dates.
- Do not add any other Markdown links.
- Do not write a Sources or References list; it is added automatically.
- If no suitable evidence is supplied, write without citations.

CURRENT EVENTS:
- In open_book mode, state specific events, dates, organizations, products,
  or policy claims only when supported by supplied evidence.
- If a requested detail cannot be verified, say so briefly.

Output only the section Markdown. Do not add a word-count note.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])

    evidence = [
        EvidenceItem(**item)
        for item in payload.get("evidence", [])
    ]

    evidence_text = get_evidence_text(evidence)
    bullets_text = "\n".join(f"- {bullet}" for bullet in task.bullets)

    context = (
        f"Blog title: {plan.blog_title}\n"
        f"Audience: {plan.audience}\n"
        f"Tone: {plan.tone}\n"
        f"Blog kind: {plan.blog_kind}\n"
        f"Blog constraints: {plan.constraints}\n"
        f"Topic: {payload.get('topic', '')}\n"
        f"Requested style: {payload.get('blog_style', 'Auto')}\n"
        f"Research mode: {payload.get('mode', 'closed_book')}\n"
        f"As-of date: {payload.get('as_of', '')}\n\n"
        f"Section title: {task.title}\n"
        f"Section goal: {task.goal}\n"
        f"Required word range: {MIN_SECTION_WORDS}-{MAX_SECTION_WORDS}\n"
        f"Target words: {task.target_words}\n"
        f"Tags: {task.tags}\n"
        f"Requires research: {task.requires_research}\n"
        f"Requires citations: {task.requires_citations}\n"
        f"Requires code: {task.requires_code}\n\n"
        f"Required bullets:\n{bullets_text}\n\n"
        f"Numbered evidence (cite only these):\n{evidence_text}"
    )

    last_markdown = ""
    last_count = 0
    logs: list[str] = []

    # Retry once if the generated section is outside the word range.
    for attempt in range(2):
        retry_note = ""

        if attempt == 1:
            if last_count < MIN_SECTION_WORDS:
                retry_note = (
                    f"\nYour previous section had {last_count} words. "
                    f"Expand it with useful, topic-specific explanation "
                    f"to at least {MIN_SECTION_WORDS} words, without filler."
                )
            else:
                retry_note = (
                    f"\nYour previous section had {last_count} words. "
                    f"Shorten it to at most {MAX_SECTION_WORDS} words "
                    f"while preserving important explanations and citations."
                )

        response = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"{context}\n\n"
                        f"{retry_note}\n\n"
                        "Write the complete section now."
                    )
                ),
            ]
        )

        candidate = clean_section_markdown(
            extract_text(response),
            task.title,
        )
        candidate = sanitize_citations(
            candidate,
            {number: item.url for number, item in enumerate(evidence, start=1)},
        )

        body = get_body_without_heading(candidate)
        count = count_words(body)

        last_markdown, last_count = candidate, count

        if MIN_SECTION_WORDS <= count <= MAX_SECTION_WORDS:
            break
    else:
        # Do not abort the entire blog for one slightly-off section.
        logs.append(
            f"Warning: section '{task.title}' has {last_count} words "
            f"(target {MIN_SECTION_WORDS}-{MAX_SECTION_WORDS}); kept as is."
        )

    if evidence and task.requires_citations and not CITE_RE.search(last_markdown):
        logs.append(
            f"Note: section '{task.title}' contains no citations."
        )

    return {
        "sections": [(task.id, last_markdown)],
        "logs": logs,
    }


# ============================================================
# 10. Merge sections
# ============================================================

def merge_content(state: State) -> dict:
    plan = state.get("plan")

    if plan is None:
        raise ValueError("merge_content called without a blog plan.")

    evidence: list[EvidenceItem] = list(state.get("evidence", []))[:MAX_EVIDENCE]
    sources = {number: item.url for number, item in enumerate(evidence, start=1)}
    logs: list[str] = []

    ordered_sections = [
        markdown
        for _, markdown in sorted(
            state.get("sections", []),
            key=lambda item: item[0],
        )
    ]

    body = "\n\n".join(ordered_sections).strip()

    # Re-check every citation and drop those whose source cannot support
    # the claim they follow.
    if evidence:
        body, verify_logs = verify_citations(body, evidence)
        logs.extend(verify_logs)

    # Renumber inline citations [1], [2], ... in order of appearance.
    body, cited_order = renumber_citations(body, sources)

    merged_markdown = f"# {plan.blog_title}\n\n{body}\n"

    if cited_order:
        source_lines = []

        for new_number, old_number in enumerate(cited_order, start=1):
            item = evidence[old_number - 1]
            title = escape_link_text(display_title(item))
            line = (
                f"{new_number}. [{title}]({md_url(item.url)}) — {item.url}"
            )

            if item.published_at:
                line += f" ({item.published_at[:10]})"

            source_lines.append(line)

        merged_markdown += "\n## Sources\n\n" + "\n".join(source_lines) + "\n"

    # Final validation: repair once, then report anything still wrong.
    problems = validate_citation_markdown(merged_markdown)

    if problems:
        merged_markdown = repair_markdown_citations(merged_markdown)
        remaining = validate_citation_markdown(merged_markdown)
        logs.append(
            f"Citation validation found {len(problems)} problem(s); "
            f"{len(remaining)} remain after repair."
        )
        logs.extend(f"  - {problem}" for problem in remaining[:10])
    else:
        logs.append("Citation validation: no problems found.")

    logs.append(
        f"Merged {len(ordered_sections)} sections with "
        f"{len(cited_order)} cited sources."
    )

    return {"merged_md": merged_markdown, "logs": logs}


# ============================================================
# 11. Image planning
# ============================================================

DECIDE_IMAGES_SYSTEM = f"""
You are a visual editor creating professional technical visuals for a blog.

Rules:
- Use at most {MAX_IMAGES} images total.
- Only create an image when it materially improves understanding.
- section_title MUST exactly match one supplied ## section title.
- Use at most one image per section.
- filename must be lowercase, hyphenated, and end in .png.
- title should be a short technical heading (max 7 words).
- labels: choose 3-6 IMPORTANT technical labels only. Keep each label 2-5 words.
  These labels will be rendered by Pillow, so spelling must be exact.
- For every label, provide approximate normalized anchor_x/anchor_y values for
  the visual feature it points to and choose a side for its callout.
- prompt is ONLY for FLUX. Describe visual objects, composition, colors,
  lighting, depth, and relationships. NEVER request text, letters, numbers,
  words, labels, captions, logos, signs, UI, or writing.
- Prefer a strong central visual with clear space near edges for callouts.
- Do not make a giant bottom text panel. Labels are integrated as small cards
  around the visual with connector lines.
- If no image is useful, return images=[].

Return a GlobalImagePlan.
"""

def list_section_titles(markdown: str) -> list[str]:
    titles = []
    in_code = False

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            continue

        match = re.match(r"^##\s+(.*?)\s*#*\s*$", line)

        if match and match.group(1).strip().lower() != "sources":
            titles.append(match.group(1).strip())

    return titles


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title or "").strip().lower()


def insert_image_placeholder(
    markdown: str,
    section_title: str,
    placeholder: str,
) -> Optional[str]:
    """
    Insert a placeholder after the first plain paragraph of the named
    section (or after the last placeholder already in that section).

    The article text and citations are never rewritten by a model.
    Returns None if the section cannot be found.
    """
    lines = markdown.split("\n")
    heading_re = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
    target = _normalize_title(section_title)

    start: Optional[int] = None
    fallback: Optional[int] = None
    in_code = False

    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            continue

        match = heading_re.match(line)

        if not match or len(match.group(1)) != 2:
            continue

        heading = _normalize_title(match.group(2))

        if heading == target:
            start = index
            break

        if fallback is None and target and (
            target in heading or heading in target
        ):
            fallback = index

    if start is None:
        start = fallback

    if start is None:
        return None

    # Find the end of the section and which lines are inside code fences.
    end = len(lines)
    in_code = False
    code_flags: dict[int, bool] = {}

    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()

        if stripped.startswith("```"):
            in_code = not in_code
            code_flags[index] = True
            continue

        code_flags[index] = in_code

        if not in_code and re.match(r"^#{1,6}\s", lines[index]):
            end = index
            break

    search_from = start + 1

    for index in range(start + 1, end):
        if re.match(r"^\[\[IMAGE_\d+\]\]$", lines[index].strip()):
            search_from = index + 1

    for index in range(search_from, end):
        line = lines[index]
        stripped = line.strip()

        if not stripped or code_flags.get(index):
            continue

        if re.match(
            r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||!\[|\[\[IMAGE_)",
            stripped,
        ):
            continue

        previous_is_boundary = (
            index == start + 1
            or not lines[index - 1].strip()
            or bool(re.match(r"^#{1,6}\s", lines[index - 1]))
        )

        if not previous_is_boundary:
            continue

        # Advance to the end of this paragraph.
        stop = index

        while stop < end and lines[stop].strip():
            stop += 1

        insertion = ["", placeholder]

        if stop >= end:
            insertion.append("")

        lines[stop:stop] = insertion
        return "\n".join(lines)

    # No plain paragraph found: place at the end of the section.
    position = end

    while position > start + 1 and not lines[position - 1].strip():
        position -= 1

    lines[position:position] = ["", placeholder]
    return "\n".join(lines)


def decide_images(state: State) -> dict:
    plan = state.get("plan")

    if plan is None:
        raise ValueError("Image planning requires a blog plan.")

    merged = state.get("merged_md", "")

    if not state.get("include_images", True):
        return {
            "md_with_placeholders": merged,
            "image_specs": [],
            "logs": ["Images: disabled by the user."],
        }

    titles = list_section_titles(merged)

    if not titles:
        return {
            "md_with_placeholders": merged,
            "image_specs": [],
            "logs": ["Images: no sections found."],
        }

    planner = llm.with_structured_output(GlobalImagePlan)

    try:
        result = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state.get('topic', '')}\n"
                        f"Blog title: {plan.blog_title}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Requested style: {state.get('blog_style', 'Auto')}\n\n"
                        "Section titles (copy exactly):\n"
                        + "\n".join(f"- {title}" for title in titles)
                        + f"\n\nBlog Markdown:\n{merged}"
                    )
                ),
            ]
        )
        proposed = result.images[:MAX_IMAGES]
    except Exception as exc:
        return {
            "md_with_placeholders": merged,
            "image_specs": [],
            "logs": [f"Images: planning failed ({exc}); no images added."],
        }

    markdown = merged
    specs: list[dict] = []
    logs: list[str] = []

    for index, image in enumerate(proposed, start=1):
        placeholder = f"[[IMAGE_{index}]]"
        updated = insert_image_placeholder(
            markdown,
            image.section_title,
            placeholder,
        )

        if updated is None:
            logs.append(
                f"Images: could not find section '{image.section_title}'; "
                "image skipped."
            )
            continue

        markdown = updated
        spec = image.model_dump()
        spec["placeholder"] = placeholder
        specs.append(spec)

    logs.append(f"Images: planned {len(specs)} image(s).")

    return {
        "md_with_placeholders": markdown,
        "image_specs": specs,
        "logs": logs,
    }


# ============================================================
# 12. Cloudflare image generation
# ============================================================

def cloudflare_generate_image_bytes(prompt: str) -> bytes:
    """
    Generate an image using Cloudflare Workers AI.

    Required .env variables:
      CLOUDFLARE_ACCOUNT_ID
      CLOUDFLARE_API_TOKEN
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set.")

    if not api_token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set.")

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    )

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        json={"prompt": prompt[:2000]},
        timeout=180,
    )

    if not response.ok:
        raise RuntimeError(
            f"Cloudflare image generation failed "
            f"({response.status_code}): {response.text[:1000]}"
        )

    content_type = response.headers.get("content-type", "").lower()

    # Some Workers AI responses return raw image bytes.
    if content_type.startswith("image/"):
        return response.content

    # Other responses return JSON containing base64 image data.
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Cloudflare returned an unrecognized image response."
        ) from exc

    result = data.get("result") or {}
    image_data = (
        result.get("image")
        or data.get("image")
        or result.get("data")
    )

    if isinstance(image_data, list) and image_data:
        image_data = image_data[0]

    if not isinstance(image_data, str) or not image_data:
        raise RuntimeError(
            "Cloudflare did not return image data: "
            f"{json.dumps(data)[:1000]}"
        )

    try:
        return base64.b64decode(image_data)
    except Exception as exc:
        raise RuntimeError("Could not decode Cloudflare image data.") from exc


def detect_image_extension(data: bytes) -> str:
    """Pick the right extension from the image's magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"

    return ".png"


# ============================================================
# 13. Generate images, place them in Markdown, save everything
# ============================================================

def safe_image_stem(filename: str, index: int) -> str:
    stem = Path(str(filename or "")).stem.strip()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-_")
    return stem[:60] or f"blog-image-{index}"


def clean_inline(text: str) -> str:
    """Single-line text that is safe inside Markdown image alt/caption."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.replace("[", "(").replace("]", ")").replace("*", "")


# ------------------------------------------------------------
# Text-free FLUX prompts + integrated technical renderer
# ------------------------------------------------------------

IMAGE_STYLE_SUFFIX = (
    "Premium technical editorial illustration, polished modern visual design, "
    "strong central focal subject, clean negative space around the edges, "
    "subtle depth, crisp shapes, sophisticated consistent color palette, "
    "high detail, professional educational technology aesthetic, "
    "absolutely no text, no letters, no numbers, no words, no labels, "
    "no captions, no logos, no watermark, no signage, no writing of any kind."
)

_TEXT_WORDS_RE = re.compile(
    r"\b(labell?ed|labell?ing|labels?|captions?|titles?|headings?|"
    r"text|words?|letters?|typography|written|writing|annotations?|"
    r"legend|formula|equation)\b", flags=re.IGNORECASE
)
_DIAGRAM_WORDS_RE = re.compile(
    r"\b(infographic|diagram|flow ?chart|chart|graph|schematic)\b",
    flags=re.IGNORECASE,
)


def build_image_prompt(topic: str, description: str) -> str:
    description = description or ""
    description = re.sub(r'["“”‘’].*?["“”‘’]', " ", description)
    description = _DIAGRAM_WORDS_RE.sub("technical illustration", description)
    description = _TEXT_WORDS_RE.sub("", description)
    description = re.sub(r"\s+", " ", description).strip(" ,.;:")
    return f"{description}. Concept: {topic}. {IMAGE_STYLE_SUFFIX}"


def _font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
    words = re.sub(r"\s+", " ", (text or "").strip()).split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_integrated_infographic(image_bytes: bytes, title: str, labels: list[dict]) -> bytes:
    """Render exact labels as integrated callouts around the FLUX artwork."""
    from io import BytesIO

    base = Image.open(BytesIO(image_bytes)).convert("RGB")
    target_w, target_h = 1400, 900
    base.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (247, 248, 250))
    x0 = (target_w - base.width) // 2
    y0 = (target_h - base.height) // 2 + 18
    canvas.paste(base, (x0, y0))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if title:
        title_font = _font(32, bold=True)
        title_text = re.sub(r"\s+", " ", title).strip()[:70]
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        tx = (target_w - tw)//2
        draw.rounded_rectangle(
            (tx-20, 10, tx+tw+20, 10+th+20), radius=16,
            fill=(255,255,255,232), outline=(220,224,230,220), width=1,
        )
        draw.text((tx, 20), title_text, font=title_font, fill=(25,31,42,255))

    palette = [
        ((37,99,235,245)), ((124,58,237,245)), ((5,150,105,245)),
        ((234,88,12,245)), ((8,145,178,245)), ((190,24,93,245)),
    ]
    card_font = _font(21, bold=True)
    card_w, card_h = 250, 62

    def anchor_xy(ax, ay):
        return int(x0 + ax*base.width), int(y0 + ay*base.height)

    groups = {"left": [], "right": [], "top": [], "bottom": []}
    for raw in labels[:6]:
        if not isinstance(raw, dict):
            continue
        text_value = re.sub(r"\s+", " ", str(raw.get("text", "")).strip())[:55]
        if not text_value:
            continue
        side = raw.get("side", "right") if raw.get("side") in groups else "right"
        ax = max(.08, min(.92, float(raw.get("anchor_x", .5))))
        ay = max(.12, min(.88, float(raw.get("anchor_y", .5))))
        groups[side].append((text_value, ax, ay))

    placements = []
    for side, items in groups.items():
        for idx, (text_value, ax, ay) in enumerate(items):
            anchor = anchor_xy(ax, ay)
            if side in ("left", "right"):
                cy = int(125 + (idx+1)*(target_h-215)/(len(items)+1))
                if side == "left":
                    box = (24, cy-card_h//2, 24+card_w, cy+card_h//2); attach=(box[2],cy)
                else:
                    box=(target_w-24-card_w,cy-card_h//2,target_w-24,cy+card_h//2); attach=(box[0],cy)
            else:
                cx=int(190+(idx+1)*(target_w-380)/(len(items)+1))
                if side=="top":
                    box=(cx-card_w//2,74,cx+card_w//2,74+card_h); attach=(cx,box[3])
                else:
                    box=(cx-card_w//2,target_h-82-card_h,cx+card_w//2,target_h-82); attach=(cx,box[1])
            placements.append((box,attach,anchor,text_value,len(placements)%len(palette)))

    for box, attach, anchor, text_value, color_idx in placements:
        accent=palette[color_idx]; ax,ay=anchor
        if abs(attach[0]-ax)>abs(attach[1]-ay):
            mid=((attach[0]+ax)//2,attach[1]); points=[attach,mid,anchor]
        else:
            mid=(attach[0],(attach[1]+ay)//2); points=[attach,mid,anchor]
        draw.line(points, fill=accent, width=4, joint="curve")
        r=7; draw.ellipse((ax-r,ay-r,ax+r,ay+r),fill=accent)

    for box, attach, anchor, text_value, color_idx in placements:
        accent=palette[color_idx]; sx1,sy1,sx2,sy2=box
        draw.rounded_rectangle((sx1+5,sy1+6,sx2+5,sy2+6),radius=16,fill=(0,0,0,55))
        draw.rounded_rectangle(box,radius=16,fill=(255,255,255,238),outline=accent,width=3)
        draw.rounded_rectangle((sx1,sy1,sx1+8,sy2),radius=7,fill=accent)
        lines=_wrap_text(draw,text_value,card_font,card_w-34)
        line_h=25; total=len(lines)*line_h; ty=(sy1+sy2-total)//2-1
        for line in lines[:2]:
            draw.text((sx1+20,ty),line,font=card_font,fill=(24,31,42,255)); ty+=line_h

    result=Image.alpha_composite(canvas.convert("RGBA"),overlay).convert("RGB")
    result=ImageEnhance.Contrast(result).enhance(1.04)
    result=ImageEnhance.Sharpness(result).enhance(1.10)
    result=result.filter(ImageFilter.UnsharpMask(radius=1.1,percent=110,threshold=3))
    out=BytesIO(); result.save(out,format="PNG",optimize=True); return out.getvalue()


def generate_and_place_images(state: State) -> dict:
    plan = state.get("plan")

    if plan is None:
        raise ValueError("Image generation requires a blog plan.")

    markdown = (
        state.get("md_with_placeholders")
        or state.get("merged_md")
        or ""
    )

    image_specs = [dict(spec) for spec in (state.get("image_specs") or [])]

    blog_slug = safe_slug(plan.blog_title)
    blog_dir = OUTPUT_ROOT / blog_slug
    images_dir = blog_dir / "images"

    blog_dir.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    used_stems: set[str] = set()

    if image_specs:
        images_dir.mkdir(parents=True, exist_ok=True)

    for index, spec in enumerate(image_specs[:MAX_IMAGES], start=1):
        placeholder = spec.get("placeholder", f"[[IMAGE_{index}]]")
        stem = safe_image_stem(spec.get("filename", ""), index)

        while stem in used_stems:
            stem = f"{stem}-{index}"

        used_stems.add(stem)

        prompt = build_image_prompt(
            state.get("topic", ""),
            spec.get("prompt", ""),
        )

        try:
            image_bytes = cloudflare_generate_image_bytes(prompt)
            final_bytes = render_integrated_infographic(
                image_bytes,
                spec.get("title", ""),
                spec.get("labels", []),
            )
            saved_name = f"{stem}.png"
            (images_dir / saved_name).write_bytes(final_bytes)
            spec["filename"] = saved_name

        except Exception as exc:
            logs.append(f"Image generation failed for {stem}: {exc}")
            spec["filename"] = None
            caption = clean_inline(spec.get("caption", ""))
            fallback = (
                f"> **Image unavailable:** {caption}\n>\n"
                "> Image generation failed. Check the image API "
                "configuration and logs."
            )
            markdown = markdown.replace(placeholder, fallback)
            continue

        image_markdown = (
            f"![{clean_inline(spec.get('alt', 'Blog illustration'))}]"
            f"(images/{saved_name})\n\n"
            f"*{clean_inline(spec.get('caption', ''))}*"
        )

        markdown = markdown.replace(placeholder, image_markdown)

    # Remove any placeholder that was never resolved.
    markdown = re.sub(r"\n*\[\[IMAGE_\d+\]\]\n*", "\n\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"

    markdown_path = blog_dir / "blog.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    logs.append(f"Saved blog to {markdown_path}")

    # Save metadata so past blogs can restore the Plan/Evidence/Logs tabs.
    all_logs = list(state.get("logs") or []) + logs

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "topic": state.get("topic", ""),
        "blog_style": state.get("blog_style", "Auto"),
        "as_of": state.get("as_of", ""),
        "audience": state.get("audience", ""),
        "tone": state.get("tone", ""),
        "mode": state.get("mode", ""),
        "queries": state.get("queries", []),
        "plan": plan.model_dump(),
        "evidence": [
            evidence_to_dict(item) for item in state.get("evidence", [])
        ],
        "image_specs": image_specs,
        "logs": all_logs,
    }

    (blog_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "final": markdown,
        "md_with_placeholders": markdown,
        "image_specs": image_specs,
        "markdown_path": str(markdown_path),
        "output_dir": str(blog_dir),
        "logs": logs,
    }


# ============================================================
# 14. Build the LangGraph
# ============================================================
# The reducer stages are regular nodes in the main graph. (Running them as
# a compiled subgraph that shares the State schema would re-apply the
# `sections` and `logs` reducers and duplicate their contents.)

graph = StateGraph(State)

graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("worker", worker_node)
graph.add_node("merge_content", merge_content)
graph.add_node("decide_images", decide_images)
graph.add_node("generate_and_place_images", generate_and_place_images)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)

graph.add_edge("research", "orchestrator")

graph.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"],
)

graph.add_edge("worker", "merge_content")
graph.add_edge("merge_content", "decide_images")
graph.add_edge("decide_images", "generate_and_place_images")
graph.add_edge("generate_and_place_images", END)

# Imported by bwa_frontend.py
app = graph.compile()