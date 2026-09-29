from __future__ import annotations

import html
import io
import json
import re
import traceback
import zipfile
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import streamlit as st

from bwa_backend import (
    app,
    repair_markdown_citations,
    split_sources,
    normalize_url,
    validate_citation_markdown,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "blog_output"

DEFAULT_STYLE = "Auto"
STYLE_OPTIONS = ["Auto", "Technical", "Non-technical"]
TONE_OPTIONS = ["Informative", "Professional", "Conversational", "Academic"]

# ![alt](src "optional title") optionally followed by an italic caption line.
IMAGE_PATTERN = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+"[^"]*")?\)'
    r"(?:[ \t]*\r?\n(?:[ \t]*\r?\n)?[ \t]*\*(?P<caption>[^*\n]+)\*[ \t]*"
    r"(?=\r?\n|\Z))?"
)

# Inline citation produced by the backend: [1](https://source.example)
CITE_PATTERN = re.compile(r"(?<![!\[])\[(\d{1,3})\]\((https?://[^\s)\]]+)\)")

FENCE_PATTERN = re.compile(r"(```.*?```)", re.DOTALL)


# ============================================================
# STYLING
# ============================================================

CUSTOM_CSS = """
<style>
/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {
    width: 100%;
}

/* ---------- Download buttons ---------- */
.stDownloadButton > button {
    width: 100%;
}

/* ---------- Blog preview ---------- */
.st-key-blog_preview {
    max-width: 880px;
    margin: 0 auto;
    padding: 2.25rem 2.75rem;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    box-shadow: 0 2px 14px rgba(0, 0, 0, 0.06);
    font-size: 1.05rem;
    line-height: 1.75;
}

.st-key-blog_preview h1 {
    font-size: 2.2rem;
    font-weight: 750;
    line-height: 1.2;
    margin: 0 0 1.25rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid rgba(255, 75, 75, 0.85);
}

.st-key-blog_preview h2 {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 2.25rem 0 0.85rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.3);
}

.st-key-blog_preview h3 {
    font-size: 1.2rem;
    font-weight: 650;
    margin: 1.5rem 0 0.5rem 0;
}

.st-key-blog_preview p,
.st-key-blog_preview li {
    font-size: 1.05rem;
    line-height: 1.75;
}

.st-key-blog_preview ul,
.st-key-blog_preview ol {
    padding-left: 1.4rem;
}

.st-key-blog_preview blockquote {
    border-left: 4px solid rgba(255, 75, 75, 0.7);
    margin: 1rem 0;
    padding: 0.4rem 1rem;
    background: rgba(128, 128, 128, 0.08);
    border-radius: 0 8px 8px 0;
}

.st-key-blog_preview a {
    text-decoration: none;
    border-bottom: 1px solid rgba(255, 75, 75, 0.5);
}

.st-key-blog_preview a:hover {
    border-bottom-width: 2px;
}

.st-key-blog_preview sup.cite {
    font-size: 0.72em;
    margin-left: 0.15em;
    vertical-align: super;
    line-height: 0;
}

.st-key-blog_preview sup.cite + sup.cite {
    margin-left: 0.1em;
}

.st-key-blog_preview sup.cite a {
    font-weight: 700;
    border-bottom: none;
    padding: 0 0.12em;
    border-radius: 4px;
}

.st-key-blog_preview sup.cite a:hover {
    background: rgba(255, 75, 75, 0.15);
}

.st-key-blog_preview pre {
    border-radius: 10px;
}

.st-key-blog_preview code {
    font-size: 0.9em;
}

.st-key-blog_preview [data-testid="stImage"] img,
.st-key-blog_preview img {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
}

.st-key-blog_preview [data-testid="stImageCaption"] {
    text-align: center;
    font-style: italic;
    opacity: 0.8;
}

@media (max-width: 768px) {
    .st-key-blog_preview {
        padding: 1.25rem 1rem;
    }

    .st-key-blog_preview h1 {
        font-size: 1.7rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# GENERIC HELPERS
# ============================================================

def to_plain(value: Any) -> Any:
    """Recursively convert pydantic models and paths to plain Python data."""
    if hasattr(value, "model_dump"):
        return to_plain(value.model_dump())

    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def format_for_display(value: Any) -> str:
    """Format structured values for display in Streamlit."""
    if value is None:
        return "No information returned."

    value = to_plain(value)

    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def transform_outside_code(text: str, func) -> str:
    """Apply `func` only to the parts of `text` that are not fenced code."""
    parts = FENCE_PATTERN.split(text)
    return "".join(
        part if index % 2 else func(part)
        for index, part in enumerate(parts)
    )


# ============================================================
# PATH AND FILE HELPERS
# ============================================================

def resolve_output_path(path_value: str | Path | None) -> Path | None:
    """
    Resolve a backend-provided path.

    Relative paths are interpreted relative to the project directory.
    """
    if not path_value:
        return None

    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = APP_DIR / path

    try:
        return path.resolve()
    except OSError:
        return path


def read_text_file(path: Path) -> str:
    """Read a text file with a helpful fallback for encoding issues."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def find_blog_directories() -> list[Path]:
    """Return blog folders containing a blog.md file, newest first."""
    if not OUTPUT_DIR.exists():
        return []

    blog_files = list(OUTPUT_DIR.glob("*/blog.md"))
    blog_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return [path.parent for path in blog_files]


def list_blog_names() -> list[str]:
    """Return folder names for previously generated blogs."""
    return [folder.name for folder in find_blog_directories()]


def load_blog_from_folder(folder_name: str) -> dict[str, Any] | None:
    """Load a previously generated blog, its images and its metadata."""
    folder = resolve_output_path(OUTPUT_DIR / folder_name)

    if folder is None:
        return None

    markdown_path = folder / "blog.md"

    if not markdown_path.exists():
        return None

    markdown, citations_repaired = prepare_markdown(
        read_text_file(markdown_path)
    )

    metadata: dict[str, Any] = {}
    metadata_path = folder / "metadata.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(read_text_file(metadata_path))
        except (ValueError, OSError):
            metadata = {}

    return {
        "folder": folder,
        "markdown_path": markdown_path,
        "markdown": markdown,
        "image_dir": folder / "images",
        "plan": metadata.get("plan"),
        "evidence": metadata.get("evidence"),
        "image_specs": metadata.get("image_specs"),
        "logs": metadata.get("logs", []),
        "mode": metadata.get("mode"),
        "queries": metadata.get("queries"),
        "topic": metadata.get("topic"),
        "as_of": metadata.get("as_of"),
        "citations_repaired": citations_repaired,
    }


def prepare_markdown(markdown: str) -> tuple[str, int]:
    """
    Repair malformed inline citations (for example in blogs saved by an
    older version) using the article's own Sources list.

    Returns (markdown, number_of_problems_fixed).
    """
    before = validate_citation_markdown(markdown)

    if not before:
        return markdown, 0

    repaired = repair_markdown_citations(markdown)
    after = validate_citation_markdown(repaired)

    return repaired, max(0, len(before) - len(after))


def resolve_image_source(src: str, blog_folder: Path | None) -> Path | None:
    """
    Resolve a local Markdown image reference to an existing file.

    Relative paths are resolved against the blog folder, which keeps
    `images/name.png` working. If the exact path is missing, the file name
    is looked up in the blog's images folder.
    """
    src = unquote(src.strip().strip("<>"))

    if not src or src.startswith(("http://", "https://", "data:")):
        return None

    path = Path(src)
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((blog_folder if blog_folder else APP_DIR) / path)

    if blog_folder:
        candidates.append(blog_folder / "images" / path.name)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue

        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def collect_local_images(
    markdown: str,
    blog_folder: Path | None,
) -> list[tuple[str, Path]]:
    """
    Return (reference_as_written, resolved_path) for every local image in
    the Markdown, in order and without duplicates.
    """
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for match in IMAGE_PATTERN.finditer(markdown or ""):
        src = match.group("src")
        path = resolve_image_source(src, blog_folder)

        if path is None or str(path) in seen:
            continue

        seen.add(str(path))
        found.append((src, path))

    return found


def make_blog_zip(markdown: str, blog_folder: Path | None) -> bytes:
    """
    Create a ZIP containing blog.md and every local image it references.
    Image references are rewritten to images/<name> inside the archive.
    """
    references = collect_local_images(markdown, blog_folder)

    zip_markdown = markdown
    used_names: set[str] = set()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        for reference, image_path in references:
            name = image_path.name
            stem, suffix = image_path.stem, image_path.suffix
            counter = 2

            while name in used_names:
                name = f"{stem}-{counter}{suffix}"
                counter += 1

            used_names.add(name)
            archive_name = f"images/{name}"

            zip_markdown = zip_markdown.replace(
                f"({reference})",
                f"({archive_name})",
            )
            zip_file.write(image_path, archive_name)

        zip_file.writestr("blog.md", zip_markdown)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def make_markdown_download(markdown: str) -> bytes:
    return markdown.encode("utf-8")


# ============================================================
# BACKEND RESULT HELPERS
# ============================================================

def extract_markdown(result: dict[str, Any]) -> str:
    """Get the final Markdown from the backend result."""
    for key in ["final", "markdown", "blog", "final_blog", "content"]:
        value = result.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def extract_output_path(result: dict[str, Any]) -> Path | None:
    """Find the saved Markdown path returned by the backend."""
    for key in ["markdown_path", "blog_path", "output_path", "saved_path"]:
        value = result.get(key)

        if isinstance(value, (str, Path)):
            path = resolve_output_path(value)

            if path and path.exists() and path.is_file():
                return path

    return None


def extract_blog_folder(
    result: dict[str, Any],
    markdown_path: Path | None,
) -> Path | None:
    """Find the blog output folder."""
    if markdown_path:
        return markdown_path.parent

    for key in ["output_dir", "blog_dir", "folder"]:
        value = result.get(key)

        if isinstance(value, (str, Path)):
            path = resolve_output_path(value)

            if path and path.exists() and path.is_dir():
                return path

    return None


# ============================================================
# MARKDOWN PREVIEW RENDERING
# ============================================================

def split_markdown_with_images(markdown: str) -> list[dict[str, str]]:
    """
    Split Markdown into ordered text and image segments so images are
    rendered exactly where the Markdown references them.
    """
    segments: list[dict[str, str]] = []
    cursor = 0

    for match in IMAGE_PATTERN.finditer(markdown):
        if match.start() > cursor:
            segments.append(
                {"type": "text", "text": markdown[cursor:match.start()]}
            )

        segments.append(
            {
                "type": "image",
                "alt": match.group("alt") or "",
                "src": match.group("src"),
                "caption": (match.group("caption") or "").strip(),
            }
        )
        cursor = match.end()

    if cursor < len(markdown):
        segments.append({"type": "text", "text": markdown[cursor:]})

    return segments


def render_text_segment(text: str, url_titles: dict[str, str]) -> None:
    """Render a Markdown text block with styled, clickable citations."""
    if not text.strip():
        return

    def cite_html(match: re.Match) -> str:
        number = match.group(1)
        url = match.group(2)
        title = url_titles.get(url, url)

        return (
            '<sup class="cite">'
            f'<a href="{html.escape(url, quote=True)}" '
            'target="_blank" rel="noopener noreferrer" '
            f'title="{html.escape(title, quote=True)}">[{number}]</a>'
            "</sup>"
        )

    text = transform_outside_code(
        text,
        lambda part: CITE_PATTERN.sub(cite_html, part),
    )

    st.markdown(text.strip("\n"), unsafe_allow_html=True)


def render_blog_preview(
    markdown: str,
    blog_folder: Path | None,
    evidence: Any,
) -> None:
    """Render the article with inline images and citations, in order."""
    url_titles: dict[str, str] = {}

    for item in to_plain(evidence) or []:
        if isinstance(item, dict) and item.get("url"):
            url_titles[str(item["url"])] = str(
                item.get("title") or item["url"]
            )

    try:
        container = st.container(key="blog_preview")
    except TypeError:
        # Older Streamlit versions do not support container keys.
        container = st.container()

    with container:
        for segment in split_markdown_with_images(markdown):
            if segment["type"] == "text":
                render_text_segment(segment["text"], url_titles)
                continue

            src = segment["src"].strip().strip("<>")
            caption = segment["caption"] or segment["alt"] or None

            if src.startswith(("http://", "https://", "data:")):
                st.image(src, caption=caption)
                continue

            image_path = resolve_image_source(src, blog_folder)

            if image_path is not None:
                st.image(str(image_path), caption=caption)
            else:
                st.caption(f"🖼️ Image not found: `{src}`")


# ============================================================
# SESSION STATE
# ============================================================

if "current_blog" not in st.session_state:
    st.session_state.current_blog = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "last_traceback" not in st.session_state:
    st.session_state.last_traceback = None


# ============================================================
# PAGE HEADER
# ============================================================

st.title("✍️ Blog Writing Agent")
st.caption(
    "Generate researched blog posts with a LangGraph workflow, "
    "supporting evidence, inline citations, and optional images."
)

st.divider()


# ============================================================
# SIDEBAR — BLOG INPUTS
# ============================================================

with st.sidebar:
    st.header("Create a Blog")

    topic = st.text_area(
        "Blog topic",
        placeholder="Example: How large language models work",
        height=120,
        key="topic_input",
    )

    style = st.selectbox(
        "Writing style",
        options=STYLE_OPTIONS,
        index=0,
        help=(
            "Auto lets the agent choose a suitable style. "
            "Technical is intended for specialist readers; "
            "Non-technical is intended for general readers."
        ),
    )

    as_of_date = st.date_input(
        "As-of date",
        value=date.today(),
        help=(
            "The agent treats this as 'today' when deciding what counts "
            "as recent and when filtering news sources."
        ),
    )

    audience = st.text_input(
        "Target audience",
        placeholder="Example: College students",
    )

    tone = st.selectbox(
        "Tone",
        options=TONE_OPTIONS,
        index=0,
    )

    include_images = st.checkbox(
        "Generate images",
        value=True,
        help="Generate up to three images with Cloudflare Workers AI.",
    )

    generate_button = st.button(
        "Generate Blog",
        type="primary",
        disabled=not topic.strip(),
    )

    st.divider()

    st.subheader("Previous Blogs")

    previous_blogs = list_blog_names()

    if previous_blogs:
        selected_blog = st.selectbox(
            "Load a previous blog",
            options=previous_blogs,
            index=None,
            placeholder="Select a saved blog",
            format_func=lambda name: name.replace("_", " ").title(),
        )

        if st.button("Load Selected Blog", disabled=not selected_blog):
            loaded_blog = load_blog_from_folder(selected_blog)

            if loaded_blog:
                st.session_state.current_blog = loaded_blog
                st.session_state.last_result = None
                st.session_state.last_error = None
                st.session_state.last_traceback = None
                st.rerun()
            else:
                st.error("Could not find blog.md in that folder.")
    else:
        st.caption("No previously generated blogs found.")


# ============================================================
# GENERATE BLOG
# ============================================================

if generate_button:
    # Keys match the backend's State: topic, blog_style, as_of, audience,
    # tone, include_images.
    inputs = {
        "topic": topic.strip(),
        "blog_style": style,
        "as_of": as_of_date.isoformat(),
        "audience": audience.strip(),
        "tone": tone,
        "include_images": include_images,
    }

    st.session_state.last_error = None
    st.session_state.last_traceback = None
    st.session_state.current_blog = None
    st.session_state.last_result = None

    try:
        with st.spinner("Researching and writing your blog..."):
            result = app.invoke(inputs)

        if not isinstance(result, dict):
            raise TypeError(
                "The backend returned an unexpected result. "
                "Expected a dictionary from app.invoke()."
            )

        markdown = extract_markdown(result)

        if not markdown.strip():
            raise ValueError(
                "The backend completed but did not return blog Markdown. "
                "Check that the backend returns the final content under "
                "the `final` key."
            )

        markdown, citations_repaired = prepare_markdown(markdown)

        markdown_path = extract_output_path(result)
        blog_folder = extract_blog_folder(result, markdown_path)

        if blog_folder is None and markdown_path:
            blog_folder = markdown_path.parent

        if blog_folder is None:
            blog_folder = OUTPUT_DIR

        if markdown_path is None:
            markdown_path = blog_folder / "blog.md"

        st.session_state.current_blog = {
            "folder": blog_folder,
            "markdown_path": markdown_path,
            "markdown": markdown,
            "image_dir": blog_folder / "images",
            "plan": to_plain(result.get("plan")),
            "evidence": to_plain(result.get("evidence")),
            "image_specs": to_plain(result.get("image_specs")),
            "logs": result.get("logs", []),
            "mode": result.get("mode"),
            "queries": result.get("queries"),
            "topic": inputs["topic"],
            "as_of": inputs["as_of"],
            "citations_repaired": citations_repaired,
        }
        st.session_state.last_result = result

    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        st.session_state.last_traceback = traceback.format_exc()


# ============================================================
# ERROR MESSAGE
# ============================================================

if st.session_state.last_error:
    st.error("Blog generation failed.")
    st.code(st.session_state.last_error)

    if st.session_state.last_traceback:
        with st.expander("Show technical details"):
            st.code(st.session_state.last_traceback)


# ============================================================
# DISPLAY CURRENT BLOG
# ============================================================

current_blog = st.session_state.current_blog

if current_blog:
    markdown = current_blog.get("markdown", "")
    markdown_path = current_blog.get("markdown_path")
    blog_folder = current_blog.get("folder")
    blog_folder_path = Path(blog_folder) if blog_folder else None

    st.success("Blog is ready.")

    if current_blog.get("citations_repaired"):
        st.info(
            "Malformed citations in this blog were repaired for display and "
            "download using its Sources list. The saved file on disk is "
            "unchanged."
        )

    missing_images = [
        match.group("src")
        for match in IMAGE_PATTERN.finditer(markdown)
        if not match.group("src").startswith(("http://", "https://", "data:"))
        and resolve_image_source(match.group("src"), blog_folder_path) is None
    ]

    if missing_images:
        st.warning(
            "These images are referenced in the blog but were not found on "
            "disk, so they are missing from the preview and the ZIP: "
            + ", ".join(f"`{name}`" for name in missing_images)
        )

    # Download controls
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Download Markdown",
            data=make_markdown_download(markdown),
            file_name="blog.md",
            mime="text/markdown",
        )

    with download_col2:
        zip_name = (
            blog_folder_path.name if blog_folder_path else "blog"
        ) or "blog"

        st.download_button(
            label="Download Blog ZIP",
            data=make_blog_zip(markdown, blog_folder_path),
            file_name=f"{zip_name}.zip",
            mime="application/zip",
            help="Includes blog.md and every local image it references.",
        )

    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
        [
            "Plan",
            "Evidence",
            "Markdown Preview",
            "Images",
            "Logs",
        ]
    )

    # --------------------------------------------------------
    # PLAN TAB
    # --------------------------------------------------------

    with tab_plan:
        st.subheader("Blog Plan")

        plan = to_plain(current_blog.get("plan"))

        if isinstance(plan, dict) and plan:
            st.markdown(f"**Title:** {plan.get('blog_title', '')}")

            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.markdown(f"**Audience:** {plan.get('audience', '')}")
            meta_col2.markdown(f"**Tone:** {plan.get('tone', '')}")
            meta_col3.markdown(f"**Kind:** {plan.get('blog_kind', '')}")

            if plan.get("constraints"):
                st.markdown(
                    "**Constraints:** " + "; ".join(plan["constraints"])
                )

            for task in plan.get("tasks", []):
                label = f"{task.get('id', '')}. {task.get('title', '')}"

                with st.expander(label):
                    st.markdown(f"**Goal:** {task.get('goal', '')}")

                    for bullet in task.get("bullets", []):
                        st.markdown(f"- {bullet}")

                    st.caption(
                        f"Target words: {task.get('target_words', '')} · "
                        f"Code: {task.get('requires_code', False)} · "
                        f"Citations: {task.get('requires_citations', True)}"
                    )

            with st.expander("Raw plan JSON"):
                st.code(format_for_display(plan), language="json")
        else:
            st.info("No plan was returned by the backend.")

    # --------------------------------------------------------
    # EVIDENCE TAB
    # --------------------------------------------------------

    with tab_evidence:
        st.subheader("Research Evidence")

        evidence = to_plain(current_blog.get("evidence"))

        # Map each source URL to the [n] it carries in the article.
        _, article_sources, _, _ = split_sources(markdown)
        cited_as = {
            normalize_url(url): number
            for number, url in article_sources.items()
        }

        if current_blog.get("mode"):
            st.caption(f"Research mode: {current_blog.get('mode')}")

        if current_blog.get("queries"):
            with st.expander("Search queries"):
                for query in current_blog["queries"]:
                    st.markdown(f"- {query}")

        if evidence:
            for index, item in enumerate(evidence, start=1):
                if not isinstance(item, dict):
                    continue

                title = item.get("title") or item.get("url") or "Untitled"
                url = item.get("url") or ""
                published = item.get("published_at") or "unknown date"
                source = item.get("source") or ""

                st.markdown(f"**{index}. [{title}]({url})**")

                article_number = cited_as.get(normalize_url(url))
                badge = (
                    f"cited in the article as [{article_number}]"
                    if article_number
                    else "not cited in the article"
                )
                st.caption(
                    (f"{source} · " if source else "")
                    + f"{published} · {badge}"
                )

                if item.get("snippet"):
                    st.write(item["snippet"])

                st.divider()
        else:
            st.info(
                "No evidence was returned. The topic may not have needed "
                "research, or no Tavily results were available."
            )

    # --------------------------------------------------------
    # MARKDOWN PREVIEW TAB
    # --------------------------------------------------------

    with tab_preview:
        render_blog_preview(
            markdown,
            blog_folder_path,
            current_blog.get("evidence"),
        )

        with st.expander("View raw Markdown"):
            st.code(markdown, language="markdown")

    # --------------------------------------------------------
    # IMAGES TAB
    # --------------------------------------------------------

    with tab_images:
        st.subheader("Generated Images")

        image_refs = collect_local_images(markdown, blog_folder_path)

        if image_refs:
            image_columns = st.columns(2)

            for index, (_, image_path) in enumerate(image_refs):
                with image_columns[index % 2]:
                    st.image(str(image_path), caption=image_path.name)
        else:
            st.info(
                "No local images were found in the blog Markdown. "
                "Images may be disabled, or generation may have failed "
                "(see the Logs tab)."
            )

        image_specs = current_blog.get("image_specs")

        if image_specs:
            with st.expander("Image specifications"):
                st.code(format_for_display(image_specs), language="json")

    # --------------------------------------------------------
    # LOGS TAB
    # --------------------------------------------------------

    with tab_logs:
        st.subheader("Execution Logs")

        logs = current_blog.get("logs")

        if logs:
            if isinstance(logs, (list, tuple)):
                st.code("\n".join(str(line) for line in logs))
            else:
                st.code(format_for_display(logs))
        else:
            st.info(
                "No execution logs were returned. "
                "Backend tracing may still be available through LangSmith."
            )

else:
    st.info(
        "Enter a topic in the sidebar and click **Generate Blog** "
        "to create a new blog."
    )