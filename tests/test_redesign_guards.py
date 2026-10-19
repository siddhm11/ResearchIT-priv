"""
Static guards for the redesign: rules that look harmless in review.

Each reads source files rather than running them, like
tests/test_event_loop_blocking.py, so a violation fails CI the moment it is
written.
"""
import ast
import pathlib
import re

APP = pathlib.Path("app")
TEMPLATES = APP / "templates"

# CPU-heavy calls that must run via asyncio.to_thread, never inside a coroutine.
HEAVY_CALLS = {("arxiv_html", "parse"), ("Image", "open")}

# Templates that used `| safe` before the redesign, with the reason it is safe.
# ai_summary.html renders HTML built and escaped in app/routers/search.py.
SAFE_ALLOWED = {"partials/ai_summary.html"}

# Modules that predate the strong-reference rule and keep their own task sets.
CREATE_TASK_ALLOWED = {"hybrid_search_svc.py", "discovery_runtime.py", "turso_sync.py", "events.py"}
# figure_prewarm.py stores its one task in a module global, which is a strong reference.


def _async_calls(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.AsyncFunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and isinstance(node.func.value, ast.Name):
                    yield fn, node, (node.func.value.id, node.func.attr)


def test_heavy_parsing_never_runs_inside_a_coroutine():
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for fn, node, name in _async_calls(tree):
            if name in HEAVY_CALLS:
                offenders.append(f"{path}:{node.lineno} {name[0]}.{name[1]}() inside async def {fn.name}")
    assert not offenders, "use asyncio.to_thread:\n" + "\n".join(offenders)


def test_the_guard_catches_a_direct_parse_call():
    tree = ast.parse("async def f(html):\n    return arxiv_html.parse(html, 'x')\n")
    assert any(name == ("arxiv_html", "parse") for _, _, name in _async_calls(tree))


def test_the_shared_http_client_is_never_closed_by_async_with():
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            # The client itself as the context manager closes the shared pool;
            # `async with client.stream(...)` only closes one response.
            if re.search(r"async\s+with\s+(http_client\.)?get_client\(\)\s*(as\b|:)", line):
                offenders.append(f"{path}:{i}")
    assert not offenders, "async with closes the shared pool for everyone:\n" + "\n".join(offenders)


def test_the_client_guard_flags_closing_the_client_but_not_a_stream():
    pattern = r"async\s+with\s+(http_client\.)?get_client\(\)\s*(as\b|:)"
    assert re.search(pattern, "async with http_client.get_client() as c:")
    assert not re.search(pattern, "async with http_client.get_client().stream('GET', u) as r:")


def test_new_modules_keep_a_reference_to_their_tasks():
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if path.name in CREATE_TASK_ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("asyncio.create_task(") or stripped.startswith("loop.create_task("):
                offenders.append(f"{path}:{i}: {stripped}")
    assert not offenders, "store the task (asyncio keeps only a weak reference):\n" + "\n".join(offenders)


def test_no_new_template_marks_output_safe():
    offenders = []
    for path in sorted(TEMPLATES.rglob("*.html")):
        rel = str(path.relative_to(TEMPLATES))
        if rel in SAFE_ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\|\s*safe\b", line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, "model output and arXiv text are untrusted; escape them:\n" + "\n".join(offenders)


def test_pages_load_scripts_and_styles_only_from_our_origin():
    offenders = []
    for path in sorted(TEMPLATES.rglob("*.html")):
        text = path.read_text()
        for m in re.finditer(r'<(?:script[^>]*\bsrc|link[^>]*\bhref)="(https?:)?//', text):
            offenders.append(f"{path.relative_to(TEMPLATES)}: {m.group(0)}")
    assert not offenders, "base.html promises no outside origins:\n" + "\n".join(offenders)
