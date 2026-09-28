"""constraints.txt must stay consistent with requirements.txt and the Dockerfile.

The image installs exact versions from constraints.txt. CI does not build the
image, so a requirements.txt change (a Dependabot floor bump, say) that no
longer fits the lock would only surface as a failed Space build. This catches
it in the pull request instead: run `make lock` and commit the result.
"""
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> list[Requirement]:
    out = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(Requirement(line))
    return out


def _pins() -> dict[str, Version]:
    pins = {}
    for line in (ROOT / "constraints.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[canonicalize_name(name)] = Version(version.split(";")[0].strip())
    return pins


def test_every_requirement_is_pinned_within_its_range():
    pins = _pins()
    problems = []
    for req in _requirements():
        pinned = pins.get(canonicalize_name(req.name))
        if pinned is None:
            problems.append(f"{req.name}: missing from constraints.txt")
        elif not req.specifier.contains(pinned, prereleases=True):
            problems.append(f"{req.name}: pinned {pinned} does not satisfy '{req.specifier}'")
    assert not problems, "Run `make lock`: " + "; ".join(problems)


def test_ml_stack_pins_match_requirements_exactly():
    """The encoding stack is pinned on purpose; the lock must not drift from it."""
    pins = _pins()
    for req in _requirements():
        if req.name.lower() in {"flagembedding", "transformers", "sentence-transformers"}:
            exact = [s.version for s in req.specifier if s.operator == "=="]
            assert exact and pins[canonicalize_name(req.name)] == Version(exact[0])

