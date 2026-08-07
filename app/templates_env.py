"""
Shared Jinja2 environment with custom filters.
Import `templates` from here instead of creating it per-router.
"""
import json
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")


def _tojson_parse(value: str | None) -> list:
    """Parse a JSON-encoded string into a Python list. Returns [] on error."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


templates.env.filters["tojson_parse"] = _tojson_parse


# Exposed to every template so the map links do not have to be threaded through
# each route's context dict. Both are inert when SPACE_APP_URL is unset: the
# templates test them and render nothing, so an undeployed map leaves no dead
# links behind.
from app import config  # noqa: E402  (after `templates` exists, avoids a cycle)

templates.env.globals["space_app_url"] = config.SPACE_APP_URL
templates.env.globals["space_paper_url"] = config.space_paper_url
