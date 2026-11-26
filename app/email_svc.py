"""Sending email, for the weekly digest (app/digest_svc.py).

Hugging Face Spaces block outbound SMTP (ports 465 and 587), so mail goes out
over HTTPS through Resend's API (https://resend.com/docs/api-reference/emails/send-email),
with the app's shared httpx client: no new dependency.

Configured by two Space secrets; without both, nothing is sent and the digest
offers only its private link and RSS feed:

    RESEND_API_KEY   the API key
    DIGEST_FROM      the sender, e.g. "ResearchIT <digest@example.org>"

Resend delivers to other people only from a domain verified in its dashboard
(SPF and DKIM records). hf.space can't be verified, so email needs a domain
the owner controls; until then the test sender onboarding@resend.dev reaches
only the account's own address.
"""
from __future__ import annotations

import os

import httpx

from app import http_client

API = "https://api.resend.com/emails"


class NotConfigured(RuntimeError):
    pass


class SendFailed(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.getenv("RESEND_API_KEY", "").strip() and os.getenv("DIGEST_FROM", "").strip())


async def send(to: str, subject: str, html: str, text: str, headers: dict | None = None) -> str:
    """Send one email; returns the provider's message id. Raises NotConfigured or SendFailed."""
    key, sender = os.getenv("RESEND_API_KEY", "").strip(), os.getenv("DIGEST_FROM", "").strip()
    if not key or not sender:
        raise NotConfigured("RESEND_API_KEY and DIGEST_FROM are not both set")
    try:
        resp = await http_client.get_client().post(
            API, timeout=20,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"from": sender, "to": [to], "subject": subject, "html": html, "text": text,
                  "headers": headers or {}})
    except (httpx.TimeoutException, httpx.TransportError) as e:
        raise SendFailed(type(e).__name__) from e
    if resp.status_code >= 300:
        # The body names the cause (unverified domain, bad key, rate limit);
        # it never contains the key.
        raise SendFailed(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return (resp.json() or {}).get("id", "")
