"""Shared HTTP session for all data-pull scripts.

This machine sits behind a FortiGate firewall that re-signs all HTTPS traffic
with its own CA. Two accommodations, both scoped to this project:
  1. Verify against data/ca_bundle.pem (certifi + the FortiGate CA).
  2. Drop OpenSSL's VERIFY_X509_STRICT flag (default-on since Python 3.13),
     because the FortiGate CA cert lacks the Authority Key Identifier
     extension that strict mode requires. Chain verification itself stays on.
"""
import ssl
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "ca_bundle.pem"


class _LenientStrictness(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=str(BUNDLE) if BUNDLE.exists() else None)
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _LenientStrictness())
    s.headers["User-Agent"] = "prem-predictor (personal project)"
    return s
