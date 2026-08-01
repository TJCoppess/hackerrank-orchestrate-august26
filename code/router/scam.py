from __future__ import annotations

import re
from urllib.parse import urlsplit

from .models import IncomingMessage, RiskLevel, ScamScanResult, ScamSignal


URL_RE = re.compile(
    r"(?i)\b((?:https?://|www\.)[^\s<>()]+|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>()]*)?)"
)
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.one",
}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "link", "site", "online", "icu", "zip"}
PROTECTIVE_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:do\s+not|don't|never|avoid)\s+(?:share|send|give|enter|reveal)\b.{0,30}\b(?:otp|one[- ]time password|password|pin|cvv|code)\b"
)
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:share|send|enter|provide|give|confirm|verify|reply\s+with)\b.{0,35}\b(?:otp|one[- ]time password|password|pin|cvv|verification code|security code)\b|\b(?:otp|one[- ]time password|password|pin|cvv|verification code|security code)\b.{0,35}\b(?:share|send|enter|provide|give|confirm|verify)\b"
)
URGENCY_RE = re.compile(
    r"(?i)\b(?:urgent|immediately|act now|right now|within \d+ (?:minutes?|hours?)|account (?:will be )?(?:blocked|closed|suspended)|final warning|last chance|expires? today)\b"
)
PAYMENT_RE = re.compile(
    r"(?i)\b(?:small|nominal|reattempt|processing|release|verification|delivery|customs?)\s+(?:fee|charge)|\bpay\s+(?:only\s+)?(?:rs\.?|₹|\$)?\s*\d+|\b(?:upi|bank transfer|gift card|crypto)\b"
)
BAIT_RE = re.compile(
    r"(?i)\b(?:won|winner|prize|lottery|reward|cashback|refund|unclaimed|free gift|claim now)\b"
)
IP_DOMAIN_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _normalize_url(raw: str) -> tuple[str, str]:
    cleaned = raw.rstrip(".,;:!?]}\"'")
    candidate = cleaned if "://" in cleaned else f"https://{cleaned}"
    parsed = urlsplit(candidate)
    domain = (parsed.hostname or "").lower().rstrip(".")
    return candidate, domain


def _domain_matches(candidate: str, official: str) -> bool:
    official = official.lower().strip().rstrip(".")
    return bool(official) and (
        candidate == official or candidate.endswith(f".{official}")
    )


def scan_message(
    message: IncomingMessage,
    effective_text: str,
    official_business_domain: str = "",
    sender_business_domain: str = "",
) -> ScamScanResult:
    """Return deterministic risk signals without choosing a routing action."""
    extracted = [_normalize_url(match.group(1)) for match in URL_RE.finditer(effective_text)]
    urls = list(dict.fromkeys(url for url, _ in extracted))
    domains = list(dict.fromkeys(domain for _, domain in extracted if domain))
    signals: list[ScamSignal] = []

    def add(code: str, weight: int, explanation: str) -> None:
        if not any(signal.code == code for signal in signals):
            signals.append(ScamSignal(code=code, weight=weight, explanation=explanation))

    if any(domain in SHORTENERS for domain in domains):
        add("url_shortener", 20, "Contains a commonly abused shortened URL.")
    if any(IP_DOMAIN_RE.fullmatch(domain) for domain in domains):
        add("raw_ip_url", 20, "Uses a raw IP address instead of a named domain.")
    if any(domain.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS for domain in domains):
        add("suspicious_tld", 12, "Contains a URL on a frequently abused TLD.")

    official = official_business_domain.lower().strip().rstrip(".")
    sender_domain = sender_business_domain.lower().strip().rstrip(".")
    if official and sender_domain and not _domain_matches(sender_domain, official):
        add(
            "business_domain_mismatch",
            30,
            "The business account is sending from a non-official domain.",
        )
    if official and domains and not any(_domain_matches(domain, official) for domain in domains):
        official_label = official.split(".")[0]
        if any(official_label in domain.replace("-", "") for domain in domains):
            add("business_domain_impersonation", 30, "URL resembles the business name but not its official domain.")
        else:
            add("business_domain_mismatch", 18, "URL does not match the business account's official domain.")

    protective = bool(PROTECTIVE_CREDENTIAL_RE.search(effective_text))
    credential_request = bool(CREDENTIAL_RE.search(effective_text)) and not protective
    if credential_request:
        add("credential_request", 45, "Requests an OTP, password, PIN, CVV, or security code.")
    if URGENCY_RE.search(effective_text):
        add("urgency_or_threat", 15, "Uses pressure, urgency, or an account threat.")
    if PAYMENT_RE.search(effective_text):
        add("payment_or_fee_bait", 15, "Requests a fee or risky payment method.")
    if BAIT_RE.search(effective_text):
        add("prize_or_refund_bait", 12, "Uses prize, reward, or refund bait.")
    if message.forwarded_count >= 5:
        add("high_forward_count", 10, "Message has been forwarded many times.")
    elif message.forwarded_count >= 2:
        add("forwarded_multiple_times", 5, "Message has been forwarded multiple times.")

    codes = {signal.code for signal in signals}
    if credential_request and domains:
        add("credential_plus_link", 20, "Combines a credential request with a link.")
    if "urgency_or_threat" in codes and (
        domains or "payment_or_fee_bait" in codes
    ):
        add("pressure_combination", 10, "Combines pressure with a link or payment request.")

    score = min(100, sum(signal.weight for signal in signals))
    level = RiskLevel.LOW if score < 30 else RiskLevel.MEDIUM if score < 60 else RiskLevel.HIGH
    return ScamScanResult(
        risk_score=score,
        risk_level=level,
        urls=urls,
        domains=domains,
        signals=signals,
    )
