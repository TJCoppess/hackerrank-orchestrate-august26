from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router.models import IncomingMessage, RiskLevel
from router.scam import scan_message


def message(forwarded_count: int = 0) -> IncomingMessage:
    return IncomingMessage(
        message_id="msg_test", user_id="u_1", conversation_type="personal",
        created_at="2026-08-01 12:00", forwarded_count=forwarded_count,
    )


class ScamScannerTests(unittest.TestCase):
    def test_protective_otp_warning_is_clean(self) -> None:
        result = scan_message(message(), "Security reminder: never share your OTP with anyone.")
        self.assertEqual(result.risk_score, 0)
        self.assertNotIn("credential_request", {signal.code for signal in result.signals})

    def test_shortener_urgency_and_otp_combination_is_high(self) -> None:
        result = scan_message(message(), "Urgent: enter your OTP at bit.ly/claim right now")
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertIn("bit.ly", result.domains)
        self.assertIn("credential_plus_link", {signal.code for signal in result.signals})
        self.assertLessEqual(result.risk_score, 100)

    def test_official_domain_match_and_mismatch(self) -> None:
        matched = scan_message(message(), "Track at https://orders.amazon.in/a", "amazon.in")
        mismatch = scan_message(message(), "Track at amazon-delivery.xyz/a", "amazon.in")
        self.assertNotIn("business_domain_mismatch", {s.code for s in matched.signals})
        self.assertTrue(any(s.code.startswith("business_domain_") for s in mismatch.signals))

    def test_raw_ip_suspicious_tld_and_forwarding(self) -> None:
        result = scan_message(
            message(forwarded_count=8),
            "Claim prize at http://192.168.1.5/pay or reward-center.top",
        )
        codes = {signal.code for signal in result.signals}
        self.assertTrue({"raw_ip_url", "suspicious_tld", "high_forward_count"} <= codes)


if __name__ == "__main__":
    unittest.main()
