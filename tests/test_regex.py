"""
Advanced unit tests for the Regex Detector.

Covers every pattern exhaustively: valid formats, invalid formats,
boundary conditions, character offset correctness, Luhn validation,
edge cases, and mixed-entity scenarios.
"""

import pytest
from app.detectors.base import EntityCategory, EntityType
from app.detectors.regex import RegexDetector, luhn_check


@pytest.fixture
def detector():
    return RegexDetector()


# ═══════════════════════════════════════════════════════════════
#  LUHN ALGORITHM
# ═══════════════════════════════════════════════════════════════


class TestLuhnAlgorithm:
    """Luhn validation: valid cards, invalid cards, edge cases."""

    @pytest.mark.parametrize(
        "number,expected",
        [
            # Valid Luhn numbers (known test vectors)
            ("4532015112830366", True),  # Visa
            ("5425233430109903", True),  # Mastercard
            ("4916338506082832", True),  # Visa
            ("6011111111111117", True),  # Discover
            ("3530111333300000", True),  # JCB 16-digit
            # Invalid — one digit off
            ("4532015112830367", False),
            ("4532015112830365", False),
            # Random invalid
            ("1234567890123456", False),
            ("9999999999999999", False),
            ("1111111111111111", False),  # sum=24, not divisible by 10
            # Wrong length — always False
            ("453201511283036", False),  # 15 digits
            ("45320151128303660", False),  # 17 digits
            ("123", False),
            ("", False),
        ],
    )
    def test_luhn_numeric(self, number, expected):
        assert luhn_check(number) == expected

    def test_luhn_strips_spaces(self):
        """luhn_check strips non-digit chars — spaced card = same as compact."""
        assert luhn_check("4532 0151 1283 0366") is True

    def test_luhn_strips_dashes(self):
        assert luhn_check("4532-0151-1283-0366") is True

    def test_luhn_all_letters_is_false(self):
        """Non-numeric string has no digits → length check fails."""
        assert luhn_check("abcdefghijklmnop") is False

    def test_luhn_mixed_alpha_digit(self):
        """Mixed string: only 4 digits extracted → length ≠ 16 → False."""
        assert luhn_check("card-4532-invalid") is False


# ═══════════════════════════════════════════════════════════════
#  EMAIL
# ═══════════════════════════════════════════════════════════════


class TestEmailDetection:
    """Email detection: standard formats, edge cases, true negatives."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text,expected_match",
        [
            ("user@example.com", "user@example.com"),
            ("john.doe@company.org", "john.doe@company.org"),
            ("user+tag@sub.domain.co.uk", "user+tag@sub.domain.co.uk"),
            ("123numeric@test.io", "123numeric@test.io"),
            ("user_name@hyphen-domain.com", "user_name@hyphen-domain.com"),
            ("UPPER@EXAMPLE.COM", "UPPER@EXAMPLE.COM"),
            ("a@b.co", "a@b.co"),  # Minimal valid
        ],
    )
    async def test_detects_valid_email(self, detector, text, expected_match):
        findings = await detector.detect(text)
        emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
        assert len(emails) >= 1
        assert any(f.matched_text == expected_match for f in emails)

    @pytest.mark.asyncio
    async def test_email_confidence_and_metadata(self, detector):
        findings = await detector.detect("hello@world.com")
        emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
        assert len(emails) == 1
        assert emails[0].confidence == 0.95
        assert emails[0].detector == "regex"
        assert emails[0].category == EntityCategory.PII

    @pytest.mark.asyncio
    async def test_email_offsets_exact(self, detector):
        text = "Send to john@test.com please."
        findings = await detector.detect(text)
        emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
        assert len(emails) == 1
        f = emails[0]
        assert text[f.start : f.end] == "john@test.com"

    @pytest.mark.asyncio
    async def test_multiple_emails_in_sentence(self, detector):
        text = "CC alice@example.com and bob@work.io on this."
        findings = await detector.detect(text)
        emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
        assert len(emails) == 2
        matched = {f.matched_text for f in emails}
        assert "alice@example.com" in matched
        assert "bob@work.io" in matched

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "no at sign here",
            "user at example dot com",  # Obfuscated — regex is blind
            "email validation checks for @-sign",
            "just a sentence about email protocols",
        ],
    )
    async def test_no_false_positive_email(self, detector, text):
        findings = await detector.detect(text)
        emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
        assert len(emails) == 0


# ═══════════════════════════════════════════════════════════════
#  PHONE
# ═══════════════════════════════════════════════════════════════


class TestPhoneDetection:
    """Phone detection: all standard US formats, true negatives."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "Call 555-123-4567",
            "Call (555) 123-4567",
            "Call 555.123.4567",
            "Call 555 123 4567",
            "Call 5551234567",
            "Call +1-555-123-4567",
            "Call +1 555 123 4567",
            "Call 1-800-555-1234",
        ],
    )
    async def test_detects_phone_format(self, detector, text):
        findings = await detector.detect(text)
        phones = [f for f in findings if f.entity_type == EntityType.PHONE]
        assert len(phones) >= 1, f"Expected phone in: {text!r}"

    @pytest.mark.asyncio
    async def test_phone_confidence_and_metadata(self, detector):
        findings = await detector.detect("Call me at 555-123-4567.")
        phones = [f for f in findings if f.entity_type == EntityType.PHONE]
        assert len(phones) == 1
        assert phones[0].confidence == 0.95
        assert phones[0].detector == "regex"
        assert phones[0].category == EntityCategory.PII

    @pytest.mark.asyncio
    async def test_phone_offsets_exact(self, detector):
        text = "Number: 555-123-4567 end"
        findings = await detector.detect(text)
        phones = [f for f in findings if f.entity_type == EntityType.PHONE]
        assert len(phones) == 1
        f = phones[0]
        assert text[f.start : f.end] == "555-123-4567"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "I have 5 items and 12 apples",
            "call me five five five one two three four five six seven",  # Spelled out
            "12345",  # Too short
            "12345678",  # 8 digits — not a valid phone
        ],
    )
    async def test_no_false_positive_phone(self, detector, text):
        findings = await detector.detect(text)
        phones = [f for f in findings if f.entity_type == EntityType.PHONE]
        assert len(phones) == 0, f"Unexpected phone in: {text!r}"


# ═══════════════════════════════════════════════════════════════
#  SSN
# ═══════════════════════════════════════════════════════════════


class TestSSNDetection:
    """SSN detection: dashed format required, alternatives should not fire."""

    @pytest.mark.asyncio
    async def test_detects_ssn_standard(self, detector):
        findings = await detector.detect("SSN: 123-45-6789")
        ssns = [f for f in findings if f.entity_type == EntityType.SSN]
        assert len(ssns) == 1
        assert ssns[0].matched_text == "123-45-6789"
        assert ssns[0].confidence == 0.95
        assert ssns[0].category == EntityCategory.PII

    @pytest.mark.asyncio
    async def test_ssn_offsets_exact(self, detector):
        text = "Her SSN is 456-78-9012 on file."
        findings = await detector.detect(text)
        ssns = [f for f in findings if f.entity_type == EntityType.SSN]
        assert len(ssns) == 1
        f = ssns[0]
        assert text[f.start : f.end] == "456-78-9012"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "Number 123456789",  # No dashes
            "Social 12-345-6789",  # Wrong grouping
            "Social 1234-56-789",  # Wrong grouping
            "Ref 123-456-789",  # 3-3-3, not 3-2-4
            "Code 12-34-5678",  # 2-2-4
        ],
    )
    async def test_no_ssn_wrong_format(self, detector, text):
        findings = await detector.detect(text)
        ssns = [f for f in findings if f.entity_type == EntityType.SSN]
        assert len(ssns) == 0, f"Unexpected SSN in: {text!r}"

    @pytest.mark.asyncio
    async def test_multiple_ssns(self, detector):
        text = "Records: 111-22-3333 and 444-55-6666"
        findings = await detector.detect(text)
        ssns = [f for f in findings if f.entity_type == EntityType.SSN]
        assert len(ssns) == 2


# ═══════════════════════════════════════════════════════════════
#  CREDIT CARD
# ═══════════════════════════════════════════════════════════════


class TestCreditCardDetection:
    """Credit card: Luhn validation, separator formats, invalid numbers."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text,card_number",
        [
            ("Card: 4532015112830366", "4532015112830366"),  # Visa compact
            ("Card: 4532 0151 1283 0366", "4532 0151 1283 0366"),  # Visa spaced
            ("Card: 4532-0151-1283-0366", "4532-0151-1283-0366"),  # Visa dashed
            ("Card: 5425233430109903", "5425233430109903"),  # Mastercard
            ("Card: 6011111111111117", "6011111111111117"),  # Discover
        ],
    )
    async def test_detects_valid_card(self, detector, text, card_number):
        findings = await detector.detect(text)
        ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
        assert len(ccs) == 1
        assert ccs[0].matched_text == card_number
        assert ccs[0].confidence == 0.99
        assert ccs[0].category == EntityCategory.PII

    @pytest.mark.asyncio
    async def test_rejects_invalid_luhn(self, detector):
        """A 16-digit number that fails Luhn must not be flagged."""
        findings = await detector.detect("Card: 1234567890123456")
        ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
        assert len(ccs) == 0

    @pytest.mark.asyncio
    async def test_rejects_15_digit_amex_style(self, detector):
        """15-digit Amex numbers don't match 16-digit pattern."""
        findings = await detector.detect("Card: 378282246310005")
        ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
        assert len(ccs) == 0

    @pytest.mark.asyncio
    async def test_credit_card_offsets_exact(self, detector):
        text = "Pay with 4532015112830366 today"
        findings = await detector.detect(text)
        ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
        assert len(ccs) == 1
        f = ccs[0]
        assert text[f.start : f.end] == "4532015112830366"


# ═══════════════════════════════════════════════════════════════
#  AWS KEY
# ═══════════════════════════════════════════════════════════════


class TestAWSKeyDetection:
    """AWS access key: AKIA prefix + 16 uppercase alphanumeric."""

    @pytest.mark.asyncio
    async def test_detects_aws_key_standard(self, detector):
        findings = await detector.detect("Key: AKIAIOSFODNN7EXAMPLE")
        aws = [f for f in findings if f.entity_type == EntityType.AWS_KEY]
        assert len(aws) == 1
        assert aws[0].matched_text == "AKIAIOSFODNN7EXAMPLE"
        assert aws[0].confidence == 0.95
        assert aws[0].category == EntityCategory.SECRET
        assert aws[0].detector == "regex"

    @pytest.mark.asyncio
    async def test_aws_key_offsets_exact(self, detector):
        text = "export AWS_KEY=AKIAIOSFODNN7EXAMPLE here"
        findings = await detector.detect(text)
        aws = [f for f in findings if f.entity_type == EntityType.AWS_KEY]
        assert len(aws) == 1
        f = aws[0]
        assert text[f.start : f.end] == "AKIAIOSFODNN7EXAMPLE"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "Key: ABIAIOSFODNN7EXAMPLE",  # Wrong prefix
            "Key: AKIAiosfodnn7example",  # Lowercase (pattern needs [0-9A-Z])
            "Key: AKIA123",  # Too short after prefix
            "we use AWS for hosting",  # AWS mention, no key
        ],
    )
    async def test_no_aws_key_wrong_format(self, detector, text):
        findings = await detector.detect(text)
        aws = [f for f in findings if f.entity_type == EntityType.AWS_KEY]
        assert len(aws) == 0, f"Unexpected AWS key in: {text!r}"


# ═══════════════════════════════════════════════════════════════
#  GITHUB TOKEN
# ═══════════════════════════════════════════════════════════════


class TestGitHubTokenDetection:
    """GitHub token: ghp_ (36 chars) and github_pat_ (22+ chars) formats."""

    @pytest.mark.asyncio
    async def test_detects_ghp_token(self, detector):
        token = "ghp_" + "A" * 36
        findings = await detector.detect(f"Token: {token}")
        gh = [f for f in findings if f.entity_type == EntityType.GITHUB_TOKEN]
        assert len(gh) == 1
        assert gh[0].matched_text == token
        assert gh[0].confidence == 0.95
        assert gh[0].category == EntityCategory.SECRET

    @pytest.mark.asyncio
    async def test_detects_github_pat_token(self, detector):
        token = "github_pat_" + "a1B2c3D4e5F6g7H8i9J0kK"  # 22 chars after prefix
        findings = await detector.detect(f"Token: {token}")
        gh = [f for f in findings if f.entity_type == EntityType.GITHUB_TOKEN]
        assert len(gh) == 1

    @pytest.mark.asyncio
    async def test_ghp_token_wrong_length(self, detector):
        """ghp_ + 35 chars = too short — no match."""
        token = "ghp_" + "A" * 35
        findings = await detector.detect(f"Token: {token}")
        gh = [f for f in findings if f.entity_type == EntityType.GITHUB_TOKEN]
        assert len(gh) == 0

    @pytest.mark.asyncio
    async def test_no_github_token_wrong_prefix(self, detector):
        findings = await detector.detect(
            "Token: gha_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
        )
        gh = [f for f in findings if f.entity_type == EntityType.GITHUB_TOKEN]
        assert len(gh) == 0


# ═══════════════════════════════════════════════════════════════
#  JWT
# ═══════════════════════════════════════════════════════════════


class TestJWTDetection:
    """JWT detection: three-part dot-separated base64url tokens."""

    @pytest.mark.asyncio
    async def test_detects_valid_jwt(self, detector):
        jwt = (
            "eyJhbGciOiJIUzI1NiJ9"
            ".eyJzdWIiOiJ1c2VyMSIsImlhdCI6MTYwMH0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        findings = await detector.detect(f"Authorization: Bearer {jwt}")
        jwts = [f for f in findings if f.entity_type == EntityType.JWT]
        assert len(jwts) == 1
        assert jwts[0].confidence == 0.95
        assert jwts[0].category == EntityCategory.SECRET

    @pytest.mark.asyncio
    async def test_no_jwt_two_parts_only(self, detector):
        """JWT requires three dot-separated parts."""
        token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9"
        findings = await detector.detect(f"Token: {token}")
        jwts = [f for f in findings if f.entity_type == EntityType.JWT]
        assert len(jwts) == 0

    @pytest.mark.asyncio
    async def test_no_jwt_wrong_prefix(self, detector):
        """JWT header must start with eyJ (base64 of '{')."""
        token = "abcXYZ1234567890ab.eyJzdWIiOiJ1c2VyMSJ9.SflKxwRJSMeKKF2QT4fw"
        findings = await detector.detect(f"Token: {token}")
        jwts = [f for f in findings if f.entity_type == EntityType.JWT]
        assert len(jwts) == 0


# ═══════════════════════════════════════════════════════════════
#  MIXED ENTITIES & EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestMixedAndEdgeCases:
    """Multi-entity texts, empty input, and offset invariants."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_no_findings(self, detector):
        findings = await detector.detect("")
        assert findings == []

    @pytest.mark.asyncio
    async def test_plain_text_returns_no_findings(self, detector):
        findings = await detector.detect("The quick brown fox jumps over the lazy dog.")
        assert findings == []

    @pytest.mark.asyncio
    async def test_detects_all_three_pii_types_in_one_string(self, detector):
        text = "Email: alice@example.com, Phone: 555-867-5309, SSN: 987-65-4321"
        findings = await detector.detect(text)
        types = {f.entity_type for f in findings}
        assert EntityType.EMAIL in types
        assert EntityType.PHONE in types
        assert EntityType.SSN in types

    @pytest.mark.asyncio
    async def test_detects_mixed_pii_and_secret(self, detector):
        text = "Contact dev@corp.com, key AKIAIOSFODNN7EXAMPLE"
        findings = await detector.detect(text)
        types = {f.entity_type for f in findings}
        assert EntityType.EMAIL in types
        assert EntityType.AWS_KEY in types

    @pytest.mark.asyncio
    async def test_all_offsets_are_valid_slices(self, detector):
        """For every finding, text[start:end] must equal matched_text."""
        text = (
            "john@example.com called 555-123-4567, "
            "SSN 123-45-6789, card 4532015112830366, "
            "key AKIAIOSFODNN7EXAMPLE"
        )
        findings = await detector.detect(text)
        assert len(findings) >= 4
        for f in findings:
            assert text[f.start : f.end] == f.matched_text, (
                f"Offset mismatch for {f.entity_type}: "
                f"text[{f.start}:{f.end}]={text[f.start : f.end]!r} "
                f"!= matched_text={f.matched_text!r}"
            )

    @pytest.mark.asyncio
    async def test_findings_have_valid_start_end_ordering(self, detector):
        """Every finding must have start < end."""
        text = "user@test.com, 555-000-1234, 123-45-6789"
        findings = await detector.detect(text)
        for f in findings:
            assert f.start < f.end, f"Invalid span for {f.entity_type}"

    @pytest.mark.asyncio
    async def test_all_entity_types_have_correct_category(self, detector):
        """Regex detector must assign correct PII or SECRET category."""
        text = (
            "user@test.com "
            "555-123-4567 "
            "123-45-6789 "
            "4532015112830366 "
            "AKIAIOSFODNN7EXAMPLE"
        )
        findings = await detector.detect(text)
        for f in findings:
            if f.entity_type in (
                EntityType.EMAIL,
                EntityType.PHONE,
                EntityType.SSN,
                EntityType.CREDIT_CARD,
            ):
                assert f.category == EntityCategory.PII, (
                    f"{f.entity_type} should be PII"
                )
            elif f.entity_type in (
                EntityType.AWS_KEY,
                EntityType.GITHUB_TOKEN,
                EntityType.JWT,
            ):
                assert f.category == EntityCategory.SECRET, (
                    f"{f.entity_type} should be SECRET"
                )
