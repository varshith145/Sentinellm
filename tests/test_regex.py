"""
Unit tests for the Regex Detector.

Tests all regex patterns: EMAIL, PHONE, SSN, CREDIT_CARD, AWS_KEY,
GITHUB_TOKEN, JWT, and Luhn validation.
"""

import pytest

from app.detectors.base import EntityType
from app.detectors.regex import RegexDetector, luhn_check


@pytest.fixture
def detector():
    return RegexDetector()


# --- Luhn Validation ---

def test_luhn_valid():
    assert luhn_check("4532015112830366") is True


def test_luhn_invalid():
    assert luhn_check("1234567890123456") is False


def test_luhn_short():
    assert luhn_check("123") is False


# --- Email Detection ---

@pytest.mark.asyncio
async def test_detect_email(detector):
    findings = await detector.detect("Contact john@example.com for info.")
    assert len(findings) == 1
    assert findings[0].entity_type == EntityType.EMAIL
    assert findings[0].matched_text == "john@example.com"
    assert findings[0].detector == "regex"
    assert findings[0].confidence == 0.95


@pytest.mark.asyncio
async def test_detect_multiple_emails(detector):
    text = "Email john@example.com or jane@company.org"
    findings = await detector.detect(text)
    emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
    assert len(emails) == 2


@pytest.mark.asyncio
async def test_no_email_in_plain_text(detector):
    findings = await detector.detect("This is just regular text without emails.")
    emails = [f for f in findings if f.entity_type == EntityType.EMAIL]
    assert len(emails) == 0


# --- Phone Detection ---

@pytest.mark.asyncio
async def test_detect_phone(detector):
    findings = await detector.detect("Call me at 555-123-4567.")
    phones = [f for f in findings if f.entity_type == EntityType.PHONE]
    assert len(phones) == 1
    assert phones[0].matched_text == "555-123-4567"


@pytest.mark.asyncio
async def test_detect_phone_with_area_code(detector):
    findings = await detector.detect("Call (212) 555-1234.")
    phones = [f for f in findings if f.entity_type == EntityType.PHONE]
    assert len(phones) == 1


@pytest.mark.asyncio
async def test_detect_phone_with_country_code(detector):
    findings = await detector.detect("Call +1-555-123-4567.")
    phones = [f for f in findings if f.entity_type == EntityType.PHONE]
    assert len(phones) == 1


# --- SSN Detection ---

@pytest.mark.asyncio
async def test_detect_ssn(detector):
    findings = await detector.detect("SSN is 123-45-6789")
    ssns = [f for f in findings if f.entity_type == EntityType.SSN]
    assert len(ssns) == 1
    assert ssns[0].matched_text == "123-45-6789"


@pytest.mark.asyncio
async def test_no_ssn_without_dashes(detector):
    findings = await detector.detect("Number 123456789 is not formatted as SSN")
    ssns = [f for f in findings if f.entity_type == EntityType.SSN]
    assert len(ssns) == 0


# --- Credit Card Detection ---

@pytest.mark.asyncio
async def test_detect_credit_card_valid_luhn(detector):
    # Valid Luhn number
    findings = await detector.detect("Card: 4532015112830366")
    ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
    assert len(ccs) == 1
    assert ccs[0].confidence == 0.99


@pytest.mark.asyncio
async def test_no_credit_card_invalid_luhn(detector):
    # Invalid Luhn number
    findings = await detector.detect("Card: 1234567890123456")
    ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
    assert len(ccs) == 0


@pytest.mark.asyncio
async def test_detect_credit_card_with_spaces(detector):
    findings = await detector.detect("Card: 4532 0151 1283 0366")
    ccs = [f for f in findings if f.entity_type == EntityType.CREDIT_CARD]
    assert len(ccs) == 1


# --- AWS Key Detection ---

@pytest.mark.asyncio
async def test_detect_aws_key(detector):
    findings = await detector.detect("Key: AKIAIOSFODNN7EXAMPLE")
    aws = [f for f in findings if f.entity_type == EntityType.AWS_KEY]
    assert len(aws) == 1
    assert aws[0].matched_text == "AKIAIOSFODNN7EXAMPLE"


@pytest.mark.asyncio
async def test_no_aws_key_wrong_prefix(detector):
    findings = await detector.detect("Key: ABCDIOSFODNN7EXAMPLE")
    aws = [f for f in findings if f.entity_type == EntityType.AWS_KEY]
    assert len(aws) == 0


# --- GitHub Token Detection ---

@pytest.mark.asyncio
async def test_detect_github_token(detector):
    token = "ghp_" + "a" * 36
    findings = await detector.detect(f"Token: {token}")
    gh = [f for f in findings if f.entity_type == EntityType.GITHUB_TOKEN]
    assert len(gh) == 1


# --- JWT Detection ---

@pytest.mark.asyncio
async def test_detect_jwt(detector):
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.abc123def456ghi789jkl"
    findings = await detector.detect(f"Token: {jwt}")
    jwts = [f for f in findings if f.entity_type == EntityType.JWT]
    assert len(jwts) == 1


# --- Mixed Content ---

@pytest.mark.asyncio
async def test_detect_multiple_entity_types(detector):
    text = "Email john@example.com, call 555-123-4567, SSN 123-45-6789"
    findings = await detector.detect(text)
    types = {f.entity_type for f in findings}
    assert EntityType.EMAIL in types
    assert EntityType.PHONE in types
    assert EntityType.SSN in types


# --- Character Offsets ---

@pytest.mark.asyncio
async def test_correct_offsets(detector):
    text = "My email is john@test.com please"
    findings = await detector.detect(text)
    assert len(findings) >= 1
    email_finding = [f for f in findings if f.entity_type == EntityType.EMAIL][0]
    assert text[email_finding.start : email_finding.end] == "john@test.com"
