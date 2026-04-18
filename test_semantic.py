"""
SentinelLM — Advanced Semantic Detector Tests

These tests send obfuscated PII that regex and Presidio CANNOT catch.
Only the semantic DistilBERT model should flag these.

Run with:
    python3 test_semantic.py
"""

import requests
import json

BASE = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen2.5:0.5b"

# ANSI colors for terminal output
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def color_decision(decision):
    if decision == "ALLOW":   return f"{GREEN}ALLOW{RESET}"
    if decision == "MASK":    return f"{YELLOW}MASK{RESET}"
    if decision == "BLOCK":   return f"{RED}BLOCK{RESET}"
    return decision

def run_test(name, message, expect, note):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{name}{RESET}")
    print(f"{DIM}Note: {note}{RESET}")
    print(f"{DIM}Message: \"{message}\"{RESET}")

    try:
        r = requests.post(BASE, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}]
        }, timeout=30)
        data = r.json()
    except Exception as e:
        print(f"{RED}  ✗ Request failed: {e}{RESET}")
        return

    ppg = data.get("ppg", {})
    input_dec  = ppg.get("input_decision", "N/A")
    output_dec = ppg.get("output_decision", "N/A")
    detectors  = ppg.get("detectors_used", [])
    reasons    = data.get("error", {}).get("reasons", []) or []
    in_redact  = ppg.get("input_redactions", {})
    latency    = ppg.get("latency_ms", {})

    # Pass/fail
    passed = input_dec == expect
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL (expected {expect}){RESET}"

    print(f"  Status:    {status}")
    print(f"  Decision:  {color_decision(input_dec)} → output: {color_decision(output_dec)}")
    print(f"  Detectors: {', '.join(detectors)}")
    print(f"  Latency:   detection={latency.get('detection')}ms  total={latency.get('total')}ms")

    if in_redact:
        print(f"  Redacted:  {in_redact}")
    if reasons:
        for r in reasons:
            print(f"  Reason:    {CYAN}{r}{RESET}")

    if "semantic" in detectors and (input_dec in ("MASK", "BLOCK")):
        print(f"  {CYAN}⚡ Semantic detector active{RESET}")

    return passed


tests = [
    # ── OBFUSCATED EMAILS ──────────────────────────────────────
    (
        "Obfuscated email (dot/at spelling)",
        "reach me at john dot smith at company dot com",
        "MASK",
        "No @ symbol — regex is blind, Presidio misses it, semantic catches it"
    ),
    (
        "Obfuscated email (spaced words)",
        "my address is john at gmail dot com for any follow up",
        "MASK",
        "Common obfuscation to avoid spam filters and PII scanners"
    ),

    # ── SPELLED-OUT PHONE NUMBERS ──────────────────────────────
    (
        "Phone number spelled out in words",
        "call me at five five five one two three four five six seven",
        "MASK",
        "No digits — regex requires digit patterns, this slips through"
    ),
    (
        "Phone number with casual phrasing",
        "my number is eight hundred five five five oh one two three",
        "MASK",
        "Mixed word/digit phone — only semantic understands the context"
    ),

    # ── SPELLED-OUT SSN ────────────────────────────────────────
    (
        "SSN spelled out in words",
        "my social is four five six dash seven eight dash nine zero one two",
        "MASK",
        "SSN said aloud — no dashes or digits for regex to pattern-match"
    ),

    # ── INFORMAL SECRETS / PASSWORDS ──────────────────────────
    (
        "Classic weak password leak",
        "the secret key is hunter two dont tell anyone",
        "BLOCK",  # GENERIC_SECRET → policy action is BLOCK, not MASK
        "hunter2 — famous leaked password, written informally"
    ),
    (
        "Password hinted in conversation",
        "my password for the server is the name of my dog followed by the year i was born",
        "BLOCK",  # Correctly caught as implied credential — BLOCK is right
        "No actual secret value but context clearly implies credential disclosure"
    ),

    # ── MIXED OBFUSCATION ─────────────────────────────────────
    (
        "API key described verbally",
        "the api key starts with aych tee tee pee ess colon slash slash",
        "BLOCK",
        "Spelled-out URL prefix for a secret — regex can't see through phonetics"
    ),
    (
        "Credit card hinted",
        "my card number starts with four one one one and ends with one one one one",
        "MASK",
        "Partial card number hinted — not a complete pattern for regex"
    ),

    # ── HARD NEGATIVES — should NOT flag ──────────────────────
    (
        "False positive check: technical email talk",
        "the email validation function checks for at-sign and dot in the domain",
        "ALLOW",
        "Talks ABOUT emails without containing one — should not flag"
    ),
    (
        "False positive check: numbers in context",
        "the function takes five arguments and returns one of three values",
        "ALLOW",
        "Numbers that are not phone numbers — semantic must understand context"
    ),
    (
        "False positive check: AWS mention",
        "we use AWS for cloud hosting and S3 for storage",
        "ALLOW",
        "AWS mentioned but no actual key — should pass through clean"
    ),
]

print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  SentinelLM — Semantic Detector Test Suite{RESET}")
print(f"{BOLD}{'═'*60}{RESET}")
print(f"{DIM}  Testing obfuscated PII that regex + Presidio cannot catch.{RESET}")
print(f"{DIM}  Each test shows which detectors fired and why.{RESET}")

passed = 0
total  = len(tests)

for name, message, expect, note in tests:
    result = run_test(name, message, expect, note)
    if result:
        passed += 1

print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  Results: {passed}/{total} passed{RESET}")

score_pct = passed / total * 100
if score_pct == 100:
    print(f"  {GREEN}Perfect score — all detectors firing correctly{RESET}")
elif score_pct >= 70:
    print(f"  {YELLOW}Good — semantic model is working, some edge cases missed{RESET}")
else:
    print(f"  {RED}Needs improvement — semantic model may need more training data{RESET}")

print(f"{BOLD}{'═'*60}{RESET}\n")
