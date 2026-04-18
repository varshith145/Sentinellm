"""
SentinelLM — Synthetic Training Data Generator

Generates 400+ labeled examples covering:
  - Obfuscated emails (dot/at spelling, no @ symbol)
  - Spelled-out phone numbers (no digits)
  - Spelled-out SSNs
  - Informal passwords and secrets
  - API keys and tokens described verbally
  - Credit card number hints
  - Implied credentials
  - Hard negatives (should NOT be flagged)

Each example has exact character-level entity spans verified
by the make_example() builder — no off-by-one errors.

Run with:
    python3 model/data/generate_training_data.py

Appends to synthetic_obfuscated.jsonl and hard_negatives.jsonl.
"""

import json
import random
from pathlib import Path
from typing import Optional

random.seed(42)

SYNTHETIC_PATH = Path(__file__).parent / "synthetic_obfuscated.jsonl"
NEGATIVES_PATH = Path(__file__).parent / "hard_negatives.jsonl"

# ── Helpers ────────────────────────────────────────────────────────────────

def make_example(parts: list) -> dict:
    """
    Build a labeled example from parts.

    parts: list of (text_segment, label_or_None)
      label_or_None: "PII", "SECRET", or None

    Builds the full text and records exact character spans for
    any segment with a non-None label. Guarantees span accuracy.
    """
    text = ""
    entities = []
    for segment, label in parts:
        if label:
            start = len(text)
            text += segment
            entities.append({"start": start, "end": len(text), "label": label})
        else:
            text += segment
    return {"text": text, "entities": entities}


def neg(text: str) -> dict:
    """Shorthand for a hard-negative (no entities)."""
    return {"text": text, "entities": []}


def verify(example: dict) -> bool:
    """Sanity-check that every entity span matches the text."""
    text = example["text"]
    for ent in example["entities"]:
        chunk = text[ent["start"]:ent["end"]]
        if not chunk.strip():
            return False
    return True


# ── Name / domain pools ────────────────────────────────────────────────────

FIRST_NAMES = [
    "john", "jane", "mike", "sarah", "david", "emily", "chris", "lisa",
    "james", "emma", "ryan", "olivia", "daniel", "sophia", "alex", "maya",
    "kevin", "anna", "tyler", "rachel", "brandon", "jessica", "carlos", "nina",
    "wei", "priya", "omar", "fatima", "liam", "zoe",
]

LAST_NAMES = [
    "smith", "jones", "brown", "davis", "wilson", "taylor", "anderson",
    "thomas", "jackson", "white", "harris", "martin", "garcia", "martinez",
    "rodriguez", "lee", "walker", "hall", "allen", "young", "king", "wright",
    "scott", "green", "baker", "adams", "nelson", "carter", "mitchell", "chen",
]

DOMAINS = [
    "gmail", "yahoo", "hotmail", "outlook", "company", "work", "corp",
    "enterprise", "acme", "startup", "tech", "dev", "io", "co",
]

TLDS = ["com", "org", "net", "io", "co", "edu"]

DIGITS_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}

AREA_CODES = ["555", "212", "415", "312", "713", "404", "206", "617", "702", "303"]

PASSWORDS = [
    "hunter2", "password123", "letmein", "qwerty", "admin123",
    "welcome1", "monkey99", "dragon7", "shadow88", "master42",
    "sunshine3", "princess1", "football9", "baseball7", "iloveyou2",
    "abc123xyz", "trustno1", "starwars9", "access99", "mustang5",
]

API_KEY_PREFIXES = [
    "sk-proj", "ghp_abc", "xoxb-123", "AKIA", "pk_live", "rk_live",
    "eyJhbGc", "Bearer abc123", "token xyz789", "key_prod_abc",
]

CONTEXT_VERBS = [
    "reach me at", "contact me at", "email me at", "send it to",
    "you can find me at", "my address is", "write to me at",
    "drop me a line at", "hit me up at", "get in touch at",
]

PHONE_CONTEXTS = [
    "call me at", "my number is", "reach me on", "phone me at",
    "you can call me at", "my cell is", "my direct line is",
    "ring me at", "text me at", "my mobile is",
]

SSN_CONTEXTS = [
    "my social security number is", "my SSN is", "my social is",
    "the last four of my social are", "my social security is",
    "for verification my SSN is", "my tax id is",
]

SECRET_CONTEXTS = [
    "the secret key is", "the password is", "my api key is",
    "the token is", "the access key is", "use the key",
    "the credential is", "the passphrase is", "the auth token is",
    "my private key is",
]


# ── Generators ─────────────────────────────────────────────────────────────

def digits_to_words(digits: str) -> str:
    """Convert a digit string to space-separated word form."""
    return " ".join(DIGITS_WORDS[d] for d in digits if d.isdigit())


def gen_obfuscated_emails(n: int) -> list[dict]:
    """Obfuscated emails — no @ symbol, written as words."""
    examples = []
    patterns = [
        # "john dot smith at gmail dot com"
        lambda f, l, d, t: make_example([
            (random.choice(CONTEXT_VERBS) + " ", None),
            (f"{f} dot {l} at {d} dot {t}", "PII"),
        ]),
        # "john at gmail dot com"
        lambda f, l, d, t: make_example([
            (random.choice(CONTEXT_VERBS) + " ", None),
            (f"{f} at {d} dot {t}", "PII"),
        ]),
        # "john underscore smith at company dot com"
        lambda f, l, d, t: make_example([
            (random.choice(CONTEXT_VERBS) + " ", None),
            (f"{f} underscore {l} at {d} dot {t}", "PII"),
        ]),
        # "john dash smith at gmail dot com"
        lambda f, l, d, t: make_example([
            (random.choice(CONTEXT_VERBS) + " ", None),
            (f"{f} dash {l} at {d} dot {t}", "PII"),
        ]),
        # mid-sentence: "my details are john at gmail dot com for follow up"
        lambda f, l, d, t: make_example([
            ("my details are ", None),
            (f"{f} at {d} dot {t}", "PII"),
            (" for any follow up", None),
        ]),
        # "the address john dot smith at work dot com was added"
        lambda f, l, d, t: make_example([
            ("the address ", None),
            (f"{f} dot {l} at {d} dot {t}", "PII"),
            (" was added to the system", None),
        ]),
        # "contact is firstname at domain dot tld"
        lambda f, l, d, t: make_example([
            ("contact is ", None),
            (f"{f} at {d} dot {t}", "PII"),
        ]),
        # with "please reach"
        lambda f, l, d, t: make_example([
            ("please reach out to ", None),
            (f"{f} dot {l} at {d} dot {t}", "PII"),
            (" at your earliest convenience", None),
        ]),
    ]

    for _ in range(n):
        f = random.choice(FIRST_NAMES)
        l = random.choice(LAST_NAMES)
        d = random.choice(DOMAINS)
        t = random.choice(TLDS)
        ex = random.choice(patterns)(f, l, d, t)
        if verify(ex):
            examples.append(ex)
    return examples


def gen_spelled_phones(n: int) -> list[dict]:
    """Phone numbers written as words — no digits."""
    examples = []
    patterns = [
        lambda ctx, num: make_example([
            (ctx + " ", None),
            (num, "PII"),
        ]),
        lambda ctx, num: make_example([
            (ctx + " ", None),
            (num, "PII"),
            (" anytime after nine am", None),
        ]),
        lambda ctx, num: make_example([
            ("feel free to ", None),
            (ctx + " ", None),
            (num, "PII"),
            (" if you have questions", None),
        ]),
        lambda ctx, num: make_example([
            ("the best way to reach me is ", None),
            (ctx + " ", None),
            (num, "PII"),
        ]),
    ]

    for _ in range(n):
        area = digits_to_words(random.choice(AREA_CODES))
        mid  = digits_to_words(str(random.randint(100, 999)))
        last = digits_to_words(str(random.randint(1000, 9999)))
        num  = f"{area} {mid} {last}"
        ctx  = random.choice(PHONE_CONTEXTS)
        ex   = random.choice(patterns)(ctx, num)
        if verify(ex):
            examples.append(ex)
    return examples


def gen_spelled_ssns(n: int) -> list[dict]:
    """SSNs spelled out in words."""
    examples = []
    patterns = [
        lambda ctx, ssn: make_example([
            (ctx + " ", None),
            (ssn, "PII"),
        ]),
        lambda ctx, ssn: make_example([
            ("for verification purposes ", None),
            (ctx + " ", None),
            (ssn, "PII"),
        ]),
        lambda ctx, ssn: make_example([
            (ctx + " ", None),
            (ssn, "PII"),
            (" please keep this confidential", None),
        ]),
        lambda ctx, ssn: make_example([
            ("the form requires ", None),
            (ctx + " ", None),
            (ssn, "PII"),
            (" to process the application", None),
        ]),
    ]

    for _ in range(n):
        a = digits_to_words(f"{random.randint(100,999)}")
        b = digits_to_words(f"{random.randint(10,99)}")
        c = digits_to_words(f"{random.randint(1000,9999)}")
        ssn = f"{a} dash {b} dash {c}"
        ctx = random.choice(SSN_CONTEXTS)
        ex  = random.choice(patterns)(ctx, ssn)
        if verify(ex):
            examples.append(ex)
    return examples


def gen_passwords(n: int) -> list[dict]:
    """Plaintext passwords and secrets leaked in conversation."""
    examples = []

    def rand_secret():
        choices = PASSWORDS + [
            f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(8,16)))}",
            f"{random.choice(FIRST_NAMES)}{random.randint(10,99)}",
            f"{random.choice(LAST_NAMES)}{''.join(random.choices('0123456789',k=4))}",
        ]
        return random.choice(choices)

    patterns = [
        lambda ctx, pw: make_example([
            (ctx + " ", None),
            (pw, "SECRET"),
        ]),
        lambda ctx, pw: make_example([
            (ctx + " ", None),
            (pw, "SECRET"),
            (" do not share this", None),
        ]),
        lambda ctx, pw: make_example([
            ("i set the password to ", None),
            (pw, "SECRET"),
            (" for now", None),
        ]),
        lambda ctx, pw: make_example([
            ("use ", None),
            (pw, "SECRET"),
            (" to log in to the staging server", None),
        ]),
        lambda ctx, pw: make_example([
            ("the default credential is ", None),
            (pw, "SECRET"),
            (" until you reset it", None),
        ]),
        lambda ctx, pw: make_example([
            ("login with username admin and password ", None),
            (pw, "SECRET"),
        ]),
        lambda ctx, pw: make_example([
            ("the root password is ", None),
            (pw, "SECRET"),
            (" change it after first login", None),
        ]),
        lambda ctx, pw: make_example([
            ("temporary password for your account is ", None),
            (pw, "SECRET"),
        ]),
    ]

    for _ in range(n):
        ctx = random.choice(SECRET_CONTEXTS)
        pw  = rand_secret()
        ex  = random.choice(patterns)(ctx, pw)
        if verify(ex):
            examples.append(ex)
    return examples


def gen_api_keys(n: int) -> list[dict]:
    """API keys and tokens described or partially revealed."""

    def rand_key():
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        prefix = random.choice(API_KEY_PREFIXES)
        suffix = "".join(random.choices(chars, k=random.randint(12, 24)))
        return f"{prefix}{suffix}"

    phonetic_prefixes = [
        "aych tee tee pee ess colon slash slash",
        "ess kay dash",
        "gee aych pee underscore",
        "ay kay eye ay",
        "pee kay underscore live underscore",
        "ex oh ex bee dash",
        "bee ee ay ar ee ar space",
    ]

    examples = []
    patterns = [
        lambda ctx, key: make_example([
            (ctx + " ", None),
            (key, "SECRET"),
        ]),
        lambda ctx, key: make_example([
            ("add this to your env file: ", None),
            (ctx + " equals ", None),
            (key, "SECRET"),
        ]),
        lambda ctx, key: make_example([
            ("the ", None),
            (ctx + " is ", None),
            (key, "SECRET"),
            (" rotate it weekly", None),
        ]),
        lambda ctx, key: make_example([
            ("export API_KEY equals ", None),
            (key, "SECRET"),
        ]),
        lambda ctx, key: make_example([
            ("authorization header should be Bearer ", None),
            (key, "SECRET"),
        ]),
        # Phonetic description
        lambda ctx, key: make_example([
            ("the key starts with ", None),
            (key, "SECRET"),
            (" followed by random characters", None),
        ]),
    ]

    for _ in range(n):
        ctx = random.choice([
            "the api key is", "my api key is", "the access token is",
            "the secret key is", "the auth token is", "the private key is",
            "the service account key is", "the bearer token is",
        ])
        key = rand_key()
        ex  = random.choice(patterns)(ctx, key)
        if verify(ex):
            examples.append(ex)

    # Add phonetic ones
    for phonetic in phonetic_prefixes * 3:
        ex = make_example([
            ("the api key starts with ", None),
            (phonetic, "SECRET"),
            (" and then more characters", None),
        ])
        if verify(ex):
            examples.append(ex)

    return examples


def gen_credit_cards(n: int) -> list[dict]:
    """Partial or hinted credit card numbers."""
    examples = []

    def rand_cc_words():
        digits = [random.choice(list(DIGITS_WORDS.values())) for _ in range(4)]
        return " ".join(digits)

    patterns = [
        lambda part: make_example([
            ("my card number starts with ", None),
            (part, "PII"),
            (" and ends with the last four i gave you", None),
        ]),
        lambda part: make_example([
            ("charge the card that ends in ", None),
            (part, "PII"),
        ]),
        lambda part: make_example([
            ("the first four digits of my card are ", None),
            (part, "PII"),
        ]),
        lambda part: make_example([
            ("my credit card is ", None),
            (part, "PII"),
            (" you have the rest on file", None),
        ]),
        lambda part: make_example([
            ("bill the card starting with ", None),
            (part, "PII"),
        ]),
    ]

    for _ in range(n):
        part = rand_cc_words()
        ex   = random.choice(patterns)(part)
        if verify(ex):
            examples.append(ex)
    return examples


def gen_implied_credentials(n: int) -> list[dict]:
    """Credential disclosure without explicit secret value."""
    templates = [
        "my password for the server is the name of my dog followed by the year i was born",
        "the wifi password is the street we grew up on plus the house number",
        "i use my mothers maiden name as my security answer",
        "the pin is the last four digits of my old phone number",
        "my password hint is my favorite team and the year they won",
        "for security questions i always use my first pets name",
        "the passphrase is the city where my parents met",
        "my backup code is stored in the notes app under personal",
        "the master password follows the pattern name plus birthday",
        "i set the recovery phrase to something only i would know from childhood",
        "the security pin is based on my wifes birthday backwards",
        "my email password is the same as my bank just with a number at the end",
        "the answer to secret question is my elementary school name",
        "i use my childhood nickname as the passphrase it has uppercase and numbers",
        "the token resets every hour and i store it in my password manager",
    ]
    # These are edge cases — some systems flag, some don't
    # Label the key phrases as SECRET since they describe credential patterns
    examples = []
    for t in random.choices(templates, k=n):
        # For implied credentials, label the whole implied secret description
        # as SECRET since the meaning is credential-related
        ex = make_example([("", None), (t, "SECRET")])
        if verify(ex):
            examples.append(ex)
    return examples


def gen_hard_negatives(n: int) -> list[dict]:
    """Non-PII sentences that look similar but contain no sensitive data."""
    templates = [
        # Email-like but technical
        "the email validation function checks for at sign and dot in the domain",
        "smtp protocol sends emails over port twenty five",
        "the email field is required and must contain an at symbol",
        "configure the email server to use tls encryption",
        "the email template uses placeholder variables for personalization",
        "email marketing campaigns have an average open rate of twenty percent",
        "the regex pattern for email validation is well documented online",
        "email threading groups related messages in the inbox",

        # Phone-like but not PII
        "call the function with five arguments and return one result",
        "the function accepts three to five parameters",
        "dial the conference bridge and enter the meeting code",
        "the telephone protocol was invented in the eighteen hundreds",
        "phone screens typically have resolutions in the thousands of pixels",
        "the call center handles five hundred calls per day",

        # Number-heavy but not sensitive
        "the algorithm runs in order of n squared time complexity",
        "the dataset contains one million two hundred thousand records",
        "version three point one four of the library was released",
        "the server processes two hundred requests per second",
        "our uptime was ninety nine point nine percent last quarter",
        "the array has one thousand and twenty four elements",

        # AWS/cloud but no keys
        "we use aws for cloud hosting and s3 for file storage",
        "aws lambda functions run serverless workloads at scale",
        "the aws console shows your billing and usage metrics",
        "deploy to aws using the cloudformation template provided",
        "aws provides over two hundred cloud services globally",

        # Password/auth context without actual secrets
        "the password field must be at least eight characters long",
        "password managers help users create strong unique passwords",
        "enable two factor authentication on your account settings",
        "the authentication flow uses oauth two point zero",
        "password hashing uses bcrypt with twelve rounds of salting",
        "the login page has a forgot password link below the form",
        "strong passwords combine uppercase lowercase numbers and symbols",

        # Social security in general context
        "social security is an important government retirement program",
        "the social security administration manages retirement benefits",
        "social security numbers are used for tax identification",
        "the social security act was signed in nineteen thirty five",

        # Token/key in technical context
        "a jwt token contains three base sixty four encoded sections",
        "api tokens expire after twenty four hours for security",
        "the csrf token prevents cross site request forgery attacks",
        "use environment variables to store api keys never hardcode them",
        "the access token is stored in an httponly cookie",
        "rotate your api keys every ninety days as a best practice",
        "never commit api keys or secrets to version control",
        "the api rate limit is one thousand requests per hour",

        # Credit card in general context
        "credit card fraud detection uses machine learning models",
        "pci dss compliance requires encrypting all card data",
        "the credit card processing fee is two point nine percent",
        "contactless payments use nfc technology for card transactions",

        # Generic technical content
        "the database schema has a users table with an email column",
        "encrypt all personally identifiable information at rest",
        "the gdpr requires consent before collecting personal data",
        "data retention policies must comply with regional regulations",
        "anonymize user data before using it for analytics",
        "the privacy policy explains how we handle user information",
        "implement data masking for non production environments",
        "access to production credentials requires approval from security",
    ]

    selected = random.choices(templates, k=n)
    return [neg(t) for t in selected]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SentinelLM — Training Data Generator")
    print("=" * 60)

    all_positive = []
    all_negative = []

    print("\nGenerating positive examples...")

    emails = gen_obfuscated_emails(90)
    print(f"  Obfuscated emails:      {len(emails)}")
    all_positive.extend(emails)

    phones = gen_spelled_phones(70)
    print(f"  Spelled-out phones:     {len(phones)}")
    all_positive.extend(phones)

    ssns = gen_spelled_ssns(50)
    print(f"  Spelled-out SSNs:       {len(ssns)}")
    all_positive.extend(ssns)

    passwords = gen_passwords(70)
    print(f"  Passwords/secrets:      {len(passwords)}")
    all_positive.extend(passwords)

    api_keys = gen_api_keys(60)
    print(f"  API keys/tokens:        {len(api_keys)}")
    all_positive.extend(api_keys)

    cards = gen_credit_cards(40)
    print(f"  Credit card hints:      {len(cards)}")
    all_positive.extend(cards)

    implied = gen_implied_credentials(30)
    print(f"  Implied credentials:    {len(implied)}")
    all_positive.extend(implied)

    print(f"\n  Total positive:         {len(all_positive)}")

    print("\nGenerating hard negatives...")
    negatives = gen_hard_negatives(120)
    print(f"  Total negatives:        {len(negatives)}")

    # Shuffle both
    random.shuffle(all_positive)
    random.shuffle(negatives)

    # Verify all examples
    bad = [ex for ex in all_positive if not verify(ex)]
    if bad:
        print(f"\n  ⚠️  {len(bad)} examples failed verification — skipping them")
        all_positive = [ex for ex in all_positive if verify(ex)]

    # Append to existing files
    print(f"\nAppending to {SYNTHETIC_PATH}...")
    with open(SYNTHETIC_PATH, "a") as f:
        for ex in all_positive:
            f.write(json.dumps(ex) + "\n")

    print(f"Appending to {NEGATIVES_PATH}...")
    with open(NEGATIVES_PATH, "a") as f:
        for ex in negatives:
            f.write(json.dumps(ex) + "\n")

    # Count totals
    with open(SYNTHETIC_PATH) as f:
        total_pos = sum(1 for l in f if l.strip())
    with open(NEGATIVES_PATH) as f:
        total_neg = sum(1 for l in f if l.strip())

    print("\n" + "=" * 60)
    print("✅ Generation complete!")
    print(f"   synthetic_obfuscated.jsonl: {total_pos} examples total")
    print(f"   hard_negatives.jsonl:       {total_neg} examples total")
    print(f"\n   Now run: ./train.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
