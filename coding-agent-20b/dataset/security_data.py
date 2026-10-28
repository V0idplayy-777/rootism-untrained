"""Security-critical instruction data (paired safe / vulnerable code).

General coding ability does not transfer automatically to security-sensitive
code: models happily emit `md5(password)`, `verify=False`, `shell=True` and
string-interpolated SQL because those patterns are common in public corpora.

This module supplies the missing signal:

- curated ``SecuritySample`` entries: a vulnerable snippet, a corrected
  snippet, the CWE class and *why* the fix works,
- generators for three SFT shapes (review, repair, implement-securely) and for
  DPO preference pairs (safe chosen / vulnerable rejected) on the same prompt,
- a review rubric the model is trained to apply, which is also what
  ``evaluation/security_eval.py`` scores.

The samples are small and dependency-free; they are meant to be upsampled
against the general mixture (see ``dataset/mixture.py``: ``security-sft``),
not to be the whole security curriculum.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SecuritySample:
    id: str
    cwe: str
    category: str
    language: str
    title: str
    vulnerable: str
    safe: str
    why: str
    keywords: tuple[str, ...] = ()


SECURITY_SAMPLES: list[SecuritySample] = [
    SecuritySample(
        "pw-hash",
        "CWE-916",
        "password_storage",
        "python",
        "Store passwords with a slow, salted KDF",
        "import hashlib\n\n\ndef store_password(password: str) -> str:\n    return hashlib.md5(password.encode()).hexdigest()\n",
        "import secrets\n\nimport argon2\n\n_HASHER = argon2.PasswordHasher()\n\n\ndef store_password(password: str) -> str:\n    return _HASHER.hash(password)\n\n\ndef verify_password(stored: str, candidate: str) -> bool:\n    try:\n        return _HASHER.verify(stored, candidate)\n    except argon2.exceptions.VerifyMismatchError:\n        return False\n",
        "MD5/SHA-1 are fast and unsalted, so a stolen database is trivially brute-forced. "
        "Argon2id is memory-hard and salts per password; bcrypt/scrypt are acceptable.",
        ("password", "hash", "md5", "salt"),
    ),
    SecuritySample(
        "timing-compare",
        "CWE-208",
        "secret_comparison",
        "python",
        "Compare secrets in constant time",
        "def check_token(expected: str, provided: str) -> bool:\n    return expected == provided\n",
        "import hmac\n\n\ndef check_token(expected: str, provided: str) -> bool:\n    return hmac.compare_digest(\n        expected.encode(\"utf-8\"), provided.encode(\"utf-8\")\n    )\n",
        "Python string equality short-circuits on the first differing byte, leaking "
        "how much of the secret matched. hmac.compare_digest is constant-time.",
        ("token", "compare", "timing", "hmac"),
    ),
    SecuritySample(
        "sql-injection",
        "CWE-89",
        "injection",
        "python",
        "Parameterise SQL instead of interpolating input",
        "def get_user(conn, email: str):\n    cur = conn.cursor()\n    cur.execute(f\"SELECT id, email FROM users WHERE email = '{email}'\")\n    return cur.fetchone()\n",
        "def get_user(conn, email: str):\n    cur = conn.cursor()\n    cur.execute(\n        \"SELECT id, email FROM users WHERE email = %s\", (email,)\n    )\n    return cur.fetchone()\n",
        "Bound parameters travel out-of-band, so quotes in the input cannot change "
        "the statement. Escaping by hand is not equivalent.",
        ("sql", "query", "injection", "cursor"),
    ),
    SecuritySample(
        "path-traversal",
        "CWE-22",
        "file_access",
        "python",
        "Resolve and confine user-supplied paths",
        "from pathlib import Path\n\n\ndef read_upload(base_dir: str, name: str) -> bytes:\n    return (Path(base_dir) / name).read_bytes()\n",
        "from pathlib import Path\n\n\ndef read_upload(base_dir: str, name: str) -> bytes:\n    base = Path(base_dir).resolve()\n    target = (base / name).resolve()\n    if not target.is_relative_to(base):\n        raise ValueError(\"path escapes upload directory\")\n    return target.read_bytes()\n",
        "`..%2f` style input climbs out of the directory. Resolving both paths and "
        "checking containment defeats traversal, symlinks included.",
        ("path", "traversal", "upload", "file"),
    ),
    SecuritySample(
        "command-injection",
        "CWE-78",
        "injection",
        "python",
        "Never pass user input through a shell",
        "import subprocess\n\n\ndef convert(filename: str) -> int:\n    return subprocess.call(f\"convert {filename} out.png\", shell=True)\n",
        "import subprocess\n\n\ndef convert(filename: str) -> int:\n    return subprocess.call(\n        [\"convert\", \"--\", filename, \"out.png\"], shell=False, timeout=30\n    )\n",
        "With shell=True, `; rm -rf ~` in the filename executes. An argument vector "
        "bypasses the shell entirely; also add a timeout.",
        ("subprocess", "shell", "command", "convert"),
    ),
    SecuritySample(
        "sql-authz",
        "CWE-639",
        "authorization",
        "python",
        "Scope queries by the authenticated principal",
        "def get_invoice(db, invoice_id: int, user_id: int):\n    return db.query(\"SELECT * FROM invoices WHERE id = %s\", (invoice_id,))\n",
        "def get_invoice(db, invoice_id: int, user_id: int):\n    return db.query(\n        \"SELECT * FROM invoices WHERE id = %s AND owner_id = %s\",\n        (invoice_id, user_id),\n    )\n",
        "Authentication is not authorization: without the owner predicate any "
        "logged-in user can read another tenant's invoice (IDOR).",
        ("authz", "idor", "invoice", "owner"),
    ),
    SecuritySample(
        "random-secret",
        "CWE-338",
        "randomness",
        "python",
        "Use a CSPRNG for tokens, not `random`",
        "import random\nimport string\n\n\ndef reset_token() -> str:\n    alphabet = string.ascii_letters + string.digits\n    return \"\".join(random.choice(alphabet) for _ in range(32))\n",
        "import secrets\n\n\ndef reset_token() -> str:\n    return secrets.token_urlsafe(32)\n",
        "`random` is a Mersenne Twister seeded from a predictable state; observing a "
        "few outputs recovers it. `secrets` uses the OS CSPRNG.",
        ("random", "token", "csprng", "secrets"),
    ),
    SecuritySample(
        "deserialization",
        "CWE-502",
        "deserialization",
        "python",
        "Do not unpickle untrusted data",
        "import pickle\n\n\ndef load_session(blob: bytes) -> dict:\n    return pickle.loads(blob)\n",
        "import json\n\n\ndef load_session(blob: bytes) -> dict:\n    data = json.loads(blob)\n    if not isinstance(data, dict):\n        raise ValueError(\"invalid session payload\")\n    return data\n",
        "pickle executes arbitrary code during load (`__reduce__`). Use a data-only "
        "format (JSON/msgpack) and validate the shape.",
        ("pickle", "deserialize", "session"),
    ),
    SecuritySample(
        "yaml-load",
        "CWE-502",
        "deserialization",
        "python",
        "Use the safe YAML loader",
        "import yaml\n\n\ndef load_config(text: str) -> dict:\n    return yaml.load(text)\n",
        "import yaml\n\n\ndef load_config(text: str) -> dict:\n    return yaml.safe_load(text)\n",
        "yaml.load with the default loader constructs arbitrary Python objects. "
        "safe_load restricts to plain data types.",
        ("yaml", "load", "config"),
    ),
    SecuritySample(
        "jwt-alg",
        "CWE-347",
        "authentication",
        "python",
        "Pin the JWT algorithm and verify the signature",
        "import jwt\n\n\ndef identity(token: str) -> dict:\n    return jwt.decode(token, options={\"verify_signature\": False})\n",
        "import jwt\n\n\ndef identity(token: str, key: str, algorithms=(\"RS256\",)) -> dict:\n    return jwt.decode(\n        token,\n        key,\n        algorithms=list(algorithms),\n        options={\"require\": [\"exp\", \"iss\", \"aud\"]},\n    )\n",
        "Skipping verification (or accepting `alg: none` / HS256 with the public key) "
        "lets anyone mint tokens. Pin the expected algorithm and required claims.",
        ("jwt", "token", "verify", "algorithm"),
    ),
    SecuritySample(
        "tls-verify",
        "CWE-295",
        "transport",
        "python",
        "Never disable certificate verification",
        "import requests\n\n\ndef fetch(url: str) -> bytes:\n    return requests.get(url, verify=False, timeout=10).content\n",
        "import requests\n\n\ndef fetch(url: str) -> bytes:\n    return requests.get(url, verify=True, timeout=10).content\n",
        "verify=False accepts any certificate, so a network attacker can impersonate "
        "the server. Fix trust by installing the CA, not by disabling checks.",
        ("requests", "tls", "verify", "https"),
    ),
    SecuritySample(
        "ssrf",
        "CWE-918",
        "ssrf",
        "python",
        "Allow-list outbound hosts and block link-local ranges",
        "import requests\n\n\ndef proxy(url: str):\n    return requests.get(url, timeout=10).content\n",
        "import ipaddress\nimport socket\nfrom urllib.parse import urlparse\n\nimport requests\n\nALLOWED_HOSTS = {\"api.example.com\", \"cdn.example.com\"}\n\n\ndef _public_ip(host: str) -> bool:\n    infos = socket.getaddrinfo(host, None)\n    for info in infos:\n        ip = ipaddress.ip_address(info[4][0])\n        if ip.is_private or ip.is_link_local or ip.is_loopback or ip.is_reserved:\n            return False\n    return True\n\n\ndef proxy(url: str) -> bytes:\n    parsed = urlparse(url)\n    if parsed.scheme not in {\"https\"} or parsed.hostname not in ALLOWED_HOSTS:\n        raise ValueError(\"host not allowed\")\n    if not _public_ip(parsed.hostname):\n        raise ValueError(\"resolved address not allowed\")\n    return requests.get(url, timeout=10).content\n",
        "An unvalidated fetch lets the caller reach 169.254.169.254 (cloud metadata) "
        "or internal admin ports. Allow-list hosts and re-check the resolved IP.",
        ("ssrf", "url", "metadata", "allowlist"),
    ),
    SecuritySample(
        "secret-in-code",
        "CWE-798",
        "secrets",
        "python",
        "Load credentials from the environment, fail closed",
        "API_KEY = \"sk-live-9f2b1c7d4a5e6f7a8b9c0d1e2f3a4b5c\"\n\n\ndef client():\n    return Client(api_key=API_KEY)\n",
        "import os\n\n\ndef client():\n    api_key = os.environ.get(\"SERVICE_API_KEY\")\n    if not api_key:\n        raise RuntimeError(\"SERVICE_API_KEY is not configured\")\n    return Client(api_key=api_key)\n",
        "Hard-coded keys leak through git history, logs and model context. Read from "
        "the environment/secret manager and fail closed when absent.",
        ("secret", "api_key", "env"),
    ),
    SecuritySample(
        "eval-injection",
        "CWE-95",
        "injection",
        "python",
        "Do not eval untrusted input",
        "def calculate(expr: str) -> float:\n    return eval(expr)\n",
        "import ast\nimport operator\n\n_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}\n\n\ndef calculate(expr: str) -> float:\n    def _eval(node):\n        if isinstance(node, ast.Expression):\n            return _eval(node.body)\n        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):\n            return node.value\n        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:\n            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))\n        raise ValueError(f\"unsupported expression: {type(node).__name__}\")\n\n    return float(_eval(ast.parse(expr, mode=\"eval\")))\n",
        "eval runs arbitrary Python (`__import__('os').system(...)`). Parse with ast "
        "and allow only the operators you need.",
        ("eval", "ast", "expression"),
    ),
    SecuritySample(
        "xxe",
        "CWE-611",
        "parsing",
        "python",
        "Disable external entities in XML parsing",
        "from lxml import etree\n\n\ndef parse(xml_bytes: bytes):\n    return etree.fromstring(xml_bytes)\n",
        "from defusedxml import ElementTree as SafeET\n\n\ndef parse(xml_bytes: bytes):\n    return SafeET.fromstring(xml_bytes)\n",
        "The default XML parser resolves external entities, enabling local file "
        "disclosure and SSRF. defusedxml turns that off.",
        ("xml", "xxe", "entities"),
    ),
    SecuritySample(
        "zip-slip",
        "CWE-22",
        "file_access",
        "python",
        "Validate archive member paths before extraction",
        "import zipfile\n\n\ndef extract(archive: str, dest: str) -> None:\n    with zipfile.ZipFile(archive) as zf:\n        zf.extractall(dest)\n",
        "import zipfile\nfrom pathlib import Path\n\n\ndef extract(archive: str, dest: str) -> None:\n    base = Path(dest).resolve()\n    with zipfile.ZipFile(archive) as zf:\n        for member in zf.infolist():\n            target = (base / member.filename).resolve()\n            if not target.is_relative_to(base):\n                raise ValueError(f\"unsafe archive member: {member.filename}\")\n            if member.is_dir():\n                target.mkdir(parents=True, exist_ok=True)\n                continue\n            target.parent.mkdir(parents=True, exist_ok=True)\n            with zf.open(member) as src, open(target, \"wb\") as dst:\n                dst.write(src.read())\n",
        "extractall honours `../` in member names (zip-slip) and can write outside the "
        "destination. Validate each resolved target.",
        ("zip", "extract", "traversal"),
    ),
    SecuritySample(
        "crypto-mode",
        "CWE-327",
        "cryptography",
        "python",
        "Use AEAD, not ECB",
        "from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes\n\n\ndef encrypt(key: bytes, plaintext: bytes) -> bytes:\n    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()\n    return encryptor.update(plaintext) + encryptor.finalize()\n",
        "import os\n\nfrom cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\n\ndef encrypt(key: bytes, plaintext: bytes, aad: bytes = b\"\") -> bytes:\n    nonce = os.urandom(12)\n    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)\n",
        "ECB is deterministic and leaks structure (identical blocks encrypt "
        "identically), and it is unauthenticated. AES-GCM provides confidentiality "
        "plus integrity with a fresh nonce.",
        ("crypto", "aes", "ecb", "gcm"),
    ),
    SecuritySample(
        "tempfile",
        "CWE-377",
        "file_access",
        "python",
        "Create temp files safely",
        "import os\n\n\ndef write_report(data: bytes) -> str:\n    path = \"/tmp/report.csv\"\n    with open(path, \"wb\") as f:\n        f.write(data)\n    return path\n",
        "import tempfile\n\n\ndef write_report(data: bytes) -> str:\n    handle = tempfile.NamedTemporaryFile(\n        mode=\"wb\", prefix=\"report-\", suffix=\".csv\", delete=False\n    )\n    with handle:\n        handle.write(data)\n    return handle.name\n",
        "A predictable path in a shared directory invites symlink attacks and races. "
        "tempfile creates the file atomically with a random name.",
        ("tempfile", "race", "symlink"),
    ),
]


def security_review_rubric() -> str:
    """The checklist the model is trained (and scored) to apply."""

    return "\n".join(
        [
            "Security review checklist:",
            "1. Input handling: is every external input validated and type-checked?",
            "2. Injection: SQL/OS/template/eval sinks reachable from input?",
            "3. AuthN/AuthZ: is the caller authenticated *and* authorised for this object?",
            "4. Secrets: any hard-coded keys, tokens, passwords, or logging of them?",
            "5. Crypto: vetted library, AEAD mode, CSPRNG, constant-time comparison?",
            "6. Transport/parsing: certificate verification on, safe loaders, entity limits?",
            "7. Filesystem/network: paths confined, outbound hosts allow-listed?",
            "8. Failure mode: does it fail closed, and are errors free of sensitive data?",
        ]
    )


class SecurityDataGenerator:
    """Turns the curated samples into SFT rows and DPO pairs.

    Three task shapes, because review, repair and implementation are different
    skills:

    - ``review``: find the problem in the vulnerable snippet (output = finding),
    - ``repair``: fix the vulnerable snippet (output = safe snippet),
    - ``implement``: implement a feature with the security requirement stated.
    """

    def __init__(self, samples: list[SecuritySample] | None = None, seed: int = 42):

        self.samples = samples or SECURITY_SAMPLES

        self.rng = random.Random(seed)

    @staticmethod
    def _finding(sample: SecuritySample) -> str:
        return (
            f"**{sample.title}** ({sample.cwe}, {sample.category})\n\n"
            f"{sample.why}\n\n"
            f"Fixed version:\n```{sample.language}\n{sample.safe}```"
        )

    def review_samples(self) -> list[dict]:
        return [
            {
                "instruction": (
                    "Review this code for security vulnerabilities. Identify the "
                    "weakness class (CWE) and show the corrected code."
                ),
                "input_context": f"```{s.language}\n{s.vulnerable}```",
                "output": self._finding(s),
                "category": "security_review",
                "language": s.language,
                "difficulty": "advanced",
                "metadata": {"cwe": s.cwe, "sample_id": s.id},
            }
            for s in self.samples
        ]

    def repair_samples(self) -> list[dict]:
        return [
            {
                "instruction": (
                    f"Fix the {s.category.replace('_', ' ')} vulnerability in this "
                    f"{s.language} code without changing its public behaviour."
                ),
                "input_context": f"```{s.language}\n{s.vulnerable}```",
                "output": f"```{s.language}\n{s.safe}```\n\n{s.why}",
                "category": "security_repair",
                "language": s.language,
                "difficulty": "advanced",
                "metadata": {"cwe": s.cwe, "sample_id": s.id},
            }
            for s in self.samples
        ]

    def implement_samples(self) -> list[dict]:
        needs = {
            "password_storage": "Passwords must be stored with a salted, memory-hard KDF.",
            "secret_comparison": "Secret comparison must be constant-time.",
            "randomness": "Tokens must come from a CSPRNG.",
            "authentication": "JWT signatures and claims must be verified with a pinned algorithm.",
            "transport": "TLS certificates must be verified.",
            "cryptography": "Encryption must be authenticated (AEAD).",
        }

        out = []

        for s in self.samples:

            need = needs.get(s.category)

            if not need:

                continue

            out.append(
                {
                    "instruction": (
                        f"Implement the {s.category.replace('_', ' ')} helper in "
                        f"{s.language}. Requirement: {need}"
                    ),
                    "input_context": security_review_rubric(),
                    "output": f"```{s.language}\n{s.safe}```\n\n{s.why}",
                    "category": "security_implement",
                    "language": s.language,
                    "difficulty": "advanced",
                    "metadata": {"cwe": s.cwe, "sample_id": s.id},
                }
            )

        return out

    def dpo_pairs(self) -> list[dict]:
        """Preference pairs: same prompt, safe answer chosen, vuln rejected."""

        pairs = []

        for s in self.samples:

            pairs.append(
                {
                    "prompt": (
                        f"Write {s.language} code for: {s.title}.\n\n"
                        f"{security_review_rubric()}"
                    ),
                    "chosen": f"```{s.language}\n{s.safe}```\n\n{s.why}",
                    "rejected": f"```{s.language}\n{s.vulnerable}```",
                    "category": "security_preference",
                    "metadata": {"cwe": s.cwe, "sample_id": s.id},
                }
            )

        return pairs

    def generate_batch(self, count: int, seed: int | None = None) -> list[dict]:
        """Mixed batch of review/repair/implement rows, sampled without repeats."""

        pool = self.review_samples() + self.repair_samples() + self.implement_samples()

        rng = random.Random(seed if seed is not None else self.rng.randrange(1 << 30))

        if count >= len(pool):

            out = pool[:]

            rng.shuffle(out)

            return out

        return rng.sample(pool, count)

    def stats(self) -> dict:
        return {
            "samples": len(self.samples),
            "cwes": sorted({s.cwe for s in self.samples}),
            "categories": sorted({s.category for s in self.samples}),
        }
