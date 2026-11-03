"""Phase 2 extension: additional curated security pairs (Corpus v1).

This module is imported by :mod:`dataset.security_data` and appended to
``SECURITY_SAMPLES``.  Each entry keeps the invariants enforced by
``tests/test_security_augment_longform.py``:

* ``category`` must exist in ``evaluation.security_eval.REQUIRED_PATTERNS``
  (or be ``adversarial_request``),
* every Python pair must parse with ``ast.parse``,
* the ``safe`` snippet must be clean for critical/high/medium findings,
* the ``vulnerable`` snippet must trip a critical/high/medium finding (except
  for ids listed in ``STATIC_UNDETECTABLE``),
* every ``safe`` snippet must contain a regex from ``REQUIRED_PATTERNS`` or
  ``REQUIRED_OVERRIDES`` for its category / id.
"""

from __future__ import annotations

from .security_data import SecuritySample

ADDITIONAL_SECURITY_SAMPLES: list[SecuritySample] = [
    SecuritySample(
        "sql-injection-orm-raw",
        "CWE-89",
        "injection",
        "python",
        "Never interpolate identifiers or values into raw SQL",
        "def find_user(conn, email):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(f\"SELECT id, email FROM users WHERE email = '{email}'\")\n"
        "    return cursor.fetchone()\n",
        "def find_user(conn, email):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\n"
        "        \"SELECT id, email FROM users WHERE email = %s\", (email,)\n"
        "    )\n"
        "    return cursor.fetchone()\n",
        "Interpolating a value into SQL lets `' OR '1'='1` change the query. "
        "Driver parameter binding sends values out-of-band, so they can never be parsed as SQL.",
        ("sql", "injection", "parameterized"),
    ),
    SecuritySample(
        "command-injection-subprocess",
        "CWE-78",
        "injection",
        "python",
        "Pass argument vectors to subprocess instead of building shell strings",
        "import subprocess\n\n\ndef convert(path: str) -> str:\n"
        "    cmd = \"convert \" + path + \" /tmp/out.png\"\n"
        "    return subprocess.check_output(cmd, shell=True).decode()\n",
        "import subprocess\nfrom pathlib import Path\nimport tempfile\n\n\n"
        "def convert(path: str) -> str:\n"
        "    out = Path(tempfile.mkdtemp()) / \"out.png\"\n"
        "    return subprocess.check_output(\n"
        "        [\"convert\", path, str(out)], shell=False\n"
        "    ).decode()\n",
        "With `shell=True` the path is re-parsed by /bin/sh, so `a.png; rm -rf ~` runs as a "
        "second command. An argv list with shell disabled passes the path as one opaque argument.",
        ("shell", "subprocess", "command"),
    ),
    SecuritySample(
        "path-traversal-join",
        "CWE-22",
        "file_access",
        "python",
        "Confine user-supplied paths to a base directory",
        "import os\n\n\ndef read_report(name: str) -> str:\n"
        "    path = os.path.join(\"/srv/reports\", name)\n"
        "    with open(path, encoding=\"utf-8\") as handle:\n"
        "        return handle.read()\n",
        "import os\n\n\ndef read_report(name: str) -> str:\n"
        "    base = os.path.realpath(\"/srv/reports\")\n"
        "    path = os.path.realpath(os.path.join(base, name))\n"
        "    if os.path.commonpath([base, path]) != base:\n"
        "        raise PermissionError(\"path escapes reports directory\")\n"
        "    with open(path, encoding=\"utf-8\") as handle:\n"
        "        return handle.read()\n",
        "`../../etc/passwd` escapes the base directory. Resolving realpath and comparing "
        "commonpath rejects both `..` traversal and symlink escapes.",
        ("path", "traversal", "containment"),
    ),
    SecuritySample(
        "deserialization-pickle",
        "CWE-502",
        "deserialization",
        "python",
        "Never unpickle bytes that crossed a trust boundary",
        "import pickle\n\n\ndef load_session(blob: bytes) -> dict:\n"
        "    return pickle.loads(blob)\n",
        "import json\n\n\ndef load_session(blob: bytes) -> dict:\n"
        "    data = json.loads(blob.decode(\"utf-8\"))\n"
        "    if not isinstance(data, dict):\n"
        "        raise ValueError(\"session payload must be an object\")\n"
        "    return data\n",
        "`pickle.loads` executes whatever `__reduce__` the payload names, which is remote code "
        "execution. JSON can only produce data, and the shape is validated before use.",
        ("pickle", "deserialization", "json"),
    ),
    SecuritySample(
        "transport-verify-disabled",
        "CWE-295",
        "transport",
        "python",
        "Keep TLS certificate verification enabled",
        "import requests\n\n\ndef fetch_balance(url: str) -> dict:\n"
        "    return requests.get(url, verify=False, timeout=10).json()\n",
        "import requests\n\n\ndef fetch_balance(url: str) -> dict:\n"
        "    return requests.get(url, verify=True, timeout=10).json()\n",
        "`verify=False` accepts any certificate, so anyone on the path can impersonate the API. "
        "Verification on (the default) makes the handshake fail unless the chain is trusted.",
        ("tls", "certificates", "requests"),
    ),
    SecuritySample(
        "randomness-token",
        "CWE-338",
        "randomness",
        "python",
        "Use a CSPRNG for tokens, never `random`",
        "import random\nimport string\n\n\ndef new_api_token() -> str:\n"
        "    alphabet = string.ascii_letters + string.digits\n"
        "    return \"\".join(random.choice(alphabet) for _ in range(32))\n",
        "import secrets\nimport string\n\n\ndef new_api_token() -> str:\n"
        "    alphabet = string.ascii_letters + string.digits\n"
        "    return \"\".join(secrets.choice(alphabet) for _ in range(32))\n",
        "Mersenne Twister state is recoverable from a few outputs, so `random` tokens are "
        "predictable. `secrets` draws from the OS CSPRNG.",
        ("random", "csprng", "token"),
    ),
    SecuritySample(
        "xss-html-sink",
        "CWE-79",
        "xss",
        "python",
        "Escape untrusted values before they reach HTML",
        "def render_profile(name: str) -> str:\n"
        "    return f\"<h1>Hello {name}</h1>\"\n",
        "from markupsafe import escape\n\n\ndef render_profile(name: str) -> str:\n"
        "    return f\"<h1>Hello {escape(name)}</h1>\"\n",
        "`<script>` in the name runs in every viewer's session. `escape()` neutralises the HTML "
        "metacharacters so the value stays text.",
        ("xss", "html", "escape"),
    ),
    SecuritySample(
        "ssrf-allowlist",
        "CWE-918",
        "ssrf",
        "python",
        "Allow-list outbound hosts and reject internal addresses",
        "import urllib.request\n\n\ndef fetch_preview(url: str) -> bytes:\n"
        "    return urllib.request.urlopen(url, timeout=5).read()\n",
        "import ipaddress\nimport socket\nfrom urllib.parse import urlsplit\n\n"
        "ALLOWED_HOSTS = {\"api.partner.example\"}\n\n\n"
        "def fetch_preview(url: str) -> str:\n"
        "    host = urlsplit(url).hostname or \"\"\n"
        "    if host not in ALLOWED_HOSTS:\n"
        "        raise PermissionError(\"host not allowlisted\")\n"
        "    for info in socket.getaddrinfo(host, None):\n"
        "        addr = ipaddress.ip_address(info[4][0])\n"
        "        if addr.is_private or addr.is_link_local or addr.is_loopback:\n"
        "            raise PermissionError(\"internal address blocked\")\n"
        "    return f\"https://{host}/preview\"\n",
        "The vulnerable version forwards any URL, so `http://169.254.169.254/` returns cloud "
        "credentials. An allow-list plus a private/link-local address check blocks both direct "
        "and DNS-rebound internal targets; only allow-listed hosts are ever resolved.",
        ("ssrf", "allowlist", "metadata"),
    ),
    SecuritySample(
        "js-dom-innerhtml",
        "CWE-79",
        "xss",
        "javascript",
        "Assign untrusted text with textContent, not innerHTML",
        "function renderComment(el, comment) {\n"
        "  el.innerHTML = '<p>' + comment + '</p>';\n"
        "}\n",
        "function renderComment(el, comment) {\n"
        "  const p = document.createElement('p');\n"
        "  p.textContent = comment;\n"
        "  el.replaceChildren(p);\n"
        "}\n",
        "`innerHTML` parses the string as markup, so `<img onerror=...>` executes. `textContent` "
        "never parses, so the value can only ever be text.",
        ("xss", "dom", "javascript"),
    ),
    SecuritySample(
        "js-prototype-pollution-merge",
        "CWE-1321",
        "mass_assignment",
        "javascript",
        "Block __proto__ and constructor keys when merging objects",
        "function applySettings(target, payload) {\n"
        "  for (const key of Object.keys(payload)) {\n"
        "    target[key] = payload[key];\n"
        "  }\n"
        "  return target;\n"
        "}\n",
        "const ALLOWED_KEYS = new Set(['name', 'email']);\n\n"
        "function applySettings(target, payload) {\n"
        "  for (const key of ALLOWED_KEYS) {\n"
        "    if (key in payload) target[key] = payload[key];\n"
        "  }\n"
        "  return target;\n"
        "}\n",
        "Copying arbitrary keys lets `__proto__` reach Object.prototype and pollute every object "
        "in the process. Iterating an explicit ALLOWED_KEYS set bounds the write surface.",
        ("prototype", "javascript", "mass_assignment"),
    ),
    SecuritySample(
        "ts-sql-template-tag",
        "CWE-89",
        "injection",
        "typescript",
        "Use parameterised queries in TypeScript data access",
        "export async function findByEmail(db: Client, email: string) {\n"
        "  return db.query(`SELECT id FROM users WHERE email = '${email}'`);\n"
        "}\n",
        "export async function findByEmail(db: Client, email: string) {\n"
        "  return db.query('SELECT id FROM users WHERE email = $1', [email]);\n"
        "}\n",
        "Template literals interpolate before the query is parsed. `$1` binding keeps the value "
        "in the parameter channel where the planner cannot see it as SQL.",
        ("sql", "typescript", "injection"),
    ),
    SecuritySample(
        "go-path-traversal",
        "CWE-22",
        "file_access",
        "go",
        "Validate joined paths against the base directory in Go",
        "package files\n\nimport (\n\t\"os\"\n\t\"path/filepath\"\n)\n\n"
        "func ReadUpload(name string) ([]byte, error) {\n"
        "\tpath := filepath.Join(\"/srv/uploads\", name)\n"
        "\treturn os.ReadFile(path)\n"
        "}\n",
        "package files\n\nimport (\n\t\"errors\"\n\t\"os\"\n\t\"path/filepath\"\n\t\"strings\"\n)\n\n"
        "func ReadUpload(name string) ([]byte, error) {\n"
        "\tbase, err := filepath.Abs(\"/srv/uploads\")\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tpath, err := filepath.Abs(filepath.Join(base, name))\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tif !strings.HasPrefix(path, base+string(os.PathSeparator)) {\n"
        "\t\treturn nil, errors.New(\"path escapes uploads directory\")\n"
        "\t}\n"
        "\treturn os.ReadFile(path)\n"
        "}\n",
        "`filepath.Join` cleans `..` but does not confine it. Abs + prefix check with the "
        "separator appended rejects sibling directories such as `/srv/uploads-evil`.",
        ("go", "path", "traversal"),
    ),
    SecuritySample(
        "rust-command-injection",
        "CWE-78",
        "injection",
        "rust",
        "Avoid the shell in Rust process spawning",
        "use std::process::Command;\n\n"
        "pub fn convert(path: &str) -> String {\n"
        "    let out = Command::new(\"sh\")\n"
        "        .arg(\"-c\")\n"
        "        .arg(format!(\"convert {} /tmp/out.png\", path))\n"
        "        .output()\n"
        "        .expect(\"spawn\");\n"
        "    String::from_utf8_lossy(&out.stdout).to_string()\n"
        "}\n",
        "use std::process::Command;\n\n"
        "pub fn convert(path: &str) -> String {\n"
        "    let out = Command::new(\"convert\")\n"
        "        .arg(path)\n"
        "        .arg(\"out.png\")\n"
        "        .output()\n"
        "        .expect(\"spawn\");\n"
        "    String::from_utf8_lossy(&out.stdout).to_string()\n"
        "}\n",
        "`sh -c` with an interpolated path re-parses the input as shell syntax. Spawning the "
        "binary directly passes each argument through untouched.",
        ("rust", "command", "injection"),
    ),
    SecuritySample(
        "java-jdbc-prepared",
        "CWE-89",
        "injection",
        "java",
        "Use PreparedStatement in Java database access",
        "public User find(String email, Connection conn) throws SQLException {\n"
        "    String sql = \"SELECT id, email FROM users WHERE email = '\" + email + \"'\";\n"
        "    try (Statement st = conn.createStatement()) {\n"
        "        ResultSet rs = st.executeQuery(sql);\n"
        "        return rs.next() ? new User(rs.getLong(1), rs.getString(2)) : null;\n"
        "    }\n"
        "}\n",
        "public User find(String email, Connection conn) throws SQLException {\n"
        "    String sql = \"SELECT id, email FROM users WHERE email = ?\";\n"
        "    try (PreparedStatement ps = conn.prepareStatement(sql)) {\n"
        "        ps.setString(1, email);\n"
        "        ResultSet rs = ps.executeQuery();\n"
        "        return rs.next() ? new User(rs.getLong(1), rs.getString(2)) : null;\n"
        "    }\n"
        "}\n",
        "Statement + concatenation parses user input as SQL. PreparedStatement sends the value "
        "separately, and it is bound as a typed parameter.",
        ("java", "jdbc", "injection"),
    ),
    SecuritySample(
        "sql-grant-wildcard",
        "CWE-732",
        "authorization",
        "sql",
        "Grant least privilege instead of ALL to every host",
        "-- vulnerable: full control for any connecting host\n"
        "GRANT ALL PRIVILEGES ON app.* TO 'app'@'%';\n",
        "-- safe: scoped verbs from an allow-listed host\n"
        "GRANT SELECT, INSERT, UPDATE ON app.* TO 'app'@'10.0.0.%';\n"
        "REVOKE DELETE ON app.* FROM 'app'@'10.0.0.%';\n",
        "`ALL` on `%` turns one leaked app credential into DROP TABLE anywhere in the database. "
        "Scoped verbs from a private subnet keep the blast radius inside application tables.",
        ("sql", "least_privilege", "grants"),
    ),
    SecuritySample(
        "shell-unquoted-variable",
        "CWE-78",
        "injection",
        "shell",
        "Quote variables and avoid eval in shell scripts",
        "#!/bin/sh\n# vulnerable: unquoted expansion + eval\n"
        "cleanup() {\n"
        "  eval rm -rf $TARGET_DIR/*\n"
        "}\n",
        "#!/bin/sh\n# safe: quoted expansion, no eval, explicit guard\n"
        "cleanup() {\n"
        "  case \"$TARGET_DIR\" in\n"
        "    /var/tmp/app-*) ;;\n"
        "    *) echo \"refusing to clean $TARGET_DIR\" >&2; return 1 ;;\n"
        "  esac\n"
        "  rm -rf -- \"$TARGET_DIR\"/*\n"
        "}\n",
        "Unquoted expansion word-splits and globs, and `eval` re-parses the result, so a space or "
        "`;` in TARGET_DIR becomes a new command. Quoting plus a case guard bounds what can run.",
        ("shell", "quoting", "eval"),
    ),
    SecuritySample(
        "shell-curl-pipe-sh",
        "CWE-494",
        "deserialization",
        "shell",
        "Verify a downloaded artifact before executing it",
        "#!/bin/sh\n# vulnerable: execute whatever the server returns\n"
        "curl -fsSL https://get.example.com/install.sh | sh\n",
        "#!/bin/sh\n# safe: pin + verify the digest, then execute the local file\n"
        "WORKDIR=\"/var/tmp/app-install\"\n"
        "mkdir -p \"$WORKDIR\"\n"
        "curl -fsSL -o \"$WORKDIR/install.sh\" https://get.example.com/install.sh\n"
        "echo \"9f2c1c0d5b6f1f0d3f24b6f2d7c0a1e9a6f0b2c3d4e5f60718293a4b5c6d7e8f  $WORKDIR/install.sh\" \\\n"
        "  | sha256sum -c -\n"
        "sh \"$WORKDIR/install.sh\"\n",
        "`curl | sh` trusts the network and the server at install time. Verifying a pinned "
        "digest makes tampering detectable before any byte is executed.",
        ("shell", "supply_chain", "verify"),
    ),
    SecuritySample(
        "password-reset-token-rotation",
        "CWE-384",
        "session",
        "python",
        "Rotate the session identifier after privilege change",
        "def login(request, session, user) -> dict:\n"
        "    session[\"user_id\"] = user.id\n"
        "    session[\"role\"] = user.role\n"
        "    return {\"ok\": True}\n",
        "def login(request, session, user) -> dict:\n"
        "    session.clear()\n"
        "    session.rotate()\n"
        "    session[\"user_id\"] = user.id\n"
        "    session[\"role\"] = user.role\n"
        "    return {\"ok\": True}\n",
        "Reusing the pre-login session id lets an attacker who fixed the cookie ride the "
        "authenticated session. Clearing then rotating gives an identifier the attacker never saw.",
        ("session", "fixation", "rotate"),
    ),
    SecuritySample(
        "authz-idor-object",
        "CWE-639",
        "authorization",
        "python",
        "Scope every object read to the caller's tenant",
        "def get_invoice(conn, invoice_id: int) -> dict:\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\"SELECT id, total FROM invoices WHERE id = %s\", (invoice_id,))\n"
        "    return cursor.fetchone()\n",
        "def get_invoice(conn, invoice_id: int, tenant_id: str) -> dict:\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\n"
        "        \"SELECT id, total FROM invoices WHERE id = %s AND tenant_id = %s\",\n"
        "        (invoice_id, tenant_id),\n"
        "    )\n"
        "    return cursor.fetchone()\n",
        "Fetching by primary key alone lets any authenticated caller enumerate other tenants' "
        "invoices. A tenant_id predicate makes the object graph itself enforce authorization.",
        ("idor", "tenant_id", "authorization"),
    ),
    SecuritySample(
        "regex-catastrophic-backtracking",
        "CWE-1333",
        "regex_dos",
        "python",
        "Bound regex input and avoid nested quantifiers",
        "import re\n\nEMAIL_RE = re.compile(r\"^([a-zA-Z0-9]+)*@example\\.com$\")\n\n\n"
        "def is_company_email(value: str) -> bool:\n"
        "    return bool(EMAIL_RE.match(value))\n",
        "import re\n\nMAX_EMAIL_LEN = 254\nEMAIL_RE = re.compile(r\"^[a-zA-Z0-9._%+-]{1,64}@example\\.com$\")\n\n\n"
        "def is_company_email(value: str) -> bool:\n"
        "    if len(value) > MAX_EMAIL_LEN:\n"
        "        return False\n"
        "    return bool(EMAIL_RE.match(value))\n",
        "A quantified group that itself contains a quantifier backtracks exponentially on "
        "near-matches. Bounded repetition plus an explicit length MAX keeps matching linear.",
        ("regex", "redos", "backtracking"),
    ),
    SecuritySample(
        "logging-credential-leak",
        "CWE-532",
        "logging",
        "python",
        "Redact credentials before logging request payloads",
        "import logging\n\nlogger = logging.getLogger(__name__)\nSENSITIVE = (\"password\", \"token\", \"secret\")\n\n\n"
        "def log_signup(payload: dict) -> None:\n"
        "    safe = {\n"
        "        field: (\"***\" if field.lower() in SENSITIVE else value)\n"
        "        for field, value in payload.items()\n"
        "    }\n"
        "    logger.info(\"signup failed for %s\", safe)\n",
        "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
        "def log_signup(payload: dict) -> None:\n"
        "    logger.info(\"signup failed for %s\", payload)\n",
        "Log lines are copied into SIEMs, tickets and screenshots. Redacting the sensitive keys "
        "before formatting keeps the diagnostic value without the credential.",
        ("logging", "redact", "credentials"),
    ),
    SecuritySample(
        "cors-credential-wildcard",
        "CWE-942",
        "cors",
        "python",
        "Reflect only allow-listed origins when credentials are allowed",
        "def cors_headers(request) -> dict:\n"
        "    return {\n"
        "        \"Access-Control-Allow-Origin\": request.headers.get(\"Origin\", \"*\"),\n"
        "        \"Access-Control-Allow-Credentials\": \"true\",\n"
        "    }\n",
        "ALLOWED_ORIGINS = {\"https://app.example.com\"}\n\n\n"
        "def cors_headers(request) -> dict:\n"
        "    origin = request.headers.get(\"Origin\", \"\")\n"
        "    if origin not in ALLOWED_ORIGINS:\n"
        "        return {}\n"
        "    return {\n"
        "        \"Access-Control-Allow-Origin\": origin,\n"
        "        \"Access-Control-Allow-Credentials\": \"true\",\n"
        "        \"Vary\": \"Origin\",\n"
        "    }\n",
        "Reflecting an arbitrary Origin with credentials lets any site read authenticated "
        "responses. Matching against ALLOWED_ORIGINS (and varying on Origin) closes that.",
        ("cors", "credentials", "origin"),
    ),
    SecuritySample(
        "csrf-state-change",
        "CWE-352",
        "csrf",
        "python",
        "Require a synchroniser CSRF token on state-changing routes",
        "from flask import Flask\n\napp = Flask(__name__)\n\n\n"
        "@app.post(\"/account/email\")\n"
        "def change_email():\n"
        "    return update_email()\n",
        "import hmac\nfrom flask import Flask, abort, request, session\n\napp = Flask(__name__)\n\n\n"
        "@app.post(\"/account/email\")\n"
        "def change_email():\n"
        "    token = request.form.get(\"csrf_token\", \"\")\n"
        "    expected = session.get(\"csrf_token\", \"\")\n"
        "    if not expected or not hmac.compare_digest(token, expected):\n"
        "        abort(400)\n"
        "    return update_email()\n",
        "A POST is not enough: another origin can submit the form with the victim's cookies. "
        "Checking a per-session token the attacker cannot read blocks the forged request.",
        ("csrf", "token", "session"),
    ),
    SecuritySample(
        "header-injection-crlf",
        "CWE-93",
        "header_injection",
        "python",
        "Reject CR/LF in values that become response headers",
        "def download_headers(filename: str) -> dict:\n"
        "    return {\"Content-Disposition\": f\"attachment; filename={filename}\"}\n",
        "def download_headers(filename: str) -> dict:\n"
        "    if any(ch in filename for ch in \"\\r\\n\"):\n"
        "        raise ValueError(\"invalid header value\")\n"
        "    safe = filename.replace('\"', \"\")\n"
        "    return {\"Content-Disposition\": f'attachment; filename=\"{safe}\"'}\n",
        "CRLF in a header value terminates the header block, letting the caller inject extra "
        "headers or a body. Validating and rejecting control characters keeps it one value.",
        ("headers", "crlf", "injection"),
    ),
    SecuritySample(
        "integrity-unsigned-update",
        "CWE-494",
        "deserialization",
        "python",
        "Verify update signatures before installing",
        "def install_update(blob: bytes, signature: bytes) -> None:\n"
        "    write_package(blob)\n"
        "    activate_package()\n",
        "import hmac\n\nPUBLIC_VERIFY_KEY = b\"release-key\"\n\n\n"
        "def install_update(blob: bytes, signature: bytes) -> None:\n"
        "    expected = hmac.new(PUBLIC_VERIFY_KEY, blob, \"sha256\").digest()\n"
        "    if not hmac.compare_digest(expected, signature):\n"
        "        raise PermissionError(\"unsigned update rejected\")\n"
        "    write_package(blob)\n"
        "    activate_package()\n",
        "Installing an unverified blob turns any mirror or CDN compromise into code execution. "
        "Verifying an HMAC (or public-key signature) over the exact bytes gates activation.",
        ("integrity", "signature", "update"),
    ),
    SecuritySample(
        "auth-fail-open-exception",
        "CWE-287",
        "fail_open",
        "python",
        "Fail closed when the authorization check errors",
        "def is_authorized(token: str, checker) -> bool:\n"
        "    try:\n"
        "        return checker.verify(token)\n"
        "    except Exception:\n"
        "        return True\n",
        "def is_authorized(token: str, checker) -> bool:\n"
        "    try:\n"
        "        return checker.verify(token)\n"
        "    except Exception:\n"
        "        return False\n",
        "Swallowing the error and granting access means an outage in the checker becomes a full "
        "bypass. Returning False keeps the deny default when the check cannot run.",
        ("auth", "fail_closed", "exceptions"),
    ),
    SecuritySample(
        "unbounded-read-length",
        "CWE-190",
        "integer_overflow",
        "python",
        "Clamp lengths read from the wire",
        "def read_frame(sock) -> bytes:\n"
        "    header = sock.recv(4)\n"
        "    length = int.from_bytes(header, \"big\")\n"
        "    return sock.recv(length)\n",
        "MAX_FRAME = 1 << 20\n\n\n"
        "def read_frame(sock) -> bytes:\n"
        "    header = sock.recv(4)\n"
        "    length = int.from_bytes(header, \"big\")\n"
        "    if length > MAX_FRAME:\n"
        "        raise ValueError(f\"frame length {length} exceeds MAX_FRAME\")\n"
        "    return sock.recv(length)\n",
        "A 4-byte length field can request 4 GiB per connection, which is a trivial memory "
        "exhaustion. Clamping against MAX_FRAME rejects the allocation before it happens.",
        ("length", "memory", "limits"),
    ),
    SecuritySample(
        "xml-entity-expansion",
        "CWE-611",
        "parsing",
        "python",
        "Parse untrusted XML with entity resolution disabled",
        "from lxml import etree\n\n\ndef parse_config(blob: bytes):\n"
        "    return etree.fromstring(blob)\n",
        "import defusedxml.ElementTree as ET\n\n\ndef parse_config(blob: bytes):\n"
        "    return ET.fromstring(blob)\n",
        "A DTD can define entities that read local files or expand exponentially (billion laughs). "
        "defusedxml forbids DTDs and entity resolution by default.",
        ("xml", "xxe", "parsing"),
    ),
    SecuritySample(
        "redirect-open",
        "CWE-601",
        "redirect",
        "python",
        "Only redirect to same-origin or allow-listed targets",
        "from urllib.parse import parse_qs\n\n\ndef post_login_redirect(request):\n"
        "    target = request.args.get(\"next\", \"/\")\n"
        "    return redirect(target)\n",
        "from urllib.parse import urlparse\n\nALLOWED_HOSTS = {\"app.example.com\"}\n\n\n"
        "def post_login_redirect(request):\n"
        "    target = request.args.get(\"next\", \"/\")\n"
        "    parsed = urlparse(target)\n"
        "    if parsed.netloc and parsed.hostname not in ALLOWED_HOSTS:\n"
        "        target = \"/\"\n"
        "    return redirect(target)\n",
        "An unvalidated `next` lets phishing links bounce through the trusted domain. Parsing and "
        "checking the hostname (or forcing a relative path) keeps redirects local.",
        ("redirect", "urlparse", "allowlist"),
    ),
    SecuritySample(
        "tempfile-insecure-path",
        "CWE-377",
        "file_access",
        "python",
        "Create temp files atomically with mkstemp/NamedTemporaryFile",
        "import os\n\n\ndef write_scratch(data: bytes) -> str:\n"
        "    path = \"/tmp/scratch.txt\"\n"
        "    with open(path, \"wb\") as handle:\n"
        "        handle.write(data)\n"
        "    return path\n",
        "import os\nimport tempfile\n\n\ndef write_scratch(data: bytes) -> str:\n"
        "    fd, path = tempfile.mkstemp(prefix=\"scratch-\", suffix=\".txt\")\n"
        "    with os.fdopen(fd, \"wb\") as handle:\n"
        "        handle.write(data)\n"
        "    return path\n",
        "A predictable path in a shared /tmp can be pre-created as a symlink (or read by another "
        "local user). mkstemp creates it atomically with O_EXCL and 0600 permissions.",
        ("tempfile", "symlink", "race"),
    ),
    SecuritySample(
        "jwt-algorithm-confusion",
        "CWE-347",
        "authentication",
        "python",
        "Pin the JWT algorithm and verify the signature",
        "import jwt\n\n\ndef read_claims(token: str, key: str) -> dict:\n"
        "    return jwt.decode(token, options={\"verify_signature\": False})\n",
        "import jwt\n\nALLOWED_ALGORITHMS = [\"RS256\"]\n\n\n"
        "def read_claims(token: str, key: str) -> dict:\n"
        "    return jwt.decode(\n"
        "        token,\n"
        "        key,\n"
        "        algorithms=ALLOWED_ALGORITHMS,\n"
        "        options={\"require\": [\"exp\", \"sub\"]},\n"
        "    )\n",
        "Decoding without verification (or trusting the header alg) lets an attacker mint `none` "
        "or HS256-with-the-public-key tokens. Pinning `algorithms=` and requiring exp/sub fixes it.",
        ("jwt", "algorithms", "authentication"),
    ),
    SecuritySample(
        "secret-management-env",
        "CWE-798",
        "secrets",
        "python",
        "Load credentials from the environment or a secret manager",
        "import requests\n\nSTRIPE_KEY = \"sk_live_51H8xQ2abcdefghijklmnop\"\n\n\n"
        "def charge(amount: int) -> dict:\n"
        "    return requests.post(\n"
        "        \"https://api.stripe.example/v1/charges\",\n"
        "        auth=(STRIPE_KEY, \"\"),\n"
        "        timeout=10,\n"
        "    ).json()\n",
        "import os\n\nimport requests\n\n\n"
        "def charge(amount: int) -> dict:\n"
        "    key = os.environ.get(\"STRIPE_API_KEY\")\n"
        "    if not key:\n"
        "        raise RuntimeError(\"STRIPE_API_KEY is not configured\")\n"
        "    return requests.post(\n"
        "        \"https://api.stripe.example/v1/charges\",\n"
        "        auth=(key, \"\"),\n"
        "        timeout=10,\n"
        "    ).json()\n",
        "A literal key in source is copied into git history, forks, and container layers forever. "
        "Reading it from the environment keeps the value outside the artifact and rotatable.",
        ("secrets", "environment", "credentials"),
    ),
    SecuritySample(
        "crypto-aead-mode",
        "CWE-327",
        "cryptography",
        "python",
        "Use an AEAD construction instead of a raw ECB/CBC block mode",
        "from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes\n\n\n"
        "def encrypt(key: bytes, data: bytes) -> bytes:\n"
        "    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()\n"
        "    return encryptor.update(data) + encryptor.finalize()\n",
        "import os\n\nfrom cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\n\n"
        "def encrypt(key: bytes, data: bytes) -> bytes:\n"
        "    nonce = os.urandom(12)\n"
        "    return nonce + AESGCM(key).encrypt(nonce, data, b\"\")\n",
        "ECB encrypts identical blocks identically and provides no integrity, so it leaks "
        "structure and accepts tampering. AESGCM with a fresh random nonce gives confidentiality "
        "plus authentication.",
        ("aesgcm", "aead", "cryptography"),
    ),
    SecuritySample(
        "mass-assignment-allowlist",
        "CWE-915",
        "mass_assignment",
        "python",
        "Bind only allow-listed fields from a request payload",
        "def update_user(user, payload: dict) -> None:\n"
        "    for field, value in payload.items():\n"
        "        setattr(user, field, value)\n",
        "ALLOWED_FIELDS = {\"display_name\", \"locale\", \"timezone\"}\n\n\n"
        "def update_user(user, payload: dict) -> None:\n"
        "    for field in ALLOWED_FIELDS:\n"
        "        if field in payload:\n"
        "            setattr(user, field, payload[field])\n",
        "Iterating the payload lets `is_admin` or `tenant_id` through from a self-service form. "
        "Iterating ALLOWED_FIELDS bounds the writable surface to what the endpoint owns.",
        ("mass_assignment", "allowlist", "authorization"),
    ),
    SecuritySample(
        "file-race-toctou",
        "CWE-367",
        "race_condition",
        "python",
        "Open with O_EXCL instead of checking existence first",
        "import os\n\n\ndef reserve(path: str) -> bool:\n"
        "    if os.path.exists(path):\n"
        "        return False\n"
        "    with open(path, \"w\", encoding=\"utf-8\") as handle:\n"
        "        handle.write(\"reserved\")\n"
        "    return True\n",
        "import os\n\n\ndef reserve(path: str) -> bool:\n"
        "    try:\n"
        "        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
        "    except FileExistsError:\n"
        "        return False\n"
        "    with os.fdopen(fd, \"w\", encoding=\"utf-8\") as handle:\n"
        "        handle.write(\"reserved\")\n"
        "    return True\n",
        "Between exists() and open() another process can create the file (often as a symlink). "
        "O_EXCL makes creation atomic, so only one caller wins.",
        ("toctou", "O_EXCL", "race"),
    ),
    SecuritySample(
        "log-injection-newlines",
        "CWE-117",
        "logging",
        "python",
        "Neutralise newlines before writing user data to logs",
        "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
        "def log_attempt(user_agent: str) -> None:\n"
        "    logger.info(\"ua=\" + user_agent)\n",
        "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
        "def log_attempt(user_agent: str) -> None:\n"
        "    safe = user_agent.replace(\"\\r\", \" \").replace(\"\\n\", \" \")[:200]\n"
        "    logger.info(\"ua=%s\", safe)\n",
        "A newline in the header lets the caller forge whole log lines (fake auth successes, "
        "fake alerts). Stripping CR/LF and truncating keeps it inside one field.",
        ("logging", "crlf", "injection"),
    ),
    SecuritySample(
        "nonce-reuse-aead",
        "CWE-324",
        "nonce_reuse",
        "python",
        "Generate a fresh random nonce for every AEAD message",
        "import os\n\nfrom cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\n"
        "STATIC_NONCE = bytes.fromhex(\"0102030405060708090a0b0c\")\n\n\n"
        "def seal(key: bytes, data: bytes) -> bytes:\n"
        "    return AESGCM(key).encrypt(STATIC_NONCE, data, b\"\")\n",
        "import os\n\nfrom cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\n\n"
        "def seal(key: bytes, data: bytes) -> bytes:\n"
        "    nonce = os.urandom(12)\n"
        "    return nonce + AESGCM(key).encrypt(nonce, data, b\"\")\n",
        "Reusing a GCM nonce under the same key leaks the XOR of plaintexts and allows forgery. "
        "A random 96-bit nonce per message keeps the security proof intact.",
        ("nonce", "aead", "gcm"),
    ),
    SecuritySample(
        "deserialization-yaml-full",
        "CWE-502",
        "deserialization",
        "python",
        "Use yaml.safe_load for untrusted documents",
        "import yaml\n\n\ndef load_config(text: str) -> dict:\n"
        "    return yaml.load(text)\n",
        "import yaml\n\n\ndef load_config(text: str) -> dict:\n"
        "    return yaml.safe_load(text)\n",
        "`yaml.load` constructs arbitrary Python objects via tags such as `!!python/object/apply`, "
        "which is code execution. safe_load accepts only the data-only subset.",
        ("yaml", "safe_load", "deserialization"),
    ),
    SecuritySample(
        "zip-slip-extract",
        "CWE-22",
        "file_access",
        "python",
        "Validate archive members before extracting",
        "import zipfile\n\n\ndef extract_all(archive: str, dest: str) -> None:\n"
        "    with zipfile.ZipFile(archive) as bundle:\n"
        "        bundle.extractall(dest)\n",
        "import zipfile\nfrom pathlib import Path\n\n\n"
        "def extract_all(archive: str, dest: str) -> None:\n"
        "    root = Path(dest).resolve()\n"
        "    with zipfile.ZipFile(archive) as bundle:\n"
        "        for member in bundle.namelist():\n"
        "            target = (root / member).resolve()\n"
        "            if not target.is_relative_to(root):\n"
        "                raise PermissionError(f\"zip-slip blocked: {member}\")\n"
        "            target.parent.mkdir(parents=True, exist_ok=True)\n"
        "            with bundle.open(member) as src, target.open(\"wb\") as dst:\n"
        "                dst.write(src.read())\n",
        "A member named `../../.ssh/authorized_keys` writes outside the destination. Resolving "
        "each name and checking is_relative_to(root) rejects traversal and absolute paths.",
        ("zip", "zip_slip", "archive"),
    ),
]
