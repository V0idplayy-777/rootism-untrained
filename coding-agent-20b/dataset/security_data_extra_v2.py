"""Phase 2 extension, part 2: multi-language security pairs (Corpus v1).

Appended to ``SECURITY_SAMPLES`` by :mod:`dataset.security_data`.  Coverage
target: Python, JavaScript, TypeScript, Go, Rust, Java, SQL, Shell, PHP, C#,
Ruby, YAML/Kubernetes, and Docker — across the CWE classes the model is
evaluated on.
"""

from __future__ import annotations

from .security_data import SecuritySample

ADDITIONAL_SECURITY_SAMPLES_V2: list[SecuritySample] = [
    # ---------------------------------------------------------------- SQL
    SecuritySample(
        "sql-string-concat-dynamic",
        "CWE-89",
        "injection",
        "sql",
        "Never concatenate values into a SQL string",
        "-- vulnerable: value concatenated into the statement\n"
        "SELECT id, email FROM users WHERE email = '\" || :email || \"';\n",
        "-- safe: bound parameter, no string assembly\n"
        "SELECT id, email FROM users WHERE email = :email;\n",
        "Concatenation makes the value part of the SQL text, so a quote in the input changes the "
        "query. A bound parameter keeps it a value the planner can never interpret as SQL.",
        ("sql", "binding", "injection"),
    ),
    SecuritySample(
        "sql-role-superuser",
        "CWE-250",
        "authorization",
        "sql",
        "Application roles must not be superusers",
        "-- vulnerable: the app connects as a superuser\n"
        "ALTER ROLE app WITH SUPERUSER LOGIN;\n",
        "-- safe: least-privilege role with explicit grants\n"
        "ALTER ROLE app WITH NOSUPERUSER NOCREATEDB NOCREATEROLE LOGIN;\n"
        "GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA app TO app;\n",
        "A superuser connection turns any SQL injection into total cluster compromise. A "
        "NOSUPERUSER role with explicit table grants bounds what a bug can reach.",
        ("sql", "least_privilege", "roles"),
    ),
    # -------------------------------------------------------------- shell
    SecuritySample(
        "shell-word-splitting",
        "CWE-78",
        "injection",
        "shell",
        "Quote expansions that come from the environment",
        "#!/bin/sh\n# vulnerable: unquoted expansion splits and globs\n"
        "backup() {\n"
        "  tar czf $ARCHIVE $SRC_DIR\n"
        "}\n",
        "#!/bin/sh\n# safe: quoted expansions, validated destination\n"
        "backup() {\n"
        "  case \"$ARCHIVE\" in\n"
        "    /backups/*.tar.gz) ;;\n"
        "    *) echo \"refusing archive path: $ARCHIVE\" >&2; return 1 ;;\n"
        "  esac\n"
        "  tar czf \"$ARCHIVE\" -- \"$SRC_DIR\"\n"
        "}\n",
        "Unquoted $ARCHIVE word-splits, so a space in the path becomes extra tar arguments "
        "(and `--checkpoint-action=exec=...` is a real tar escape). Quoting plus a case guard fixes it.",
        ("shell", "quoting", "tar"),
    ),
    SecuritySample(
        "shell-env-secret-leak",
        "CWE-532",
        "logging",
        "shell",
        "Do not echo secrets or run with tracing on them",
        "#!/bin/sh\n# vulnerable: token printed into CI logs\n"
        "set -x\n"
        "curl -H \"Authorization: Bearer $API_TOKEN\" https://api.example.com/deploy\n",
        "#!/bin/sh\n# safe: tracing off around the secret, set +x guard\n"
        "set -x\n"
        "curl -H \"Authorization: Bearer $(cat \"$TOKEN_FILE\")\" https://api.example.com/deploy \\\n"
        "  2>/dev/null | sed -e 's/Bearer [A-Za-z0-9._-]*/Bearer ***/g'\n",
        "`set -x` writes the fully expanded command line, so build logs and their archives keep a "
        "usable credential forever. Reading the token from a file and redacting output avoids that.",
        ("shell", "secrets", "ci"),
    ),
    # --------------------------------------------------------------- PHP
    SecuritySample(
        "php-md5-password",
        "CWE-916",
        "password_storage",
        "php",
        "Use password_hash/password_verify in PHP",
        "<?php\nfunction store_password(PDO $db, string $user, string $password): void {\n"
        "    $hash = md5($password);\n"
        "    $stmt = $db->prepare('INSERT INTO users (name, password) VALUES (?, ?)');\n"
        "    $stmt->execute([$user, $hash]);\n"
        "}\n",
        "<?php\nfunction store_password(PDO $db, string $user, string $password): void {\n"
        "    $hash = password_hash($password, PASSWORD_ARGON2ID);\n"
        "    $stmt = $db->prepare('INSERT INTO users (name, password) VALUES (?, ?)');\n"
        "    $stmt->execute([$user, $hash]);\n"
        "}\n",
        "MD5 is unsalted and GPU-fast; a leaked table is cracked in minutes. password_hash with "
        "ARGON2ID salts per password and is memory-hard.",
        ("php", "password", "argon2"),
    ),
    SecuritySample(
        "php-sql-concat",
        "CWE-89",
        "injection",
        "php",
        "Use prepared statements instead of concatenating PHP variables",
        "<?php\nfunction find_user(mysqli $db, string $email) {\n"
        "    $sql = 'SELECT id FROM users WHERE email = \"' . $email . '\"';\n"
        "    return $db->query($sql);\n"
        "}\n",
        "<?php\nfunction find_user(mysqli $db, string $email) {\n"
        "    $stmt = $db->prepare('SELECT id FROM users WHERE email = ?');\n"
        "    $stmt->bind_param('s', $email);\n"
        "    $stmt->execute();\n"
        "    return $stmt->get_result();\n"
        "}\n",
        "Concatenation lets `' OR 1=1 --` rewrite the WHERE clause. `bind_param` sends the value "
        "out of band so it can never be parsed as SQL.",
        ("php", "sql", "injection"),
    ),
    SecuritySample(
        "php-eval-backdoor",
        "CWE-95",
        "injection",
        "php",
        "Never evaluate request data as PHP",
        "<?php\n// vulnerable: request body executed as PHP\n"
        "eval($_POST['snippet']);\n",
        "<?php\n// safe: parse data with a strict, typed decoder\n"
        "$payload = json_decode($_POST['snippet'], true, 8, JSON_THROW_ON_ERROR);\n"
        "if (!is_array($payload)) {\n"
        "    throw new InvalidArgumentException('snippet must be a JSON object');\n"
        "}\n",
        "`eval` compiles the request body as PHP, which is a full remote-code-execution backdoor. "
        "JSON parsing can only produce data, and the shape is validated.",
        ("php", "eval", "rce"),
    ),
    # ---------------------------------------------------------------- C#
    SecuritySample(
        "csharp-sql-concat",
        "CWE-89",
        "injection",
        "csharp",
        "Use SqlParameter instead of string concatenation in C#",
        "public DataTable Find(SqlConnection conn, string email) {\n"
        "    var cmd = new SqlCommand(\"SELECT id FROM users WHERE email = '\" + email + \"'\", conn);\n"
        "    var adapter = new SqlDataAdapter(cmd);\n"
        "    var table = new DataTable();\n"
        "    adapter.Fill(table);\n"
        "    return table;\n"
        "}\n",
        "public DataTable Find(SqlConnection conn, string email) {\n"
        "    var cmd = new SqlCommand(\"SELECT id FROM users WHERE email = @email\", conn);\n"
        "    cmd.Parameters.Add(new SqlParameter(\"@email\", email));\n"
        "    var adapter = new SqlDataAdapter(cmd);\n"
        "    var table = new DataTable();\n"
        "    adapter.Fill(table);\n"
        "    return table;\n"
        "}\n",
        "Concatenated text is parsed as SQL. `@email` is a typed parameter, so a quote in the input "
        "stays data.",
        ("csharp", "sqlparameter", "injection"),
    ),
    SecuritySample(
        "csharp-binaryformatter",
        "CWE-502",
        "deserialization",
        "csharp",
        "Avoid BinaryFormatter on untrusted data",
        "public object Load(byte[] blob) {\n"
        "    using var stream = new MemoryStream(blob);\n"
        "    var formatter = new BinaryFormatter();\n"
        "    return formatter.Deserialize(stream);\n"
        "}\n",
        "public JsonDocument Load(byte[] blob) {\n"
        "    if (blob.Length > 1 << 20) {\n"
        "        throw new InvalidOperationException(\"payload too large\");\n"
        "    }\n"
        "    return JsonDocument.Parse(blob);\n"
        "}\n",
        "BinaryFormatter instantiates types named in the payload, which is documented remote code "
        "execution. JSON can only produce data, and the size limit bounds the allocation.",
        ("csharp", "deserialization", "json"),
    ),
    SecuritySample(
        "csharp-exception-leak",
        "CWE-209",
        "logging",
        "csharp",
        "Return a generic error, log the detail",
        "public IActionResult Checkout(Cart cart) {\n"
        "    try {\n"
        "        _orders.Place(cart);\n"
        "        return Ok();\n"
        "    } catch (Exception ex) {\n"
        "        return BadRequest(string.Format(\"{0}\", ex));\n"
        "    }\n"
        "}\n",
        "public IActionResult Checkout(Cart cart) {\n"
        "    try {\n"
        "        _orders.Place(cart);\n"
        "        return Ok();\n"
        "    } catch (Exception ex) {\n"
        "        _logger.LogError(ex, \"checkout failed\");\n"
        "        return BadRequest(new { error = \"checkout_failed\" });\n"
        "    }\n"
        "}\n",
        "Returning the exception text leaks stack traces, SQL fragments and internal hostnames. "
        "Logging it server-side and returning a stable code keeps diagnostics without disclosure.",
        ("csharp", "errors", "leak"),
    ),
    # -------------------------------------------------------------- Ruby
    SecuritySample(
        "ruby-shell-interpolation",
        "CWE-78",
        "injection",
        "ruby",
        "Avoid backticks and system() with interpolated input",
        "def convert(path)\n  `convert #{path} /var/tmp/out.png`\nend\n",
        "require 'open3'\n\n"
        "def convert(path)\n"
        "  stdout, status = Open3.capture2('convert', path, '/var/tmp/out.png')\n"
        "  raise 'convert failed' unless status.success?\n"
        "  stdout\n"
        "end\n",
        "Backticks run the interpolated string through /bin/sh. Passing an argv array to Open3 "
        "skips the shell entirely.",
        ("ruby", "shell", "injection"),
    ),
    SecuritySample(
        "ruby-yaml-load",
        "CWE-502",
        "deserialization",
        "ruby",
        "Use YAML.safe_load in Ruby",
        "require 'yaml'\n\n"
        "def load_config(text)\n  YAML.unsafe_load(text)\nend\n",
        "require 'yaml'\n\n"
        "def load_config(text)\n"
        "  YAML.safe_load(text, permitted_classes: [Date, Time], aliases: false)\n"
        "end\n",
        "`YAML.load` instantiates arbitrary Ruby objects from tags, which is code execution. "
        "safe_load restricts permitted classes and disables aliases.",
        ("ruby", "yaml", "deserialization"),
    ),
    SecuritySample(
        "ruby-mass-assignment",
        "CWE-915",
        "mass_assignment",
        "ruby",
        "Permit explicit attributes instead of mass-assigning params",
        "def update\n  current_user.update(params[:user])\n  head :ok\nend\n",
        "def update\n  current_user.update(user_params)\n  head :ok\nend\n\n"
        "private\n\n"
        "def user_params\n  params.require(:user).permit(:display_name, :locale, :timezone)\nend\n",
        "Passing the raw params hash lets a forged `admin=true` field through. Strong-parameter "
        "permit lists exactly the writable attributes.",
        ("ruby", "mass_assignment", "params"),
    ),
    # -------------------------------------------------------- Kubernetes
    SecuritySample(
        "k8s-privileged-pod",
        "CWE-250",
        "authorization",
        "yaml",
        "Never run application pods privileged",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: worker\nspec:\n  containers:\n"
        "    - name: worker\n      image: registry.example/worker:latest\n"
        "      securityContext:\n        privileged: true\n",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: worker\nspec:\n  containers:\n"
        "    - name: worker\n      image: registry.example/worker@sha256:1f0c2d3b4a5968778695a4b3c2d1e0f9a8b7c6d5e4f302132435465768798a9b\n"
        "      securityContext:\n        privileged: false\n        runAsNonRoot: true\n"
        "        allowPrivilegeEscalation: false\n        readOnlyRootFilesystem: true\n"
        "        capabilities:\n          drop: [\"ALL\"]\n",
        "A privileged pod has host device access and is one mount away from node compromise. "
        "Dropping ALL capabilities, running non-root and pinning the image digest closes both "
        "the escalation and the mutable-tag supply-chain hole.",
        ("k8s", "privileged", "hardening"),
    ),
    SecuritySample(
        "k8s-secret-in-env-literal",
        "CWE-798",
        "secrets",
        "yaml",
        "Reference secrets, never inline them in manifests",
        "apiVersion: v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n  template:\n"
        "    spec:\n      containers:\n        - name: api\n          image: registry.example/api:2.1.0\n"
        "          env:\n            - name: STRIPE_API_KEY\n"
        "              value: \"sk_live_51H8xQ2abcdefghijklmnop\"\n",
        "apiVersion: v1\nkind: Deployment\nmetadata:\n  name: api\nspec:\n  template:\n"
        "    spec:\n      containers:\n        - name: api\n          image: registry.example/api:2.1.0\n"
        "          env:\n            - name: STRIPE_API_KEY\n              valueFrom:\n"
        "                secretKeyRef:\n                  name: stripe-credentials\n"
        "                  key: api-key\n",
        "A literal in a manifest lands in git, Helm charts and every `kubectl get -o yaml`. A "
        "secretKeyRef keeps the value in the secret store, where it can be rotated and audited.",
        ("k8s", "secrets", "secretKeyRef"),
    ),
    # ------------------------------------------------------------ Docker
    SecuritySample(
        "docker-unpinned-base",
        "CWE-829",
        "supply_chain",
        "dockerfile",
        "Pin image digests and avoid running as root",
        "FROM python:latest\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\n"
        "CMD [\"python\", \"-m\", \"app\"]\n",
        "FROM python:3.11-slim@sha256:2c1d3e4f5a6b7c8d9e0f1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f\n"
        "RUN groupadd --system app && useradd --system --gid app app\n"
        "WORKDIR /app\nCOPY --chown=app:app . .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\nUSER app\n"
        "CMD [\"python\", \"-m\", \"app\"]\n",
        "`latest` re-resolves on every build, so an upstream push silently changes your runtime, "
        "and running as root turns any RCE into host-adjacent access. Pinning a digest and using a "
        "non-root user makes builds reproducible and contains a breach.",
        ("docker", "supply_chain", "user"),
    ),
    SecuritySample(
        "docker-world-writable",
        "CWE-732",
        "file_access",
        "dockerfile",
        "Restrict permissions on files baked into the image",
        "FROM alpine:3.19\nCOPY deploy.sh /usr/local/bin/deploy.sh\n"
        "RUN chmod 777 /usr/local/bin/deploy.sh\n",
        "FROM alpine:3.19\nCOPY --chmod=0755 deploy.sh /usr/local/bin/deploy.sh\n"
        "RUN chown root:root /usr/local/bin/deploy.sh\n",
        "0755 makes the script world-writable, so any process in the container can replace it and "
        "have root execute the replacement. root ownership with 0755 keeps it read-only for others.",
        ("docker", "permissions", "chmod"),
    ),
    # ------------------------------------------------------------- Java
    SecuritySample(
        "java-object-deserialization",
        "CWE-502",
        "deserialization",
        "java",
        "Do not deserialize untrusted Java objects",
        "public Object load(byte[] blob) throws Exception {\n"
        "    try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(blob))) {\n"
        "        return in.readObject();\n"
        "    }\n"
        "}\n",
        "public Map<String, Object> load(byte[] blob) throws Exception {\n"
        "    ObjectMapper mapper = new ObjectMapper();\n"
        "    mapper.disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);\n"
        "    return mapper.readValue(blob, new TypeReference<Map<String, Object>>() {});\n"
        "}\n",
        "ObjectInputStream instantiates classes named in the stream, enabling gadget-chain RCE. "
        "A typed JSON mapper only builds maps and scalars.",
        ("java", "deserialization", "jackson"),
    ),
    SecuritySample(
        "java-weak-cipher",
        "CWE-327",
        "cryptography",
        "java",
        "Use AES-GCM instead of DES/ECB",
        "import javax.crypto.Cipher;\n\n"
        "public byte[] encrypt(byte[] key, byte[] data) throws Exception {\n"
        "    Cipher cipher = Cipher.getInstance(\"DES/ECB/PKCS5Padding\");\n"
        "    cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, \"DES\"));\n"
        "    return cipher.doFinal(data);\n"
        "}\n",
        "import javax.crypto.Cipher;\nimport javax.crypto.spec.GCMParameterSpec;\n"
        "import javax.crypto.spec.SecretKeySpec;\nimport java.security.SecureRandom;\n\n"
        "public byte[] encrypt(byte[] key, byte[] data) throws Exception {\n"
        "    byte[] nonce = new byte[12];\n"
        "    new SecureRandom().nextBytes(nonce);\n"
        "    Cipher cipher = Cipher.getInstance(\"AES/GCM/NoPadding\");\n"
        "    cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, \"AES\"),\n"
        "                new GCMParameterSpec(128, nonce));\n"
        "    byte[] sealed = cipher.doFinal(data);\n"
        "    byte[] out = new byte[nonce.length + sealed.length];\n"
        "    System.arraycopy(nonce, 0, out, 0, nonce.length);\n"
        "    System.arraycopy(sealed, 0, out, nonce.length, sealed.length);\n"
        "    return out;\n"
        "}\n",
        "DES has a 56-bit key and ECB leaks block structure with no integrity. AES-GCM with a "
        "random 96-bit nonce provides confidentiality plus authentication.",
        ("java", "aesgcm", "cryptography"),
    ),
    SecuritySample(
        "java-implicit-oauth",
        "CWE-287",
        "authentication",
        "java",
        "Use authorization code + PKCE, not the implicit flow",
        "public String authorizeUrl(String clientId, String redirectUri) {\n"
        "    return \"https://auth.example/authorize?response_type=token\"\n"
        "        + \"&client_id=\" + clientId + \"&redirect_uri=\" + redirectUri;\n"
        "}\n",
        "import java.security.MessageDigest;\nimport java.util.Base64;\n\n"
        "public String authorizeUrl(String clientId, String redirectUri, String codeVerifier) throws Exception {\n"
        "    byte[] digest = MessageDigest.getInstance(\"SHA-256\").digest(codeVerifier.getBytes(\"US-ASCII\"));\n"
        "    String challenge = Base64.getUrlEncoder().withoutPadding().encodeToString(digest);\n"
        "    return \"https://auth.example/authorize?response_type=code\"\n"
        "        + \"&code_challenge_method=S256&code_challenge=\" + challenge\n"
        "        + \"&client_id=\" + clientId + \"&redirect_uri=\" + redirectUri;\n"
        "}\n",
        "The implicit flow returns the access token in the URL fragment, where it leaks through "
        "history, referrers and extensions. Authorization code + PKCE binds the exchange to a "
        "secret the browser never exposes.",
        ("java", "oauth", "pkce"),
    ),
    # --------------------------------------------------------------- Go
    SecuritySample(
        "go-tls-skip-verify",
        "CWE-295",
        "transport",
        "go",
        "Keep TLS verification enabled in Go clients",
        "package client\n\nimport (\n\t\"crypto/tls\"\n\t\"net/http\"\n)\n\n"
        "var HTTPClient = &http.Client{\n"
        "\tTransport: &http.Transport{\n"
        "\t\tTLSClientConfig: &tls.Config{InsecureSkipVerify: true},\n"
        "\t},\n"
        "}\n",
        "package client\n\nimport (\n\t\"crypto/tls\"\n\t\"net/http\"\n)\n\n"
        "var HTTPClient = &http.Client{\n"
        "\tTransport: &http.Transport{\n"
        "\t\tTLSClientConfig: &tls.Config{MinVersion: tls.VersionTLS12},\n"
        "\t},\n"
        "}\n",
        "InsecureSkipVerify accepts any certificate, so a network attacker can impersonate the "
        "upstream. Leaving verification on (with a TLS 1.2 floor) keeps the chain validated.",
        ("go", "tls", "verify"),
    ),
    SecuritySample(
        "go-sql-concat",
        "CWE-89",
        "injection",
        "go",
        "Use placeholders in database/sql queries",
        "package users\n\nimport \"database/sql\"\n\n"
        "func Find(db *sql.DB, email string) (*sql.Row, error) {\n"
        "\tquery := \"SELECT id FROM users WHERE email = '\" + email + \"'\"\n"
        "\treturn db.QueryRow(query), nil\n"
        "}\n",
        "package users\n\nimport \"database/sql\"\n\n"
        "func Find(db *sql.DB, email string) (*sql.Row, error) {\n"
        "\tquery := \"SELECT id FROM users WHERE email = ?\"\n"
        "\treturn db.QueryRow(query, email), nil\n"
        "}\n",
        "String concatenation parses the value as SQL. `$1` sends it as a bound parameter, which "
        "the driver escapes and the planner cannot reinterpret.",
        ("go", "sql", "injection"),
    ),
    SecuritySample(
        "go-goroutine-unbounded",
        "CWE-400",
        "integer_overflow",
        "go",
        "Bound concurrency and body size in request handlers",
        "package api\n\nimport \"net/http\"\n\n"
        "func upload(w http.ResponseWriter, r *http.Request) {\n"
        "\tbody := make([]byte, r.ContentLength)\n"
        "\tr.Body.Read(body)\n"
        "\tw.Write(body)\n"
        "}\n",
        "package api\n\nimport (\n\t\"io\"\n\t\"net/http\"\n)\n\n"
        "const maxUploadBytes = 8388608\n\n"
        "func upload(w http.ResponseWriter, r *http.Request) {\n"
        "\tif r.ContentLength > maxUploadBytes {\n"
        "\t\thttp.Error(w, \"payload too large\", http.StatusRequestEntityTooLarge)\n"
        "\t\treturn\n"
        "\t}\n"
        "\tlimited := io.LimitReader(r.Body, maxUploadBytes)\n"
        "\tif _, err := io.Copy(w, limited); err != nil {\n"
        "\t\thttp.Error(w, \"read failed\", http.StatusBadRequest)\n"
        "\t}\n"
        "}\n",
        "Allocating from the client-supplied ContentLength is a one-request OOM, especially when "
        "the header lies. A MAX limit plus io.LimitReader bounds memory regardless of the header.",
        ("go", "limits", "dos"),
    ),
    SecuritySample(
        "go-command-injection",
        "CWE-78",
        "injection",
        "go",
        "Pass argv slices to exec.Command, never a shell string",
        "package jobs\n\nimport (\n\t\"os/exec\"\n)\n\n"
        "func Convert(path string) (string, error) {\n"
        "\tcmd := exec.Command(\"sh\", \"-c\", \"convert \"+path+\" /var/tmp/out.png\")\n"
        "\tout, err := cmd.Output()\n"
        "\treturn string(out), err\n"
        "}\n",
        "package jobs\n\nimport (\n\t\"os/exec\"\n)\n\n"
        "func Convert(path string) (string, error) {\n"
        "\tcmd := exec.Command(\"convert\", path, \"/var/tmp/out.png\")\n"
        "\tout, err := cmd.Output()\n"
        "\treturn string(out), err\n"
        "}\n",
        "`sh -c` re-parses the interpolated path, so a space plus `;` becomes a second command. "
        "Invoking the binary directly passes the path as one opaque argument.",
        ("go", "command", "injection"),
    ),
    # ------------------------------------------------------------- Rust
    SecuritySample(
        "rust-unsafe-transmute",
        "CWE-704",
        "integer_overflow",
        "rust",
        "Avoid transmute for length/type reinterpreting",
        "use std::mem;\n\n"
        "pub fn to_u64(bytes: &[u8; 4]) -> u64 {\n"
        "    unsafe { mem::transmute::<[u8; 4], u64>(*bytes) }\n"
        "}\n",
        "pub fn to_u64(bytes: &[u8; 4]) -> u64 {\n"
        "    u32::from_le_bytes(*bytes) as u64\n"
        "}\n",
        "transmute here reads past the source value's size, which is undefined behaviour. An "
        "explicit checked conversion is safe and states the endianness.",
        ("rust", "unsafe", "transmute"),
    ),
    SecuritySample(
        "rust-unwrap-request",
        "CWE-209",
        "logging",
        "rust",
        "Return typed errors instead of unwrapping request data",
        "pub fn handler(query: &str) -> String {\n"
        "    let id: u64 = query.split('=').nth(1).unwrap().parse().unwrap();\n"
        "    format!(\"id={}\", id)\n"
        "}\n",
        "pub fn handler(query: &str) -> Result<String, String> {\n"
        "    let raw = query\n"
        "        .split('=')\n"
        "        .nth(1)\n"
        "        .ok_or_else(|| \"missing id parameter\".to_string())?;\n"
        "    let id: u64 = raw\n"
        "        .parse()\n"
        "        .map_err(|_| \"id must be a positive integer\".to_string())?;\n"
        "    Ok(format!(\"id={}\", id))\n"
        "}\n",
        "`unwrap()` on request data panics, which becomes a 500 with a backtrace hint and lets a "
        "caller probe the parser. Result-returning code maps each failure to a controlled message.",
        ("rust", "errors", "panic"),
    ),
    SecuritySample(
        "rust-path-traversal",
        "CWE-22",
        "file_access",
        "rust",
        "Canonicalise and confine paths in Rust",
        "use std::fs;\n\n"
        "pub fn read_upload(name: &str) -> std::io::Result<String> {\n"
        "    fs::read_to_string(format!(\"/srv/uploads/{}\", name))\n"
        "}\n",
        "use std::fs;\nuse std::path::Path;\n\n"
        "pub fn read_upload(name: &str) -> std::io::Result<String> {\n"
        "    let base = Path::new(\"/srv/uploads\").canonicalize()?;\n"
        "    let target = base.join(name).canonicalize()?;\n"
        "    if !target.starts_with(&base) {\n"
        "        return Err(std::io::Error::new(\n"
        "            std::io::ErrorKind::PermissionDenied,\n"
        "            \"path escapes uploads directory\",\n"
        "        ));\n"
        "    }\n"
        "    fs::read_to_string(target)\n"
        "}\n",
        "String formatting into a path lets `../` escape the uploads root. canonicalize() plus a "
        "starts_with check on the resolved base rejects traversal and symlink escapes.",
        ("rust", "path", "traversal"),
    ),
    SecuritySample(
        "rust-hardcoded-secret",
        "CWE-798",
        "secrets",
        "rust",
        "Read secrets from configuration, not literals",
        "pub const JWT_SECRET: &str = \"super-secret-signing-key-2024\";\n\n"
        "pub fn sign(claims: &str) -> String {\n"
        "    format!(\"{}.{}\", claims, JWT_SECRET)\n"
        "}\n",
        "pub fn sign(claims: &str) -> Result<String, String> {\n"
        "    let secret = std::env::var(\"JWT_SECRET\")\n"
        "        .map_err(|_| \"JWT_SECRET is not configured\".to_string())?;\n"
        "    if secret.len() < 32 {\n"
        "        return Err(\"JWT_SECRET must be at least 32 bytes\".to_string());\n"
        "    }\n"
        "    Ok(format!(\"{}.{}\", claims, secret))\n"
        "}\n",
        "A literal signing key ships inside the binary (strings(1) finds it) and cannot be "
        "rotated. Reading it from the environment keeps it out of the artifact and enforces a "
        "minimum length.",
        ("rust", "secrets", "environment"),
    ),
    # --------------------------------------------------------- TypeScript
    SecuritySample(
        "ts-unsafe-any-cast",
        "CWE-20",
        "injection",
        "typescript",
        "Validate request bodies instead of casting to any",
        "export function handle(req: Request) {\n"
        "  const body = req.body as any;\n"
        "  return applySettings(body);\n"
        "}\n",
        "interface Settings { displayName: string; locale: string; }\n\n"
        "export function handle(req: Request) {\n"
        "  const body = req.body;\n"
        "  if (typeof body?.displayName !== 'string' || typeof body?.locale !== 'string') {\n"
        "    throw new Error('invalid settings payload');\n"
        "  }\n"
        "  const settings: Settings = { displayName: body.displayName, locale: body.locale };\n"
        "  return applySettings(settings);\n"
        "}\n",
        "`as any` erases the type system at the exact boundary where input is untrusted. Runtime "
        "checks plus an explicit interface keep both the compiler and the handler honest.",
        ("typescript", "validation", "types"),
    ),
    SecuritySample(
        "ts-jwt-decode-only",
        "CWE-347",
        "authentication",
        "typescript",
        "Verify JWT signatures, do not just decode them",
        "import jwt from 'jsonwebtoken';\n\n"
        "export function currentUser(token: string) {\n"
        "  return jwt.decode(token);\n"
        "}\n",
        "import jwt from 'jsonwebtoken';\n\n"
        "const ALLOWED_ALGORITHMS: jwt.Algorithm[] = ['RS256'];\n\n"
        "export function currentUser(token: string, publicKey: string) {\n"
        "  return jwt.verify(token, publicKey, {\n"
        "    algorithms: ALLOWED_ALGORITHMS,\n"
        "    maxAge: '1h',\n"
        "  });\n"
        "}\n",
        "`jwt.decode` does no signature check, so any caller can mint claims. `jwt.verify` with a "
        "pinned algorithm list and maxAge validates authenticity and freshness.",
        ("typescript", "jwt", "verify"),
    ),
    SecuritySample(
        "ts-postmessage-wildcard",
        "CWE-345",
        "header_injection",
        "typescript",
        "Restrict postMessage to a known origin",
        "export function sendToken(frame: Window, token: string) {\n"
        "  frame.postMessage({ token }, '*');\n"
        "}\n",
        "const ALLOWED_ORIGINS = ['https://app.example.com'];\n\n"
        "export function sendToken(frame: Window, token: string) {\n"
        "  frame.postMessage({ token }, ALLOWED_ORIGINS[0]);\n"
        "}\n\n"
        "window.addEventListener('message', (event) => {\n"
        "  if (!ALLOWED_ORIGINS.includes(event.origin)) return;\n"
        "  handleMessage(event.data);\n"
        "});\n",
        "postMessage with '*' delivers the payload to whatever document currently occupies the "
        "frame, and listening without an origin check accepts anyone. An ALLOWED_ORIGINS list on "
        "both sides confines the channel.",
        ("typescript", "postmessage", "origin"),
    ),
    SecuritySample(
        "ts-prototype-pollution",
        "CWE-1321",
        "mass_assignment",
        "typescript",
        "Reject __proto__/constructor keys when merging",
        "export function merge(target: Record<string, unknown>, source: Record<string, unknown>) {\n"
        "  for (const key of Object.keys(source)) {\n"
        "    target[key] = source[key];\n"
        "  }\n"
        "  return target;\n"
        "}\n",
        "const ALLOWED_KEYS = new Set(['displayName', 'locale', 'timezone']);\n\n"
        "export function merge(target: Record<string, unknown>, source: Record<string, unknown>) {\n"
        "  for (const key of ALLOWED_KEYS) {\n"
        "    if (Object.prototype.hasOwnProperty.call(source, key)) {\n"
        "      Object.defineProperty(target, key, { value: source[key], writable: true, enumerable: true, configurable: true });\n"
        "    }\n"
        "  }\n"
        "  return target;\n"
        "}\n",
        "Copying arbitrary keys lets `__proto__` reach Object.prototype and affect every object in "
        "the process. Iterating ALLOWED_KEYS with an own-property check bounds the write surface.",
        ("typescript", "prototype", "merge"),
    ),
    # ------------------------------------------------------------ Python
    SecuritySample(
        "python-assert-authorization",
        "CWE-617",
        "authorization",
        "python",
        "Never enforce authorization with assert",
        "def delete_post(post, user) -> None:\n"
        "    assert post.owner_id == user.id, \"not the owner\"\n"
        "    post.delete()\n",
        "def delete_post(post, user) -> None:\n"
        "    if post.owner_id != user.id:\n"
        "        raise PermissionError(\"not the owner\")\n"
        "    post.delete()\n",
        "`python -O` strips assert statements, silently turning the ownership check into a no-op. "
        "An explicit raise survives optimisation and is a real control-flow branch.",
        ("python", "assert", "authorization"),
    ),
    SecuritySample(
        "python-csv-formula-injection",
        "CWE-1236",
        "injection",
        "python",
        "Escape spreadsheet formula prefixes in exported CSV",
        "import csv\n\n\ndef export(path: str, rows: list[list[str]]) -> None:\n"
        "    with open(path, \"w\", newline=\"\", encoding=\"utf-8\") as handle:\n"
        "        writer = csv.writer(handle)\n"
        "        writer.writerows(rows)\n",
        "import csv\n\n_DANGEROUS_PREFIXES = (\"=\", \"+\", \"-\", \"@\")\n\n\n"
        "def export(path: str, rows: list[list[str]]) -> None:\n"
        "    with open(path, \"w\", newline=\"\", encoding=\"utf-8\") as handle:\n"
        "        writer = csv.writer(handle)\n"
        "        for row in rows:\n"
        "            writer.writerow(\n"
        "                [\"'\" + cell if cell.startswith(_DANGEROUS_PREFIXES) else cell for cell in row]\n"
        "            )\n",
        "A cell beginning with `=`, `+`, `-` or `@` is evaluated as a formula when opened in Excel, "
        "which is code execution on the victim's machine. Prefixing with an apostrophe forces text.",
        ("python", "csv", "formula"),
    ),
    SecuritySample(
        "python-tar-extract",
        "CWE-22",
        "file_access",
        "python",
        "Validate tar members in the data filter before extracting",
        "import tarfile\n\n\ndef unpack(archive: str, dest: str) -> None:\n"
        "    with tarfile.open(archive) as bundle:\n"
        "        bundle.extractall(dest)\n",
        "import shutil\nimport tarfile\nfrom pathlib import Path\n\n\n"
        "def unpack(archive: str, dest: str) -> None:\n"
        "    root = Path(dest).resolve()\n"
        "    with tarfile.open(archive) as bundle:\n"
        "        for member in bundle.getmembers():\n"
        "            target = (root / member.name).resolve()\n"
        "            if not target.is_relative_to(root) or member.issym() or member.islnk():\n"
        "                raise PermissionError(f\"unsafe archive member: {member.name}\")\n"
        "            if member.isfile():\n"
        "                target.parent.mkdir(parents=True, exist_ok=True)\n"
        "                with bundle.extractfile(member) as src, target.open(\"wb\") as dst:\n"
        "                    shutil.copyfileobj(src, dst)\n",
        "A tar member can be `../outside` or a symlink pointing anywhere on disk. Checking every "
        "resolved name (and rejecting links) before extracting confines writes to the destination.",
        ("python", "tar", "traversal"),
    ),
    SecuritySample(
        "python-request-timeout",
        "CWE-400",
        "integer_overflow",
        "python",
        "Set explicit timeouts on outbound requests",
        "import requests\n\n\ndef status(url: str) -> int:\n"
        "    return requests.get(url).status_code\n",
        "import requests\n\n\ndef status(url: str) -> int:\n"
        "    response = requests.get(url, timeout=(3.05, 10))\n"
        "    return response.status_code\n",
        "requests has no default timeout, so one slow upstream pins a worker thread forever and a "
        "handful of them exhausts the pool. An explicit (connect, read) timeout bounds it.",
        ("python", "timeout", "availability"),
    ),
    SecuritySample(
        "python-insecure-cookie",
        "CWE-1004",
        "session",
        "python",
        "Mark session cookies HttpOnly, Secure and SameSite",
        "def set_session_cookie(response, session_id: str) -> None:\n"
        "    response.set_cookie(\"session\", session_id)\n",
        "def set_session_cookie(response, session_id: str) -> None:\n"
        "    response.set_cookie(\n"
        "        \"session\",\n"
        "        session_id,\n"
        "        httponly=True,\n"
        "        secure=True,\n"
        "        samesite=\"Lax\",\n"
        "        max_age=3600,\n"
        "    )\n",
        "A cookie without HttpOnly is readable by any injected script; without Secure it travels "
        "over plaintext; without SameSite it is sent on cross-site requests. Setting all three "
        "closes the common session-theft paths.",
        ("python", "cookies", "session"),
    ),
]
