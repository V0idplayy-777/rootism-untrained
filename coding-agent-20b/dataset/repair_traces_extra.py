"""Repair traces, part 2: execution-grounded, multi-language (Corpus v1).

Each entry is ``(id, language, task, buggy, fixed, failure, diagnosis, rejected)``:

* ``failure`` is the real shape a runner prints (``pytest``/``jest``/``go test``/
  ``cargo test``/``mvn test``/``psql``), including the assertion diff,
* ``rejected`` is a plausible patch that still fails — the DPO negative,
* ``fixed`` is the minimal change that makes the failing test green.

The train/held-out split is computed from a hash of the bug id (see
``RepairTraceGenerator.split``), so adding entries never re-shuffles existing
assignments.
"""

from __future__ import annotations

#: (id, language, task, buggy, fixed, failure, diagnosis, rejected)
EXTRA_CURATED: list[tuple[str, str, str, str, str, str, str, str]] = [
    # ------------------------------------------------------------- JavaScript
    (
        "js-async-foreach",
        "javascript",
        "`sumPrices` must resolve to the total of all items fetched asynchronously.",
        "async function sumPrices(ids, fetchPrice) {\n"
        "  let total = 0;\n"
        "  ids.forEach(async (id) => {\n"
        "    total += await fetchPrice(id);\n"
        "  });\n"
        "  return total;\n"
        "}\n",
        "async function sumPrices(ids, fetchPrice) {\n"
        "  const prices = await Promise.all(ids.map((id) => fetchPrice(id)));\n"
        "  return prices.reduce((sum, price) => sum + price, 0);\n"
        "}\n",
        "FAIL src/pricing.test.js\n"
        "  ● sumPrices › totals every price\n"
        "    expect(received).toBe(expected) // Object.is equality\n"
        "    Expected: 60\n"
        "    Received: 0\n",
        "forEach ignores the promise returned by an async callback, so the function returned "
        "before any await resolved and the total was still 0.",
        "async function sumPrices(ids, fetchPrice) {\n"
        "  let total = 0;\n"
        "  for (const id of ids) total += await fetchPrice(id);\n"
        "  return total;\n"
        "}\n",
    ),
    (
        "js-float-money",
        "javascript",
        "`addMoney` must sum currency amounts without drifting.",
        "function addMoney(a, b) {\n  return a + b;\n}\n",
        "function addMoney(a, b) {\n"
        "  return Math.round((a + b) * 100) / 100;\n"
        "}\n",
        "FAIL src/money.test.js\n"
        "  ● addMoney › adds two cents-safe amounts\n"
        "    expect(received).toBe(expected)\n"
        "    Expected: 0.3\n"
        "    Received: 0.30000000000000004\n",
        "IEEE-754 binary floats cannot represent 0.1 and 0.2 exactly, so the sum carries "
        "representation error. Rounding to cents after summation makes the result exact.",
        "function addMoney(a, b) {\n  return (a + b).toFixed(2) * 1.0;\n}\n",
    ),
    (
        "js-array-sort-numeric",
        "javascript",
        "`sortScores` must return scores in ascending numeric order.",
        "function sortScores(scores) {\n  return scores.sort();\n}\n",
        "function sortScores(scores) {\n  return [...scores].sort((a, b) => a - b);\n}\n",
        "FAIL src/scores.test.js\n"
        "  ● sortScores › orders numerically\n"
        "    expect(received).toEqual(expected)\n"
        "    Expected: [2, 10, 100]\n"
        "    Received: [10, 100, 2]\n",
        "The default comparator coerces to strings, so \"100\" sorts before \"2\". A numeric "
        "comparator fixes the order; copying first avoids mutating the caller's array.",
        "function sortScores(scores) {\n  return scores.sort((a, b) => a - b);\n}\n",
    ),
    (
        "js-json-parse-unguarded",
        "javascript",
        "`readConfig` must fall back to defaults when the payload is not valid JSON.",
        "function readConfig(raw) {\n  const parsed = JSON.parse(raw);\n  return { retries: parsed.retries };\n}\n",
        "function readConfig(raw) {\n"
        "  let parsed;\n"
        "  try {\n"
        "    parsed = JSON.parse(raw);\n"
        "  } catch {\n"
        "    return { retries: 3 };\n"
        "  }\n"
        "  return { retries: Number.isInteger(parsed.retries) ? parsed.retries : 3 };\n"
        "}\n",
        "FAIL src/config.test.js\n"
        "  ● readConfig › falls back on malformed input\n"
        "    SyntaxError: Unexpected token } in JSON at position 4\n",
        "JSON.parse throws on malformed input instead of returning null, so the fallback path "
        "was unreachable. Catching the parse error (and validating the field type) restores it.",
        "function readConfig(raw) {\n  const parsed = JSON.parse(raw || '{}');\n"
        "  return { retries: parsed.retries };\n}\n",
    ),
    (
        "js-event-listener-leak",
        "javascript",
        "`attach` must stop growing the listener list when called repeatedly.",
        "function attach(target, onChange) {\n"
        "  target.addEventListener('input', onChange);\n"
        "  return () => {};\n"
        "}\n",
        "function attach(target, onChange) {\n"
        "  target.addEventListener('input', onChange);\n"
        "  return () => target.removeEventListener('input', onChange);\n"
        "}\n",
        "FAIL src/attach.test.js\n"
        "  ● attach › detaches on dispose\n"
        "    expect(jest.fn()).toHaveBeenCalledTimes(expected)\n"
        "    Expected number of calls: 0\n"
        "    Received number of calls: 1\n",
        "The returned disposer was a no-op, so the listener stayed registered and the handler "
        "ran after dispose. Returning a real removeEventListener call fixes it.",
        "function attach(target, onChange) {\n  target.addEventListener('input', onChange);\n"
        "  return () => undefined;\n}\n",
    ),
    (
        "js-date-month-index",
        "javascript",
        "`monthLabel` must render the month the date actually falls in.",
        "const LABELS = ['Jan', 'Feb', 'Mar', 'Apr'];\n\n"
        "function monthLabel(date) {\n  return LABELS[date.getMonth()];\n}\n",
        "const LABELS = ['Jan', 'Feb', 'Mar', 'Apr'];\n\n"
        "function monthLabel(date) {\n  const index = date.getMonth();\n"
        "  return LABELS[index] ?? 'unknown';\n}\n",
        "FAIL src/format.test.js\n"
        "  ● monthLabel › labels April\n"
        "    expect(received).toBe(expected)\n"
        "    Expected: \"Apr\"\n"
        "    Received: undefined\n",
        "getMonth() is zero-based while the test data was built from a one-based month, so "
        "April (index 3) was read as May. Deriving the index once and guarding the lookup keeps "
        "the mapping explicit and never returns undefined.",
        "function monthLabel(date) {\n  return LABELS[date.getMonth() - 1];\n}\n",
    ),
    (
        "js-object-mutation",
        "javascript",
        "`withTimeout` must return a new options object and leave the input untouched.",
        "function withTimeout(options, ms) {\n  options.timeout = ms;\n  return options;\n}\n",
        "function withTimeout(options, ms) {\n  return { ...options, timeout: ms };\n}\n",
        "FAIL src/http.test.js\n"
        "  ● withTimeout › does not mutate the caller's object\n"
        "    expect(received).toEqual(expected)\n"
        "    Expected: {\"timeout\": 1000}\n"
        "    Received: {\"timeout\": 5000}\n",
        "The function wrote into the caller's object, so a shared default was permanently "
        "changed by the first call. Spreading into a new object removes the aliasing.",
        "function withTimeout(options, ms) {\n  options.timeout = ms;\n  return { ...options };\n}\n",
    ),
    # ------------------------------------------------------------- TypeScript
    (
        "ts-optional-chain-crash",
        "typescript",
        "`displayName` must not throw when the profile has no address.",
        "export function displayName(profile: Profile): string {\n"
        "  return profile.address.city.toUpperCase();\n"
        "}\n",
        "export function displayName(profile: Profile): string {\n"
        "  return profile.address?.city?.toUpperCase() ?? 'unknown';\n"
        "}\n",
        "FAIL src/profile.test.ts\n"
        "  ● displayName › handles a missing address\n"
        "    TypeError: Cannot read properties of undefined (reading 'city')\n",
        "`address` is optional in the schema but dereferenced as if it were required, so the "
        "call throws before any fallback runs. Optional chaining plus a nullish fallback makes "
        "the missing case a value, not an exception.",
        "export function displayName(profile: Profile): string {\n"
        "  if (!profile.address) return 'unknown';\n"
        "  return profile.address.city.toUpperCase();\n"
        "}\n",
    ),
    (
        "ts-narrowing-any",
        "typescript",
        "`parseCount` must accept the string form and reject non-numeric input.",
        "export function parseCount(value: any): number {\n"
        "  return value;\n"
        "}\n",
        "export function parseCount(value: unknown): number {\n"
        "  const parsed = typeof value === 'number' ? value : Number(value);\n"
        "  if (!Number.isInteger(parsed) || parsed < 0) {\n"
        "    throw new RangeError(`not a count: ${String(value)}`);\n"
        "  }\n"
        "  return parsed;\n"
        "}\n",
        "FAIL src/count.test.ts\n"
        "  ● parseCount › rejects a non-numeric string\n"
        "    expect(received).toThrow(expected)\n"
        "    Expected: RangeError\n"
        "    Received function did not throw\n",
        "`any` let the string flow straight through as a \"number\", so callers doing arithmetic "
        "got string concatenation. Narrowing from `unknown` and validating the integer range "
        "makes the failure a throw instead of silent corruption.",
        "export function parseCount(value: any): number {\n  return Number(value);\n}\n",
    ),
    (
        "ts-generic-unbounded",
        "typescript",
        "`chunk` must never return a chunk longer than the requested size.",
        "export function chunk<T>(items: T[], size: number): T[][] {\n"
        "  const out: T[][] = [];\n"
        "  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));\n"
        "  return out;\n"
        "}\n",
        "export function chunk<T>(items: T[], size: number): T[][] {\n"
        "  if (!Number.isInteger(size) || size <= 0) {\n"
        "    throw new RangeError('size must be a positive integer');\n"
        "  }\n"
        "  const out: T[][] = [];\n"
        "  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));\n"
        "  return out;\n"
        "}\n",
        "FAIL src/chunk.test.ts\n"
        "  ● chunk › rejects a zero size\n"
        "    RangeError: Invalid array length\n",
        "`i += 0` never advances, so the loop pushed the same slice until the array length "
        "exploded. Validating `size` up front turns an infinite loop into a clear error.",
        "export function chunk<T>(items: T[], size: number): T[][] {\n"
        "  const out: T[][] = [];\n"
        "  for (let i = 0; i < items.length; i += Math.max(1, size)) out.push(items.slice(i, i + size));\n"
        "  return out;\n"
        "}\n",
    ),
    (
        "ts-enum-exhaustive",
        "typescript",
        "`label` must return a string for every status variant.",
        "export function label(status: Status): string {\n"
        "  if (status === 'open') return 'Open';\n"
        "  if (status === 'closed') return 'Closed';\n"
        "}\n",
        "export function label(status: Status): string {\n"
        "  switch (status) {\n"
        "    case 'open':\n"
        "      return 'Open';\n"
        "    case 'closed':\n"
        "      return 'Closed';\n"
        "    case 'merged':\n"
        "      return 'Merged';\n"
        "    default: {\n"
        "      const never: never = status;\n"
        "      throw new Error(`unhandled status: ${String(never)}`);\n"
        "    }\n"
        "  }\n"
        "}\n",
        "FAIL src/status.test.ts\n"
        "  ● label › labels a merged request\n"
        "    expect(received).toBe(expected)\n"
        "    Expected: \"Merged\"\n"
        "    Received: undefined\n",
        "The union gained a 'merged' member but the if-chain silently fell through and returned "
        "undefined. An exhaustive switch with a `never` default makes the compiler flag the "
        "missing case and the runtime throw instead of returning undefined.",
        "export function label(status: Status): string {\n"
        "  return status === 'open' ? 'Open' : 'Closed';\n"
        "}\n",
    ),
    (
        "ts-promise-misuse",
        "typescript",
        "`loadAll` must surface a rejected load instead of resolving with 'undefined'.",
        "export async function loadAll(ids: string[], load: (id: string) => Promise<Item>) {\n"
        "  const results: Item[] = [];\n"
        "  ids.forEach(async (id) => {\n"
        "    results.push(await load(id));\n"
        "  });\n"
        "  return results;\n"
        "}\n",
        "export async function loadAll(ids: string[], load: (id: string) => Promise<Item>) {\n"
        "  return Promise.all(ids.map((id) => load(id)));\n"
        "}\n",
        "FAIL src/load.test.ts\n"
        "  ● loadAll › propagates a load failure\n"
        "    expect(received).rejects.toThrow()\n"
        "    Received promise resolved instead of rejected\n",
        "The async callback's rejection happened in a floating promise, so the outer function "
        "resolved with an empty array. Promise.all keeps the rejection on the returned chain.",
        "export async function loadAll(ids: string[], load: (id: string) => Promise<Item>) {\n"
        "  const results: Item[] = [];\n"
        "  for (const id of ids) results.push(await load(id));\n"
        "  return results;\n"
        "}\n",
    ),
    # --------------------------------------------------------------------- Go
    (
        "go-loop-variable-capture",
        "go",
        "`StartAll` must give each goroutine its own job.",
        "func StartAll(jobs []Job) {\n"
        "\tfor _, job := range jobs {\n"
        "\t\tgo func() {\n"
        "\t\t\tjob.Run()\n"
        "\t\t}()\n"
        "\t}\n"
        "}\n",
        "func StartAll(jobs []Job) {\n"
        "\tfor _, job := range jobs {\n"
        "\t\tjob := job\n"
        "\t\tgo func() {\n"
        "\t\t\tjob.Run()\n"
        "\t\t}()\n"
        "\t}\n"
        "}\n",
        "--- FAIL: TestStartAllRunsEachJob (0.00s)\n"
        "    scheduler_test.go:31: job c ran 3 times, job a and b never ran\n",
        "Before Go 1.22 the range variable is reused across iterations, so every closure "
        "captured the same `job`. Shadowing it inside the loop gives each goroutine its own copy.",
        "func StartAll(jobs []Job) {\n"
        "\tfor i := range jobs {\n"
        "\t\tgo func() {\n"
        "\t\t\tjobs[i].Run()\n"
        "\t\t}()\n"
        "\t}\n"
        "}\n",
    ),
    (
        "go-nil-map-write",
        "go",
        "`Add` must record labels on a fresh Store without panicking.",
        "type Store struct {\n\tlabels map[string]string\n}\n\n"
        "func (s *Store) Add(key, value string) {\n\ts.labels[key] = value\n}\n",
        "type Store struct {\n\tlabels map[string]string\n}\n\n"
        "func NewStore() *Store {\n\treturn &Store{labels: make(map[string]string)}\n}\n\n"
        "func (s *Store) Add(key, value string) {\n"
        "\tif s.labels == nil {\n"
        "\t\ts.labels = make(map[string]string)\n"
        "\t}\n"
        "\ts.labels[key] = value\n"
        "}\n",
        "--- FAIL: TestStoreAdd (0.00s)\n"
        "panic: assignment to entry in nil map [recovered]\n"
        "\tstore_test.go:18 +0x1d\n",
        "A zero-value struct has a nil map and writing to it panics. Lazily initialising the "
        "map (and providing a constructor) makes the zero value usable.",
        "func (s *Store) Add(key, value string) {\n\ts.labels = map[string]string{key: value}\n}\n",
    ),
    (
        "go-sql-rows-unclosed",
        "go",
        "`Count` must return every row and leave no connection checked out.",
        "func Count(db *sql.DB) (int, error) {\n"
        "\trows, err := db.Query(\"SELECT id FROM jobs\")\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\tn := 0\n"
        "\tfor rows.Next() {\n"
        "\t\tn++\n"
        "\t}\n"
        "\treturn n, nil\n"
        "}\n",
        "func Count(db *sql.DB) (int, error) {\n"
        "\trows, err := db.Query(\"SELECT id FROM jobs\")\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\tdefer rows.Close()\n"
        "\tn := 0\n"
        "\tfor rows.Next() {\n"
        "\t\tn++\n"
        "\t}\n"
        "\treturn n, rows.Err()\n"
        "}\n",
        "--- FAIL: TestCountDoesNotLeakConnections (1.20s)\n"
        "    db_test.go:44: timed out waiting for a pooled connection\n",
        "Rows holds its connection until Close, so the pool drained after a few calls. "
        "`defer rows.Close()` returns it, and `rows.Err()` surfaces a mid-iteration failure.",
        "func Count(db *sql.DB) (int, error) {\n"
        "\trows, err := db.Query(\"SELECT id FROM jobs\")\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\tn := 0\n"
        "\tfor rows.Next() {\n"
        "\t\tn++\n"
        "\t}\n"
        "\trows.Close()\n"
        "\treturn n, nil\n"
        "}\n",
    ),
    (
        "go-error-shadowed",
        "go",
        "`Load` must report a scan failure instead of returning a zero value.",
        "func Load(db *sql.DB, id int) (string, error) {\n"
        "\tvar name string\n"
        "\terr := db.QueryRow(\"SELECT name FROM jobs WHERE id=$1\", id).Scan(&name)\n"
        "\tif err == sql.ErrNoRows {\n"
        "\t\treturn \"\", fmt.Errorf(\"job %d: %w\", id, err)\n"
        "\t}\n"
        "\treturn name, nil\n"
        "}\n",
        "func Load(db *sql.DB, id int) (string, error) {\n"
        "\tvar name string\n"
        "\terr := db.QueryRow(\"SELECT name FROM jobs WHERE id=$1\", id).Scan(&name)\n"
        "\tif err != nil {\n"
        "\t\treturn \"\", fmt.Errorf(\"job %d: %w\", id, err)\n"
        "\t}\n"
        "\treturn name, nil\n"
        "}\n",
        "--- FAIL: TestLoadPropagatesError (0.00s)\n"
        "    jobs_test.go:52: got error <nil>, want a wrapped error\n",
        "Only ErrNoRows was handled; every other error (bad SQL, closed connection) fell "
        "through as success. Checking `err != nil` covers all of them.",
        "func Load(db *sql.DB, id int) (string, error) {\n"
        "\tvar name string\n"
        "\terr := db.QueryRow(\"SELECT name FROM jobs WHERE id=$1\", id).Scan(&name)\n"
        "\tif err == sql.ErrNoRows {\n"
        "\t\treturn \"\", fmt.Errorf(\"job %d: %w\", id, err)\n"
        "\t}\n"
        "\tif err != nil {\n"
        "\t\treturn name, err\n"
        "\t}\n"
        "\treturn name, nil\n"
        "}\n",
    ),
    (
        "go-context-ignored",
        "go",
        "`Fetch` must give up when the caller's context is cancelled.",
        "func Fetch(ctx context.Context, client *http.Client, url string) ([]byte, error) {\n"
        "\tresp, err := client.Get(url)\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tdefer resp.Body.Close()\n"
        "\treturn io.ReadAll(resp.Body)\n"
        "}\n",
        "func Fetch(ctx context.Context, client *http.Client, url string) ([]byte, error) {\n"
        "\treq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tresp, err := client.Do(req)\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tdefer resp.Body.Close()\n"
        "\treturn io.ReadAll(resp.Body)\n"
        "}\n",
        "--- FAIL: TestFetchHonoursCancellation (2.00s)\n"
        "    fetch_test.go:61: Fetch returned <nil> after context was cancelled\n",
        "`client.Get` builds a request with a background context, so cancellation never reached "
        "the transport. `NewRequestWithContext` threads the caller's deadline through.",
        "func Fetch(ctx context.Context, client *http.Client, url string) ([]byte, error) {\n"
        "\tselect {\n"
        "\tcase <-ctx.Done():\n"
        "\t\treturn nil, ctx.Err()\n"
        "\tdefault:\n"
        "\t}\n"
        "\tresp, err := client.Get(url)\n"
        "\tif err != nil {\n"
        "\t\treturn nil, err\n"
        "\t}\n"
        "\tdefer resp.Body.Close()\n"
        "\treturn io.ReadAll(resp.Body)\n"
        "}\n",
    ),
    (
        "go-json-tag-missing",
        "go",
        "`Encode` must emit the documented snake_case field names.",
        "type Job struct {\n\tID       int    `json:\"id\"`\n"
        "\tRetryMax int\n}\n",
        "type Job struct {\n\tID       int    `json:\"id\"`\n"
        "\tRetryMax int    `json:\"retry_max\"`\n}\n",
        "--- FAIL: TestEncodeUsesSnakeCase (0.00s)\n"
        "    encode_test.go:27: got {\"id\":1,\"RetryMax\":3}, want {\"id\":1,\"retry_max\":3}\n",
        "Without a json tag, encoding/json falls back to the Go field name, which breaks the "
        "published API contract. Adding the tag pins the wire format.",
        "type Job struct {\n\tID       int    `json:\"id\"`\n"
        "\tRetryMax int    `json:\"retryMax\"`\n}\n",
    ),
    # ------------------------------------------------------------------- Rust
    (
        "rust-index-out-of-bounds",
        "rust",
        "`first_word` must return None for an empty string instead of panicking.",
        "pub fn first_word(text: &str) -> Option<&str> {\n"
        "    let end = text.find(' ').unwrap();\n"
        "    Some(&text[..end])\n"
        "}\n",
        "pub fn first_word(text: &str) -> Option<&str> {\n"
        "    let end = text.find(' ').unwrap_or(text.len());\n"
        "    Some(&text[..end])\n"
        "}\n",
        "---- words::first_word_handles_single_word stdout ----\n"
        "thread 'main' panicked at src/words.rs:2:32:\n"
        "called `Option::unwrap()` on a `None` value\n",
        "`find` returns None when there is no space, and `unwrap` turned that ordinary case "
        "into a panic. Defaulting to the full length returns the whole string.",
        "pub fn first_word(text: &str) -> Option<&str> {\n"
        "    Some(text.split(' ').next().unwrap())\n"
        "}\n",
    ),
    (
        "rust-mutex-deadlock",
        "rust",
        "`bump` must not deadlock when it reads and writes the same counter.",
        "pub fn bump(state: &Mutex<Counter>) {\n"
        "    let mut guard = state.lock().unwrap();\n"
        "    let current = state.lock().unwrap().value;\n"
        "    guard.value = current + 1;\n"
        "}\n",
        "pub fn bump(state: &Mutex<Counter>) {\n"
        "    let mut guard = state.lock().unwrap();\n"
        "    let current = guard.value;\n"
        "    guard.value = current + 1;\n"
        "}\n",
        "---- state::bump_increments stdout ----\n"
        "test state::bump_increments has been running for over 60 seconds\n"
        "thread 'state::bump_increments' panicked at 'lock attempt would block'\n",
        "Re-locking a std Mutex on the same thread is not reentrant, so it blocks forever. "
        "Reading through the guard already held removes the second lock.",
        "pub fn bump(state: &Mutex<Counter>) {\n"
        "    let current = state.lock().unwrap().value;\n"
        "    state.lock().unwrap().value = current + 1;\n"
        "}\n",
    ),
    (
        "rust-clone-vs-move",
        "rust",
        "`labels` must include the caller's tag without consuming it.",
        "pub fn labels(base: Vec<String>, tag: String) -> Vec<String> {\n"
        "    let mut out = base;\n"
        "    out.push(tag);\n"
        "    out\n"
        "}\n",
        "pub fn labels(mut base: Vec<String>, tag: String) -> Vec<String> {\n"
        "    base.push(tag);\n"
        "    base\n"
        "}\n",
        "---- labels::caller_keeps_base stdout ----\n"
        "error[E0382]: borrow of moved value: `base`\n"
        "  --> tests/labels.rs:12:20\n",
        "The function took ownership of the caller's Vec, so the test could not read it "
        "afterwards. Taking `mut base` by value but documenting the move (or borrowing) matches "
        "the caller's expectation; the test wants the base back unchanged, so accept `&mut`.",
        "pub fn labels(base: Vec<String>, tag: String) -> Vec<String> {\n"
        "    let mut out = base.clone();\n"
        "    out.push(tag.clone());\n"
        "    out\n"
        "}\n",
    ),
    (
        "rust-utf8-slice",
        "rust",
        "`truncate_label` must not panic on multi-byte characters.",
        "pub fn truncate_label(label: &str, max: usize) -> String {\n"
        "    label[..max].to_string()\n"
        "}\n",
        "pub fn truncate_label(label: &str, max: usize) -> String {\n"
        "    let mut end = max.min(label.len());\n"
        "    while end > 0 && !label.is_char_boundary(end) {\n"
        "        end -= 1;\n"
        "    }\n"
        "    label[..end].to_string()\n"
        "}\n",
        "---- text::truncate_multibyte stdout ----\n"
        "thread 'main' panicked at src/text.rs:2:11:\n"
        "byte index 5 is not a char boundary; it is inside 'é' (bytes 4..6)\n",
        "Slicing a &str by byte offset panics when the offset lands inside a multi-byte "
        "character. Walking back to the nearest char boundary makes the cut safe.",
        "pub fn truncate_label(label: &str, max: usize) -> String {\n"
        "    label.chars().take(max).collect()\n"
        "}\n",
    ),
    (
        "rust-error-swallowed",
        "rust",
        "`read_port` must return an error instead of a default on bad input.",
        "pub fn read_port(raw: &str) -> Result<u16, String> {\n"
        "    Ok(raw.parse::<u16>().unwrap_or(8080))\n"
        "}\n",
        "pub fn read_port(raw: &str) -> Result<u16, String> {\n"
        "    raw.parse::<u16>()\n"
        "        .map_err(|err| format!(\"invalid port {raw:?}: {err}\"))\n"
        "}\n",
        "---- config::rejects_bad_port stdout ----\n"
        "assertion `left == right` failed\n"
        "  left: Ok(8080)\n"
        " right: Err(\"invalid port \\\"abc\\\"\")\n",
        "`unwrap_or` turned a configuration error into a silently wrong default, which hides "
        "typos in production. Propagating the parse error keeps the failure visible.",
        "pub fn read_port(raw: &str) -> Result<u16, String> {\n"
        "    let port = raw.parse::<u16>().map_err(|err| err.to_string())?;\n"
        "    Ok(port)\n"
        "}\n",
    ),
    (
        "rust-iterator-consumed",
        "rust",
        "`total` must sum the iterator and still let the caller reuse it.",
        "pub fn total(items: impl Iterator<Item = i64>) -> i64 {\n"
        "    items.sum()\n"
        "}\n",
        "pub fn total(items: &mut impl Iterator<Item = i64>) -> i64 {\n"
        "    items.sum()\n"
        "}\n",
        "---- stats::reuses_iterator stdout ----\n"
        "error[E0382]: use of moved value: `values`\n"
        "  --> tests/stats.rs:9:18\n",
        "`sum` consumes the iterator, so the caller's binding was moved and unusable "
        "afterwards. Taking `&mut impl Iterator` borrows it for the duration of the call.",
        "pub fn total(items: impl Iterator<Item = i64>) -> i64 {\n"
        "    items.collect::<Vec<_>>().iter().sum()\n"
        "}\n",
    ),
    # ------------------------------------------------------------------- Java
    (
        "java-integer-equality",
        "java",
        "`isSameUser` must compare ids by value for ids above the Integer cache.",
        "public boolean isSameUser(Integer a, Integer b) {\n"
        "    return a == b;\n"
        "}\n",
        "public boolean isSameUser(Integer a, Integer b) {\n"
        "    return Objects.equals(a, b);\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] UserTest.isSameUserForLargeIds:24 expected:<true> but was:<false>\n",
        "`==` on boxed Integers compares references; values above the -128..127 cache are "
        "distinct objects. Objects.equals compares the wrapped values.",
        "public boolean isSameUser(Integer a, Integer b) {\n    return a.intValue() == b.intValue();\n}\n",
    ),
    (
        "java-string-split-regex",
        "java",
        "`parsePath` must split on a literal dot, not a regex wildcard.",
        "public String[] parsePath(String path) {\n    return path.split(\".\");\n}\n",
        "public String[] parsePath(String path) {\n    return path.split(\"\\\\.\");\n}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] PathTest.parsePath:18 expected:<3> but was:<0>\n",
        "split takes a regular expression, and `.` matches any character, so every character "
        "was a delimiter and the result was empty. Escaping it splits on literal dots.",
        "public String[] parsePath(String path) {\n    return path.split(\"\\\\.\", -1);\n}\n",
    ),
    (
        "java-stream-reuse",
        "java",
        "`summary` must not reuse an already-consumed stream.",
        "public String summary(List<String> items) {\n"
        "    Stream<String> stream = items.stream().filter(s -> !s.isBlank());\n"
        "    long count = stream.count();\n"
        "    return count + \":\" + stream.findFirst().orElse(\"\");\n"
        "}\n",
        "public String summary(List<String> items) {\n"
        "    List<String> present = items.stream().filter(s -> !s.isBlank()).toList();\n"
        "    return present.size() + \":\" + present.stream().findFirst().orElse(\"\");\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] SummaryTest.summary:31 java.lang.IllegalStateException: stream has already been operated upon or closed\n",
        "A Stream is single-use; the terminal count() closed it, so the second consumer threw. "
        "Materialising the filtered list once lets both readers work off data.",
        "public String summary(List<String> items) {\n"
        "    long count = items.stream().filter(s -> !s.isBlank()).count();\n"
        "    return count + \":\" + items.stream().findFirst().orElse(\"\");\n"
        "}\n",
    ),
    (
        "java-file-encoding",
        "java",
        "`read` must preserve non-ASCII characters from a UTF-8 file.",
        "public String read(Path path) throws IOException {\n"
        "    return new String(Files.readAllBytes(path));\n"
        "}\n",
        "public String read(Path path) throws IOException {\n"
        "    return Files.readString(path, StandardCharsets.UTF_8);\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] ReadTest.readsUtf8:22 expected:<café> but was:<cafÃ©>\n",
        "`new String(bytes)` decodes with the platform default charset, which is not UTF-8 on "
        "every CI image. Naming the charset explicitly makes the result deterministic.",
        "public String read(Path path) throws IOException {\n"
        "    return new String(Files.readAllBytes(path), StandardCharsets.ISO_8859_1);\n"
        "}\n",
    ),
    (
        "java-null-autounbox",
        "java",
        "`total` must treat a null amount as zero rather than throwing.",
        "public int total(List<Integer> amounts) {\n"
        "    int sum = 0;\n"
        "    for (Integer amount : amounts) {\n"
        "        sum += amount;\n"
        "    }\n"
        "    return sum;\n"
        "}\n",
        "public int total(List<Integer> amounts) {\n"
        "    int sum = 0;\n"
        "    for (Integer amount : amounts) {\n"
        "        if (amount != null) {\n"
        "            sum += amount;\n"
        "        }\n"
        "    }\n"
        "    return sum;\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] TotalTest.nullAmountIsZero:19 java.lang.NullPointerException\n"
        "\tat Total.total(Total.java:5)\n",
        "Auto-unboxing a null Integer throws NPE at the assignment. Guarding the null keeps "
        "the column's nullable contract from crashing the aggregate.",
        "public int total(List<Integer> amounts) {\n"
        "    return amounts.stream().mapToInt(Integer::intValue).sum();\n"
        "}\n",
    ),
    (
        "java-transaction-missing-commit",
        "java",
        "`save` must commit the row so a later connection can read it.",
        "public void save(Connection conn, String name) throws SQLException {\n"
        "    try (PreparedStatement ps = conn.prepareStatement(\"INSERT INTO jobs(name) VALUES (?)\")) {\n"
        "        ps.setString(1, name);\n"
        "        ps.executeUpdate();\n"
        "    }\n"
        "}\n",
        "public void save(Connection conn, String name) throws SQLException {\n"
        "    try (PreparedStatement ps = conn.prepareStatement(\"INSERT INTO jobs(name) VALUES (?)\")) {\n"
        "        ps.setString(1, name);\n"
        "        ps.executeUpdate();\n"
        "    }\n"
        "    conn.commit();\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] JobRepositoryTest.savedRowIsVisible:41 expected:<1> but was:<0>\n",
        "The connection runs with autocommit disabled, so the insert stayed in an uncommitted "
        "transaction and the reader's snapshot did not see it. Committing ends the unit of work.",
        "public void save(Connection conn, String name) throws SQLException {\n"
        "    try (PreparedStatement ps = conn.prepareStatement(\"INSERT INTO jobs(name) VALUES (?)\")) {\n"
        "        ps.setString(1, name);\n"
        "        ps.executeUpdate();\n"
        "    }\n"
        "    conn.setAutoCommit(true);\n"
        "}\n",
    ),
    # -------------------------------------------------------------------- SQL
    (
        "sql-left-join-filter",
        "sql",
        "`active_members` must keep members with no orders, counting them as 0.",
        "SELECT m.id, COUNT(o.id) AS orders\n"
        "FROM members m\n"
        "LEFT JOIN orders o ON o.member_id = m.id\n"
        "WHERE o.status = 'paid'\n"
        "GROUP BY m.id;\n",
        "SELECT m.id, COUNT(o.id) AS orders\n"
        "FROM members m\n"
        "LEFT JOIN orders o ON o.member_id = m.id AND o.status = 'paid'\n"
        "GROUP BY m.id;\n",
        "ERROR: 1 of 3 rows returned\n"
        "  expected rows: (1,0) (2,3) (3,0)\n"
        "  actual rows:   (2,3)\n",
        "The WHERE predicate ran after the join, so rows with a NULL order were filtered out — "
        "turning the outer join into an inner one. Moving the predicate into the ON clause keeps "
        "members without qualifying orders.",
        "SELECT m.id, COUNT(o.id) AS orders\n"
        "FROM members m\n"
        "LEFT JOIN orders o ON o.member_id = m.id\n"
        "GROUP BY m.id\n"
        "HAVING COUNT(o.id) >= 0;\n",
    ),
    (
        "sql-null-comparison",
        "sql",
        "`unassigned_tickets` must include tickets whose assignee is NULL.",
        "SELECT id FROM tickets WHERE assignee != 42;\n",
        "SELECT id FROM tickets\n"
        "WHERE assignee IS NULL OR assignee <> 42;\n",
        "ERROR: 1 of 4 rows returned\n"
        "  expected rows: 1, 3, 4\n"
        "  actual rows:   3\n",
        "`NULL != 42` evaluates to UNKNOWN, never TRUE, so unassigned tickets disappeared. "
        "Testing IS NULL explicitly restores them.",
        "SELECT id FROM tickets WHERE COALESCE(assignee, -1) != 42;\n",
    ),
    (
        "sql-implicit-group-by",
        "sql",
        "`latest_status` must return one row per order with its newest event.",
        "SELECT order_id, status\n"
        "FROM events\n"
        "GROUP BY order_id;\n",
        "SELECT order_id, status\n"
        "FROM events e\n"
        "WHERE occurred_at = (\n"
        "    SELECT MAX(occurred_at) FROM events x WHERE x.order_id = e.order_id\n"
        ");\n",
        "ERROR: column \"events.status\" must appear in the GROUP BY clause\n"
        "  or be used in an aggregate function\n",
        "Selecting a bare column next to GROUP BY makes the engine pick an arbitrary row from "
        "each group in permissive engines, or reject it in strict ones. A correlated MAX lookup "
        "selects the row the report actually means.",
        "SELECT order_id, MIN(status) AS status FROM events GROUP BY order_id;\n",
    ),
    (
        "sql-offset-pagination",
        "sql",
        "`next_page` must not skip or duplicate rows under concurrent inserts.",
        "SELECT id, created_at\n"
        "FROM posts\n"
        "ORDER BY created_at DESC\n"
        "LIMIT 20 OFFSET 40;\n",
        "SELECT id, created_at\n"
        "FROM posts\n"
        "WHERE (created_at, id) < (:last_created_at, :last_id)\n"
        "ORDER BY created_at DESC, id DESC\n"
        "LIMIT 20;\n",
        "ERROR: page 3 returned rows also returned on page 2\n"
        "  overlapping ids: 41, 42, 43\n",
        "OFFSET counts rows in the current ordering, so an insert above the cursor shifts every "
        "row down and the next page repeats them. Keyset pagination anchors on the last row seen.",
        "SELECT id, created_at\n"
        "FROM posts\n"
        "ORDER BY created_at DESC, id DESC\n"
        "LIMIT 20 OFFSET 20;\n",
    ),
    (
        "sql-avg-integer",
        "sql",
        "`avg_latency` must report fractional milliseconds.",
        "SELECT AVG(latency_ms) AS avg_latency FROM requests;\n",
        "SELECT AVG(latency_ms::numeric) AS avg_latency FROM requests;\n",
        "ERROR: 1 of 1 rows returned\n"
        "  expected: 2.5\n"
        "  actual:   2\n",
        "AVG over an integer column truncates to integer in PostgreSQL. Casting to numeric keeps "
        "the fractional part the metric is about.",
        "SELECT CAST(AVG(latency_ms) AS INTEGER) AS avg_latency FROM requests;\n",
    ),
    # --------------------------------------------------------------- mixed/other
    (
        "js-dto-extra-field",
        "javascript",
        "`toPublicUser` must never expose the password hash.",
        "function toPublicUser(row) {\n  return row;\n}\n",
        "function toPublicUser(row) {\n"
        "  const { password_hash, ...rest } = row;\n"
        "  return rest;\n"
        "}\n",
        "FAIL src/user.test.js\n"
        "  ● toPublicUser › omits the password hash\n"
        "    expect(received).not.toHaveProperty('password_hash')\n",
        "Returning the row directly forwards every column the SELECT happened to include, so a "
        "future `SELECT *` would leak credentials. Destructuring out the secret makes the "
        "response shape explicit.",
        "function toPublicUser(row) {\n  return { ...row, password_hash: undefined };\n}\n",
    ),
    (
        "go-slice-alias",
        "go",
        "`Append` must not overwrite the caller's backing array.",
        "func Append(items []int, value int) []int {\n"
        "\treturn append(items[:len(items)], value)\n"
        "}\n",
        "func Append(items []int, value int) []int {\n"
        "\tout := make([]int, len(items), len(items)+1)\n"
        "\tcopy(out, items)\n"
        "\treturn append(out, value)\n"
        "}\n",
        "--- FAIL: TestAppendDoesNotMutateInput (0.00s)\n"
        "    slice_test.go:19: original slice became [1 2 9], want [1 2]\n",
        "Slicing to len(items) keeps the original capacity, so append wrote the new value into "
        "the caller's backing array. Copying into a fresh slice removes the aliasing.",
        "func Append(items []int, value int) []int {\n"
        "\treturn append(append([]int{}, items...), value)\n"
        "}\n",
    ),
    (
        "rust-option-phantom",
        "rust",
        "`parse_id` must reject a missing id instead of inventing one.",
        "pub fn parse_id(raw: Option<&str>) -> u32 {\n"
        "    raw.and_then(|value| value.parse().ok()).unwrap_or(0)\n"
        "}\n",
        "pub fn parse_id(raw: Option<&str>) -> Result<u32, String> {\n"
        "    let value = raw.ok_or_else(|| \"missing id\".to_string())?;\n"
        "    value\n"
        "        .parse::<u32>()\n"
        "        .map_err(|err| format!(\"invalid id {value:?}: {err}\"))\n"
        "}\n",
        "---- params::rejects_missing_id stdout ----\n"
        "assertion `left == right` failed\n"
        "  left: Ok(0)\n"
        " right: Err(\"missing id\")\n",
        "`unwrap_or(0)` silently manufactured a valid-looking primary key for missing input, "
        "which then queried row 0. Returning Result keeps the absence visible at the call site.",
        "pub fn parse_id(raw: Option<&str>) -> u32 {\n"
        "    raw.and_then(|value| value.parse().ok()).unwrap_or(1)\n"
        "}\n",
    ),
    (
        "java-mutable-static",
        "java",
        "`register` must not accumulate handlers across test cases.",
        "public final class Registry {\n"
        "    public static final List<Handler> HANDLERS = new ArrayList<>();\n\n"
        "    public static void register(Handler handler) {\n"
        "        HANDLERS.add(handler);\n"
        "    }\n"
        "}\n",
        "public final class Registry {\n"
        "    private static final List<Handler> HANDLERS = new ArrayList<>();\n\n"
        "    public static void register(Handler handler) {\n"
        "        HANDLERS.add(handler);\n"
        "    }\n\n"
        "    public static void reset() {\n"
        "        HANDLERS.clear();\n"
        "    }\n\n"
        "    public static List<Handler> handlers() {\n"
        "        return List.copyOf(HANDLERS);\n"
        "    }\n"
        "}\n",
        "Tests run: 2, Failures: 1\n"
        "[ERROR] RegistryTest.secondTestSeesOneHandler:31 expected:<1> but was:<2>\n",
        "A static mutable list is shared by every test in the JVM, so state leaked between "
        "cases. Exposing a copy and an explicit reset makes the test isolation possible.",
        "public final class Registry {\n"
        "    public static final List<Handler> HANDLERS = new CopyOnWriteArrayList<>();\n\n"
        "    public static void register(Handler handler) {\n"
        "        HANDLERS.add(handler);\n"
        "    }\n"
        "}\n",
    ),
    (
        "js-promise-all-settled",
        "javascript",
        "`bestEffort` must return every successful result when some loads fail.",
        "async function bestEffort(ids, load) {\n"
        "  return Promise.all(ids.map((id) => load(id)));\n"
        "}\n",
        "async function bestEffort(ids, load) {\n"
        "  const settled = await Promise.allSettled(ids.map((id) => load(id)));\n"
        "  return settled\n"
        "    .filter((result) => result.status === 'fulfilled')\n"
        "    .map((result) => result.value);\n"
        "}\n",
        "FAIL src/batch.test.js\n"
        "  ● bestEffort › returns partial results\n"
        "    expect(received).rejects.toThrow()\n"
        "    Received promise rejected instead of resolved\n",
        "Promise.all fails fast on the first rejection, discarding the results that already "
        "succeeded. allSettled lets the caller keep the partial batch.",
        "async function bestEffort(ids, load) {\n"
        "  return Promise.all(ids.map((id) => load(id).catch(() => undefined)));\n"
        "}\n",
    ),
    (
        "ts-mapped-readonly",
        "typescript",
        "`freeze` must make nested config objects immutable too.",
        "export function freeze<T extends object>(value: T): T {\n"
        "  return Object.freeze(value);\n"
        "}\n",
        "export function freeze<T extends object>(value: T): T {\n"
        "  const clone = structuredClone(value);\n"
        "  const deepFreeze = (node: object): void => {\n"
        "    Object.freeze(node);\n"
        "    for (const key of Object.keys(node)) {\n"
        "      const child = (node as Record<string, unknown>)[key];\n"
        "      if (child && typeof child === 'object' && !Object.isFrozen(child)) {\n"
        "        deepFreeze(child as object);\n"
        "      }\n"
        "    }\n"
        "  };\n"
        "  deepFreeze(clone);\n"
        "  return clone;\n"
        "}\n",
        "FAIL src/freeze.test.ts\n"
        "  ● freeze › protects nested objects\n"
        "    TypeError: Cannot assign to read only property 'timeout' of object '#<Object>'\n"
        "    (expected the write to be rejected at the nested level, not only the root)\n",
        "Object.freeze is shallow, so the nested settings object was still writable. Recursing "
        "over own keys (and cloning first so the caller's object is untouched) makes it deep.",
        "export function freeze<T extends object>(value: T): T {\n"
        "  return Object.freeze({ ...value });\n}\n",
    ),
    (
        "go-time-format",
        "go",
        "`FormatDay` must emit an ISO-8601 date.",
        "func FormatDay(t time.Time) string {\n\treturn t.Format(\"YYYY-MM-DD\")\n}\n",
        "func FormatDay(t time.Time) string {\n\treturn t.Format(\"2006-01-02\")\n}\n",
        "--- FAIL: TestFormatDay (0.00s)\n"
        "    format_test.go:14: got \"YYYY-11-DD\", want \"2024-11-03\"\n",
        "Go's layout string is a reference time, not a pattern language, so only the literal "
        "digits that match 2006-01-02 are substituted. Using the reference layout formats correctly.",
        "func FormatDay(t time.Time) string {\n\treturn t.Format(\"2006-1-2\")\n}\n",
    ),
    (
        "rust-lock-order",
        "rust",
        "`transfer` must not deadlock when two threads move value in opposite directions.",
        "pub fn transfer(from: &Mutex<u64>, to: &Mutex<u64>, amount: u64) {\n"
        "    let mut a = from.lock().unwrap();\n"
        "    let mut b = to.lock().unwrap();\n"
        "    *a -= amount;\n"
        "    *b += amount;\n"
        "}\n",
        "pub fn transfer(low: &Mutex<u64>, high: &Mutex<u64>, amount: u64) {\n"
        "    // Total order by address avoids the lock-order inversion.\n"
        "    let (first, second) = if (low as *const _) < (high as *const _) {\n"
        "        (low, high)\n"
        "    } else {\n"
        "        (high, low)\n"
        "    };\n"
        "    let mut a = first.lock().unwrap();\n"
        "    let mut b = second.lock().unwrap();\n"
        "    if std::ptr::eq(first, low) && !std::ptr::eq(low, high) {\n"
        "        *a -= amount;\n"
        "        *b += amount;\n"
        "    } else if !std::ptr::eq(low, high) {\n"
        "        *b -= amount;\n"
        "        *a += amount;\n"
        "    }\n"
        "}\n",
        "---- bank::transfer_is_deadlock_free stdout ----\n"
        "test bank::transfer_is_deadlock_free has been running for over 60 seconds\n",
        "Two threads acquiring the same pair in opposite order deadlock. Establishing one global "
        "lock order (here by address) means every thread takes them the same way.",
        "pub fn transfer(a: &Mutex<u64>, b: &Mutex<u64>, amount: u64) {\n"
        "    let mut first = a.lock().unwrap();\n"
        "    let mut second = b.lock().unwrap();\n"
        "    *first -= amount;\n"
        "    *second += amount;\n"
        "}\n",
    ),
    (
        "java-string-builder-loop",
        "java",
        "`join` must build the result in linear time for large inputs.",
        "public String join(List<String> parts) {\n"
        "    String out = \"\";\n"
        "    for (String part : parts) {\n"
        "        out += part + \",\";\n"
        "    }\n"
        "    return out;\n"
        "}\n",
        "public String join(List<String> parts) {\n"
        "    return String.join(\",\", parts);\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] JoinTest.joinLargeInput:37 expected completion under 2000ms but took 8421ms\n",
        "String is immutable, so every `+=` copies the whole prefix — quadratic in the number of "
        "parts. String.join (or a StringBuilder) appends in place.",
        "public String join(List<String> parts) {\n"
        "    String out = \"\";\n"
        "    for (String part : parts) {\n"
        "        out = out.concat(part).concat(\",\");\n"
        "    }\n"
        "    return out;\n"
        "}\n",
    ),
    (
        "js-timeout-not-cleared",
        "javascript",
        "`withDeadline` must clear its timer so the process can exit.",
        "function withDeadline(work, ms) {\n"
        "  return new Promise((resolve, reject) => {\n"
        "    setTimeout(() => reject(new Error('deadline exceeded')), ms);\n"
        "    work().then(resolve, reject);\n"
        "  });\n"
        "}\n",
        "function withDeadline(work, ms) {\n"
        "  return new Promise((resolve, reject) => {\n"
        "    const timer = setTimeout(() => reject(new Error('deadline exceeded')), ms);\n"
        "    work().then(\n"
        "      (value) => {\n"
        "        clearTimeout(timer);\n"
        "        resolve(value);\n"
        "      },\n"
        "      (err) => {\n"
        "        clearTimeout(timer);\n"
        "        reject(err);\n"
        "      },\n"
        "    );\n"
        "  });\n"
        "}\n",
        "FAIL src/deadline.test.js\n"
        "  ● withDeadline › exits promptly after success\n"
        "    A worker process has failed to exit gracefully\n",
        "The pending timer keeps the event loop alive after the work settles, so jest hangs and "
        "the timer also fires a spurious rejection. Clearing it on both paths ends the race.",
        "function withDeadline(work, ms) {\n"
        "  return Promise.race([\n"
        "    work(),\n"
        "    new Promise((_, reject) => setTimeout(() => reject(new Error('deadline exceeded')), ms)),\n"
        "  ]);\n"
        "}\n",
    ),
    (
        "sql-union-dedup",
        "sql",
        "`all_events` must keep duplicate rows from the two source tables.",
        "SELECT id, kind FROM app_events\n"
        "UNION\n"
        "SELECT id, kind FROM audit_events;\n",
        "SELECT id, kind FROM app_events\n"
        "UNION ALL\n"
        "SELECT id, kind FROM audit_events;\n",
        "ERROR: 1 of 1 rows returned\n"
        "  expected: 4 rows (2 app + 2 audit, one id present in both)\n"
        "  actual:   3 rows\n",
        "UNION applies an implicit DISTINCT over the combined result, so the row that legitimately "
        "appears in both sources was collapsed. UNION ALL preserves every source row.",
        "SELECT DISTINCT id, kind FROM app_events\n"
        "UNION\n"
        "SELECT id, kind FROM audit_events;\n",
    ),
    (
        "ts-json-parse-date",
        "typescript",
        "`parseEvent` must return a Date, not the JSON string.",
        "export function parseEvent(raw: string): Event {\n"
        "  return JSON.parse(raw);\n"
        "}\n",
        "export function parseEvent(raw: unknown): Event {\n"
        "  const payload = JSON.parse(String(raw)) as { id?: unknown; at?: unknown };\n"
        "  if (typeof payload.id !== 'string' || typeof payload.at !== 'string') {\n"
        "    throw new TypeError('event requires id and at');\n"
        "  }\n"
        "  const at = new Date(payload.at);\n"
        "  if (Number.isNaN(at.getTime())) {\n"
        "    throw new RangeError(`invalid timestamp: ${payload.at}`);\n"
        "  }\n"
        "  return { id: payload.id, at };\n"
        "}\n",
        "FAIL src/event.test.ts\n"
        "  ● parseEvent › returns a Date instance\n"
        "    expect(received).toBeInstanceOf(Date)\n"
        "    Received: \"2024-11-03T00:00:00Z\"\n",
        "JSON has no date type, so the timestamp stayed a string and every later comparison "
        "silently compared text. Parsing into a Date (and rejecting unparseable input) restores "
        "the contract.",
        "export function parseEvent(raw: string): Event {\n"
        "  return JSON.parse(raw) as Event;\n}\n",
    ),
    (
        "go-http-body-close",
        "go",
        "`GetStatus` must close the response body.",
        "func GetStatus(client *http.Client, url string) (int, error) {\n"
        "\tresp, err := client.Get(url)\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\treturn resp.StatusCode, nil\n"
        "}\n",
        "func GetStatus(client *http.Client, url string) (int, error) {\n"
        "\tresp, err := client.Get(url)\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\tdefer resp.Body.Close()\n"
        "\t_, _ = io.Copy(io.Discard, resp.Body)\n"
        "\treturn resp.StatusCode, nil\n"
        "}\n",
        "--- FAIL: TestGetStatusReusesConnections (3.01s)\n"
        "    client_test.go:38: only 1 of 5 connections were reused\n",
        "An unread, unclosed body prevents the transport from returning the connection to the "
        "pool, so every call dials a new socket. Draining and closing it enables keep-alive.",
        "func GetStatus(client *http.Client, url string) (int, error) {\n"
        "\tresp, err := client.Get(url)\n"
        "\tif err != nil {\n"
        "\t\treturn 0, err\n"
        "\t}\n"
        "\tresp.Body.Close()\n"
        "\treturn resp.StatusCode, nil\n"
        "}\n",
    ),
    (
        "rust-rc-cycle",
        "rust",
        "`Detach` must let the node be dropped instead of leaking it.",
        "pub struct Node {\n    pub parent: Rc<RefCell<Option<Rc<Node>>>>,\n"
        "    pub child: Rc<RefCell<Option<Rc<Node>>>>,\n"
        "}\n\n"
        "pub fn detach(node: &Rc<Node>) {\n"
        "    *node.child.borrow_mut() = None;\n"
        "}\n",
        "pub struct Node {\n    pub parent: Weak<RefCell<Option<Rc<Node>>>>,\n"
        "    pub child: Rc<RefCell<Option<Rc<Node>>>>,\n"
        "}\n\n"
        "pub fn detach(node: &Rc<Node>) {\n"
        "    if let Some(child) = node.child.borrow_mut().take() {\n"
        "        *child.parent.upgrade().map(|p| p).unwrap_or_else(|| Rc::new(RefCell::new(None))).borrow_mut() = None;\n"
        "    }\n"
        "}\n",
        "---- tree::detach_drops_node stdout ----\n"
        "assertion `left == right` failed\n"
        "  left: 2\n"
        " right: 0\n"
        "  (Rc strong count never reached zero after detach)\n",
        "Two Rc fields pointing at each other form a reference cycle, so neither node's strong "
        "count ever reaches zero and both leak. The parent link must be a Weak.",
        "pub struct Node {\n    pub parent: Option<Rc<RefCell<Node>>>,\n"
        "    pub child: Option<Rc<RefCell<Node>>>,\n"
        "}\n\n"
        "pub fn detach(node: &mut Node) {\n"
        "    node.child = None;\n}\n",
    ),
    (
        "java-equals-hashcode",
        "java",
        "`Set` must de-duplicate users by id.",
        "public final class User {\n"
        "    private final int id;\n\n"
        "    public User(int id) {\n"
        "        this.id = id;\n"
        "    }\n\n"
        "    @Override\n"
        "    public boolean equals(Object other) {\n"
        "        return other instanceof User && ((User) other).id == id;\n"
        "    }\n"
        "}\n",
        "public final class User {\n"
        "    private final int id;\n\n"
        "    public User(int id) {\n"
        "        this.id = id;\n"
        "    }\n\n"
        "    @Override\n"
        "    public boolean equals(Object other) {\n"
        "        return other instanceof User && ((User) other).id == id;\n"
        "    }\n\n"
        "    @Override\n"
        "    public int hashCode() {\n"
        "        return Integer.hashCode(id);\n"
        "    }\n"
        "}\n",
        "Tests run: 1, Failures: 1\n"
        "[ERROR] UserSetTest.deduplicatesById:22 expected:<1> but was:<2>\n",
        "HashSet hashes the object first and only compares within a bucket, so an overridden "
        "equals without hashCode sends equal objects to different buckets. Implementing hashCode "
        "from the same field makes them collide and compare.",
        "public final class User {\n"
        "    private final int id;\n\n"
        "    public User(int id) {\n"
        "        this.id = id;\n"
        "    }\n"
        "}\n",
    ),
]
