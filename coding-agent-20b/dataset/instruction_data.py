"""
Instruction-Following Dataset Generator for Coding Agent.

Generates diverse instruction-response pairs for training:
- Code generation instructions
- Bug fixing instructions
- Code review instructions
- Code explanation instructions
- Refactoring instructions
- Test generation instructions
- Multi-turn coding conversations
- Tool use / function calling examples
"""



import random

import json

from typing import List, Dict, Any, Optional, Tuple

from dataclasses import dataclass





@dataclass

class InstructionSample:

    """A single instruction-following training example."""

    instruction: str

    input_context: str

    output: str

    category: str

    language: Optional[str] = None

    difficulty: str = "intermediate"





class InstructionGenerator:

    """Generates instruction-following training samples."""



    def __init__(self):

        self.generators = [

            self._generate_code_gen_instructions,

            self._generate_bug_fix_instructions,

            self._generate_code_review_instructions,

            self._generate_explanation_instructions,

            self._generate_refactor_instructions,

            self._generate_test_gen_instructions,

            self._generate_multiturn_conversations,

            self._generate_tool_use_instructions,

            self._generate_algorithm_instructions,

            self._generate_debug_instructions,

        ]



    def generate_batch(self, count: int) -> List[InstructionSample]:

        """Generate a batch of instruction samples."""

        samples = []

        for _ in range(count):

            gen = random.choice(self.generators)

            batch = gen()

            samples.extend(batch)

        return samples[:count]



    def _generate_code_gen_instructions(self) -> List[InstructionSample]:

        """Generate code generation instruction samples."""

        templates = [

            {

                "instruction": "Write a function that finds all prime numbers up to n using the Sieve of Eratosthenes algorithm.",

                "input_context": "",

                "output": '''def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    Find all prime numbers up to n using the Sieve of Eratosthenes.
    
    Time Complexity: O(n log log n)
    Space Complexity: O(n)
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]''',

                "category": "code_generation",

                "language": "python",

            },

            {

                "instruction": "Implement a thread-safe singleton pattern in Python.",

                "input_context": "",

                "output": '''import threading
from typing import Optional

class SingletonMeta(type):
    """Thread-safe singleton metaclass."""
    
    _instances: Dict[type, Any] = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class DatabaseConnection(metaclass=SingletonMeta):
    """Example singleton class."""
    
    def __init__(self):
        self.connected = False
        self._connection = None
    
    def connect(self, connection_string: str):
        self._connection = connection_string
        self.connected = True
    
    def query(self, sql: str):
        if not self.connected:
            raise RuntimeError("Not connected")
        return f"Executing: {sql}"


# Usage
db1 = DatabaseConnection()
db2 = DatabaseConnection()
assert db1 is db2  # Same instance''',

                "category": "code_generation",

                "language": "python",

            },

            {

                "instruction": "Create a Python decorator that caches function results with TTL (time-to-live) expiration.",

                "input_context": "",

                "output": '''import time
import functools
import hashlib
import pickle
from typing import Any, Callable, Optional

def ttl_cache(ttl_seconds: float = 300, max_size: int = 1000):
    """
    Decorator that caches function results with TTL expiration.
    
    Args:
        ttl_seconds: How long to cache results (in seconds)
        max_size: Maximum number of cached results
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Tuple[float, Any]] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_data = (args, tuple(sorted(kwargs.items())))
            key = hashlib.md5(pickle.dumps(key_data)).hexdigest()
            
            # Check cache
            if key in cache:
                timestamp, result = cache[key]
                if time.time() - timestamp < ttl_seconds:
                    return result
                else:
                    del cache[key]
            
            # Compute and cache
            result = func(*args, **kwargs)
            
            # Evict oldest if at capacity
            if len(cache) >= max_size:
                oldest_key = min(cache, key=lambda k: cache[k][0])
                del cache[oldest_key]
            
            cache[key] = (time.time(), result)
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "max_size": max_size,
            "ttl": ttl_seconds,
        }
        
        return wrapper
    return decorator


# Usage
@ttl_cache(ttl_seconds=60, max_size=100)
def expensive_computation(x: int, y: int) -> int:
    """A function that takes time to compute."""
    time.sleep(1)  # Simulate expensive operation
    return x ** y

# First call: computes (slow)
result1 = expensive_computation(2, 10)  # Returns 1024

# Second call: cached (fast)
result2 = expensive_computation(2, 10)  # Returns 1024 instantly''',

                "category": "code_generation",

                "language": "python",

            },

            {

                "instruction": "Write a JavaScript function that debounces another function with configurable delay and immediate option.",

                "input_context": "",

                "output": '''/**
 * Creates a debounced version of the provided function.
 * 
 * @param {Function} func - The function to debounce
 * @param {number} wait - Milliseconds to delay
 * @param {boolean} immediate - If true, trigger on leading edge
 * @returns {Function} Debounced function with cancel method
 */
function debounce(func, wait = 300, immediate = false) {
    let timeoutId = null;
    let lastArgs = null;
    let lastThis = null;
    let result;
    
    function debounced(...args) {
        lastArgs = args;
        lastThis = this;
        
        const callNow = immediate && !timeoutId;
        
        clearTimeout(timeoutId);
        
        timeoutId = setTimeout(() => {
            timeoutId = null;
            
            if (!immediate) {
                result = func.apply(lastThis, lastArgs);
            }
            
            lastArgs = null;
            lastThis = null;
        }, wait);
        
        if (callNow) {
            result = func.apply(lastThis, lastArgs);
        }
        
        return result;
    }
    
    debounced.cancel = function() {
        clearTimeout(timeoutId);
        timeoutId = null;
        lastArgs = null;
        lastThis = null;
    };
    
    debounced.flush = function() {
        if (timeoutId) {
            clearTimeout(timeoutId);
            timeoutId = null;
            result = func.apply(lastThis, lastArgs);
        }
        return result;
    };
    
    debounced.pending = function() {
        return timeoutId !== null;
    };
    
    return debounced;
}

// Usage example
const handleSearch = debounce((query) => {
    console.log("Searching for:", query);
    fetch(`/api/search?q=${encodeURIComponent(query)}`);
}, 500);

// Attach to input
document.getElementById("search").addEventListener("input", (e) => {
    handleSearch(e.target.value);
});''',

                "category": "code_generation",

                "language": "javascript",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_bug_fix_instructions(self) -> List[InstructionSample]:

        """Generate bug fixing instruction samples."""

        templates = [

            {

                "instruction": "The following code has a bug where the linked list reverse function doesn't work correctly. Find and fix the bug.",

                "input_context": '''def reverse_linked_list(head):
    """Reverse a singly linked list."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        # Bug: forgot to advance current
    return prev''',

                "output": '''The bug is that `current` is never advanced to `next_node` inside the loop,
causing an infinite loop. Here's the fix:

```python
def reverse_linked_list(head):
    """Reverse a singly linked list."""
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node  # <-- FIX: advance current pointer
    return prev
```

**Root cause:** After reassigning `current.next = prev` and `prev = current`,
we must move `current` to the saved `next_node` to continue traversing.
Without this, `current` stays the same and the loop never terminates.

**Time Complexity:** O(n) - single pass through the list
**Space Complexity:** O(1) - only three pointers used''',

                "category": "bug_fixing",

                "language": "python",

            },

            {

                "instruction": "This Python function is supposed to find the maximum subarray sum (Kadane's algorithm) but returns wrong results for all-negative arrays. Fix it.",

                "input_context": '''def max_subarray_sum(arr):
    """Find maximum contiguous subarray sum."""
    max_sum = 0
    current_sum = 0
    for num in arr:
        current_sum += num
        if current_sum < 0:
            current_sum = 0
        if current_sum > max_sum:
            max_sum = current_sum
    return max_sum''',

                "output": '''The bug is that initializing `max_sum = 0` causes all-negative arrays to
return 0 instead of the least negative element. Here's the fix:

```python
def max_subarray_sum(arr):
    """Find maximum contiguous subarray sum (Kadane's algorithm)."""
    if not arr:
        return 0
    
    max_sum = arr[0]       # FIX: initialize to first element, not 0
    current_sum = arr[0]   # FIX: start from first element
    
    for num in arr[1:]:    # FIX: skip first element since we used it
        current_sum = max(num, current_sum + num)  # FIX: restart or extend
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**What changed:**
1. `max_sum` initialized to `arr[0]` instead of `0` — handles all-negative case
2. `current_sum` starts at `arr[0]` — properly seeds the running sum  
3. `current_sum = max(num, current_sum + num)` — decides whether to start a
   new subarray at `num` or extend the existing one
4. Loop starts from `arr[1:]` since the first element is already accounted for

**Test cases:**
- `[1, -2, 3, 4, -1]` → `7` (subarray [3, 4])
- `[-3, -2, -5, -1]` → `-1` (subarray [-1]) ← previously returned 0
- `[5, -1, 5]` → `9` (subarray [5, -1, 5])''',

                "category": "bug_fixing",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_code_review_instructions(self) -> List[InstructionSample]:

        """Generate code review instruction samples."""

        templates = [

            {

                "instruction": "Review this code for performance issues, security vulnerabilities, and best practices. Provide specific recommendations.",

                "input_context": '''from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name LIKE '%{query}%'")
    results = cursor.fetchall()
    return str(results)

@app.route('/user/<id>')
def get_user(id):
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = " + id)
    return str(cursor.fetchone())''',

                "output": '''## Code Review

### 🔴 Critical: SQL Injection Vulnerability

Both endpoints are vulnerable to SQL injection attacks. The user input is
directly concatenated into SQL queries.

**Fix:** Use parameterized queries:

```python
# BEFORE (vulnerable):
cursor.execute(f"SELECT * FROM users WHERE name LIKE '%{query}%'")

# AFTER (safe):
cursor.execute("SELECT * FROM users WHERE name LIKE ?", (f"%{query}%",))
```

### 🔴 Critical: Resource Leak

Database connections are never closed, causing connection pool exhaustion.

**Fix:** Use context managers:

```python
@app.route('/search')
def search():
    query = request.args.get('q', '')
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name LIKE ?", (f"%{query}%",))
        results = cursor.fetchall()
    return jsonify(results)
```

### 🟡 Performance: N+1 Query Risk

Creating a new database connection per request is expensive.

**Fix:** Use connection pooling:

```python
import sqlite3
from contextlib import contextmanager

# Create pool at startup
_pool = sqlite3.connect('app.db', check_same_thread=False)

@contextmanager
def get_db():
    yield _pool
```

### 🟡 Security: Information Disclosure

Returning raw `str(results)` exposes the database schema.

**Fix:** Return structured JSON with only necessary fields:

```python
from flask import jsonify

@app.route('/search')
def search():
    # ... (safe query)
    users = [{"id": r[0], "name": r[1]} for r in results]
    return jsonify({"users": users, "count": len(users)})
```

### 🟡 Best Practice: Input Validation

No validation on the `q` parameter.

```python
query = request.args.get('q', '').strip()
if len(query) > 100:
    return jsonify({"error": "Query too long"}), 400
if len(query) < 2:
    return jsonify({"error": "Query too short"}), 400
```

### Summary
| Issue | Severity | Category |
|-------|----------|----------|
| SQL Injection | Critical | Security |
| Unclosed connections | Critical | Resources |
| No connection pooling | Medium | Performance |
| Raw data exposure | Medium | Security |
| No input validation | Medium | Best Practice |''',

                "category": "code_review",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_explanation_instructions(self) -> List[InstructionSample]:

        """Generate code explanation samples."""

        templates = [

            {

                "instruction": "Explain how Python's GIL (Global Interpreter Lock) works and its implications for multi-threaded programs.",

                "input_context": "",

                "output": '''# Python's Global Interpreter Lock (GIL)

## What is the GIL?

The GIL is a mutex (mutual exclusion lock) in CPython that allows only one
thread to execute Python bytecode at a time, even on multi-core systems.

## How it works

```
Thread A ──→ [Acquires GIL] ──→ Executes bytecode ──→ [Releases GIL]
                                                        ↓
Thread B ──→ [Waits for GIL] ←──────────────────────────┘
                ↓
            [Acquires GIL] ──→ Executes bytecode ──→ [Releases GIL]
```

The GIL is released and reacquired:
- Every N bytecode instructions (check interval, ~100 by default)
- During I/O operations (file reads, network calls)
- When calling C extensions that release it explicitly

## Implications

### ✅ Where threading still works:
- **I/O-bound tasks:** Network requests, file operations, database queries
  The GIL is released during I/O, so threads can run concurrently.

```python
import threading
import requests

def fetch(url):
    return requests.get(url).status_code

# This IS faster with threads (I/O-bound)
urls = ["https://example.com"] * 10
threads = [threading.Thread(target=fetch, args=(u,)) for u in urls]
```

### ❌ Where threading doesn't help:
- **CPU-bound tasks:** Math, data processing, image manipulation
  Threads run sequentially due to GIL — no parallelism.

```python
# This is NOT faster with threads (CPU-bound)
def compute():
    return sum(i * i for i in range(10_000_000))

# Use multiprocessing instead
import multiprocessing
with multiprocessing.Pool(4) as pool:
    results = pool.map(compute, range(4))
```

## Python 3.13+ (No-GIL)

PEP 703 introduced an experimental free-threaded build where the GIL can
be disabled, enabling true multi-threaded parallelism. This is still
experimental and may change C extension compatibility.

## Rule of Thumb
| Task Type | Use | Why |
|-----------|-----|-----|
| I/O-bound | `threading` or `asyncio` | GIL released during I/O |
| CPU-bound | `multiprocessing` | Separate processes, separate GILs |
| Mixed | `concurrent.futures` | Choose executor based on task |''',

                "category": "explanation",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_refactor_instructions(self) -> List[InstructionSample]:

        templates = [

            {

                "instruction": "Refactor this code to follow SOLID principles and improve maintainability. The current code mixes concerns and has tight coupling.",

                "input_context": '''class OrderProcessor:
    def process(self, order):
        # Validate
        if not order.get("items"):
            return {"error": "No items"}
        for item in order["items"]:
            if item["price"] < 0:
                return {"error": "Invalid price"}
        
        # Calculate total
        total = sum(item["price"] * item["quantity"] for item in order["items"])
        tax = total * 0.1
        discount = total * order.get("discount_rate", 0)
        final = total + tax - discount
        
        # Save to database
        import sqlite3
        conn = sqlite3.connect("orders.db")
        conn.execute("INSERT INTO orders (total, tax, status) VALUES (?, ?, ?)",
                     (final, tax, "processed"))
        conn.commit()
        conn.close()
        
        # Send email
        import smtplib
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login("store@example.com", "password123")
        server.sendmail("store@example.com", order["email"],
                       f"Order confirmed! Total: ${final:.2f}")
        server.quit()
        
        return {"status": "success", "total": final}''',

                "output": '''"""
Refactored Order Processor following SOLID principles:
- S: Each class has a single responsibility
- O: Open for extension via interfaces
- L: Substitutable implementations
- D: Depend on abstractions, not concretions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol
from decimal import Decimal


# ===== Domain Models =====

@dataclass
class OrderItem:
    name: str
    price: Decimal
    quantity: int
    
    @property
    def subtotal(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class Order:
    items: List[OrderItem]
    email: str
    discount_rate: Decimal = Decimal("0")
    
    @property
    def subtotal(self) -> Decimal:
        return sum(item.subtotal for item in self.items)


@dataclass
class OrderResult:
    total: Decimal
    tax: Decimal
    discount: Decimal
    final_total: Decimal


# ===== Interfaces (Dependency Inversion) =====

class OrderValidator(Protocol):
    def validate(self, order: Order) -> Optional[str]: ...

class PriceCalculator(Protocol):
    def calculate(self, order: Order) -> OrderResult: ...

class OrderRepository(Protocol):
    def save(self, order: Order, result: OrderResult) -> None: ...

class NotificationService(Protocol):
    def send_confirmation(self, email: str, result: OrderResult) -> None: ...


# ===== Implementations =====

class StandardOrderValidator:
    """Validates order data (Single Responsibility)."""
    
    def validate(self, order: Order) -> Optional[str]:
        if not order.items:
            return "Order must contain at least one item"
        
        for item in order.items:
            if item.price < 0:
                return f"Invalid price for item: {item.name}"
            if item.quantity < 1:
                return f"Invalid quantity for item: {item.name}"
        
        if not order.email or "@" not in order.email:
            return "Invalid email address"
        
        return None


class TaxInclusiveCalculator:
    """Calculates order totals with tax and discounts."""
    
    def __init__(self, tax_rate: Decimal = Decimal("0.10")):
        self.tax_rate = tax_rate
    
    def calculate(self, order: Order) -> OrderResult:
        subtotal = order.subtotal
        tax = subtotal * self.tax_rate
        discount = subtotal * order.discount_rate
        final = subtotal + tax - discount
        
        return OrderResult(
            total=subtotal,
            tax=tax,
            discount=discount,
            final_total=final,
        )


class SqliteOrderRepository:
    """Persists orders to SQLite database."""
    
    def __init__(self, db_path: str = "orders.db"):
        self.db_path = db_path
    
    def save(self, order: Order, result: OrderResult) -> None:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO orders (total, tax, discount, status) VALUES (?, ?, ?, ?)",
                (float(result.final_total), float(result.tax),
                 float(result.discount), "processed"),
            )


class EmailNotificationService:
    """Sends order confirmation emails."""
    
    def __init__(self, smtp_host: str, smtp_port: int):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
    
    def send_confirmation(self, email: str, result: OrderResult) -> None:
        # In production: use proper email library with templates
        pass  # Email sending logic


# ===== Orchestrator (Dependency Injection) =====

class OrderProcessor:
    """
    Orchestrates order processing using injected dependencies.
    Each dependency can be swapped for testing or different environments.
    """
    
    def __init__(
        self,
        validator: OrderValidator,
        calculator: PriceCalculator,
        repository: OrderRepository,
        notifier: NotificationService,
    ):
        self.validator = validator
        self.calculator = calculator
        self.repository = repository
        self.notifier = notifier
    
    def process(self, order: Order) -> dict:
        # Validate
        error = self.validator.validate(order)
        if error:
            return {"status": "error", "message": error}
        
        # Calculate
        result = self.calculator.calculate(order)
        
        # Persist
        self.repository.save(order, result)
        
        # Notify
        self.notifier.send_confirmation(order.email, result)
        
        return {"status": "success", "total": float(result.final_total)}


# ===== Factory for easy setup =====

def create_order_processor() -> OrderProcessor:
    """Create a fully configured OrderProcessor."""
    return OrderProcessor(
        validator=StandardOrderValidator(),
        calculator=TaxInclusiveCalculator(tax_rate=Decimal("0.10")),
        repository=SqliteOrderRepository(),
        notifier=EmailNotificationService("smtp.gmail.com", 587),
    )''',

                "category": "refactoring",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_test_gen_instructions(self) -> List[InstructionSample]:

        templates = [

            {

                "instruction": "Write comprehensive unit tests for this function, including edge cases, error cases, and boundary conditions.",

                "input_context": '''def parse_duration(duration_str: str) -> int:
    """
    Parse a duration string into total seconds.
    
    Supports formats: "2h30m", "45m", "90s", "1h", "1h30m15s"
    Returns total seconds as integer.
    Raises ValueError for invalid formats.
    """
    import re
    
    if not duration_str or not isinstance(duration_str, str):
        raise ValueError("Duration must be a non-empty string")
    
    pattern = r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?'
    match = re.fullmatch(pattern, duration_str.strip())
    
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid duration format: {duration_str}")
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds''',

                "output": '''import pytest
from parse_duration import parse_duration


class TestParseDuration:
    """Comprehensive tests for parse_duration function."""
    
    # === Happy Path Tests ===
    
    def test_hours_only(self):
        assert parse_duration("2h") == 7200
    
    def test_minutes_only(self):
        assert parse_duration("45m") == 2700
    
    def test_seconds_only(self):
        assert parse_duration("90s") == 90
    
    def test_hours_and_minutes(self):
        assert parse_duration("2h30m") == 9000
    
    def test_hours_minutes_seconds(self):
        assert parse_duration("1h30m15s") == 5415
    
    def test_all_components(self):
        assert parse_duration("23h59m59s") == 86399
    
    def test_zero_values(self):
        assert parse_duration("0h0m0s") == 0
    
    def test_large_values(self):
        assert parse_duration("100h") == 360000
    
    def test_whitespace_handling(self):
        assert parse_duration("  2h30m  ") == 9000
    
    # === Edge Cases ===
    
    def test_one_second(self):
        assert parse_duration("1s") == 1
    
    def test_one_minute(self):
        assert parse_duration("1m") == 60
    
    def test_one_hour(self):
        assert parse_duration("1h") == 3600
    
    def test_hours_and_seconds_no_minutes(self):
        assert parse_duration("1h10s") == 3610
    
    def test_minutes_and_seconds_no_hours(self):
        assert parse_duration("5m30s") == 330
    
    # === Error Cases ===
    
    def test_empty_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_duration("")
    
    def test_none_input(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_duration(None)
    
    def test_integer_input(self):
        with pytest.raises(ValueError, match="non-empty string"):
            parse_duration(123)
    
    def test_invalid_format_no_units(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("123")
    
    def test_invalid_format_text(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("two hours")
    
    def test_invalid_format_wrong_order(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("30m2h")  # Wrong order
    
    def test_negative_values(self):
        with pytest.raises(ValueError):
            parse_duration("-1h")
    
    # === Parametrized Tests ===
    
    @pytest.mark.parametrize("input_str,expected", [
        ("0s", 0),
        ("1s", 1),
        ("60s", 60),
        ("1m", 60),
        ("1h", 3600),
        ("1h1m1s", 3661),
        ("12h", 43200),
        ("24h", 86400),
    ])
    def test_various_durations(self, input_str, expected):
        assert parse_duration(input_str) == expected
    
    @pytest.mark.parametrize("invalid_input", [
        "",
        None,
        "abc",
        "1x",
        "h",
        "1.5h",
        "1h 30m",  # Space between components
    ])
    def test_invalid_inputs(self, invalid_input):
        with pytest.raises(ValueError):
            parse_duration(invalid_input)''',

                "category": "test_generation",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_multiturn_conversations(self) -> List[InstructionSample]:

        """Generate multi-turn coding conversations."""

        conversations = [

            {

                "instruction": "multi_turn_conversation",

                "input_context": "",

                "output": json.dumps({

                    "messages": [

                        {"role": "user", "content": "I need to implement a REST API for a todo app. What framework should I use in Python?"},

                        {"role": "assistant", "content": "For a Python REST API, I recommend **FastAPI** — it's modern, fast, has automatic docs, and great type support.\n\nHere's a starter:\n\n```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI(title=\"Todo API\")\n\nclass Todo(BaseModel):\n    id: int | None = None\n    title: str\n    completed: bool = False\n\ntodos = []\n\n@app.post(\"/todos\", response_model=Todo)\nasync def create_todo(todo: Todo):\n    todo.id = len(todos) + 1\n    todos.append(todo)\n    return todo\n\n@app.get(\"/todos\")\nasync def list_todos():\n    return todos\n\n@app.get(\"/todos/{todo_id}\")\nasync def get_todo(todo_id: int):\n    for todo in todos:\n        if todo.id == todo_id:\n            return todo\n    raise HTTPException(status_code=404, detail=\"Todo not found\")\n```\n\nWould you like me to add database persistence or authentication?"},

                        {"role": "user", "content": "Yes, please add SQLite database persistence using SQLAlchemy."},

                        {"role": "assistant", "content": "Here's the FastAPI app with SQLAlchemy persistence:\n\n```python\nfrom fastapi import FastAPI, HTTPException, Depends\nfrom pydantic import BaseModel\nfrom sqlalchemy import create_engine, Column, Integer, String, Boolean\nfrom sqlalchemy.orm import declarative_base, sessionmaker, Session\n\n# Database setup\nengine = create_engine(\"sqlite:///todos.db\")\nSessionLocal = sessionmaker(bind=engine)\nBase = declarative_base()\n\n# SQLAlchemy Model\nclass TodoModel(Base):\n    __tablename__ = \"todos\"\n    id = Column(Integer, primary_key=True, index=True)\n    title = Column(String, nullable=False)\n    completed = Column(Boolean, default=False)\n\nBase.metadata.create_all(engine)\n\n# Pydantic schemas\nclass TodoCreate(BaseModel):\n    title: str\n\nclass TodoResponse(BaseModel):\n    id: int\n    title: str\n    completed: bool\n    class Config:\n        from_attributes = True\n\n# Dependency\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n\napp = FastAPI(title=\"Todo API\")\n\n@app.post(\"/todos\", response_model=TodoResponse)\nasync def create_todo(todo: TodoCreate, db: Session = Depends(get_db)):\n    db_todo = TodoModel(title=todo.title)\n    db.add(db_todo)\n    db.commit()\n    db.refresh(db_todo)\n    return db_todo\n\n@app.get(\"/todos\", response_model=list[TodoResponse])\nasync def list_todos(db: Session = Depends(get_db)):\n    return db.query(TodoModel).all()\n\n@app.get(\"/todos/{todo_id}\", response_model=TodoResponse)\nasync def get_todo(todo_id: int, db: Session = Depends(get_db)):\n    todo = db.query(TodoModel).filter(TodoModel.id == todo_id).first()\n    if not todo:\n        raise HTTPException(404, \"Todo not found\")\n    return todo\n```"},

                    ]

                }),

                "category": "multi_turn",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=c["instruction"],

                input_context=c["input_context"],

                output=c["output"],

                category=c["category"],

                language=c.get("language"),

            )

            for c in conversations

        ]



    def _generate_tool_use_instructions(self) -> List[InstructionSample]:

        """Generate tool/function calling examples."""

        templates = [

            {

                "instruction": "The user wants to search for Python documentation about decorators. Show how you would use the available tools to help.",

                "input_context": 'Available tools: [search_docs(query: str), run_code(code: str), read_file(path: str), write_file(path: str, content: str)]',

                "output": json.dumps({

                    "thought": "The user wants to learn about Python decorators. I'll search the documentation first, then provide a code example they can run.",

                    "tool_calls": [

                        {"tool": "search_docs", "args": {"query": "Python decorators how to use"}},

                    ],

                    "response": "Let me look up the documentation on decorators for you.",

                    "follow_up": {

                        "thought": "Based on the documentation results, I'll now provide a working example.",

                        "tool_calls": [

                            {"tool": "run_code", "args": {"code": "def my_decorator(func):\n    def wrapper(*args, **kwargs):\n        print(f'Calling {func.__name__}')\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} returned {result}')\n        return result\n    return wrapper\n\n@my_decorator\ndef add(a, b):\n    return a + b\n\nprint(add(3, 4))"}},

                        ],

                        "response": "Here's a practical decorator example you can try.",

                    },

                }),

                "category": "tool_use",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

            )

            for t in templates

        ]



    def _generate_algorithm_instructions(self) -> List[InstructionSample]:

        templates = [

            {

                "instruction": "Explain the difference between BFS and DFS, when to use each, and implement both for a graph.",

                "input_context": "",

                "output": '''# BFS vs DFS: Graph Traversal Algorithms

## Key Differences

| Aspect | BFS (Breadth-First) | DFS (Depth-First) |
|--------|--------------------|--------------------|
| Data Structure | Queue (FIFO) | Stack (LIFO) / Recursion |
| Exploration | Level by level | Deep first, backtrack |
| Memory | O(V) for queue | O(V) for stack/recursion |
| Shortest Path | ✅ Finds shortest (unweighted) | ❌ Not guaranteed |
| Use Cases | Shortest path, level-order | Topological sort, cycle detection |

## When to Use Each

**Use BFS when:**
- Finding shortest path in unweighted graphs
- Finding all nodes at distance k
- Level-order traversal
- Checking bipartiteness

**Use DFS when:**
- Topological sorting
- Detecting cycles
- Finding connected components
- Solving mazes/puzzles
- Memory is limited (DFS uses less memory for deep, narrow graphs)

## Implementations

```python
from collections import deque, defaultdict
from typing import Dict, List, Set

class Graph:
    def __init__(self):
        self.adj: Dict[str, List[str]] = defaultdict(list)
    
    def add_edge(self, u: str, v: str, bidirectional: bool = True):
        self.adj[u].append(v)
        if bidirectional:
            self.adj[v].append(u)

def bfs(graph: Graph, start: str) -> List[str]:
    """Breadth-First Search - explores level by level."""
    visited: Set[str] = set()
    order: List[str] = []
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        order.append(node)
        
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return order

def dfs(graph: Graph, start: str) -> List[str]:
    """Depth-First Search - explores as deep as possible."""
    visited: Set[str] = set()
    order: List[str] = []
    
    def _dfs(node: str):
        visited.add(node)
        order.append(node)
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                _dfs(neighbor)
    
    _dfs(start)
    return order

def bfs_shortest_path(graph: Graph, start: str, end: str) -> List[str]:
    """BFS finds shortest path in unweighted graph."""
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        
        for neighbor in graph.adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found
```''',

                "category": "algorithm_explanation",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]



    def _generate_debug_instructions(self) -> List[InstructionSample]:

        templates = [

            {

                "instruction": "My Python program is running slowly. Here's the profiling output. Help me identify and fix the performance bottleneck.",

                "input_context": '''# Profiling output (cProfile):
# ncalls  tottime  percall  cumtime  percall filename:lineno(function)
# 1       0.001    0.001    12.345   12.345  main.py:1(main)
# 10000   8.234    0.001    8.234    0.001   utils.py:15(find_duplicates)
# 50000   2.100    0.000    2.100    0.000   utils.py:32(check_exists)
# 1       0.500    0.500    1.500    1.500   db.py:10(load_data)
# 1       0.010    0.010    0.010    0.010   main.py:50(save_results)

# The problematic function:
def find_duplicates(items):
    """Find duplicate items in a list."""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates''',

                "output": '''## Performance Analysis

### Bottleneck Identified: `find_duplicates()` in utils.py:15

This function accounts for **67% of total execution time** (8.234s / 12.345s).

### Root Cause: O(n²) Algorithm + O(n) List Lookup

```python
# Current: O(n² × n) = O(n³)
for i in range(len(items)):          # O(n)
    for j in range(i + 1, len(items)):  # O(n)
        if items[i] == items[j] and items[i] not in duplicates:  # O(n) list scan!
            duplicates.append(items[i])
```

Three nested operations:
1. Outer loop: O(n)
2. Inner loop: O(n) → combined O(n²)
3. `not in duplicates`: O(n) list scan → **total O(n³)**

### Fix: Use a Counter (O(n))

```python
from collections import Counter
from typing import List, Any

def find_duplicates(items: List[Any]) -> List[Any]:
    """Find duplicate items in O(n) time."""
    counts = Counter(items)
    return [item for item, count in counts.items() if count > 1]
```

### Performance Comparison

| Input Size | Original O(n³) | Fixed O(n) | Speedup |
|-----------|----------------|------------|---------|
| 1,000     | ~1 second      | ~0.1ms     | 10,000x |
| 10,000    | ~8 seconds     | ~1ms       | 8,000x  |
| 100,000   | ~8000 seconds  | ~10ms      | 800,000x |

### Secondary: `check_exists()` (utils.py:32)

Also check if `check_exists` is doing linear scans. If it checks membership
in a list, convert to a set:

```python
# Before: O(n) per check
if item in my_list:  # Linear scan

# After: O(1) per check  
if item in my_set:   # Hash lookup
```

### General Tips
1. **Always profile first** — don't guess where the bottleneck is
2. **Check algorithm complexity** — O(n²) or worse is usually the culprit
3. **Use appropriate data structures** — sets for membership, dicts for lookup
4. **Consider vectorization** — NumPy for numeric operations''',

                "category": "debugging",

                "language": "python",

            },

        ]



        return [

            InstructionSample(

                instruction=t["instruction"],

                input_context=t["input_context"],

                output=t["output"],

                category=t["category"],

                language=t.get("language"),

            )

            for t in templates

        ]

