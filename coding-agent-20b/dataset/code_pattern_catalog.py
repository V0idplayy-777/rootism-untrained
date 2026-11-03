"""Domain- and complexity-faithful Python code pattern catalog (Corpus v1).

Provides 360+ distinct, compilable Python implementations organized strictly by
``(complexity, domain)`` across 4 complexity tiers (``beginner``,
``intermediate``, ``advanced``, ``expert``) and 10 engineering domains
(``algorithms``, ``data``, ``web``, ``systems``, ``testing``, ``database``,
``cli``, ``security``, ``async_io``, ``ml``).

Used as a supplementary minority source alongside real permissively licensed
code — never as a standalone corpus, and never with cross-domain random
fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PatternSpec:
    pattern_id: str
    complexity: str
    domain: str
    description: str
    tags: tuple[str, ...]
    code: str


COMPLEXITIES: tuple[str, ...] = ("beginner", "intermediate", "advanced", "expert")

DOMAINS: tuple[str, ...] = (
    "algorithms",
    "data",
    "web",
    "systems",
    "testing",
    "database",
    "cli",
    "security",
    "async_io",
    "ml",
)

DOMAIN_ALIASES: dict[str, str] = {
    "data_science": "data",
    "data_structures": "algorithms",
    "backend": "web",
    "http": "web",
    "infra": "systems",
    "db": "database",
    "sql": "database",
    "auth": "security",
    "crypto": "security",
    "async": "async_io",
    "concurrency": "async_io",
    "ai": "ml",
}


def normalize_domain(domain: str) -> str:
    key = (domain or "").strip().lower()
    return DOMAIN_ALIASES.get(key, key)


def _build_algorithms_patterns() -> list[PatternSpec]:
    out: list[PatternSpec] = []

    # Beginner algorithms (9 distinct patterns)
    beginner = [
        (
            "alg-beg-two-sum-sorted",
            "Two-pointer pair lookup on a sorted integer slice",
            ("two_pointers", "arrays", "search"),
            '''def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int] | None:
    """Return 0-based indices (i, j) with i < j and nums[i] + nums[j] == target."""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return (left, right)
        if total < target:
            left += 1
        else:
            right -= 1
    return None
''',
        ),
        (
            "alg-beg-prefix-sums",
            "Immutable prefix-sum range query helper",
            ("prefix_sum", "arrays"),
            '''class RangeSum:
    """O(1) inclusive range-sum queries after O(n) preprocessing."""

    def __init__(self, values: list[int]) -> None:
        self._prefix = [0]
        for v in values:
            self._prefix.append(self._prefix[-1] + v)

    def query(self, left: int, right: int) -> int:
        if left < 0 or right >= len(self._prefix) - 1 or left > right:
            raise IndexError("invalid inclusive range")
        return self._prefix[right + 1] - self._prefix[left]
''',
        ),
        (
            "alg-beg-run-length-encode",
            "Lossless run-length encoding and decoding of strings",
            ("strings", "compression"),
            '''def run_length_encode(items: list[str]) -> list[tuple[str, int]]:
    """Collapse consecutive identical elements into (value, count) pairs."""
    if not items:
        return []
    encoded: list[tuple[str, int]] = []
    cur, count = items[0], 1
    for item in items[1:]:
        if item == cur:
            count += 1
        else:
            encoded.append((cur, count))
            cur, count = item, 1
    encoded.append((cur, count))
    return encoded
''',
        ),
        (
            "alg-beg-anagram-groups",
            "Group words into canonical character-count anagram buckets",
            ("hashing", "strings"),
            '''from collections import defaultdict


def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group strings that are character permutations of each other."""
    buckets: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for word in words:
        counts = [0] * 26
        for ch in word.lower():
            if "a" <= ch <= "z":
                counts[ord(ch) - ord("a")] += 1
        buckets[tuple(counts)].append(word)
    return list(buckets.values())
''',
        ),
        (
            "alg-beg-rotate-slice",
            "In-place k-step array rotation via three reversals",
            ("arrays", "in_place"),
            '''def rotate_in_place(nums: list[int], k: int) -> None:
    """Rotate nums to the right by k positions in O(1) auxiliary space."""
    n = len(nums)
    if n <= 1:
        return
    k %= n
    if k == 0:
        return

    def _rev(lo: int, hi: int) -> None:
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo += 1
            hi -= 1

    _rev(0, n - 1)
    _rev(0, k - 1)
    _rev(k, n - 1)
''',
        ),
        (
            "alg-beg-valid-brackets",
            "Stack-based balanced delimiter verifier",
            ("stack", "parsing"),
            '''def has_balanced_delimiters(text: str) -> bool:
    """Verify (), [], and {} delimiters close in LIFO order."""
    matching = {")": "(", "]": "[", "}": "{"}
    openers = set(matching.values())
    stack: list[str] = []
    for ch in text:
        if ch in openers:
            stack.append(ch)
        elif ch in matching:
            if not stack or stack.pop() != matching[ch]:
                return False
    return not stack
''',
        ),
        (
            "alg-beg-gcd-lcm",
            "Iterative Euclidean GCD and overflow-safe LCM",
            ("number_theory", "math"),
            '''def gcd(a: int, b: int) -> int:
    """Greatest common non-negative divisor of a and b."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least common non-negative multiple of a and b."""
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)
''',
        ),
        (
            "alg-beg-insertion-sort-key",
            "Stable insertion sort with custom key function",
            ("sorting", "stability"),
            '''from typing import Callable, TypeVar

T = TypeVar("T")


def insertion_sort_by(items: list[T], key: Callable[[T], object]) -> list[T]:
    """Return a new stably sorted list using insertion sort."""
    out = list(items)
    for i in range(1, len(out)):
        cur = out[i]
        cur_key = key(cur)
        j = i - 1
        while j >= 0 and key(out[j]) > cur_key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = cur
    return out
''',
        ),
        (
            "alg-beg-lower-bound",
            "Monotonic bisect lower-bound index search",
            ("binary_search", "monotonic"),
            '''def lower_bound(nums: list[int], target: int) -> int:
    """First index i where nums[i] >= target, or len(nums) if none."""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        ),
    ]

    # Intermediate algorithms (9 distinct patterns)
    intermediate = [
        (
            "alg-int-merge-intervals",
            "Sort-and-sweep interval coalescing",
            ("intervals", "sweep_line"),
            '''def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce overlapping or touching closed intervals [start, end]."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda span: (span[0], span[1]))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged
''',
        ),
        (
            "alg-int-monotonic-next-greater",
            "Decreasing monotonic stack for next-greater element indices",
            ("monotonic_stack", "arrays"),
            '''def next_greater_indices(values: list[int]) -> list[int]:
    """For each index i, return the smallest j > i with values[j] > values[i], or -1."""
    result = [-1] * len(values)
    stack: list[int] = []
    for idx, val in enumerate(values):
        while stack and values[stack[-1]] < val:
            prev = stack.pop()
            result[prev] = idx
        stack.append(idx)
    return result
''',
        ),
        (
            "alg-int-sliding-window-distinct",
            "Longest substring with at most k distinct characters",
            ("sliding_window", "hash_map"),
            '''def longest_k_distinct(text: str, k: int) -> int:
    """Length of the longest contiguous slice containing <= k distinct chars."""
    if k <= 0 or not text:
        return 0
    counts: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(text):
        counts[ch] = counts.get(ch, 0) + 1
        while len(counts) > k:
            drop = text[left]
            counts[drop] -= 1
            if counts[drop] == 0:
                del counts[drop]
            left += 1
        best = max(best, right - left + 1)
    return best
''',
        ),
        (
            "alg-int-kahn-toposort",
            "Kahn topological ordering with cycle detection",
            ("graphs", "dag", "topological_sort"),
            '''from collections import deque


def topological_sort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Return topological order for a DAG or raise ValueError on cycle."""
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    indegree: dict[str, int] = {n: 0 for n in nodes}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
        indegree[dst] = indegree.get(dst, 0) + 1
        indegree.setdefault(src, 0)
    queue = deque(sorted(n for n, deg in indegree.items() if deg == 0))
    order: list[str] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in adj.get(cur, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(indegree):
        raise ValueError("dependency graph contains a cycle")
    return order
''',
        ),
        (
            "alg-int-quickselect",
            "Iterative Hoare/Lomuto quickselect for k-th smallest element",
            ("selection", "partition"),
            '''def kth_smallest(values: list[int], k: int) -> int:
    """Return 0-indexed k-th smallest value without full sorting."""
    if not 0 <= k < len(values):
        raise IndexError("k out of range")
    arr = list(values)
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        pivot = arr[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return arr[k]
    return arr[lo]
''',
        ),
        (
            "alg-int-trie-prefix",
            "Prefix tree supporting exact lookup, prefix count, and deletion",
            ("trie", "strings", "trees"),
            '''class PrefixTrie:
    """Compact dictionary-backed prefix tree with prefix counters."""

    def __init__(self) -> None:
        self._children: dict[str, PrefixTrie] = {}
        self._pass_count = 0
        self._terminal = False

    def insert(self, word: str) -> None:
        node = self
        node._pass_count += 1
        for ch in word:
            node = node._children.setdefault(ch, PrefixTrie())
            node._pass_count += 1
        node._terminal = True

    def count_prefix(self, prefix: str) -> int:
        node = self
        for ch in prefix:
            nxt = node._children.get(ch)
            if nxt is None:
                return 0
            node = nxt
        return node._pass_count
''',
        ),
        (
            "alg-int-bipartite-check",
            "BFS 2-coloring bipartite verifier returning partition sets",
            ("graphs", "bfs", "bipartite"),
            '''from collections import deque


def two_color_partition(adj: dict[int, list[int]]) -> tuple[set[int], set[int]] | None:
    """Partition undirected graph into two independent sets, or None if odd cycle."""
    color: dict[int, int] = {}
    for start in adj:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj.get(u, ()):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return None
    left = {u for u, c in color.items() if c == 0}
    right = {u for u, c in color.items() if c == 1}
    return left, right
''',
        ),
        (
            "alg-int-edit-distance",
            "Space-optimized O(min(n, m)) Levenshtein edit distance",
            ("dynamic_programming", "strings"),
            '''def levenshtein_distance(a: str, b: str) -> int:
    """Minimum single-character edits (insert, delete, substitute) between a and b."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]
''',
        ),
        (
            "alg-int-disjoint-set",
            "Disjoint-set union-find with path compression and union by size",
            ("union_find", "graphs"),
            '''class DisjointSet:
    """Near-constant-time union-find tracking connected component sizes."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True
''',
        ),
    ]

    # Advanced algorithms (9 distinct patterns)
    advanced = [
        (
            "alg-adv-kmp-matcher",
            "Knuth-Morris-Pratt linear-time substring search",
            ("kmp", "strings", "automata"),
            '''def kmp_find_all(text: str, pattern: str) -> list[int]:
    """Return all 0-based start offsets of non-empty pattern in text in O(n + m)."""
    if not pattern:
        raise ValueError("pattern must be non-empty")
    lps = [0] * len(pattern)
    length = 0
    for i in range(1, len(pattern)):
        while length and pattern[i] != pattern[length]:
            length = lps[length - 1]
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length

    matches: list[int] = []
    j = 0
    for i, ch in enumerate(text):
        while j and ch != pattern[j]:
            j = lps[j - 1]
        if ch == pattern[j]:
            j += 1
            if j == len(pattern):
                matches.append(i - j + 1)
                j = lps[j - 1]
    return matches
''',
        ),
        (
            "alg-adv-tarjan-scc",
            "Iterative-safe Tarjan strongly connected components",
            ("graphs", "scc", "tarjan"),
            '''def strongly_connected_components(adj: dict[int, list[int]]) -> list[list[int]]:
    """Compute SCCs in topological reverse order using Tarjan low-link values."""
    index = 0
    indices: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    sccs: list[list[int]] = []

    def visit(v: int) -> None:
        nonlocal index
        indices[v] = lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, ()):
            if w not in indices:
                visit(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: list[int] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for node in adj:
        if node not in indices:
            visit(node)
    return sccs
''',
        ),
        (
            "alg-adv-dijkstra-path",
            "Dijkstra shortest path with predecessor reconstruction",
            ("graphs", "shortest_path", "heap"),
            '''import heapq


def shortest_path(
    adj: dict[str, list[tuple[str, float]]], start: str, goal: str
) -> tuple[float, list[str]] | None:
    """Return (cost, node_path) for non-negative edge weights, or None if unreachable."""
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, str] = {}
    pq: list[tuple[float, str]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            path = [goal]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            path.reverse()
            return d, path
        if d > dist.get(u, float("inf")):
            continue
        for v, weight in adj.get(u, ()):
            if weight < 0:
                raise ValueError("negative weight unsupported")
            nd = d + weight
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None
''',
        ),
        (
            "alg-adv-fenwick-tree",
            "Fenwick (Binary Indexed) Tree supporting point updates and prefix sums",
            ("fenwick", "range_query", "trees"),
            '''class FenwickTree:
    """1-indexed Binary Indexed Tree for O(log n) point updates and prefix sums."""

    def __init__(self, size: int) -> None:
        self._n = size
        self._bit = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        i = index + 1
        while i <= self._n:
            self._bit[i] += delta
            i += i & -i

    def prefix_sum(self, index: int) -> int:
        total = 0
        i = index + 1
        while i > 0:
            total += self._bit[i]
            i -= i & -i
        return total

    def range_sum(self, left: int, right: int) -> int:
        if left > right:
            return 0
        return self.prefix_sum(right) - (self.prefix_sum(left - 1) if left > 0 else 0)
''',
        ),
        (
            "alg-adv-lis-patience",
            "O(n log n) longest strictly increasing subsequence with witness recovery",
            ("dynamic_programming", "binary_search", "lis"),
            '''from bisect import bisect_left


def longest_increasing_subsequence(nums: list[int]) -> list[int]:
    """Recover one strictly increasing subsequence of maximum length in O(n log n)."""
    if not nums:
        return []
    tails: list[int] = []
    tail_idx: list[int] = []
    parent = [-1] * len(nums)
    for i, x in enumerate(nums):
        pos = bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[pos] = x
            tail_idx[pos] = i
        if pos > 0:
            parent[i] = tail_idx[pos - 1]
    seq: list[int] = []
    cur = tail_idx[-1]
    while cur != -1:
        seq.append(nums[cur])
        cur = parent[cur]
    seq.reverse()
    return seq
''',
        ),
        (
            "alg-adv-bellman-ford",
            "Bellman-Ford shortest distances with negative-cycle witness detection",
            ("graphs", "negative_cycles", "shortest_path"),
            '''def bellman_ford(
    num_vertices: int, edges: list[tuple[int, int, float]], source: int
) -> list[float]:
    """Compute shortest distances from source or raise ValueError on negative cycle."""
    dist = [float("inf")] * num_vertices
    dist[source] = 0.0
    for step in range(num_vertices):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                if step == num_vertices - 1:
                    raise ValueError("reachable negative-weight cycle detected")
                dist[v] = dist[u] + w
                updated = True
        if not updated:
            break
    return dist
''',
        ),
        (
            "alg-adv-rabin-karp",
            "Rolling polynomial hash Rabin-Karp multi-pattern matcher",
            ("rolling_hash", "strings"),
            '''def rabin_karp_multi(text: str, patterns: set[str]) -> dict[str, list[int]]:
    """Find occurrences of equal-length patterns using a 61-bit Mersenne rolling hash."""
    if not patterns:
        return {}
    lengths = {len(p) for p in patterns}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("all patterns must share a positive length")
    m = next(iter(lengths))
    if len(text) < m:
        return {p: [] for p in patterns}

    mod = (1 << 61) - 1
    base = 257
    high = pow(base, m - 1, mod)

    def _h(s: str) -> int:
        acc = 0
        for ch in s:
            acc = (acc * base + ord(ch)) % mod
        return acc

    by_hash: dict[int, list[str]] = {}
    for p in patterns:
        by_hash.setdefault(_h(p), []).append(p)

    out: dict[str, list[int]] = {p: [] for p in patterns}
    cur = _h(text[:m])
    for idx in range(len(text) - m + 1):
        for cand in by_hash.get(cur, ()):
            if text[idx : idx + m] == cand:
                out[cand].append(idx)
        if idx + m < len(text):
            cur = ((cur - ord(text[idx]) * high) * base + ord(text[idx + m])) % mod
    return out
''',
        ),
        (
            "alg-adv-convex-hull",
            "Andrew monotone-chain 2D convex hull in O(n log n)",
            ("computational_geometry", "convex_hull"),
            '''Point = tuple[float, float]


def convex_hull(points: list[Point]) -> list[Point]:
    """Counter-clockwise vertices of the lower+upper convex hull (collinear pruned)."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
''',
        ),
        (
            "alg-adv-lru-doubly-linked",
            "Strict O(1) LRU cache using sentinel doubly-linked nodes",
            ("lru", "linked_list", "data_structures"),
            '''class _Node:
    __slots__ = ("key", "val", "prev", "next")

    def __init__(self, key: str = "", val: int = 0) -> None:
        self.key = key
        self.val = val
        self.prev: _Node | None = None
        self.next: _Node | None = None


class StrictLRUCache:
    """O(1) get/put LRU cache without OrderedDict."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.map: dict[str, _Node] = {}
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _unlink(self, node: _Node) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        first = self.head.next
        assert first is not None
        node.next = first
        node.prev = self.head
        first.prev = node
        self.head.next = node

    def get(self, key: str) -> int | None:
        node = self.map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.val

    def put(self, key: str, value: int) -> None:
        if key in self.map:
            node = self.map[key]
            node.val = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.map) >= self.capacity:
            lru = self.tail.prev
            assert lru is not None and lru is not self.head
            self._unlink(lru)
            del self.map[lru.key]
        fresh = _Node(key, value)
        self.map[key] = fresh
        self._push_front(fresh)
''',
        ),
    ]

    # Expert algorithms (9 distinct patterns)
    expert = [
        (
            "alg-exp-dinic-maxflow",
            "Dinic level-graph + current-arc blocking flow max-flow solver",
            ("max_flow", "dinic", "graphs"),
            '''from collections import deque


class DinicMaxFlow:
    """O(E * V^2) maximum s-t flow with level BFS and current-arc pruning."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.graph: list[list[list[int]]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        fwd = [v, capacity, len(self.graph[v])]
        rev = [u, 0, len(self.graph[u])]
        self.graph[u].append(fwd)
        self.graph[v].append(rev)

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        while True:
            level = [-1] * self.n
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, _ in self.graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        q.append(v)
            if level[sink] < 0:
                return flow
            it = [0] * self.n

            def dfs(u: int, pushed: int) -> int:
                if pushed == 0 or u == sink:
                    return pushed
                while it[u] < len(self.graph[u]):
                    edge = self.graph[u][it[u]]
                    v, cap, rev = edge
                    if cap > 0 and level[v] == level[u] + 1:
                        tr = dfs(v, min(pushed, cap))
                        if tr > 0:
                            edge[1] -= tr
                            self.graph[v][rev][1] += tr
                            return tr
                    it[u] += 1
                return 0

            while True:
                pushed = dfs(source, 10**18)
                if pushed == 0:
                    break
                flow += pushed
''',
        ),
        (
            "alg-exp-lazy-segtree",
            "Lazy-propagation segment tree supporting range add and range max",
            ("segment_tree", "lazy_propagation"),
            '''class LazyRangeMaxTree:
    """Range-addition updates + range-maximum queries in O(log n)."""

    def __init__(self, values: list[int]) -> None:
        self.n = len(values)
        self.tree = [0] * (4 * max(self.n, 1))
        self.lazy = [0] * (4 * max(self.n, 1))
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, idx: int, lo: int, hi: int, vals: list[int]) -> None:
        if lo == hi:
            self.tree[idx] = vals[lo]
            return
        mid = (lo + hi) // 2
        self._build(idx * 2, lo, mid, vals)
        self._build(idx * 2 + 1, mid + 1, hi, vals)
        self.tree[idx] = max(self.tree[idx * 2], self.tree[idx * 2 + 1])

    def _push(self, idx: int) -> None:
        delta = self.lazy[idx]
        if delta:
            for child in (idx * 2, idx * 2 + 1):
                self.tree[child] += delta
                self.lazy[child] += delta
            self.lazy[idx] = 0

    def range_add(self, ql: int, qr: int, delta: int) -> None:
        def _upd(idx: int, lo: int, hi: int) -> None:
            if qr < lo or hi < ql:
                return
            if ql <= lo and hi <= qr:
                self.tree[idx] += delta
                self.lazy[idx] += delta
                return
            self._push(idx)
            mid = (lo + hi) // 2
            _upd(idx * 2, lo, mid)
            _upd(idx * 2 + 1, mid + 1, hi)
            self.tree[idx] = max(self.tree[idx * 2], self.tree[idx * 2 + 1])

        _upd(1, 0, self.n - 1)

    def range_max(self, ql: int, qr: int) -> int:
        def _qry(idx: int, lo: int, hi: int) -> int:
            if qr < lo or hi < ql:
                return -(10**18)
            if ql <= lo and hi <= qr:
                return self.tree[idx]
            self._push(idx)
            mid = (lo + hi) // 2
            return max(_qry(idx * 2, lo, mid), _qry(idx * 2 + 1, mid + 1, hi))

        return _qry(1, 0, self.n - 1)
''',
        ),
        (
            "alg-exp-aho-corasick",
            "Aho-Corasick failure-link multi-pattern automaton",
            ("aho_corasick", "automata", "strings"),
            '''from collections import deque


class AhoCorasick:
    """Linear-time multi-keyword dictionary matcher with failure links."""

    def __init__(self, keywords: list[str]) -> None:
        self.goto: list[dict[str, int]] = [{}]
        self.output: list[list[str]] = [[]]
        self.fail: list[int] = [0]
        for word in keywords:
            if not word:
                continue
            state = 0
            for ch in word:
                if ch not in self.goto[state]:
                    self.goto[state][ch] = len(self.goto)
                    self.goto.append({})
                    self.output.append([])
                    self.fail.append(0)
                state = self.goto[state][ch]
            self.output[state].append(word)
        self._build_failures()

    def _build_failures(self) -> None:
        q: deque[int] = deque()
        for ch, nxt in self.goto[0].items():
            self.fail[nxt] = 0
            q.append(nxt)
        while q:
            r = q.popleft()
            for ch, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(ch, 0)
                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text: str) -> list[tuple[int, str]]:
        state = 0
        hits: list[tuple[int, str]] = []
        for idx, ch in enumerate(text):
            while state and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for kw in self.output[state]:
                hits.append((idx - len(kw) + 1, kw))
        return hits
''',
        ),
        (
            "alg-exp-hopcroft-karp",
            "Hopcroft-Karp O(E * sqrt(V)) maximum bipartite matching",
            ("bipartite_matching", "hopcroft_karp", "graphs"),
            '''from collections import deque


def hopcroft_karp(adj: dict[int, list[int]], left_nodes: list[int]) -> dict[int, int]:
    """Return left->right maximum cardinality matching in O(E * sqrt(V))."""
    pair_u: dict[int, int | None] = {u: None for u in left_nodes}
    pair_v: dict[int, int | None] = {}
    dist: dict[int | None, float] = {}

    def bfs() -> bool:
        q: deque[int] = deque()
        for u in left_nodes:
            if pair_u[u] is None:
                dist[u] = 0.0
                q.append(u)
            else:
                dist[u] = float("inf")
        dist[None] = float("inf")
        while q:
            u = q.popleft()
            if dist[u] < dist[None]:
                for v in adj.get(u, ()):
                    nxt = pair_v.get(v)
                    if dist.get(nxt, float("inf")) == float("inf"):
                        dist[nxt] = dist[u] + 1.0
                        if nxt is not None:
                            q.append(nxt)
        return dist[None] != float("inf")

    def dfs(u: int | None) -> bool:
        if u is None:
            return True
        for v in adj.get(u, ()):
            nxt = pair_v.get(v)
            if dist.get(nxt) == dist[u] + 1.0 and dfs(nxt):
                pair_v[v] = u
                pair_u[u] = v
                return True
        dist[u] = float("inf")
        return False

    while bfs():
        for u in left_nodes:
            if pair_u[u] is None:
                dfs(u)
    return {u: v for u, v in pair_u.items() if v is not None}
''',
        ),
        (
            "alg-exp-suffix-array-kasai",
            "Prefix-doubling suffix array + Kasai linear-time LCP array",
            ("suffix_array", "lcp", "strings"),
            '''def suffix_and_lcp_arrays(text: str) -> tuple[list[int], list[int]]:
    """Build suffix array in O(n log^2 n) and Kasai LCP array in O(n)."""
    n = len(text)
    if n == 0:
        return [], []
    sa = list(range(n))
    rank = [ord(c) for c in text]
    k = 1
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp = [0] * n
        for idx in range(1, n):
            prev, cur = sa[idx - 1], sa[idx]
            prev_key = (rank[prev], rank[prev + k] if prev + k < n else -1)
            cur_key = (rank[cur], rank[cur + k] if cur + k < n else -1)
            tmp[cur] = tmp[prev] + (1 if cur_key != prev_key else 0)
        rank = tmp
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1

    lcp = [0] * max(n - 1, 0)
    inv = [0] * n
    for idx, pos in enumerate(sa):
        inv[pos] = idx
    h = 0
    for i in range(n):
        r = inv[i]
        if r == n - 1:
            h = 0
            continue
        j = sa[r + 1]
        while i + h < n and j + h < n and text[i + h] == text[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return sa, lcp
''',
        ),
        (
            "alg-exp-binary-lifting-lca",
            "Binary lifting tree ancestor jump table for O(log V) LCA and distance",
            ("trees", "lca", "binary_lifting"),
            '''from collections import deque


class BinaryLiftingLCA:
    """Preprocesses rooted tree in O(V log V) for O(log V) lowest common ancestor."""

    def __init__(self, n: int, adj: dict[int, list[int]], root: int = 0) -> None:
        self.log = max(1, n.bit_length())
        self.up = [[-1] * n for _ in range(self.log)]
        self.depth = [0] * n
        visited = {root}
        q = deque([root])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):
                if v not in visited:
                    visited.add(v)
                    self.depth[v] = self.depth[u] + 1
                    self.up[0][v] = u
                    q.append(v)
        for k in range(1, self.log):
            for v in range(n):
                mid = self.up[k - 1][v]
                self.up[k][v] = self.up[k - 1][mid] if mid != -1 else -1

    def lca(self, a: int, b: int) -> int:
        if self.depth[a] < self.depth[b]:
            a, b = b, a
        diff = self.depth[a] - self.depth[b]
        for k in range(self.log):
            if diff & (1 << k):
                a = self.up[k][a]
        if a == b:
            return a
        for k in range(self.log - 1, -1, -1):
            if self.up[k][a] != self.up[k][b]:
                a = self.up[k][a]
                b = self.up[k][b]
        return self.up[0][a]
''',
        ),
        (
            "alg-exp-persistent-treap",
            "Path-copying persistent randomized treap for versioned ordered maps",
            ("persistent_ds", "treap", "trees"),
            '''from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class _TreapNode:
    key: int
    value: str
    prio: int
    left: _TreapNode | None = None
    right: _TreapNode | None = None


class PersistentTreap:
    """Immutable versioned ordered map via functional split/merge."""

    def __init__(self, root: _TreapNode | None = None) -> None:
        self.root = root

    @staticmethod
    def _prio(key: int) -> int:
        return int(hashlib.blake2b(str(key).encode(), digest_size=4).hexdigest(), 16)

    @classmethod
    def _split(cls, node: _TreapNode | None, key: int) -> tuple[_TreapNode | None, _TreapNode | None]:
        if node is None:
            return None, None
        if key < node.key:
            left, right = cls._split(node.left, key)
            return left, _TreapNode(node.key, node.value, node.prio, right, node.right)
        left, right = cls._split(node.right, key)
        return _TreapNode(node.key, node.value, node.prio, node.left, left), right

    @classmethod
    def _merge(cls, a: _TreapNode | None, b: _TreapNode | None) -> _TreapNode | None:
        if a is None or b is None:
            return a or b
        if a.prio > b.prio:
            return _TreapNode(a.key, a.value, a.prio, a.left, cls._merge(a.right, b))
        return _TreapNode(b.key, b.value, b.prio, cls._merge(a, b.left), b.right)

    def set(self, key: int, value: str) -> PersistentTreap:
        less_eq, greater = self._split(self.root, key)
        less, _equal = self._split(less_eq, key - 1)
        fresh = _TreapNode(key, value, self._prio(key))
        return PersistentTreap(self._merge(self._merge(less, fresh), greater))

    def get(self, key: int) -> str | None:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return cur.value
            cur = cur.left if key < cur.key else cur.right
        return None
''',
        ),
        (
            "alg-exp-cdq-2d-dominance",
            "Divide-and-conquer 2D dominance point counting (merge-count)",
            ("divide_and_conquer", "geometry", "inversions"),
            '''def count_2d_dominated(points: list[tuple[int, int]]) -> list[int]:
    """For each point i, count j != i with x_j <= x_i and y_j <= y_i in O(n log n)."""
    n = len(points)
    indexed = sorted(range(n), key=lambda i: (points[i][0], points[i][1]))
    ans = [0] * n

    def solve(arr: list[int]) -> list[int]:
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = solve(arr[:mid])
        right = solve(arr[mid:])
        merged: list[int] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if points[left[i]][1] <= points[right[j]][1]:
                merged.append(left[i])
                i += 1
            else:
                ans[right[j]] += i
                merged.append(right[j])
                j += 1
        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            ans[right[j]] += i
            merged.append(right[j])
            j += 1
        return merged

    solve(indexed)
    return ans
''',
        ),
        (
            "alg-exp-two-sat",
            "Linear-time 2-SAT solver via implication graph and Kosaraju/Tarjan SCC",
            ("2sat", "boolean_satisfiability", "scc"),
            '''class TwoSatSolver:
    """Solve conjunctions of (a OR b) boolean clauses in O(V + E)."""

    def __init__(self, num_vars: int) -> None:
        self.n = num_vars
        self.adj: list[list[int]] = [[] for _ in range(2 * num_vars)]
        self.radj: list[list[int]] = [[] for _ in range(2 * num_vars)]

    def _lit(self, var: int, is_true: bool) -> int:
        return 2 * var + (0 if is_true else 1)

    def add_or(self, var_a: int, val_a: bool, var_b: int, val_b: bool) -> None:
        a = self._lit(var_a, val_a)
        b = self._lit(var_b, val_b)
        self.adj[a ^ 1].append(b)
        self.adj[b ^ 1].append(a)
        self.radj[b].append(a ^ 1)
        self.radj[a].append(b ^ 1)

    def solve(self) -> list[bool] | None:
        vis = [False] * (2 * self.n)
        order: list[int] = []

        def dfs1(u: int) -> None:
            vis[u] = True
            for v in self.adj[u]:
                if not vis[v]:
                    dfs1(v)
            order.append(u)

        for i in range(2 * self.n):
            if not vis[i]:
                dfs1(i)

        comp = [-1] * (2 * self.n)

        def dfs2(u: int, cid: int) -> None:
            comp[u] = cid
            for v in self.radj[u]:
                if comp[v] == -1:
                    dfs2(v, cid)

        cid = 0
        for u in reversed(order):
            if comp[u] == -1:
                dfs2(u, cid)
                cid += 1

        assignment = [False] * self.n
        for i in range(self.n):
            if comp[2 * i] == comp[2 * i + 1]:
                return None
            assignment[i] = comp[2 * i] > comp[2 * i + 1]
        return assignment
''',
        ),
    ]

    for comp, bucket in (
        ("beginner", beginner),
        ("intermediate", intermediate),
        ("advanced", advanced),
        ("expert", expert),
    ):
        for pid, desc, tags, code in bucket:
            out.append(PatternSpec(pid, comp, "algorithms", desc, tags, code))
    return out


# Concise, structurally rich domain builders for the remaining 9 domains (9 per tier * 4 tiers * 9 = 324)
def _domain_specs(domain: str, tiers: dict[str, list[tuple[str, str, tuple[str, ...], str]]]) -> list[PatternSpec]:
    out: list[PatternSpec] = []
    for comp, entries in tiers.items():
        for pid, desc, tags, code in entries:
            out.append(PatternSpec(pid, comp, domain, desc, tags, code))
    return out


def _build_data_patterns() -> list[PatternSpec]:
    return _domain_specs(
        "data",
        {
            "beginner": [
                ("data-beg-csv-agg", "Streaming CSV numeric column summary", ("csv", "aggregation"),
                 '''import csv
import io


def summarize_csv_column(csv_text: str, column: str) -> dict[str, float]:
    """Compute count, total, min, max, and mean of a numeric CSV column."""
    reader = csv.DictReader(io.StringIO(csv_text))
    count = 0
    total = 0.0
    lo = float("inf")
    hi = float("-inf")
    for row in reader:
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        val = float(raw)
        count += 1
        total += val
        lo = min(lo, val)
        hi = max(hi, val)
    return {"count": float(count), "sum": total, "min": lo if count else 0.0, "max": hi if count else 0.0, "mean": total / max(count, 1)}
'''),
                ("data-beg-missing-impute", "Column-wise median imputation for missing records", ("imputation", "cleaning"),
                 '''from statistics import median


def impute_numeric_median(rows: list[dict[str, float | None]], field: str) -> list[dict[str, float]]:
    """Fill None entries in field using the observed median."""
    observed = [float(r[field]) for r in rows if r.get(field) is not None]
    fill = median(observed) if observed else 0.0
    return [{**r, field: float(r[field]) if r.get(field) is not None else fill} for r in rows]
'''),
                ("data-beg-minmax-scale", "Feature min-max normalization to [0, 1]", ("scaling", "features"),
                 '''def minmax_normalize(values: list[float]) -> list[float]:
    """Scale values linearly into [0.0, 1.0]; constant inputs map to 0.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0.0:
        return [0.0 for _ in values]
    return [(v - lo) / span for v in values]
'''),
                ("data-beg-dedup-records", "Order-preserving record deduplication by composite key", ("deduplication", "etl"),
                 '''def dedup_by_keys(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    """Keep first occurrence per composite key tuple."""
    seen: set[tuple[object, ...]] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        sig = tuple(row.get(k) for k in keys)
        if sig not in seen:
            seen.add(sig)
            unique.append(row)
    return unique
'''),
                ("data-beg-histogram-bins", "Equal-width numeric histogram binning", ("histogram", "statistics"),
                 '''def equal_width_histogram(values: list[float], bins: int = 5) -> list[int]:
    """Assign values into `bins` equal-width buckets across [min, max]."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    counts = [0] * bins
    if not values:
        return counts
    lo, hi = min(values), max(values)
    if lo == hi:
        counts[0] = len(values)
        return counts
    width = (hi - lo) / bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    return counts
'''),
                ("data-beg-flatten-json", "Nested dictionary flattener with dot-separated keys", ("json", "normalization"),
                 '''def flatten_record(record: dict[str, object], sep: str = ".") -> dict[str, object]:
    """Flatten nested dicts into single-level dotted keys."""
    flat: dict[str, object] = {}

    def _walk(prefix: str, obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}{sep}{k}" if prefix else str(k), v)
        else:
            flat[prefix] = obj

    _walk("", record)
    return flat
'''),
                ("data-beg-moving-avg", "Simple trailing moving average over a numeric series", ("time_series", "window"),
                 '''from collections import deque


def trailing_moving_average(series: list[float], window: int) -> list[float]:
    """Compute trailing mean with O(1) amortized updates per step."""
    if window <= 0:
        raise ValueError("window must be positive")
    buf: deque[float] = deque()
    running = 0.0
    out: list[float] = []
    for x in series:
        buf.append(x)
        running += x
        if len(buf) > window:
            running -= buf.popleft()
        out.append(running / len(buf))
    return out
'''),
                ("data-beg-schema-cast", "Strict row type-coercion with row-index error reporting", ("validation", "etl"),
                 '''def coerce_rows(rows: list[dict[str, str]], schema: dict[str, type]) -> list[dict[str, object]]:
    """Cast string fields according to schema; attach row number on ValueError."""
    converted: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        item: dict[str, object] = {}
        for col, caster in schema.items():
            try:
                item[col] = caster(row[col])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"row {idx} column {col!r}: {exc}") from exc
        converted.append(item)
    return converted
'''),
                ("data-beg-pivot-counts", "Two-dimensional contingency count table", ("pivot", "crosstab"),
                 '''def crosstab_counts(rows: list[dict[str, str]], row_key: str, col_key: str) -> dict[str, dict[str, int]]:
    """Build nested count table rows[row_key][col_key]."""
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        rk = str(row.get(row_key, ""))
        ck = str(row.get(col_key, ""))
        bucket = table.setdefault(rk, {})
        bucket[ck] = bucket.get(ck, 0) + 1
    return table
'''),
            ],
            "intermediate": [
                ("data-int-welford-stats", "Welford numerically stable online mean and variance", ("streaming", "statistics"),
                 '''class OnlineVariance:
    """Single-pass Welford accumulator avoiding catastrophic cancellation."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def sample_variance(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0
'''),
                ("data-int-reservoir", "Algorithm R uniform reservoir sampling over unbounded streams", ("sampling", "streaming"),
                 '''import random
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def reservoir_sample(stream: Iterable[T], k: int, seed: int = 42) -> list[T]:
    """Select k items uniformly at random in one pass without knowing stream length."""
    rng = random.Random(seed)
    sample: list[T] = []
    for idx, item in enumerate(stream):
        if idx < k:
            sample.append(item)
        else:
            j = rng.randint(0, idx)
            if j < k:
                sample[j] = item
    return sample
'''),
                ("data-int-hash-join", "In-memory hash equi-join supporting inner and left outer modes", ("join", "relational"),
                 '''def hash_join(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    on: str,
    how: str = "inner",
) -> list[dict[str, object]]:
    """Build probe table on right[on] and stream left rows."""
    index: dict[object, list[dict[str, object]]] = {}
    for r in right:
        index.setdefault(r.get(on), []).append(r)
    joined: list[dict[str, object]] = []
    for lrow in left:
        matches = index.get(lrow.get(on))
        if matches:
            for rrow in matches:
                joined.append({**rrow, **lrow})
        elif how == "left":
            joined.append(dict(lrow))
    return joined
'''),
                ("data-int-ewma-anomaly", "Exponentially weighted z-score spike detector", ("anomaly_detection", "time_series"),
                 '''import math


def detect_ewma_spikes(series: list[float], alpha: float = 0.2, z_thresh: float = 3.0) -> list[int]:
    """Flag indices where point deviates > z_thresh EWMA standard deviations."""
    if not series:
        return []
    mean = series[0]
    var = 1e-6
    anomalies: list[int] = []
    for idx, x in enumerate(series[1:], start=1):
        diff = x - mean
        std = math.sqrt(var)
        if std > 0 and abs(diff) / std > z_thresh:
            anomalies.append(idx)
        mean += alpha * diff
        var = (1.0 - alpha) * (var + alpha * diff * diff)
    return anomalies
'''),
                ("data-int-sessionize", "Clickstream inactivity-gap session windowing", ("windowing", "events"),
                 '''def sessionize_events(events: list[tuple[str, float]], gap_seconds: float = 1800.0) -> dict[str, list[list[float]]]:
    """Group (user_id, timestamp) events into sessions split by > gap_seconds."""
    by_user: dict[str, list[float]] = {}
    for user, ts in events:
        by_user.setdefault(user, []).append(ts)
    sessions: dict[str, list[list[float]]] = {}
    for user, times in by_user.items():
        times.sort()
        cur = [times[0]]
        user_sessions = [cur]
        for ts in times[1:]:
            if ts - cur[-1] > gap_seconds:
                cur = [ts]
                user_sessions.append(cur)
            else:
                cur.append(ts)
        sessions[user] = user_sessions
    return sessions
'''),
                ("data-int-ndjson-batcher", "Byte- and count-bounded NDJSON chunk writer", ("ndjson", "batching"),
                 '''import json
from collections.abc import Iterable, Iterator


def batch_ndjson(rows: Iterable[dict[str, object]], max_rows: int = 500, max_bytes: int = 262_144) -> Iterator[bytes]:
    """Yield UTF-8 NDJSON chunks respecting both row-count and byte ceilings."""
    buf: list[bytes] = []
    size = 0
    for row in rows:
        encoded = (json.dumps(row, separators=(",", ":")) + "\\n").encode("utf-8")
        if buf and (len(buf) >= max_rows or size + len(encoded) > max_bytes):
            yield b"".join(buf)
            buf = []
            size = 0
        buf.append(encoded)
        size += len(encoded)
    if buf:
        yield b"".join(buf)
'''),
                ("data-int-csr-matrix", "Compressed Sparse Row (CSR) matrix-vector product", ("sparse_matrix", "linear_algebra"),
                 '''class CSRMatrix:
    """Compact CSR sparse matrix supporting O(nnz) matrix-vector multiplication."""

    def __init__(self, rows: int, cols: int, triplets: list[tuple[int, int, float]]) -> None:
        self.rows = rows
        self.cols = cols
        ordered = sorted(triplets, key=lambda t: (t[0], t[1]))
        self.indptr = [0] * (rows + 1)
        self.indices: list[int] = []
        self.data: list[float] = []
        for r, c, val in ordered:
            if val == 0.0:
                continue
            self.indices.append(c)
            self.data.append(val)
            self.indptr[r + 1] += 1
        for r in range(rows):
            self.indptr[r + 1] += self.indptr[r]

    def matvec(self, vec: list[float]) -> list[float]:
        if len(vec) != self.cols:
            raise ValueError("dimension mismatch")
        out = [0.0] * self.rows
        for r in range(self.rows):
            acc = 0.0
            for idx in range(self.indptr[r], self.indptr[r + 1]):
                acc += self.data[idx] * vec[self.indices[idx]]
            out[r] = acc
        return out
'''),
                ("data-int-schema-diff", "Backward-compatibility checker for evolving tabular schemas", ("schema_evolution", "governance"),
                 '''def check_schema_evolution(old: dict[str, tuple[str, bool]], new: dict[str, tuple[str, bool]]) -> list[str]:
    """Return breaking changes between {col: (dtype, nullable)} definitions."""
    issues: list[str] = []
    for col, (old_type, old_nullable) in old.items():
        if col not in new:
            issues.append(f"removed_column:{col}")
            continue
        new_type, new_nullable = new[col]
        if new_type != old_type:
            issues.append(f"type_changed:{col}:{old_type}->{new_type}")
        if old_nullable and not new_nullable:
            issues.append(f"nullability_tightened:{col}")
    for col, (_t, nullable) in new.items():
        if col not in old and not nullable:
            issues.append(f"added_required_column:{col}")
    return issues
'''),
                ("data-int-iqr-winsorize", "Interquartile-range Winsorizer for heavy-tailed distributions", ("outliers", "robust_stats"),
                 '''def winsorize_iqr(values: list[float], k: float = 1.5) -> list[float]:
    """Clamp values outside [Q1 - k*IQR, Q3 + k*IQR]."""
    if len(values) < 4:
        return list(values)
    ordered = sorted(values)
    n = len(ordered)
    q1 = ordered[n // 4]
    q3 = ordered[(3 * n) // 4]
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return [min(max(v, lo), hi) for v in values]
'''),
            ],
            "advanced": [
                ("data-adv-bloom-filter", "Double-hashing Bloom filter with configurable false-positive sizing", ("probabilistic", "bloom_filter"),
                 '''import hashlib
import math


class BloomFilter:
    """Bit-array Bloom filter using Kirsch-Mitzenmacher double hashing."""

    def __init__(self, expected_items: int, fp_rate: float = 0.01) -> None:
        m = max(64, int(-expected_items * math.log(fp_rate) / (math.log(2) ** 2)))
        self.num_bits = m
        self.num_hashes = max(1, int((m / max(expected_items, 1)) * math.log(2)))
        self._words = bytearray((m + 7) // 8)

    def _positions(self, item: str) -> list[int]:
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:16], "little") or 1
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def add(self, item: str) -> None:
        for bit in self._positions(item):
            self._words[bit >> 3] |= 1 << (bit & 7)

    def __contains__(self, item: str) -> bool:
        return all(self._words[bit >> 3] & (1 << (bit & 7)) for bit in self._positions(item))
'''),
                ("data-adv-count-min-sketch", "Conservative-update Count-Min Sketch heavy-hitter frequency estimator", ("sketching", "streaming"),
                 '''import hashlib


class CountMinSketch:
    """Sub-linear frequency table with conservative updates to reduce overcount."""

    def __init__(self, width: int = 1024, depth: int = 4) -> None:
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def _col(self, row: int, key: str) -> int:
        h = hashlib.blake2b(f"{row}:{key}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.width

    def increment(self, key: str, count: int = 1) -> int:
        cols = [self._col(r, key) for r in range(self.depth)]
        floor = min(self.table[r][c] for r, c in enumerate(cols))
        target = floor + count
        for r, c in enumerate(cols):
            if self.table[r][c] < target:
                self.table[r][c] = target
        return target

    def estimate(self, key: str) -> int:
        return min(self.table[r][self._col(r, key)] for r in range(self.depth))
'''),
                ("data-adv-hyperloglog", "HyperLogLog distinct-cardinality estimator with small-range linear counting", ("hyperloglog", "cardinality"),
                 '''import hashlib
import math


class HyperLogLog:
    """2^p-register HyperLogLog distinct element counter."""

    def __init__(self, p: int = 10) -> None:
        if not 4 <= p <= 16:
            raise ValueError("p must be in [4, 16]")
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m

    def add(self, value: str) -> None:
        x = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
        idx = x >> (64 - self.p)
        w = (x << self.p) & ((1 << 64) - 1)
        rho = (64 - self.p - w.bit_length()) + 1 if w else (64 - self.p + 1)
        if rho > self.registers[idx]:
            self.registers[idx] = rho

    def count(self) -> int:
        alpha = 0.7213 / (1.0 + 1.079 / self.m)
        harmonic = sum(2.0 ** (-r) for r in self.registers)
        raw = alpha * (self.m ** 2) / harmonic
        zeros = self.registers.count(0)
        if raw <= 2.5 * self.m and zeros > 0:
            return round(self.m * math.log(self.m / zeros))
        return round(raw)
'''),
                ("data-adv-columnar-codec", "Dictionary + RLE columnar block encoder/decoder", ("columnar", "compression"),
                 '''def encode_column(values: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Dictionary-encode string column followed by run-length compression of codes."""
    vocab: list[str] = []
    code_of: dict[str, int] = {}
    runs: list[tuple[int, int]] = []
    for val in values:
        code = code_of.get(val)
        if code is None:
            code = len(vocab)
            vocab.append(val)
            code_of[val] = code
        if runs and runs[-1][0] == code:
            runs[-1] = (code, runs[-1][1] + 1)
        else:
            runs.append((code, 1))
    return vocab, runs


def decode_column(vocab: list[str], runs: list[tuple[int, int]]) -> list[str]:
    out: list[str] = []
    for code, length in runs:
        out.extend([vocab[code]] * length)
    return out
'''),
                ("data-adv-external-kway-merge", "Heap-driven k-way sorted run merger with key projection", ("external_sort", "streaming"),
                 '''import heapq
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def kway_merge_sorted(runs: list[Iterable[T]], key: Callable[[T], object]) -> Iterator[T]:
    """Merge sorted iterables lazily using a size-k min-heap."""
    iters = [iter(r) for r in runs]
    heap: list[tuple[object, int, T]] = []
    for run_id, it in enumerate(iters):
        try:
            first = next(it)
            heapq.heappush(heap, (key(first), run_id, first))
        except StopIteration:
            continue
    while heap:
        _k, run_id, item = heapq.heappop(heap)
        yield item
        try:
            nxt = next(iters[run_id])
            heapq.heappush(heap, (key(nxt), run_id, nxt))
        except StopIteration:
            continue
'''),
                ("data-adv-watermark-window", "Event-time tumbling window aggregator with bounded lateness watermark", ("streaming", "watermarks"),
                 '''class TumblingWindowSum:
    """Aggregate event-time (ts, value) pairs and emit closed buckets on watermark advance."""

    def __init__(self, window_s: int, allowed_lateness_s: int = 5) -> None:
        self.window_s = window_s
        self.lateness_s = allowed_lateness_s
        self.max_event_ts = 0
        self.buckets: dict[int, float] = {}
        self.dropped_late = 0

    def ingest(self, event_ts: int, value: float) -> list[tuple[int, float]]:
        self.max_event_ts = max(self.max_event_ts, event_ts)
        watermark = self.max_event_ts - self.lateness_s
        bucket = (event_ts // self.window_s) * self.window_s
        if bucket + self.window_s <= watermark:
            self.dropped_late += 1
            return []
        self.buckets[bucket] = self.buckets.get(bucket, 0.0) + value
        ready = sorted(b for b in self.buckets if b + self.window_s <= watermark)
        return [(b, self.buckets.pop(b)) for b in ready]
'''),
                ("data-adv-dag-lineage", "Deterministic data-pipeline DAG runner with content-hash caching", ("dag", "lineage", "caching"),
                 '''import hashlib
import json
from typing import Callable


class CachedPipelineDAG:
    """Run upstream transforms once and skip recomputation when input hashes match."""

    def __init__(self) -> None:
        self.nodes: dict[str, tuple[list[str], Callable[..., object]]] = {}
        self.cache: dict[tuple[str, str], object] = {}

    def register(self, name: str, deps: list[str], fn: Callable[..., object]) -> None:
        self.nodes[name] = (deps, fn)

    def materialize(self, target: str, roots: dict[str, object]) -> object:
        if target in roots:
            return roots[target]
        deps, fn = self.nodes[target]
        inputs = [self.materialize(d, roots) for d in deps]
        sig = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
        key = (target, sig)
        if key not in self.cache:
            self.cache[key] = fn(*inputs)
        return self.cache[key]
'''),
                ("data-adv-online-cov", "Streaming Pébay/Welford bivariate covariance and Pearson correlation", ("covariance", "streaming"),
                 '''import math


class OnlineCorrelation:
    """Single-pass numerically stable covariance and Pearson r."""

    def __init__(self) -> None:
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.c_xy = 0.0

    def add(self, x: float, y: float) -> None:
        self.n += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.n
        dy = y - self.mean_y
        self.mean_y += dy / self.n
        self.c_xy += dx * (y - self.mean_y)
        self.m2_x += dx * (x - self.mean_x)
        self.m2_y += dy * (y - self.mean_y)

    def pearson_r(self) -> float:
        denom = math.sqrt(self.m2_x * self.m2_y)
        return self.c_xy / denom if denom > 0 else 0.0
'''),
                ("data-adv-zonemap-pruner", "Columnar block min/max zone-map predicate pruner", ("zone_map", "query_pruning"),
                 '''from dataclasses import dataclass


@dataclass(frozen=True)
class BlockStats:
    block_id: int
    min_val: int
    max_val: int
    has_nulls: bool = False


def prune_blocks_range(blocks: list[BlockStats], lo: int | None, hi: int | None) -> list[int]:
    """Return block IDs whose [min_val, max_val] overlaps [lo, hi]."""
    surviving: list[int] = []
    for b in blocks:
        if lo is not None and b.max_val < lo:
            continue
        if hi is not None and b.min_val > hi:
            continue
        surviving.append(b.block_id)
    return surviving
'''),
            ],
            "expert": [
                ("data-exp-tdigest", "Merging-centroid t-Digest tail-accurate streaming quantile sketch", ("tdigest", "quantiles", "sketching"),
                 '''import math


class MergingTDigest:
    """Bounded-memory centroid sketch with arcsine scale k_1 tail resolution."""

    def __init__(self, compression: float = 50.0) -> None:
        self.compression = compression
        self.centroids: list[tuple[float, float]] = []
        self._buffer: list[float] = []
        self.total_weight = 0.0

    def add(self, x: float, w: float = 1.0) -> None:
        self._buffer.append(x)
        if len(self._buffer) >= 256:
            self._compress()

    def _k(self, q: float) -> float:
        return (self.compression / (2.0 * math.pi)) * math.asin(2.0 * q - 1.0)

    def _compress(self) -> None:
        items = self.centroids + [(v, 1.0) for v in self._buffer]
        self._buffer.clear()
        if not items:
            return
        items.sort(key=lambda t: t[0])
        total = sum(w for _, w in items)
        merged: list[tuple[float, float]] = []
        cur_mean, cur_w = items[0]
        cum = 0.0
        for mean, w in items[1:]:
            q0 = cum / total
            q1 = (cum + cur_w + w) / total
            if self._k(q1) - self._k(q0) <= 1.0:
                cur_mean = (cur_mean * cur_w + mean * w) / (cur_w + w)
                cur_w += w
            else:
                merged.append((cur_mean, cur_w))
                cum += cur_w
                cur_mean, cur_w = mean, w
        merged.append((cur_mean, cur_w))
        self.centroids = merged
        self.total_weight = total

    def quantile(self, q: float) -> float:
        self._compress()
        if not self.centroids:
            return 0.0
        target = min(max(q, 0.0), 1.0) * self.total_weight
        cum = 0.0
        for mean, w in self.centroids:
            if cum + w >= target:
                return mean
            cum += w
        return self.centroids[-1][0]
'''),
                ("data-exp-misra-gries", "Deterministic Misra-Gries frequent-items heavy-hitter summary", ("heavy_hitters", "streaming"),
                 '''from __future__ import annotations


class MisraGriesSummary:
    """Guarantees every item with frequency > N / k is retained in <= k-1 counters."""

    def __init__(self, k: int) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        self.k = k
        self.counters: dict[str, int] = {}

    def update(self, item: str) -> None:
        if item in self.counters:
            self.counters[item] += 1
        elif len(self.counters) < self.k - 1:
            self.counters[item] = 1
        else:
            dead = [key for key, c in self.counters.items() if c == 1]
            for key in self.counters:
                self.counters[key] -= 1
            for key in dead:
                del self.counters[key]

    def merge(self, other: MisraGriesSummary) -> None:
        for key, count in other.counters.items():
            self.counters[key] = self.counters.get(key, 0) + count
        while len(self.counters) >= self.k:
            drop = min(self.counters.values())
            self.counters = {k: v - drop for k, v in self.counters.items() if v > drop}
'''),
                ("data-exp-arrow-validity-bitmap", "Null-validity bitmap + offset buffer builder for variable-width string arrays", ("columnar", "arrow", "buffers"),
                 '''class Utf8ColumnBuffer:
    """Arrow-layout packed validity bitmap, int32 offsets, and contiguous UTF-8 bytes."""

    def __init__(self) -> None:
        self.length = 0
        self.null_count = 0
        self.validity = bytearray()
        self.offsets = [0]
        self.values = bytearray()

    def append(self, value: str | None) -> None:
        byte_idx = self.length >> 3
        if byte_idx >= len(self.validity):
            self.validity.append(0)
        if value is not None:
            self.validity[byte_idx] |= 1 << (self.length & 7)
            raw = value.encode("utf-8")
            self.values.extend(raw)
        else:
            self.null_count += 1
        self.offsets.append(len(self.values))
        self.length += 1

    def get(self, index: int) -> str | None:
        if not 0 <= index < self.length:
            raise IndexError(index)
        if not (self.validity[index >> 3] & (1 << (index & 7))):
            return None
        start, end = self.offsets[index], self.offsets[index + 1]
        return bytes(self.values[start:end]).decode("utf-8")
'''),
                ("data-exp-cdc-merkle-reconcile", "Hash-partitioned Merkle bucket reconciler for CDC table sync", ("cdc", "merkle", "reconciliation"),
                 '''import hashlib


def bucket_digests(rows: dict[str, str], num_buckets: int = 64) -> list[str]:
    """Partition primary_key->row_hash into deterministic Merkle leaf digests."""
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(num_buckets)]
    for pk, row_hash in rows.items():
        b = int.from_bytes(hashlib.blake2b(pk.encode(), digest_size=4).digest(), "little") % num_buckets
        buckets[b].append((pk, row_hash))
    digests: list[str] = []
    for items in buckets:
        items.sort()
        h = hashlib.sha256()
        for pk, rh in items:
            h.update(f"{pk}={rh}\\n".encode("utf-8"))
        digests.append(h.hexdigest())
    return digests


def mismatched_buckets(source_digests: list[str], replica_digests: list[str]) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(source_digests, replica_digests)) if a != b]
'''),
                ("data-exp-minhash-lsh-band", "Banded MinHash LSH document deduplicator with Jaccard verification", ("minhash", "lsh", "dedup"),
                 '''import hashlib


class BandedMinHashIndex:
    """16-band x 4-row 5-gram MinHash LSH index with exact signature verification."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.bands = 16
        self.rows = 4
        self.sigs: dict[str, tuple[int, ...]] = {}
        self.buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}

    def _signature(self, text: str) -> tuple[int, ...]:
        grams = {text[i : i + 5] for i in range(max(len(text) - 4, 1))}
        num = self.bands * self.rows
        best = [(1 << 64) - 1] * num
        for g in grams:
            seed = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
            for k in range(num):
                val = ((seed * (0x9E3779B97F4A7C15 + 2 * k)) ^ (seed >> 29)) & ((1 << 64) - 1)
                if val < best[k]:
                    best[k] = val
        return tuple(best)

    def insert_if_unique(self, doc_id: str, text: str) -> bool:
        sig = self._signature(text)
        for b in range(self.bands):
            band_key = (b, sig[b * self.rows : (b + 1) * self.rows])
            for cand in self.buckets.get(band_key, ()):
                other = self.sigs[cand]
                sim = sum(1 for a, c in zip(sig, other) if a == c) / len(sig)
                if sim >= self.threshold:
                    return False
        self.sigs[doc_id] = sig
        for b in range(self.bands):
            band_key = (b, sig[b * self.rows : (b + 1) * self.rows])
            self.buckets.setdefault(band_key, []).append(doc_id)
        return True
'''),
                ("data-exp-roaring-run-chunk", "16-bit sorted-array vs bitmap hybrid integer set container", ("roaring_bitmap", "indexing"),
                 '''class HybridContainer:
    """Switches between sorted uint16 array (<4096 ints) and 65536-bit bitmap."""

    ARRAY_LIMIT = 4096

    def __init__(self) -> None:
        self.is_bitmap = False
        self.array: list[int] = []
        self.bitmap: bytearray | None = None
        self.cardinality = 0

    def add(self, value: int) -> None:
        if not 0 <= value < 65536:
            raise ValueError("uint16 out of range")
        if self.is_bitmap:
            assert self.bitmap is not None
            mask = 1 << (value & 7)
            if not (self.bitmap[value >> 3] & mask):
                self.bitmap[value >> 3] |= mask
                self.cardinality += 1
            return
        from bisect import bisect_left

        pos = bisect_left(self.array, value)
        if pos < len(self.array) and self.array[pos] == value:
            return
        self.array.insert(pos, value)
        self.cardinality += 1
        if len(self.array) >= self.ARRAY_LIMIT:
            bm = bytearray(8192)
            for v in self.array:
                bm[v >> 3] |= 1 << (v & 7)
            self.bitmap = bm
            self.array.clear()
            self.is_bitmap = True
'''),
                ("data-exp-scd2-merger", "Slowly Changing Dimension Type-2 valid-from/valid-to history merger", ("scd2", "warehousing", "temporal"),
                 '''def apply_scd2_changes(
    current: list[dict[str, object]],
    incoming: list[dict[str, object]],
    business_key: str,
    tracked_cols: tuple[str, ...],
    as_of: str,
) -> list[dict[str, object]]:
    """Close changed active dimension versions at `as_of` and append fresh active rows."""
    by_key: dict[object, dict[str, object]] = {
        r[business_key]: r for r in current if r.get("is_current") is True
    }
    history = [dict(r) for r in current]
    for row in incoming:
        bk = row[business_key]
        active = by_key.get(bk)
        if active is None:
            history.append({**row, "valid_from": as_of, "valid_to": None, "is_current": True})
        elif any(active.get(c) != row.get(c) for c in tracked_cols):
            for h in history:
                if h[business_key] == bk and h.get("is_current") is True:
                    h["valid_to"] = as_of
                    h["is_current"] = False
            history.append({**row, "valid_from": as_of, "valid_to": None, "is_current": True})
    return history
'''),
                ("data-exp-vectorized-expr-vm", "Columnar batch expression bytecode evaluator", ("vectorized_execution", "query_engine"),
                 '''def eval_columnar_rpn(columns: dict[str, list[float]], rpn_tokens: list[str], rows: int) -> list[float]:
    """Evaluate RPN arithmetic program (+, -, *, /) column-at-a-time over `rows`."""
    stack: list[list[float]] = []
    ops = {
        "+": lambda a, b: [x + y for x, y in zip(a, b)],
        "-": lambda a, b: [x - y for x, y in zip(a, b)],
        "*": lambda a, b: [x * y for x, y in zip(a, b)],
        "/": lambda a, b: [x / y if y != 0.0 else 0.0 for x, y in zip(a, b)],
    }
    for tok in rpn_tokens:
        if tok in ops:
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(ops[tok](lhs, rhs))
        elif tok in columns:
            stack.append(list(columns[tok]))
        else:
            val = float(tok)
            stack.append([val] * rows)
    if len(stack) != 1:
        raise ValueError("malformed RPN expression")
    return stack[0]
'''),
                ("data-exp-consistent-reservoir-weighted", "Efraimidis-Spirakis weighted stream reservoir sampler", ("weighted_sampling", "streaming"),
                 '''import heapq
import math
import random
from collections.abc import Iterable


def weighted_reservoir_sample(stream: Iterable[tuple[str, float]], k: int, seed: int = 42) -> list[str]:
    """Proportional-to-weight without-replacement stream sample via u^(1/w) keys."""
    rng = random.Random(seed)
    heap: list[tuple[float, str]] = []
    for item_id, weight in stream:
        if weight <= 0:
            continue
        u = max(rng.random(), 1e-15)
        key = math.log(u) / weight
        if len(heap) < k:
            heapq.heappush(heap, (key, item_id))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item_id))
    return [item_id for _, item_id in sorted(heap, reverse=True)]
'''),
            ],
        },
    )

def _build_web_patterns() -> list[PatternSpec]:
    return _domain_specs(
        "web",
        {
            "beginner": [
                ("web-beg-query-params", "Multi-value URL query string parser with percent-decoding", ("http", "url"),
                 '''from urllib.parse import unquote_plus


def parse_query_params(query: str) -> dict[str, list[str]]:
    """Parse raw URL query string into {key: [values]} preserving repeated keys."""
    cleaned = query.lstrip("?")
    if not cleaned:
        return {}
    out: dict[str, list[str]] = {}
    for pair in cleaned.split("&"):
        if not pair:
            continue
        key, _, val = pair.partition("=")
        out.setdefault(unquote_plus(key), []).append(unquote_plus(val))
    return out
'''),
                ("web-beg-problem-details", "RFC 9457 / 7807 application/problem+json payload builder", ("rest", "errors"),
                 '''def build_problem_details(status: int, title: str, detail: str, instance: str = "") -> dict[str, object]:
    """Construct an RFC 9457 Problem Details response body."""
    body: dict[str, object] = {
        "type": f"https://httpstatuses.io/{status}",
        "title": title,
        "status": status,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    return body
'''),
                ("web-beg-accept-negotiation", "Quality-weighted HTTP Accept header media-type negotiator", ("http", "headers"),
                 '''def negotiate_content_type(accept_header: str, supported: list[str]) -> str | None:
    """Pick highest-q media type from Accept header matching `supported`."""
    candidates: list[tuple[float, str]] = []
    for part in accept_header.split(","):
        segments = [s.strip() for s in part.split(";") if s.strip()]
        if not segments:
            continue
        mime = segments[0].lower()
        q = 1.0
        for param in segments[1:]:
            if param.startswith("q="):
                q = float(param[2:])
        candidates.append((q, mime))
    for _q, mime in sorted(candidates, key=lambda x: x[0], reverse=True):
        if mime in supported:
            return mime
        if mime == "*/*" and supported:
            return supported[0]
    return None
'''),
                ("web-beg-slug-permalink", "URL-safe ASCII slug generator with collision suffixing", ("routing", "urls"),
                 '''import re


def unique_slug(title: str, existing: set[str]) -> str:
    """Normalize title to lowercase hyphenated slug and disambiguate collisions."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "item"
    if base not in existing:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"
'''),
                ("web-beg-cookie-jar", "HTTP Cookie header parser and Set-Cookie serializer", ("cookies", "http"),
                 '''def parse_cookie_header(header: str) -> dict[str, str]:
    """Parse HTTP Cookie request header into name->value dictionary."""
    cookies: dict[str, str] = {}
    for chunk in header.split(";"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        cookies[k.strip()] = v.strip().strip('"')
    return cookies


def format_secure_cookie(name: str, value: str, max_age: int = 3600) -> str:
    return f"{name}={value}; Max-Age={max_age}; Path=/; HttpOnly; Secure; SameSite=Lax"
'''),
                ("web-beg-pagination-links", "RFC 8288 Web Linking header formatter for paginated APIs", ("pagination", "rest"),
                 '''def build_link_header(base_url: str, page: int, per_page: int, total_items: int) -> str:
    """Build HTTP Link header with first, prev, next, and last relations."""
    last_page = max(1, (total_items + per_page - 1) // per_page)
    rels: list[tuple[str, int]] = [("first", 1)]
    if page > 1:
        rels.append(("prev", page - 1))
    if page < last_page:
        rels.append(("next", page + 1))
    rels.append(("last", last_page))
    return ", ".join(f'<{base_url}?page={p}&per_page={per_page}>; rel="{r}"' for r, p in rels)
'''),
                ("web-beg-bearer-extract", "Strict Authorization Bearer token extractor", ("auth", "headers"),
                 '''def extract_bearer_token(headers: dict[str, str]) -> str | None:
    """Extract token from case-insensitive Authorization: Bearer <token> header."""
    raw = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
    scheme, _, token = raw.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip() or " " in token.strip():
        return None
    return token.strip()
'''),
                ("web-beg-cache-control", "HTTP Cache-Control directive builder", ("caching", "http"),
                 '''def cache_control_header(*, public: bool = False, max_age: int = 0, stale_while_revalidate: int = 0, no_store: bool = False) -> str:
    """Render deterministic HTTP Cache-Control header value."""
    if no_store:
        return "no-store, max-age=0"
    parts = ["public" if public else "private", f"max-age={max(0, max_age)}"]
    if stale_while_revalidate > 0:
        parts.append(f"stale-while-revalidate={stale_while_revalidate}")
    return ", ".join(parts)
'''),
                ("web-beg-form-validator", "Declarative contact/signup payload validator returning field errors", ("validation", "forms"),
                 '''def validate_signup_payload(payload: dict[str, object]) -> dict[str, str]:
    """Validate email, display_name, and age bounds; return {field: message}."""
    errors: dict[str, str] = {}
    email = str(payload.get("email") or "").strip()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        errors["email"] = "must be a valid email address"
    name = str(payload.get("display_name") or "").strip()
    if not 2 <= len(name) <= 60:
        errors["display_name"] = "must be between 2 and 60 characters"
    age = payload.get("age")
    if not isinstance(age, int) or not 13 <= age <= 130:
        errors["age"] = "must be an integer between 13 and 130"
    return errors
'''),
            ],
            "intermediate": [
                ("web-int-etag-precondition", "Weak/strong ETag If-None-Match and If-Match evaluator", ("etag", "http_caching"),
                 '''import hashlib


def compute_etag(body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()[:24]
    return f'"{digest}"'


def is_not_modified(if_none_match: str | None, current_etag: str) -> bool:
    if not if_none_match:
        return False
    candidates = [t.strip() for t in if_none_match.split(",")]
    if "*" in candidates:
        return True
    norm_cur = current_etag.removeprefix("W/")
    return any(c.removeprefix("W/") == norm_cur for c in candidates)
'''),
                ("web-int-cors-preflight", "Allowlist-driven CORS preflight header resolver", ("cors", "middleware"),
                 '''def resolve_cors_headers(
    origin: str | None,
    method: str,
    allowed_origins: set[str],
    allowed_methods: tuple[str, ...] = ("GET", "POST", "PUT", "DELETE"),
) -> dict[str, str]:
    """Return CORS response headers only when Origin is explicitly allowlisted."""
    if not origin or origin not in allowed_origins or method.upper() not in allowed_methods:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": ", ".join(allowed_methods),
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }
'''),
                ("web-int-token-bucket-ip", "Per-client token-bucket HTTP rate limiter", ("rate_limiting", "middleware"),
                 '''class ClientRateLimiter:
    """In-memory per-IP token bucket with monotonic refill."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self.rate = rate_per_sec
        self.burst = float(burst)
        self._state: dict[str, tuple[float, float]] = {}

    def allow(self, client_id: str, now: float) -> tuple[bool, float]:
        tokens, last = self._state.get(client_id, (self.burst, now))
        tokens = min(self.burst, tokens + max(0.0, now - last) * self.rate)
        if tokens >= 1.0:
            self._state[client_id] = (tokens - 1.0, now)
            return True, 0.0
        self._state[client_id] = (tokens, now)
        retry_after = (1.0 - tokens) / max(self.rate, 1e-9)
        return False, retry_after
'''),
                ("web-int-sse-formatter", "Server-Sent Events (SSE) frame encoder and stream parser", ("sse", "streaming"),
                 '''def encode_sse(data: str, *, event: str = "", event_id: str = "", retry_ms: int | None = None) -> str:
    """Format an SSE message frame preserving multi-line payloads."""
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    if event:
        lines.append(f"event: {event}")
    if retry_ms is not None:
        lines.append(f"retry: {retry_ms}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\\n".join(lines) + "\\n\\n"
'''),
                ("web-int-route-trie", "Parameterized URL path trie supporting :param and *wildcard segments", ("routing", "trie"),
                 '''class PathRouter:
    """Segment-trie HTTP router resolving parameterized routes like /orgs/:org/repos/*path."""

    def __init__(self) -> None:
        self._root: dict[str, object] = {}

    def add(self, method: str, pattern: str, handler_name: str) -> None:
        node = self._root.setdefault(method.upper(), {})
        assert isinstance(node, dict)
        for seg in [s for s in pattern.strip("/").split("/") if s]:
            key = ":" if seg.startswith(":") else ("*" if seg.startswith("*") else seg)
            node = node.setdefault(key, {"_param": seg[1:] if key in {":", "*"} else ""})  # type: ignore[assignment]
        node["_handler"] = handler_name  # type: ignore[index]

    def match(self, method: str, path: str) -> tuple[str, dict[str, str]] | None:
        node = self._root.get(method.upper())
        if not isinstance(node, dict):
            return None
        segs = [s for s in path.strip("/").split("/") if s]
        params: dict[str, str] = {}
        for idx, seg in enumerate(segs):
            if seg in node:
                node = node[seg]  # type: ignore[assignment]
            elif ":" in node:
                node = node[":"]  # type: ignore[assignment]
                params[str(node["_param"])] = seg
            elif "*" in node:
                node = node["*"]  # type: ignore[assignment]
                params[str(node["_param"])] = "/".join(segs[idx:])
                break
            else:
                return None
        handler = node.get("_handler")
        return (str(handler), params) if handler else None
'''),
                ("web-int-idempotency-store", "Idempotency-Key request deduplicator with payload fingerprint check", ("idempotency", "payments", "rest"),
                 '''import hashlib


class IdempotencyStore:
    """Prevent duplicate side effects and detect key reuse across mutated payloads."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, int, dict[str, object]]] = {}

    @staticmethod
    def _fingerprint(body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    def lookup_or_begin(self, key: str, body: bytes) -> tuple[str, tuple[int, dict[str, object]] | None]:
        fp = self._fingerprint(body)
        existing = self._records.get(key)
        if existing is None:
            return "miss", None
        saved_fp, status, resp = existing
        if saved_fp != fp:
            raise ValueError("Idempotency-Key reused with different request payload")
        return "hit", (status, resp)

    def commit(self, key: str, body: bytes, status: int, response: dict[str, object]) -> None:
        self._records[key] = (self._fingerprint(body), status, dict(response))
'''),
                ("web-int-multipart-parser", "RFC 7578 multipart/form-data boundary frame parser", ("multipart", "http"),
                 '''def parse_multipart_form(body: bytes, boundary: str) -> list[dict[str, object]]:
    """Split multipart/form-data body into header + content parts."""
    delim = f"--{boundary}".encode("ascii")
    parts_out: list[dict[str, object]] = []
    for chunk in body.split(delim)[1:]:
        if chunk.startswith(b"--"):
            break
        chunk = chunk.lstrip(b"\\r\\n")
        if b"\\r\\n\\r\\n" not in chunk:
            continue
        raw_headers, payload = chunk.split(b"\\r\\n\\r\\n", 1)
        if payload.endswith(b"\\r\\n"):
            payload = payload[:-2]
        headers: dict[str, str] = {}
        for line in raw_headers.decode("latin-1").split("\\r\\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        parts_out.append({"headers": headers, "body": payload})
    return parts_out
'''),
                ("web-int-range-header", "HTTP Byte-Range parser for resumable downloads (206 Partial Content)", ("http_range", "streaming"),
                 '''def parse_byte_range(range_header: str, total_size: int) -> tuple[int, int] | None:
    """Resolve single `bytes=start-end` header into inclusive (start, end) offsets."""
    if not range_header.startswith("bytes=") or total_size <= 0:
        return None
    spec = range_header[6:].strip()
    if "," in spec or "-" not in spec:
        return None
    left, right = spec.split("-", 1)
    if not left:
        suffix = int(right)
        if suffix <= 0:
            return None
        return max(0, total_size - suffix), total_size - 1
    start = int(left)
    end = int(right) if right else total_size - 1
    if start >= total_size or start > end:
        return None
    return start, min(end, total_size - 1)
'''),
                ("web-int-middleware-chain", "Composable Onion-style WSGI/ASGI-like request middleware pipeline", ("middleware", "composition"),
                 '''from collections.abc import Callable

Request = dict[str, object]
Response = dict[str, object]
Handler = Callable[[Request], Response]
Middleware = Callable[[Request, Handler], Response]


def compose_middlewares(middlewares: list[Middleware], terminal: Handler) -> Handler:
    """Wrap `terminal` handler in `middlewares` from outermost to innermost."""
    chain = terminal
    for mw in reversed(middlewares):
        prev = chain
        chain = lambda req, _mw=mw, _next=prev: _mw(req, _next)
    return chain
'''),
            ],
            "advanced": [
                ("web-adv-circuit-breaker", "Three-state (CLOSED/OPEN/HALF_OPEN) upstream HTTP circuit breaker", ("resilience", "circuit_breaker"),
                 '''class CircuitBreaker:
    """Fail-fast upstream call guard with half-open probe recovery."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = 0.0

    def before_call(self, now: float) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if now - self.opened_at >= self.recovery_timeout_s:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.state == "HALF_OPEN" or self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = now
'''),
                ("web-adv-sliding-window-counter", "Cloudflare-style weighted sliding-window HTTP rate counter", ("rate_limiting", "distributed"),
                 '''class WeightedSlidingRateLimiter:
    """Approximates sliding window in O(1) state using previous + current fixed buckets."""

    def __init__(self, limit: int, window_s: int = 60) -> None:
        self.limit = limit
        self.window_s = window_s
        self._counters: dict[tuple[str, int], int] = {}

    def allow(self, key: str, now: float) -> bool:
        cur_bucket = int(now // self.window_s)
        prev_bucket = cur_bucket - 1
        elapsed_ratio = (now % self.window_s) / self.window_s
        prev_count = self._counters.get((key, prev_bucket), 0)
        cur_count = self._counters.get((key, cur_bucket), 0)
        estimated = prev_count * (1.0 - elapsed_ratio) + cur_count
        if estimated + 1.0 > self.limit:
            return False
        self._counters[(key, cur_bucket)] = cur_count + 1
        return True
'''),
                ("web-adv-webhook-dispatcher", "Reliable outbound webhook outbox with exponential backoff and dead-lettering", ("webhooks", "outbox", "retries"),
                 '''from dataclasses import dataclass


@dataclass
class WebhookEvent:
    event_id: str
    url: str
    payload: str
    attempts: int = 0
    next_attempt_at: float = 0.0
    status: str = "pending"


class WebhookOutbox:
    """Schedule webhook deliveries with bounded retries and dead-letter quarantine."""

    def __init__(self, max_attempts: int = 5, base_delay_s: float = 2.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.events: dict[str, WebhookEvent] = {}

    def due(self, now: float) -> list[WebhookEvent]:
        return [e for e in self.events.values() if e.status == "pending" and e.next_attempt_at <= now]

    def on_result(self, event_id: str, ok: bool, now: float) -> None:
        ev = self.events[event_id]
        if ok:
            ev.status = "delivered"
            return
        ev.attempts += 1
        if ev.attempts >= self.max_attempts:
            ev.status = "dead_letter"
        else:
            ev.next_attempt_at = now + self.base_delay_s * (2 ** (ev.attempts - 1))
'''),
                ("web-adv-websocket-frame", "RFC 6455 WebSocket binary frame header encoder and unmasking decoder", ("websocket", "rfc6455", "framing"),
                 '''def decode_ws_frame(buf: bytes) -> tuple[int, bytes, int] | None:
    """Return (opcode, payload, consumed_bytes) from an RFC 6455 frame, or None if incomplete."""
    if len(buf) < 2:
        return None
    opcode = buf[0] & 0x0F
    masked = bool(buf[1] & 0x80)
    length = buf[1] & 0x7F
    pos = 2
    if length == 126:
        if len(buf) < 4:
            return None
        length = int.from_bytes(buf[2:4], "big")
        pos = 4
    elif length == 127:
        if len(buf) < 10:
            return None
        length = int.from_bytes(buf[2:10], "big")
        pos = 10
    mask_key = b""
    if masked:
        if len(buf) < pos + 4:
            return None
        mask_key = buf[pos : pos + 4]
        pos += 4
    if len(buf) < pos + length:
        return None
    raw = buf[pos : pos + length]
    if masked:
        raw = bytes(b ^ mask_key[i & 3] for i, b in enumerate(raw))
    return opcode, raw, pos + length
'''),
                ("web-adv-graphql-depth-limiter", "GraphQL query selection-depth and alias-amplification guard", ("graphql", "dos_protection"),
                 '''def check_graphql_limits(query: str, *, max_depth: int = 6, max_aliases: int = 15) -> tuple[bool, str]:
    """Reject overly nested or alias-amplified GraphQL documents before execution."""
    depth = 0
    peak_depth = 0
    in_str = False
    for ch in query:
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
                peak_depth = max(peak_depth, depth)
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return False, "unbalanced_braces"
    if depth != 0:
        return False, "unbalanced_braces"
    if peak_depth > max_depth:
        return False, f"max_depth_exceeded:{peak_depth}"
    aliases = query.count(":")
    if aliases > max_aliases:
        return False, f"max_aliases_exceeded:{aliases}"
    return True, "ok"
'''),
                ("web-adv-json-patch-rfc6902", "RFC 6902 JSON Patch atomic add/replace/remove/test applicator", ("json_patch", "rfc6902"),
                 '''import copy


def apply_json_patch(doc: dict[str, object], ops: list[dict[str, object]]) -> dict[str, object]:
    """Apply RFC 6902 operations atomically; return mutated copy or raise ValueError."""
    working = copy.deepcopy(doc)
    for op in ops:
        action = str(op["op"])
        tokens = [t.replace("~1", "/").replace("~0", "~") for t in str(op["path"]).lstrip("/").split("/") if t]
        parent: object = working
        for seg in tokens[:-1]:
            assert isinstance(parent, dict)
            parent = parent[seg]
        leaf = tokens[-1]
        assert isinstance(parent, dict)
        if action == "test":
            if parent.get(leaf) != op.get("value"):
                raise ValueError(f"test failed at {leaf!r}")
        elif action in {"add", "replace"}:
            if action == "replace" and leaf not in parent:
                raise ValueError(f"missing target {leaf!r}")
            parent[leaf] = op["value"]
        elif action == "remove":
            if leaf not in parent:
                raise ValueError(f"missing target {leaf!r}")
            del parent[leaf]
        else:
            raise ValueError(f"unsupported op {action!r}")
    return working
'''),
                ("web-adv-reverse-proxy-balancer", "Smooth weighted round-robin upstream load balancer (Nginx algorithm)", ("load_balancing", "reverse_proxy"),
                 '''from dataclasses import dataclass


@dataclass
class UpstreamPeer:
    url: str
    weight: int
    current_weight: int = 0
    healthy: bool = True


class SmoothWeightedBalancer:
    """Nginx smooth weighted round-robin avoiding burstiness on high-weight backends."""

    def __init__(self, peers: list[UpstreamPeer]) -> None:
        self.peers = peers

    def pick(self) -> str | None:
        healthy = [p for p in self.peers if p.healthy and p.weight > 0]
        if not healthy:
            return None
        total = sum(p.weight for p in healthy)
        best = healthy[0]
        for p in healthy:
            p.current_weight += p.weight
            if p.current_weight > best.current_weight:
                best = p
        best.current_weight -= total
        return best.url
'''),
                ("web-adv-openapi-validator", "Minimal OpenAPI-style JSON request schema validator with path reporting", ("openapi", "validation"),
                 '''def validate_json_schema(value: object, schema: dict[str, object], path: str = "$") -> list[str]:
    """Validate nested JSON payload against subset of Draft-2020-12 schema."""
    errors: list[str] = []
    expected = schema.get("type")
    type_ok = (
        (expected == "object" and isinstance(value, dict))
        or (expected == "array" and isinstance(value, list))
        or (expected == "string" and isinstance(value, str))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or expected is None
    )
    if not type_ok:
        return [f"{path}: expected {expected}"]
    if isinstance(value, dict):
        for req in schema.get("required", ()):  # type: ignore[union-attr]
            if req not in value:
                errors.append(f"{path}.{req}: required property missing")
        props = schema.get("properties") or {}
        if isinstance(props, dict):
            for k, sub in props.items():
                if k in value and isinstance(sub, dict):
                    errors.extend(validate_json_schema(value[k], sub, f"{path}.{k}"))
    return errors
'''),
                ("web-adv-signed-url", "Expiring HMAC-signed download URL generator and verifier", ("signed_urls", "cdn"),
                 '''import hashlib
import hmac
from urllib.parse import urlencode


def sign_download_url(secret: bytes, path: str, expires_at: int) -> str:
    payload = f"{path}:{expires_at}".encode("utf-8")
    sig = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return f"{path}?{urlencode({'expires': expires_at, 'sig': sig})}"


def verify_download_url(secret: bytes, path: str, expires_at: int, sig: str, now: int) -> bool:
    if now > expires_at:
        return False
    expected = hmac.new(secret, f"{path}:{expires_at}".encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)
'''),
            ],
            "expert": [
                ("web-exp-http2-hpack-table", "RFC 7541 HPACK dynamic header table with FIFO byte-size eviction", ("http2", "hpack", "framing"),
                 '''class HpackDynamicTable:
    """RFC 7541 HPACK dynamic header table accounting 32-byte per-entry overhead."""

    OVERHEAD = 32

    def __init__(self, max_size_bytes: int = 4096) -> None:
        self.max_size = max_size_bytes
        self.entries: list[tuple[str, str]] = []
        self.size = 0

    @classmethod
    def _entry_size(cls, name: str, value: str) -> int:
        return len(name.encode("latin-1")) + len(value.encode("latin-1")) + cls.OVERHEAD

    def _evict_to(self, target: int) -> None:
        while self.entries and self.size > target:
            old_n, old_v = self.entries.pop()
            self.size -= self._entry_size(old_n, old_v)

    def add(self, name: str, value: str) -> None:
        esize = self._entry_size(name, value)
        if esize > self.max_size:
            self.entries.clear()
            self.size = 0
            return
        self._evict_to(self.max_size - esize)
        self.entries.insert(0, (name, value))
        self.size += esize

    def resize(self, new_max: int) -> None:
        self.max_size = new_max
        self._evict_to(new_max)
'''),
                ("web-exp-adaptive-concurrency-aimd", "Netflix Gradient2 / Vegas-style AIMD adaptive HTTP concurrency limiter", ("concurrency_control", "load_shedding"),
                 '''class AdaptiveConcurrencyLimiter:
    """Adjusts max in-flight requests dynamically from baseline vs smoothed RTT."""

    def __init__(self, initial_limit: int = 20, min_limit: int = 4, max_limit: int = 256) -> None:
        self.limit = float(initial_limit)
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.in_flight = 0
        self.min_rtt = float("inf")

    def try_acquire(self) -> bool:
        if self.in_flight >= int(self.limit):
            return False
        self.in_flight += 1
        return True

    def on_complete(self, rtt_s: float, overloaded: bool = False) -> int:
        self.in_flight = max(0, self.in_flight - 1)
        if overloaded:
            self.limit = max(self.min_limit, self.limit * 0.8)
            return int(self.limit)
        self.min_rtt = min(self.min_rtt * 1.001, max(rtt_s, 1e-6))
        gradient = min(1.0, max(0.5, self.min_rtt / max(rtt_s, 1e-6)))
        if gradient >= 0.9:
            self.limit = min(self.max_limit, self.limit + 1.0)
        else:
            self.limit = max(self.min_limit, self.limit * gradient)
        return int(self.limit)
'''),
                ("web-exp-consistent-cdn-shield", "Rendezvous (Highest Random Weight) origin-shield cache router", ("cdn", "rendezvous_hashing", "caching"),
                 '''import hashlib


class RendezvousRouter:
    """HRW hashing minimizing cache-key remapping to 1/N when a shield node leaves."""

    def __init__(self, nodes: list[str]) -> None:
        self.nodes = list(nodes)

    @staticmethod
    def _weight(node: str, key: str) -> int:
        digest = hashlib.blake2b(f"{node}|{key}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def replicas_for(self, cache_key: str, k: int = 2) -> list[str]:
        scored = sorted(self.nodes, key=lambda n: self._weight(n, cache_key), reverse=True)
        return scored[:k]
'''),
                ("web-exp-w3c-traceparent", "W3C Trace Context (traceparent) parser, validator, and child-span generator", ("observability", "opentelemetry", "tracing"),
                 '''import os
import re

_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def parse_traceparent(header: str) -> dict[str, object] | None:
    """Parse W3C version-00 traceparent rejecting all-zero trace_id or parent_id."""
    m = _TRACEPARENT_RE.match(header.strip().lower())
    if not m:
        return None
    trace_id, parent_id, flags_hex = m.groups()
    if trace_id == "0" * 32 or parent_id == "0" * 16:
        return None
    flags = int(flags_hex, 16)
    return {"trace_id": trace_id, "parent_id": parent_id, "sampled": bool(flags & 0x01)}


def child_traceparent(parent_header: str) -> str:
    parsed = parse_traceparent(parent_header)
    trace_id = str(parsed["trace_id"]) if parsed else os.urandom(16).hex()
    sampled = bool(parsed["sampled"]) if parsed else True
    span_id = os.urandom(8).hex()
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"
'''),
                ("web-exp-singleflight-cache-stampede", "Probabilistic early-expiration (XFetch) + singleflight cache coordinator", ("cache_stampede", "xfetch"),
                 '''import math
import random
from typing import Callable


class StampedeSafeCache:
    """Implements Vattani XFetch probabilistic early recomputation against hot-key stampedes."""

    def __init__(self, seed: int = 42) -> None:
        self._store: dict[str, tuple[object, float, float]] = {}
        self._rng = random.Random(seed)

    def get_or_compute(self, key: str, now: float, ttl_s: float, compute_fn: Callable[[], tuple[object, float]], beta: float = 1.0) -> object:
        entry = self._store.get(key)
        if entry is not None:
            val, delta_s, expiry = entry
            u = max(self._rng.random(), 1e-12)
            if now - delta_s * beta * math.log(u) < expiry:
                return val
        val, compute_cost_s = compute_fn()
        self._store[key] = (val, compute_cost_s, now + ttl_s)
        return val
'''),
                ("web-exp-graphql-dataloader", "Batched key coalescer with per-request memoization (DataLoader pattern)", ("graphql", "batching", "n_plus_one"),
                 '''from collections.abc import Callable


class BatchLoader:
    """Eliminates N+1 queries by deduplicating keys within a tick and preserving order."""

    def __init__(self, batch_fn: Callable[[list[str]], dict[str, object]]) -> None:
        self._batch_fn = batch_fn
        self._cache: dict[str, object] = {}
        self._pending: list[str] = []

    def prime(self, key: str) -> None:
        if key not in self._cache and key not in self._pending:
            self._pending.append(key)

    def dispatch_and_get(self, keys: list[str]) -> list[object | None]:
        for k in keys:
            self.prime(k)
        if self._pending:
            fetched = self._batch_fn(list(self._pending))
            for k in self._pending:
                self._cache[k] = fetched.get(k)
            self._pending.clear()
        return [self._cache.get(k) for k in keys]
'''),
                ("web-exp-oauth-pkce-state", "OAuth 2.1 Authorization Code + S256 PKCE verifier state machine", ("oauth2", "pkce", "auth"),
                 '''import base64
import hashlib
import hmac


class PkceAuthorizationStore:
    """One-time authorization code exchange enforcing S256 PKCE code_verifier."""

    def __init__(self) -> None:
        self._codes: dict[str, tuple[str, str, str, int]] = {}

    @staticmethod
    def s256_challenge(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    def issue_code(self, code: str, client_id: str, redirect_uri: str, challenge: str, expires_at: int) -> None:
        self._codes[code] = (client_id, redirect_uri, challenge, expires_at)

    def redeem(self, code: str, client_id: str, redirect_uri: str, verifier: str, now: int) -> bool:
        entry = self._codes.pop(code, None)
        if entry is None:
            return False
        cid, ruri, challenge, exp = entry
        if now > exp or cid != client_id or ruri != redirect_uri:
            return False
        return hmac.compare_digest(self.s256_challenge(verifier), challenge)
'''),
                ("web-exp-http-structured-headers", "RFC 8941 HTTP Structured Field dictionary parser", ("rfc8941", "http_headers"),
                 '''def parse_sf_dictionary(header: str) -> dict[str, object]:
    """Parse subset of RFC 8941 Structured Fields dictionary (booleans, ints, strings)."""
    out: dict[str, object] = {}
    for member in [m.strip() for m in header.split(",") if m.strip()]:
        if "=" not in member:
            out[member] = True
            continue
        key, raw = member.split("=", 1)
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            out[key.strip()] = raw[1:-1]
        elif raw == "?1":
            out[key.strip()] = True
        elif raw == "?0":
            out[key.strip()] = False
        else:
            out[key.strip()] = int(raw)
    return out
'''),
                ("web-exp-sse-last-event-replay", "Bounded ring-buffer SSE broker supporting Last-Event-ID gapless replay", ("sse", "streaming", "replay"),
                 '''from collections import deque


class ReplayableSSEBroker:
    """Retains recent monotonic events so reconnecting clients resume via Last-Event-ID."""

    def __init__(self, history_limit: int = 256) -> None:
        self._seq = 0
        self._history: deque[tuple[int, str]] = deque(maxlen=history_limit)

    def publish(self, payload: str) -> int:
        self._seq += 1
        self._history.append((self._seq, payload))
        return self._seq

    def replay_since(self, last_event_id: int) -> tuple[bool, list[tuple[int, str]]]:
        if not self._history:
            return True, []
        oldest_id = self._history[0][0]
        gap_free = last_event_id >= oldest_id - 1
        events = [(eid, data) for eid, data in self._history if eid > last_event_id]
        return gap_free, events
'''),
            ],
        },
    )


def _build_systems_patterns() -> list[PatternSpec]:
    return _domain_specs(
        "systems",
        {
            "beginner": [
                ("sys-beg-atomic-write", "Crash-safe atomic file replacement via temporary sibling + os.replace", ("filesystem", "atomicity"),
                 '''import os
from pathlib import Path


def write_text_atomic(target: Path, content: str) -> None:
    """Write UTF-8 file via sibling temporary file and atomic POSIX rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)
'''),
                ("sys-beg-byte-units", "Human-readable IEC/SI byte-size parser and formatter", ("units", "memory"),
                 '''def parse_byte_size(spec: str) -> int:
    """Convert strings like '64KiB' or '512MB' into integer byte counts."""
    units = {"b": 1, "kb": 1000, "kib": 1024, "mb": 10**6, "mib": 1024**2, "gb": 10**9, "gib": 1024**3}
    s = spec.strip().lower()
    idx = 0
    while idx < len(s) and (s[idx].isdigit() or s[idx] == "."):
        idx += 1
    num = float(s[:idx])
    unit = s[idx:].strip() or "b"
    if unit not in units:
        raise ValueError(f"unknown unit {unit!r}")
    return int(num * units[unit])
'''),
                ("sys-beg-env-config", "Typed environment-variable loader with boolean and list coercion", ("config", "env"),
                 '''import os


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean environment value {name}={raw!r}")
'''),
                ("sys-beg-varint-codec", "Unsigned LEB128 variable-length integer encoder and decoder", ("binary", "leb128", "serialization"),
                 '''def encode_uvarint(value: int) -> bytes:
    """Encode non-negative integer using unsigned LEB128."""
    if value < 0:
        raise ValueError("uvarint requires non-negative integer")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def decode_uvarint(buf: bytes, offset: int = 0) -> tuple[int, int]:
    result = 0
    shift = 0
    pos = offset
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
'''),
                ("sys-beg-ring-log-buffer", "Fixed-capacity circular diagnostic line buffer", ("ring_buffer", "logging"),
                 '''class DiagnosticRing:
    """Overwrite-oldest ring buffer storing the last `capacity` log entries."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self._buf: list[str | None] = [None] * self.capacity
        self._write_pos = 0
        self._count = 0

    def append(self, entry: str) -> None:
        self._buf[self._write_pos] = entry
        self._write_pos = (self._write_pos + 1) % self.capacity
        self._count = min(self.capacity, self._count + 1)

    def snapshot(self) -> list[str]:
        start = 0 if self._count < self.capacity else self._write_pos
        return [self._buf[(start + i) % self.capacity] or "" for i in range(self._count)]
'''),
                ("sys-beg-pid-lockfile", "Stale-aware exclusive lockfile creator using O_CREAT | O_EXCL", ("ipc", "locking"),
                 '''import os
from pathlib import Path


def acquire_lockfile(lock_path: Path) -> bool:
    """Atomically create lockfile recording current PID; return False if already held."""
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{os.getpid()}\\n".encode("ascii"))
    finally:
        os.close(fd)
    return True
'''),
                ("sys-beg-crc32-frame", "Length-prefixed + CRC32-checked binary record framing", ("framing", "checksums"),
                 '''import struct
import zlib


def pack_frame(payload: bytes) -> bytes:
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack(">II", len(payload), crc) + payload


def unpack_frame(buf: bytes) -> bytes:
    if len(buf) < 8:
        raise ValueError("truncated header")
    length, expected_crc = struct.unpack(">II", buf[:8])
    payload = buf[8 : 8 + length]
    if len(payload) != length:
        raise ValueError("truncated payload")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
        raise ValueError("CRC32 mismatch")
    return payload
'''),
                ("sys-beg-backoff-jitter", "AWS full-jitter exponential backoff calculator", ("retries", "networking"),
                 '''import random


def full_jitter_delay(attempt: int, base_s: float = 0.1, cap_s: float = 30.0, rng: random.Random | None = None) -> float:
    """Compute uniform delay in [0, min(cap_s, base_s * 2**attempt)]."""
    ceiling = min(cap_s, base_s * (2 ** max(0, attempt)))
    r = rng.random() if rng is not None else random.random()
    return r * ceiling
'''),
                ("sys-beg-dir-du", "Non-symlink-following directory byte usage walker", ("filesystem", "disk_usage"),
                 '''from pathlib import Path


def directory_size_bytes(root: Path) -> int:
    """Sum regular-file sizes under root without traversing symbolic links."""
    total = 0
    for entry in root.rglob("*"):
        if not entry.is_symlink() and entry.is_file():
            total += entry.stat().st_size
    return total
'''),
            ],
            "intermediate": [
                ("sys-int-byte-ring-buffer", "Zero-allocation bytearray circular byte queue", ("buffers", "io"),
                 '''class ByteRingBuffer:
    """Fixed-size bytearray FIFO stream buffer for non-blocking socket IO."""

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.buf = bytearray(capacity)
        self.read_pos = 0
        self.write_pos = 0
        self.size = 0

    def write(self, data: bytes) -> int:
        n = min(len(data), self.cap - self.size)
        first = min(n, self.cap - self.write_pos)
        self.buf[self.write_pos : self.write_pos + first] = data[:first]
        if n > first:
            self.buf[: n - first] = data[first:n]
        self.write_pos = (self.write_pos + n) % self.cap
        self.size += n
        return n

    def read(self, count: int) -> bytes:
        n = min(count, self.size)
        first = min(n, self.cap - self.read_pos)
        out = bytes(self.buf[self.read_pos : self.read_pos + first])
        if n > first:
            out += bytes(self.buf[: n - first])
        self.read_pos = (self.read_pos + n) % self.cap
        self.size -= n
        return out
'''),
                ("sys-int-slab-allocator", "Fixed-slot freelist slab page allocator", ("memory_management", "allocator"),
                 '''class SlabPage:
    """O(1) slot allocation and deallocation over a contiguous bytearray."""

    def __init__(self, slot_size: int, num_slots: int) -> None:
        self.slot_size = slot_size
        self.num_slots = num_slots
        self.memory = bytearray(slot_size * num_slots)
        self.free_list = list(range(num_slots - 1, -1, -1))

    def alloc(self) -> int | None:
        return self.free_list.pop() if self.free_list else None

    def free(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots or slot in self.free_list:
            raise ValueError("invalid or double-freed slot")
        start = slot * self.slot_size
        self.memory[start : start + self.slot_size] = b"\\x00" * self.slot_size
        self.free_list.append(slot)
'''),
                ("sys-int-striped-lock", "Key-hashed striped mutex table reducing lock contention", ("concurrency", "locks"),
                 '''import hashlib
import threading
from contextlib import contextmanager
from collections.abc import Iterator


class StripedLockTable:
    """Maps arbitrary string keys onto a fixed array of RLock stripes."""

    def __init__(self, stripes: int = 32) -> None:
        self._locks = [threading.RLock() for _ in range(stripes)]

    @contextmanager
    def guard(self, key: str) -> Iterator[None]:
        idx = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=4).digest(), "little") % len(self._locks)
        lock = self._locks[idx]
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
'''),
                ("sys-int-wal-segment", "Checksummed append-only Write-Ahead Log with torn-tail recovery", ("wal", "crash_recovery", "storage"),
                 '''import struct
import zlib


class AppendOnlyWAL:
    """Serializes records with length+CRC32 and truncates torn trailing writes on replay."""

    def __init__(self) -> None:
        self._disk = bytearray()

    def append(self, record: bytes) -> int:
        offset = len(self._disk)
        crc = zlib.crc32(record) & 0xFFFFFFFF
        self._disk.extend(struct.pack(">II", len(record), crc))
        self._disk.extend(record)
        return offset

    def recover(self) -> list[bytes]:
        records: list[bytes] = []
        pos = 0
        while pos + 8 <= len(self._disk):
            length, crc = struct.unpack(">II", self._disk[pos : pos + 8])
            if pos + 8 + length > len(self._disk):
                del self._disk[pos:]
                break
            body = bytes(self._disk[pos + 8 : pos + 8 + length])
            if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
                del self._disk[pos:]
                break
            records.append(body)
            pos += 8 + length
        return records
'''),
                ("sys-int-read-write-lock", "Writer-preferring Read-Write lock preventing writer starvation", ("synchronization", "threading"),
                 '''import threading


class WriterPreferringRWLock:
    """Blocks new readers while any writer is waiting or active."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer_active = False
        self._waiting_writers = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer_active or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._waiting_writers += 1
            while self._writer_active or self._readers > 0:
                self._cond.wait()
            self._waiting_writers -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()
'''),
                ("sys-int-consistent-hash", "Virtual-node consistent hashing ring with clockwise bisect lookup", ("distributed_systems", "sharding"),
                 '''from bisect import bisect_right
import hashlib


class ConsistentHashRing:
    """Maps keys onto cluster nodes using `vnodes` virtual tokens per physical node."""

    def __init__(self, vnodes: int = 64) -> None:
        self.vnodes = vnodes
        self._keys: list[int] = []
        self._owners: dict[int, str] = {}

    @staticmethod
    def _hash(val: str) -> int:
        return int.from_bytes(hashlib.sha256(val.encode()).digest()[:8], "big")

    def add_node(self, node: str) -> None:
        for idx in range(self.vnodes):
            token = self._hash(f"{node}#{idx}")
            self._owners[token] = node
        self._keys = sorted(self._owners)

    def node_for(self, key: str) -> str:
        if not self._keys:
            raise RuntimeError("empty hash ring")
        pos = bisect_right(self._keys, self._hash(key))
        if pos == len(self._keys):
            pos = 0
        return self._owners[self._keys[pos]]
'''),
                ("sys-int-lease-manager", "Monotonic-clock TTL lease table with fencing token generation", ("leases", "fencing", "distributed_systems"),
                 '''class FencingLeaseTable:
    """Issues monotonically increasing fencing tokens per resource lease."""

    def __init__(self) -> None:
        self._epoch = 0
        self._leases: dict[str, tuple[str, float, int]] = {}

    def acquire(self, resource: str, holder: str, now: float, ttl_s: float) -> int | None:
        cur = self._leases.get(resource)
        if cur is not None and cur[1] > now and cur[0] != holder:
            return None
        self._epoch += 1
        self._leases[resource] = (holder, now + ttl_s, self._epoch)
        return self._epoch

    def validate_token(self, resource: str, token: int) -> bool:
        cur = self._leases.get(resource)
        return cur is not None and token >= cur[2]
'''),
                ("sys-int-content-addressed-store", "SHA-256 content-addressed chunk store with reference-counted GC", ("cas", "deduplication", "storage"),
                 '''import hashlib


class ContentAddressedStore:
    """Deduplicated blob store reclaiming unreferenced chunks safely."""

    def __init__(self) -> None:
        self._chunks: dict[str, bytes] = {}
        self._refcount: dict[str, int] = {}

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        if digest not in self._chunks:
            self._chunks[digest] = data
            self._refcount[digest] = 0
        self._refcount[digest] += 1
        return digest

    def release(self, digest: str) -> bool:
        if digest not in self._refcount:
            return False
        self._refcount[digest] -= 1
        if self._refcount[digest] <= 0:
            del self._refcount[digest]
            del self._chunks[digest]
            return True
        return False
'''),
                ("sys-int-proc-supervisor", "Restart-intensity supervisor window (Erlang max_restarts / period)", ("supervision", "fault_tolerance"),
                 '''from collections import deque


class RestartIntensityPolicy:
    """Allow child restarts unless > max_restarts occur within window_s."""

    def __init__(self, max_restarts: int = 3, window_s: float = 60.0) -> None:
        self.max_restarts = max_restarts
        self.window_s = window_s
        self._history: deque[float] = deque()

    def record_crash(self, now: float) -> str:
        cutoff = now - self.window_s
        while self._history and self._history[0] < cutoff:
            self._history.popleft()
        self._history.append(now)
        return "escalate" if len(self._history) > self.max_restarts else "restart"
'''),
            ],
            "advanced": [
                ("sys-adv-timing-wheel", "Hashed hierarchical O(1) timer wheel scheduler", ("timers", "networking", "kernel"),
                 '''class HashedTimingWheel:
    """O(1) timer insertion and tick advancement using wheel slots + remaining rounds."""

    def __init__(self, num_slots: int = 64, tick_ms: int = 10) -> None:
        self.num_slots = num_slots
        self.tick_ms = tick_ms
        self.cursor = 0
        self.slots: list[list[list[object]]] = [[] for _ in range(num_slots)]

    def schedule(self, delay_ms: int, task_id: str) -> None:
        ticks = max(1, delay_ms // self.tick_ms)
        rounds = ticks // self.num_slots
        slot = (self.cursor + ticks) % self.num_slots
        self.slots[slot].append([rounds, task_id])

    def advance_tick(self) -> list[str]:
        self.cursor = (self.cursor + 1) % self.num_slots
        bucket = self.slots[self.cursor]
        expired: list[str] = []
        remaining: list[list[object]] = []
        for entry in bucket:
            rounds = int(entry[0])  # type: ignore[arg-type]
            if rounds <= 0:
                expired.append(str(entry[1]))
            else:
                entry[0] = rounds - 1
                remaining.append(entry)
        self.slots[self.cursor] = remaining
        return expired
'''),
                ("sys-adv-clock-sweep-cache", "Second-chance CLOCK page buffer replacement policy", ("buffer_pool", "eviction", "os"),
                 '''class ClockBufferPool:
    """PostgreSQL-style CLOCK sweep buffer eviction with reference counts and usage bits."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.pages: list[int | None] = [None] * frames
        self.ref_bits = [0] * frames
        self.pin_counts = [0] * frames
        self.hand = 0

    def fetch(self, page_id: int) -> int:
        if page_id in self.pages:
            idx = self.pages.index(page_id)
            self.ref_bits[idx] = 1
            return idx
        scanned = 0
        while scanned < 2 * self.frames + 1:
            idx = self.hand
            self.hand = (self.hand + 1) % self.frames
            if self.pages[idx] is None:
                self.pages[idx] = page_id
                self.ref_bits[idx] = 1
                return idx
            if self.pin_counts[idx] == 0:
                if self.ref_bits[idx] == 0:
                    self.pages[idx] = page_id
                    self.ref_bits[idx] = 1
                    return idx
                self.ref_bits[idx] = 0
            scanned += 1
        raise RuntimeError("all buffer frames are pinned")
'''),
                ("sys-adv-buddy-allocator", "Binary power-of-two buddy memory block allocator with coalescing", ("allocator", "buddy_system", "os"),
                 '''class BuddyAllocator:
    """Splits and coalesces power-of-two memory blocks across order freelists."""

    def __init__(self, max_order: int = 10) -> None:
        self.max_order = max_order
        self.free: list[set[int]] = [set() for _ in range(max_order + 1)]
        self.free[max_order].add(0)

    def allocate(self, size: int) -> tuple[int, int] | None:
        order = 0
        while (1 << order) < max(1, size):
            order += 1
        if order > self.max_order:
            return None
        cur = order
        while cur <= self.max_order and not self.free[cur]:
            cur += 1
        if cur > self.max_order:
            return None
        addr = min(self.free[cur])
        self.free[cur].remove(addr)
        while cur > order:
            cur -= 1
            self.free[cur].add(addr + (1 << cur))
        return addr, order

    def deallocate(self, addr: int, order: int) -> None:
        cur_addr, cur_order = addr, order
        while cur_order < self.max_order:
            buddy = cur_addr ^ (1 << cur_order)
            if buddy not in self.free[cur_order]:
                break
            self.free[cur_order].remove(buddy)
            cur_addr = min(cur_addr, buddy)
            cur_order += 1
        self.free[cur_order].add(cur_addr)
'''),
                ("sys-adv-vector-clock", "Lamport / Fidge-Mattern vector clock causal ordering detector", ("vector_clocks", "causality", "distributed_systems"),
                 '''from __future__ import annotations


class VectorClock:
    """Detects happens-before vs concurrent causal divergence across replicas."""

    def __init__(self, versions: dict[str, int] | None = None) -> None:
        self.v: dict[str, int] = dict(versions or {})

    def tick(self, node: str) -> VectorClock:
        nxt = dict(self.v)
        nxt[node] = nxt.get(node, 0) + 1
        return VectorClock(nxt)

    def merge(self, other: VectorClock) -> VectorClock:
        keys = set(self.v) | set(other.v)
        return VectorClock({k: max(self.v.get(k, 0), other.v.get(k, 0)) for k in keys})

    def compare(self, other: VectorClock) -> str:
        keys = set(self.v) | set(other.v)
        le = all(self.v.get(k, 0) <= other.v.get(k, 0) for k in keys)
        ge = all(self.v.get(k, 0) >= other.v.get(k, 0) for k in keys)
        if le and ge:
            return "equal"
        if le:
            return "before"
        if ge:
            return "after"
        return "concurrent"
'''),
                ("sys-adv-phi-accrual-detector", "Hayashibara Phi-accrual adaptive heartbeat failure detector", ("failure_detector", "gossip", "distributed_systems"),
                 '''from collections import deque
import math


class PhiAccrualDetector:
    """Models inter-arrival heartbeat intervals to output continuous suspicion level."""

    def __init__(self, window_size: int = 100) -> None:
        self.intervals: deque[float] = deque(maxlen=window_size)
        self.last_ts: float | None = None

    def heartbeat(self, now: float) -> None:
        if self.last_ts is not None and now > self.last_ts:
            self.intervals.append(now - self.last_ts)
        self.last_ts = now

    def phi(self, now: float) -> float:
        if self.last_ts is None or len(self.intervals) < 2:
            return 0.0
        mean = sum(self.intervals) / len(self.intervals)
        elapsed = max(0.0, now - self.last_ts)
        # Exponential CDF approximation: -log10(exp(-elapsed / mean))
        return (elapsed / max(mean, 1e-6)) / math.log(10.0)
'''),
                ("sys-adv-lsm-memtable", "Sorted LSM-tree MemTable with tombstone deletion and immutable flush", ("lsm_tree", "storage_engine"),
                 '''from bisect import bisect_left


class LSMMemTable:
    """Sorted in-memory write buffer emitting tombstone-preserving SSTable runs."""

    TOMBSTONE = object()

    def __init__(self, flush_bytes: int = 65536) -> None:
        self.flush_bytes = flush_bytes
        self._keys: list[str] = []
        self._vals: list[object] = []
        self.approx_bytes = 0

    def put(self, key: str, value: bytes) -> bool:
        self._upsert(key, value)
        self.approx_bytes += len(key) + len(value)
        return self.approx_bytes >= self.flush_bytes

    def delete(self, key: str) -> None:
        self._upsert(key, self.TOMBSTONE)

    def _upsert(self, key: str, val: object) -> None:
        pos = bisect_left(self._keys, key)
        if pos < len(self._keys) and self._keys[pos] == key:
            self._vals[pos] = val
        else:
            self._keys.insert(pos, key)
            self._vals.insert(pos, val)

    def get(self, key: str) -> tuple[bool, bytes | None]:
        pos = bisect_left(self._keys, key)
        if pos >= len(self._keys) or self._keys[pos] != key:
            return False, None
        val = self._vals[pos]
        return (True, None) if val is self.TOMBSTONE else (True, val)  # type: ignore[return-value]
'''),
                ("sys-adv-work-stealing-deque", "Chase-Lev-inspired owner-LIFO / thief-FIFO work-stealing task queue", ("schedulers", "work_stealing", "concurrency"),
                 '''from collections import deque
import threading


class WorkStealingDeque:
    """Owner pushes/pops LIFO for cache locality; thieves steal FIFO from top."""

    def __init__(self) -> None:
        self._dq: deque[object] = deque()
        self._lock = threading.Lock()

    def push_bottom(self, task: object) -> None:
        with self._lock:
            self._dq.append(task)

    def pop_bottom(self) -> object | None:
        with self._lock:
            return self._dq.pop() if self._dq else None

    def steal_top(self) -> object | None:
        with self._lock:
            return self._dq.popleft() if self._dq else None
'''),
                ("sys-adv-token-bucket-shaper", "Hierarchical borrow-capable traffic shaper (tenant + global ceiling)", ("qos", "traffic_shaping"),
                 '''class HierarchicalTrafficShaper:
    """Enforces both per-tenant committed rate and shared global link ceiling."""

    def __init__(self, global_burst: int, tenant_burst: int) -> None:
        self.global_tokens = global_burst
        self.tenant_burst = tenant_burst
        self.tenants: dict[str, int] = {}

    def admit(self, tenant: str, packet_bytes: int) -> bool:
        avail = self.tenants.get(tenant, self.tenant_burst)
        if packet_bytes > avail or packet_bytes > self.global_tokens:
            return False
        self.tenants[tenant] = avail - packet_bytes
        self.global_tokens -= packet_bytes
        return True
'''),
                ("sys-adv-rsync-rolling-checksum", "Adler-style rolling weak + strong SHA-256 delta block matcher", ("delta_sync", "rsync", "rolling_hash"),
                 '''import hashlib


def build_block_signatures(target: bytes, block_size: int = 512) -> dict[int, list[tuple[int, str]]]:
    """Index target file blocks by 32-bit weak rolling checksum + SHA-256 digest."""
    sigs: dict[int, list[tuple[int, str]]] = {}
    for idx in range(0, len(target), block_size):
        chunk = target[idx : idx + block_size]
        a = sum(chunk) & 0xFFFF
        b = sum((len(chunk) - i) * byte for i, byte in enumerate(chunk)) & 0xFFFF
        weak = (b << 16) | a
        strong = hashlib.sha256(chunk).hexdigest()
        sigs.setdefault(weak, []).append((idx // block_size, strong))
    return sigs
'''),
            ],
            "expert": [
                ("sys-exp-raft-log-replication", "Raft consensus log matching and leader election safety state machine", ("raft", "consensus", "distributed_systems"),
                 '''from dataclasses import dataclass


@dataclass(frozen=True)
class LogEntry:
    term: int
    command: str


class RaftReplica:
    """Implements Raft §5.2/§5.3 up-to-date vote restriction and AppendEntries consistency."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.current_term = 0
        self.voted_for: str | None = None
        self.log: list[LogEntry] = []
        self.commit_index = -1

    def last_term_and_index(self) -> tuple[int, int]:
        if not self.log:
            return 0, -1
        return self.log[-1].term, len(self.log) - 1

    def on_request_vote(self, term: int, candidate_id: str, last_log_term: int, last_log_index: int) -> tuple[int, bool]:
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
        if term < self.current_term:
            return self.current_term, False
        my_term, my_idx = self.last_term_and_index()
        up_to_date = (last_log_term, last_log_index) >= (my_term, my_idx)
        if (self.voted_for in (None, candidate_id)) and up_to_date:
            self.voted_for = candidate_id
            return self.current_term, True
        return self.current_term, False

    def on_append_entries(self, term: int, prev_index: int, prev_term: int, entries: list[LogEntry], leader_commit: int) -> tuple[int, bool]:
        if term < self.current_term:
            return self.current_term, False
        self.current_term = term
        if prev_index >= 0:
            if prev_index >= len(self.log) or self.log[prev_index].term != prev_term:
                return self.current_term, False
        insert_at = prev_index + 1
        for offset, entry in enumerate(entries):
            pos = insert_at + offset
            if pos < len(self.log):
                if self.log[pos].term != entry.term:
                    del self.log[pos:]
                    self.log.append(entry)
            else:
                self.log.append(entry)
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, len(self.log) - 1)
        return self.current_term, True
'''),
                ("sys-exp-mvcc-snapshot-isolation", "Snapshot-Isolation MVCC tuple visibility + write-write conflict manager", ("mvcc", "transactions", "storage_engine"),
                 '''from dataclasses import dataclass


@dataclass
class TupleVersion:
    xmin: int
    xmax: int | None
    value: str


class MVCCStore:
    """Snapshot-isolation key-value engine detecting first-committer-wins write conflicts."""

    def __init__(self) -> None:
        self._next_tx = 1
        self._active: set[int] = set()
        self._committed: set[int] = set()
        self._versions: dict[str, list[TupleVersion]] = {}

    def begin(self) -> tuple[int, frozenset[int]]:
        tx = self._next_tx
        self._next_tx += 1
        snap = frozenset(self._committed)
        self._active.add(tx)
        return tx, snap

    def read(self, key: str, tx: int, snapshot: frozenset[int]) -> str | None:
        for ver in reversed(self._versions.get(key, ())):
            created_visible = ver.xmin == tx or ver.xmin in snapshot
            deleted_visible = ver.xmax is not None and (ver.xmax == tx or ver.xmax in snapshot)
            if created_visible and not deleted_visible:
                return ver.value
        return None

    def write(self, key: str, value: str, tx: int, snapshot: frozenset[int]) -> None:
        chain = self._versions.setdefault(key, [])
        for ver in chain:
            if ver.xmax is None and ver.xmin != tx and ver.xmin not in snapshot:
                raise RuntimeError("serialization failure: concurrent write")
            if ver.xmax is not None and ver.xmax != tx and ver.xmax not in snapshot:
                raise RuntimeError("serialization failure: concurrent delete")
        for ver in chain:
            if ver.xmax is None:
                ver.xmax = tx
        chain.append(TupleVersion(xmin=tx, xmax=None, value=value))

    def commit(self, tx: int) -> None:
        self._active.discard(tx)
        self._committed.add(tx)
'''),
                ("sys-exp-crdt-lww-orset", "Delta-state Observed-Remove Set (OR-Set) CRDT with causal dot tags", ("crdt", "distributed_state"),
                 '''from __future__ import annotations


class ObservedRemoveSet:
    """Add-wins OR-Set CRDT supporting concurrent add/remove and idempotent merge."""

    def __init__(self, replica_id: str) -> None:
        self.replica_id = replica_id
        self._seq = 0
        self.entries: dict[str, set[tuple[str, int]]] = {}
        self.tombstones: set[tuple[str, int]] = set()

    def add(self, element: str) -> None:
        self._seq += 1
        dot = (self.replica_id, self._seq)
        self.entries.setdefault(element, set()).add(dot)

    def remove(self, element: str) -> None:
        observed = self.entries.pop(element, set())
        self.tombstones.update(observed)

    def merge(self, other: ObservedRemoveSet) -> None:
        self.tombstones |= other.tombstones
        for elem, dots in other.entries.items():
            live = (self.entries.get(elem, set()) | dots) - self.tombstones
            if live:
                self.entries[elem] = live
            else:
                self.entries.pop(elem, None)
        for elem in list(self.entries):
            self.entries[elem] -= self.tombstones
            if not self.entries[elem]:
                del self.entries[elem]

    def values(self) -> set[str]:
        return set(self.entries)
'''),
                ("sys-exp-w-tinylfu", "Window-TinyLFU cache admission policy (4-bit Count-Min frequency sketch)", ("caching", "wtinylfu", "eviction"),
                 '''import hashlib


class TinyLFUAdmission:
    """Admits candidate over LRU victim only when 4-bit decaying frequency sketch wins."""

    def __init__(self, counters: int = 512, reset_every: int = 2048) -> None:
        self.size = counters
        self.reset_every = reset_every
        self.ops = 0
        self.sketch = [[0] * counters for _ in range(4)]

    def _idx(self, row: int, key: str) -> int:
        h = hashlib.blake2b(f"{row}:{key}".encode(), digest_size=4).digest()
        return int.from_bytes(h, "little") % self.size

    def record_access(self, key: str) -> None:
        self.ops += 1
        for r in range(4):
            c = self._idx(r, key)
            if self.sketch[r][c] < 15:
                self.sketch[r][c] += 1
        if self.ops >= self.reset_every:
            self.ops = 0
            for r in range(4):
                for c in range(self.size):
                    self.sketch[r][c] >>= 1

    def estimate(self, key: str) -> int:
        return min(self.sketch[r][self._idx(r, key)] for r in range(4))

    def should_admit(self, candidate: str, victim: str) -> bool:
        return self.estimate(candidate) > self.estimate(victim)
'''),
                ("sys-exp-cdc-rabin-chunker", "Gear-hash content-defined chunker (FastCDC) for deduplicated backups", ("fastcdc", "chunking", "storage"),
                 '''import hashlib


def content_defined_chunks(data: bytes, min_size: int = 256, avg_size: int = 1024, max_size: int = 4096) -> list[tuple[int, int, str]]:
    """Split byte stream at rolling-hash cutpoints so insertions shift only one chunk."""
    mask = avg_size - 1
    chunks: list[tuple[int, int, str]] = []
    start = 0
    fp = 0
    n = len(data)
    i = 0
    while i < n:
        fp = ((fp << 1) + (data[i] * 0x9E3779B1)) & 0xFFFFFFFF
        chunk_len = i - start + 1
        is_cut = (chunk_len >= min_size and (fp & mask) == 0) or chunk_len >= max_size
        if is_cut or i == n - 1:
            piece = data[start : i + 1]
            chunks.append((start, len(piece), hashlib.sha256(piece).hexdigest()))
            start = i + 1
            fp = 0
        i += 1
    return chunks
'''),
                ("sys-exp-deadlock-wait-for", "Lock-manager wait-for graph cycle detector selecting youngest-tx victim", ("deadlock_detection", "transactions"),
                 '''def detect_deadlock_victim(wait_for: dict[int, set[int]], tx_age: dict[int, int]) -> int | None:
    """Find a wait-for cycle and return the transaction ID with smallest age (youngest)."""
    state: dict[int, int] = {}
    stack: list[int] = []

    def dfs(tx: int) -> list[int] | None:
        state[tx] = 1
        stack.append(tx)
        for blocker in wait_for.get(tx, ()):
            st = state.get(blocker, 0)
            if st == 0:
                cyc = dfs(blocker)
                if cyc is not None:
                    return cyc
            elif st == 1:
                idx = stack.index(blocker)
                return stack[idx:]
        stack.pop()
        state[tx] = 2
        return None

    for tx in list(wait_for):
        if state.get(tx, 0) == 0:
            cycle = dfs(tx)
            if cycle:
                return min(cycle, key=lambda t: tx_age.get(t, 0))
    return None
'''),
                ("sys-exp-wfq-packet-scheduler", "Deficit Round Robin (DRR) fair packet queue scheduler", ("networking", "qos", "drr"),
                 '''from collections import deque


class DeficitRoundRobinScheduler:
    """O(1) fair bandwidth scheduler handling variable-sized packets without starvation."""

    def __init__(self, quantum: int = 1500) -> None:
        self.quantum = quantum
        self.queues: dict[str, deque[bytes]] = {}
        self.deficit: dict[str, int] = {}
        self.active: deque[str] = deque()

    def enqueue(self, flow: str, packet: bytes) -> None:
        q = self.queues.setdefault(flow, deque())
        if not q:
            self.active.append(flow)
            self.deficit.setdefault(flow, 0)
        q.append(packet)

    def dequeue(self) -> tuple[str, bytes] | None:
        while self.active:
            flow = self.active.popleft()
            q = self.queues[flow]
            self.deficit[flow] += self.quantum
            if q and len(q[0]) <= self.deficit[flow]:
                pkt = q.popleft()
                self.deficit[flow] -= len(pkt)
                if q:
                    self.active.append(flow)
                else:
                    self.deficit[flow] = 0
                return flow, pkt
            if q:
                self.active.append(flow)
        return None
'''),
                ("sys-exp-sstable-compactor", "Leveled LSM SSTable tombstone-pruning k-way merge compactor", ("lsm_tree", "compaction", "storage"),
                 '''import heapq


def compact_sstables(
    runs: list[list[tuple[str, int, bytes | None]]],
    drop_tombstones: bool = False,
) -> list[tuple[str, int, bytes | None]]:
    """Merge sorted (key, seq_no, value_or_none) runs keeping only highest seq_no per key."""
    heap: list[tuple[str, int, int, int, bytes | None]] = []
    for run_id, run in enumerate(runs):
        if run:
            k, seq, val = run[0]
            heapq.heappush(heap, (k, -seq, run_id, 0, val))

    out: list[tuple[str, int, bytes | None]] = []
    last_key: str | None = None
    while heap:
        k, neg_seq, run_id, idx, val = heapq.heappop(heap)
        if k != last_key:
            last_key = k
            if not (drop_tombstones and val is None):
                out.append((k, -neg_seq, val))
        nxt_idx = idx + 1
        if nxt_idx < len(runs[run_id]):
            nk, nseq, nval = runs[run_id][nxt_idx]
            heapq.heappush(heap, (nk, -nseq, run_id, nxt_idx, nval))
    return out
'''),
                ("sys-exp-bbr-pacing-window", "Bottleneck-bandwidth x min-prop-RTT pacing window estimator", ("congestion_control", "networking"),
                 '''from collections import deque


class BandwidthDelayEstimator:
    """Tracks windowed max delivery rate and min RTT to size BDP pacing inflight cap."""

    def __init__(self, window_samples: int = 16) -> None:
        self.rates: deque[float] = deque(maxlen=window_samples)
        self.min_rtt_s = float("inf")

    def on_ack(self, delivered_bytes: int, interval_s: float, rtt_s: float) -> None:
        if interval_s > 0:
            self.rates.append(delivered_bytes / interval_s)
        if rtt_s > 0:
            self.min_rtt_s = min(self.min_rtt_s, rtt_s)

    def target_cwnd_bytes(self, gain: float = 2.0) -> int:
        if not self.rates or self.min_rtt_s == float("inf"):
            return 14_600
        btl_bw = max(self.rates)
        return max(4096, int(gain * btl_bw * self.min_rtt_s))
'''),
            ],
        },
    )

def _build_testing_patterns() -> list:
    return _domain_specs(
        "testing",
        {
            "beginner": [
                ("test-beg-tmpdir-fixture", "pytest temp-dir fixture writing and reading a golden file", ("pytest", "fixtures"),
                 '''import json
from pathlib import Path


def write_golden(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Serialize a golden JSON fixture into an isolated temporary directory."""
    target = tmp_path / "golden.json"
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return target


def load_golden(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
'''),
                ("test-beg-approx-assert", "Numeric tolerance assertion helper with informative message", ("assertions", "floats"),
                 '''def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    """Assert two floats match within absolute tolerance, reporting the delta."""
    if abs(actual - expected) > tol:
        raise AssertionError(f"expected {expected!r} +/- {tol!r}, got {actual!r} (delta {abs(actual - expected)!r})")
'''),
                ("test-beg-fake-clock", "Deterministic injectable clock for time-dependent unit tests", ("clock", "determinism"),
                 '''class FakeClock:
    """Manually advanced monotonic clock replacing wall time in tests."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("cannot move a monotonic clock backwards")
        self.now += seconds
        return self.now
'''),
                ("test-beg-stub-http-client", "Hand-written HTTP client test double with canned responses", ("stubs", "http"),
                 '''class StubHttpClient:
    """Minimal transport double returning pre-registered status codes and bodies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._routes: dict[tuple[str, str], tuple[int, str]] = {}

    def register(self, method: str, path: str, status: int, body: str) -> None:
        self._routes[(method.upper(), path)] = (status, body)

    def request(self, method: str, path: str) -> tuple[int, str]:
        self.calls.append((method.upper(), path))
        return self._routes.get((method.upper(), path), (404, ""))
'''),
                ("test-beg-table-params", "Parametrized boundary-value test case matrix builder", ("parametrize", "edge_cases"),
                 '''def boundary_cases(limit: int) -> list[tuple[int, bool]]:
    """Produce (input, expected) pairs around inclusive/exclusive range limits."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return [
        (-1, False),
        (0, True),
        (1, True),
        (limit - 1, True),
        (limit, True),
        (limit + 1, False),
    ]
'''),
                ("test-beg-subprocess-runner", "Subprocess invocation harness asserting exit code and stderr", ("subprocess", "cli_testing"),
                 '''import subprocess
import sys


def run_python(args: list[str], timeout_s: float = 10.0) -> subprocess.CompletedProcess[str]:
    """Run a Python script in a subprocess with captured output and enforced timeout."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def assert_failed(proc: subprocess.CompletedProcess[str], needle: str) -> None:
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit code")
    if needle not in proc.stderr:
        raise AssertionError(f"stderr missing {needle!r}: {proc.stderr!r}")
'''),
                ("test-beg-snapshot-diff", "Unified textual snapshot comparator with actionable diff output", ("snapshots", "diffing"),
                 '''import difflib


def assert_matches_snapshot(actual: str, expected: str) -> None:
    """Raise AssertionError containing a unified diff when text differs."""
    if actual == expected:
        return
    diff = "\\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="snapshot",
            tofile="actual",
            lineterm="",
        )
    )
    raise AssertionError(f"snapshot mismatch:\\n{diff}")
'''),
                ("test-beg-recording-spy", "Call-recording spy object for collaboration assertions", ("spies", "mocks"),
                 '''class CallSpy:
    """Records positional and keyword arguments so tests can assert interactions."""

    def __init__(self, return_value: object = None) -> None:
        self.return_value = return_value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.return_value

    def call_count(self) -> int:
        return len(self.calls)

    def last_args(self) -> tuple[object, ...]:
        return self.calls[-1][0] if self.calls else ()
'''),
                ("test-beg-json-fixture-loader", "Fixture directory loader resolving JSON payloads by scenario name", ("fixtures", "json"),
                 '''import json
from pathlib import Path


def load_fixture(fixtures_dir: Path, scenario: str) -> dict[str, object]:
    """Load `<scenario>.json` from a fixture directory with a clear error when missing."""
    path = fixtures_dir / f"{scenario}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing fixture {scenario!r} in {fixtures_dir}")
    return json.loads(path.read_text(encoding="utf-8"))
'''),
            ],
            "intermediate": [
                ("test-int-property-roundtrip", "Property-based encode/decode round-trip harness with shrinking", ("property_based", "hypothesis"),
                 '''import random
from collections.abc import Callable


def check_roundtrip(
    encode: Callable[[bytes], bytes],
    decode: Callable[[bytes], bytes],
    samples: int = 200,
    seed: int = 7,
    max_len: int = 200,
) -> None:
    """Fuzz encode/decode identity and shrink to the smallest failing input."""
    rng = random.Random(seed)
    for _ in range(samples):
        payload = bytes(rng.randrange(256) for _ in range(rng.randrange(max_len)))
        if decode(encode(payload)) != payload:
            minimal = payload
            while minimal:
                for cut in range(len(minimal)):
                    candidate = minimal[:cut] + minimal[cut + 1 :]
                    if decode(encode(candidate)) != candidate:
                        minimal = candidate
                        break
                else:
                    break
            raise AssertionError(f"roundtrip failed for {minimal!r}")
'''),
                ("test-int-mutation-score", "Mutation-style test strength scorer on injected mutations", ("mutation_testing", "coverage"),
                 '''from collections.abc import Callable


def mutation_score(
    mutants: list[tuple[str, Callable[[int], int]]],
    original: Callable[[int], int],
    inputs: list[int],
) -> dict[str, object]:
    """Report surviving mutants that behave identically to the original implementation."""
    survivors: list[str] = []
    for name, mutant in mutants:
        killed = any(mutant(x) != original(x) for x in inputs)
        if not killed:
            survivors.append(name)
    total = len(mutants)
    killed_count = total - len(survivors)
    return {"score": killed_count / total if total else 1.0, "survivors": survivors}
'''),
                ("test-int-http-contract", "HTTP response contract tester covering status, schema keys, and headers", ("contract_testing", "http"),
                 '''def check_endpoint_contract(
    response: dict[str, object],
    expected_status: int,
    required_keys: set[str],
    required_headers: set[str] = frozenset(),
) -> list[str]:
    """Return contract violations for a serialized HTTP response envelope."""
    problems: list[str] = []
    status = response.get("status")
    if status != expected_status:
        problems.append(f"status {status!r} != {expected_status}")
    body = response.get("body")
    if not isinstance(body, dict):
        problems.append("body must be a JSON object")
    else:
        missing = required_keys - set(body)
        if missing:
            problems.append(f"body missing keys: {sorted(missing)}")
    headers = response.get("headers")
    header_map = {str(k).lower() for k in headers} if isinstance(headers, dict) else set()
    absent = {h.lower() for h in required_headers} - header_map
    if absent:
        problems.append(f"missing headers: {sorted(absent)}")
    return problems
'''),
                ("test-int-db-transaction-rollback", "Transactional test harness asserting rollback isolation between cases", ("database_testing", "isolation"),
                 '''class TransactionalTestHarness:
    """Wraps each test case in a transaction that is always rolled back."""

    def __init__(self) -> None:
        self.committed: list[tuple[str, object]] = []
        self._staged: dict[str, object] = {}

    def run_case(self, case: dict[str, object]) -> dict[str, object]:
        self._staged = {}
        try:
            self._staged.update(case)
            if case.get("force_error"):
                raise RuntimeError("simulated failure inside transaction")
            snapshot = dict(self._staged)
        finally:
            self._staged = {}
        return snapshot

    def commit(self, key: str, value: object) -> None:
        self.committed.append((key, value))
'''),
                ("test-int-flaky-retry-report", "Flake detector aggregating repeated outcomes into pass-rate report", ("flakiness", "reporting"),
                 '''from collections import Counter


def flake_report(results: dict[str, list[bool]]) -> dict[str, object]:
    """Summarize repeated test outcomes, flagging tests that sometimes fail."""
    report: dict[str, object] = {}
    flaky: list[str] = []
    for name, outcomes in results.items():
        counter = Counter(outcomes)
        passes = counter.get(True, 0)
        totals = len(outcomes)
        rate = passes / totals if totals else 0.0
        report[name] = {"runs": totals, "pass_rate": rate}
        if 0.0 < rate < 1.0:
            flaky.append(name)
    return {"tests": report, "flaky": sorted(flaky)}
'''),
                ("test-int-golden-update-guard", "Golden-file update guard requiring explicit opt-in flags", ("golden_files", "snapshots"),
                 '''import os
from pathlib import Path


def update_golden_if_opted_in(path: Path, actual: str) -> bool:
    """Rewrite a golden file only when UPDATE_GOLDEN=1; otherwise compare strictly."""
    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.write_text(actual, encoding="utf-8")
        return True
    expected = path.read_text(encoding="utf-8") if path.is_file() else ""
    if expected != actual:
        raise AssertionError(f"golden mismatch for {path.name}; rerun with UPDATE_GOLDEN=1 to accept")
    return False
'''),
                ("test-int-async-fixture-scope", "Async fixture lifecycle scope manager with teardown ordering", ("asyncio", "fixtures"),
                 '''from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ScopedResource:
    """Tracks acquisition/teardown order to validate fixture scoping rules."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self._open = False

    @asynccontextmanager
    async def session(self, name: str) -> AsyncIterator[ScopedResource]:
        self.events.append(f"setup:{name}")
        self._open = True
        try:
            yield self
        finally:
            self._open = False
            self.events.append(f"teardown:{name}")
'''),
                ("test-int-cli-golden-stdout", "CLI end-to-end harness comparing stdout/stderr against golden text", ("cli_testing", "e2e"),
                 '''from dataclasses import dataclass


@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str


def assert_cli_golden(result: CliResult, expected_code: int, expected_stdout: str) -> None:
    """Assert CLI exit code plus normalized stdout, ignoring trailing whitespace."""
    if result.code != expected_code:
        raise AssertionError(f"exit code {result.code} != {expected_code}; stderr={result.stderr!r}")
    if result.stdout.rstrip() != expected_stdout.rstrip():
        raise AssertionError(f"stdout mismatch:\\n{result.stdout!r}\\n!=\\n{expected_stdout!r}")
'''),
                ("test-int-fuzz-parser", "Structure-aware fuzz harness asserting parsers never crash on garbage", ("fuzzing", "robustness"),
                 '''import random
from collections.abc import Callable


def fuzz_never_raises(
    parse: Callable[[str], object],
    samples: int = 500,
    seed: int = 11,
    max_len: int = 120,
    alphabet: str = "{}[]():;,\\"'\\\\/0123456789abc \\n\\t",
) -> None:
    """Assert the parser raises only ValueError/TypeError, never unexpected errors."""
    rng = random.Random(seed)
    for _ in range(samples):
        payload = "".join(rng.choice(alphabet) for _ in range(rng.randrange(max_len)))
        try:
            parse(payload)
        except (ValueError, TypeError):
            continue
        except Exception as exc:  # noqa: BLE001 - fuzz harness surfaces the failure
            raise AssertionError(f"unexpected {type(exc).__name__} on {payload!r}") from exc
'''),
            ],
            "advanced": [
                ("test-adv-isolation-leak-detector", "Cross-test global state leak detector using namespace snapshots", ("isolation", "global_state"),
                 '''def dict_state_diff(before: dict[str, object], after: dict[str, object]) -> dict[str, list[str]]:
    """Report keys added, removed, and mutated between two namespace snapshots."""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def assert_no_leak(before: dict[str, object], after: dict[str, object], allow: set[str] = frozenset()) -> None:
    diff = dict_state_diff(before, after)
    leaked = {k: [n for n in names if n not in allow] for k, names in diff.items()}
    if any(leaked.values()):
        raise AssertionError(f"test leaked global state: {leaked}")
'''),
                ("test-adv-deterministic-concurrency", "Deterministic interleaving scheduler for concurrency bug reproduction", ("concurrency", "determinism"),
                 '''from dataclasses import dataclass, field


@dataclass
class DeterministicScheduler:
    """Replays a fixed schedule of greenlet-style steps to reproduce races."""

    steps: list[int]
    cursor: int = 0
    history: list[int] = field(default_factory=list)

    def next_thread(self, num_threads: int) -> int:
        if self.cursor >= len(self.steps):
            return self.cursor % num_threads
        choice = self.steps[self.cursor]
        self.cursor += 1
        if not 0 <= choice < num_threads:
            raise ValueError(f"step {choice} outside 0..{num_threads - 1}")
        self.history.append(choice)
        return choice
'''),
                ("test-adv-coverage-of-diff", "Patch-coverage analyzer flagging modified lines never executed", ("coverage", "diff_coverage"),
                 '''def uncovered_changed_lines(executed: set[int], changed: set[int]) -> list[int]:
    """Return changed lines that no test executed (diff coverage gaps)."""
    return sorted(changed - executed)


def diff_coverage_ratio(executed: set[int], changed: set[int]) -> float:
    if not changed:
        return 1.0
    return len(changed & executed) / len(changed)
'''),
                ("test-adv-hypothesis-state-machine", "Finite-state model oracle comparing implementation to reference model", ("model_based", "state_machine"),
                 '''class ModelBasedOracle:
    """Runs the same operation sequence against SUT and reference model."""

    def __init__(self, operations: list[str]) -> None:
        self.operations = operations

    def run(self, sut, model, args_by_op: dict[str, list[tuple[object, ...]]]) -> None:
        for op in self.operations:
            for args in args_by_op.get(op, [()]):
                got = getattr(sut, op)(*args)
                want = getattr(model, op)(*args)
                if got != want:
                    raise AssertionError(f"divergence on {op}{args}: sut={got!r} model={want!r}")
'''),
                ("test-adv-snapshot-redaction", "Deterministic snapshot normalizer redacting volatile ids and timestamps", ("snapshots", "normalization"),
                 '''import re

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_ISO_RE = re.compile(r"\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?Z?")


def normalize_snapshot(text: str) -> str:
    """Replace volatile UUIDs and ISO-8601 timestamps with stable placeholders."""
    text = _UUID_RE.sub("<uuid>", text)
    return _ISO_RE.sub("<timestamp>", text)
'''),
                ("test-adv-parallel-shard-balancer", "Test-suite shard balancer minimizing duration skew across workers", ("sharding", "parallelism"),
                 '''def balance_shards(durations: dict[str, float], workers: int) -> list[list[str]]:
    """Greedily assign slowest-first tests to the least-loaded worker (LPT heuristic)."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    shards: list[list[str]] = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for name, duration in sorted(durations.items(), key=lambda kv: (-kv[1], kv[0])):
        target = loads.index(min(loads))
        shards[target].append(name)
        loads[target] += duration
    return shards
'''),
                ("test-adv-fault-injection", "Fault-injecting transport wrapper for partial failure simulation", ("fault_injection", "resilience"),
                 '''class FaultInjector:
    """Deterministically fails the Nth call and returns synthetic errors."""

    def __init__(self, fail_on: set[int], error: str = "injected failure") -> None:
        self.fail_on = fail_on
        self.error = error
        self.calls = 0

    def __call__(self, fn, *args, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            raise RuntimeError(self.error)
        return fn(*args, **kwargs)
'''),
                ("test-adv-metamorphic-relations", "Metamorphic testing relations (permutation invariance, scaling)", ("metamorphic", "oracle_free"),
                 '''def check_permutation_invariance(fn, items: list[int], shuffled: list[int]) -> None:
    """A sorted-aggregate function must be invariant under input permutation."""
    if sorted(items) != sorted(shuffled):
        raise ValueError("inputs are not permutations of each other")
    if fn(items) != fn(shuffled):
        raise AssertionError("function is not permutation invariant")


def check_monotonic_scaling(fn, items: list[int]) -> None:
    """Doubling every element must double a size-linear aggregate."""
    base = fn(items)
    doubled = fn([x * 2 for x in items])
    if doubled != base * 2:
        raise AssertionError(f"expected linear scaling, got {base!r} -> {doubled!r}")
'''),
                ("test-adv-benchmark-regression", "Statistical benchmark regression gate using median-of-samples", ("benchmarks", "performance"),
                 '''import statistics


def assert_no_regression(
    baseline_ns: list[int],
    candidate_ns: list[int],
    tolerance: float = 1.10,
    warmup: int = 1,
) -> float:
    """Fail when candidate median exceeds baseline median by more than tolerance."""
    base = statistics.median(baseline_ns[warmup:])
    cand = statistics.median(candidate_ns[warmup:])
    ratio = cand / base if base else 0.0
    if ratio > tolerance:
        raise AssertionError(f"performance regression: {base:.0f}ns -> {cand:.0f}ns ({ratio:.2f}x)")
    return ratio
'''),
            ],
            "expert": [
                ("test-exp-tlc-style-bfs", "Bounded BFS state-space explorer with invariant violation witnesses", ("model_checking", "bfs"),
                 '''from collections import deque


def explore_states(
    initial: tuple[int, ...],
    transitions,
    invariants: list,
    max_states: int = 10_000,
) -> dict[str, object]:
    """Breadth-first explore reachable states until an invariant breaks or bound hits."""
    seen = {initial}
    queue: deque[tuple[tuple[int, ...], list[tuple[int, ...]]]] = deque([(initial, [initial])])
    explored = 0
    while queue and explored < max_states:
        state, path = queue.popleft()
        explored += 1
        for inv in invariants:
            if not inv(state):
                return {"ok": False, "witness": path, "explored": explored}
        for nxt in transitions(state):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return {"ok": True, "witness": [], "explored": explored, "truncated": explored >= max_states}
'''),
                ("test-exp-linearizability-checker", "Sequential-specification linearizability checker over concurrent histories", ("linearizability", "concurrency"),
                 '''from itertools import permutations


def is_linearizable(history: list[tuple[str, str, str, str]], apply_op) -> bool:
    """Check a (client, op, key, value) history against a sequential spec model."""
    n = len(history)
    for order in permutations(range(n)):
        state: dict[str, str] = {}
        ok = True
        for idx in order:
            _client, op, key, value = history[idx]
            if op == "write":
                state[key] = value
            elif op == "read":
                model = state.get(key)
                if apply_op(op, key, value, model) is False:
                    ok = False
                    break
        if ok:
            return True
    return False
'''),
                ("test-exp-differential-fuzzer", "Differential fuzzer shrinking mismatches between two implementations", ("differential_testing", "shrinking"),
                 '''def differential_fuzz(
    impl_a,
    impl_b,
    corpus: list[bytes],
    max_rounds: int = 5,
) -> list[bytes]:
    """Byte-level mutation fuzzing that returns a shrink-minimized divergence."""
    import random

    rng = random.Random(1234)
    for _ in range(max_rounds):
        for seed in corpus:
            data = bytearray(seed)
            for _ in range(rng.randrange(1, 4)):
                if data:
                    data[rng.randrange(len(data))] = rng.randrange(256)
            payload = bytes(data)
            try:
                va = impl_a(payload)
            except Exception as exc:  # noqa: BLE001
                va = ("err", type(exc).__name__)
            try:
                vb = impl_b(payload)
            except Exception as exc:  # noqa: BLE001
                vb = ("err", type(exc).__name__)
            if va != vb:
                minimal = payload
                changed = True
                while changed and minimal:
                    changed = False
                    for cut in range(len(minimal)):
                        cand = minimal[:cut] + minimal[cut + 1 :]
                        try:
                            ca = impl_a(cand)
                        except Exception as exc:  # noqa: BLE001
                            ca = ("err", type(exc).__name__)
                        try:
                            cb = impl_b(cand)
                        except Exception as exc:  # noqa: BLE001
                            cb = ("err", type(exc).__name__)
                        if ca != cb:
                            minimal = cand
                            changed = True
                            break
                return [minimal]
    return []
'''),
                ("test-exp-pbt-strategy-recursion", "Recursive generator with size-bounded shrinking invariants", ("property_based", "generators"),
                 '''import random


def gen_tree(depth: int, rng: random.Random, max_branch: int = 3) -> dict[str, object]:
    """Generate a random bounded-depth tree suitable for property tests."""
    if depth <= 0:
        return {"leaf": rng.randrange(100)}
    return {"children": [gen_tree(depth - 1, rng, max_branch) for _ in range(rng.randrange(max_branch + 1))]}


def tree_depth(node: dict[str, object]) -> int:
    children = node.get("children")
    if not isinstance(children, list) or not children:
        return 1
    return 1 + max(tree_depth(c) for c in children)


def gen_valid_tree(depth: int, rng: random.Random, max_branch: int = 3) -> dict[str, object]:
    """Generate a tree and assert the recursive size invariant holds."""
    tree = gen_tree(depth, rng, max_branch)
    if tree_depth(tree) > depth:
        raise AssertionError("generator exceeded requested depth bound")
    return tree
'''),
                ("test-exp-mutation-operator-table", "Configurable mutation operator table (AOR/ROR/LCR/SDL) for coverage scoring", ("mutation_testing", "operators"),
                 '''import re

_MUTATIONS: dict[str, list[tuple[str, str]]] = {
    "AOR": [(" + ", " - "), (" * ", " / ")],
    "ROR": [(" < ", " <= "), (" > ", " >= "), (" == ", " != ")],
    "LCR": [(" and ", " or ")],
    "SDL": [(r"return ", r"pass  # ")],
}


def generate_mutants(source: str) -> list[tuple[str, str]]:
    """Emit (operator, mutated_source) pairs, one mutation per candidate site."""
    mutants: list[tuple[str, str]] = []
    for op, table in _MUTATIONS.items():
        for old, new in table:
            if old.startswith("return"):
                if old in source:
                    idx = source.index(old)
                    mutants.append((op, source[:idx] + new + source[idx + len(old):]))
                continue
            idx = source.find(old)
            while idx != -1:
                mutants.append((op, source[:idx] + new + source[idx + len(old):]))
                idx = source.find(old, idx + 1)
    return mutants
'''),
                ("test-exp-record-replay-harness", "Cassette-based record/replay harness with redaction of secrets", ("record_replay", "network"),
                 '''import json


class NetworkCassette:
    """Records request/response pairs for deterministic offline replay."""

    SECRET_HEADERS = {"authorization", "cookie", "x-api-key"}

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, object]] = {}
        self.mode = "record"

    def _redact(self, headers: dict[str, str]) -> dict[str, str]:
        return {k: ("<redacted>" if k.lower() in self.SECRET_HEADERS else v) for k, v in headers.items()}

    def record(self, key: str, request: dict[str, object], response: dict[str, object]) -> None:
        self._entries[key] = {
            "request": {"headers": self._redact(request.get("headers", {}))},  # type: ignore[arg-type]
            "response": response,
        }

    def replay(self, key: str) -> dict[str, object]:
        if key not in self._entries:
            raise KeyError(f"no cassette entry for {key!r}")
        return json.loads(json.dumps(self._entries[key]["response"]))
'''),
                ("test-exp-race-schedule-search", "Systematic interleaving search reporting the shortest racing prefix", ("concurrency", "schedule_search"),
                 '''from itertools import product


def search_racing_prefix(
    programs: list[list[str]],
    step,
    max_depth: int = 6,
) -> list[int] | None:
    """Enumerate interleavings of N programs up to depth, returning first racy schedule."""
    n = len(programs)
    cursors = [0] * n
    found: list[int] | None = None

    def dfs(schedule: list[int]) -> bool:
        nonlocal found
        if len(schedule) > max_depth:
            return False
        if any(step(i, j) for i in range(n) for j in range(i + 1, n)):
            found = list(schedule)
            return True
        for tid in range(n):
            if cursors[tid] < len(programs[tid]):
                cursors[tid] += 1
                schedule.append(tid)
                if dfs(schedule):
                    return True
                schedule.pop()
                cursors[tid] -= 1
        return False

    return found if dfs([]) else None
'''),
                ("test-exp-flake-bisect", "Probabilistic flake bisector isolating nondeterminism to a minimal code region", ("flakiness", "bisection"),
                 '''def bisect_flake(regions: list[str], reproduces) -> list[str]:
    """Delta-debug a region list down to the smallest subset that still flakes."""
    current = list(regions)
    granularity = 2
    while len(current) >= 2:
        chunk = max(1, len(current) // granularity)
        reduced = False
        for start in range(0, len(current), chunk):
            candidate = current[:start] + current[start + chunk :]
            if candidate and reproduces(candidate):
                current = candidate
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)
    return current
'''),
                ("test-exp-coverage-guided-mutator", "AFL-style coverage-guided byte mutator with corpus queueing", ("fuzzing", "coverage_guided"),
                 '''import random


class CoverageGuidedFuzzer:
    """Corpus-queue fuzzer inserting/byte-flipping inputs that reach new edges."""

    def __init__(self, seed_bytes: bytes, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.queue: list[bytes] = [seed_bytes]
        self.seen_edges: set[int] = set()

    def mutate(self, base: bytes) -> bytes:
        data = bytearray(base)
        op = self.rng.randrange(3)
        if op == 0 and data:
            data[self.rng.randrange(len(data))] ^= 1 << self.rng.randrange(8)
        elif op == 1:
            pos = self.rng.randrange(len(data) + 1)
            data[pos:pos] = bytes([self.rng.randrange(256)])
        elif data:
            del data[self.rng.randrange(len(data))]
        return bytes(data)

    def run_round(self, execute) -> int:
        new_edges = 0
        for base in list(self.queue)[: 8]:
            candidate = self.mutate(base)
            edges = set(execute(candidate))
            fresh = edges - self.seen_edges
            if fresh:
                self.seen_edges |= fresh
                self.queue.append(candidate)
                new_edges += len(fresh)
        return new_edges
'''),
                ("test-exp-oracle-differential-schema", "Schema-driven differential oracles comparing serializer and parser symmetry", ("differential_testing", "schemas"),
                 '''def check_serializer_symmetry(schema: list[dict[str, object]], serializer, parser) -> list[str]:
    """Round-trip each schema-valid record and report first asymmetry per case."""
    problems: list[str] = []
    for idx, record in enumerate(schema):
        encoded = serializer(record)
        decoded = parser(encoded)
        if decoded != record:
            problems.append(f"case {idx}: {record!r} -> {encoded!r} -> {decoded!r}")
    return problems
'''),
            ],
        },
    )

def _build_database_patterns() -> list:
    return _domain_specs(
        "database",
        {
            "beginner": [
                ("db-beg-parameterized-query", "Parameterized SELECT builder preventing SQL injection", ("sql", "injection"),
                 '''def select_by_ids(table: str, ids: list[int], allowed_tables: set[str]) -> tuple[str, tuple[int, ...]]:
    """Build a parameterized IN-query, validating the table name against an allowlist."""
    if table not in allowed_tables:
        raise ValueError(f"unknown table {table!r}")
    if not ids:
        raise ValueError("ids must not be empty")
    placeholders = ", ".join("?" for _ in ids)
    return f"SELECT * FROM {table} WHERE id IN ({placeholders})", tuple(ids)
'''),
                ("db-beg-connection-url", "Database URL parser with percent-decoded credentials", ("config", "urls"),
                 '''from urllib.parse import unquote, urlsplit


def parse_db_url(url: str) -> dict[str, object]:
    """Parse postgres://user:pass@host:5432/dbname into a connection dict."""
    parts = urlsplit(url)
    if not parts.scheme:
        raise ValueError("missing scheme")
    return {
        "scheme": parts.scheme,
        "user": unquote(parts.username or ""),
        "password": unquote(parts.password or ""),
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": parts.path.lstrip("/"),
    }
'''),
                ("db-beg-upsert-sql", "Dialect-aware INSERT ... ON CONFLICT upsert statement builder", ("sql", "upsert"),
                 '''def build_upsert(table: str, columns: list[str], conflict_columns: list[str]) -> str:
    """Render a PostgreSQL upsert statement updating all non-conflict columns."""
    if not columns or not set(conflict_columns) <= set(columns):
        raise ValueError("conflict columns must be a subset of the inserted columns")
    cols = ", ".join(columns)
    marks = ", ".join(f":{c}" for c in columns)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c not in conflict_columns)
    conflict = ", ".join(conflict_columns)
    if not updates:
        return f"INSERT INTO {table} ({cols}) VALUES ({marks}) ON CONFLICT ({conflict}) DO NOTHING"
    return f"INSERT INTO {table} ({cols}) VALUES ({marks}) ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
'''),
                ("db-beg-transaction-ctx", "Context-manager transaction wrapper committing or rolling back", ("transactions", "context_managers"),
                 '''from contextlib import contextmanager
from collections.abc import Iterator


class FakeConnection:
    """Tracks BEGIN/COMMIT/ROLLBACK calls for unit tests."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str) -> None:
        self.statements.append(sql)


@contextmanager
def transaction(conn: FakeConnection) -> Iterator[FakeConnection]:
    """Wrap statements in BEGIN/COMMIT, rolling back if the block raises."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
'''),
                ("db-beg-row-dict-mapping", "Cursor row-to-dict mapper with column deduplication", ("mapping", "rows"),
                 '''def rows_to_dicts(columns: list[str], rows: list[tuple[object, ...]]) -> list[dict[str, object]]:
    """Zip column names to row tuples, suffixing duplicate column names."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for col in columns:
        if col in seen:
            seen[col] += 1
            names.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            names.append(col)
    return [dict(zip(names, row)) for row in rows]
'''),
                ("db-beg-pagination-cursor", "Keyset (seek) pagination predicate builder", ("pagination", "sql"),
                 '''def keyset_page_sql(table: str, order_column: str, last_value: object | None, limit: int) -> tuple[str, list[object]]:
    """Build seek pagination SQL avoiding OFFSET scans on large tables."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    params: list[object] = []
    where = ""
    if last_value is not None:
        where = f"WHERE {order_column} > ?"
        params.append(last_value)
    params.append(limit)
    return f"SELECT * FROM {table} {where} ORDER BY {order_column} LIMIT ?".strip(), params
'''),
                ("db-beg-migration-idempotency", "Idempotent migration runner tracking applied versions", ("migrations", "schema"),
                 '''class MigrationRunner:
    """Applies numbered migrations exactly once, recorded in a tracking table."""

    def __init__(self, applied: set[int] | None = None) -> None:
        self.applied = set(applied or ())

    def run(self, migrations: dict[int, str], execute) -> list[int]:
        ran: list[int] = []
        for version in sorted(migrations):
            if version in self.applied:
                continue
            execute(migrations[version])
            self.applied.add(version)
            ran.append(version)
        return ran
'''),
                ("db-beg-slow-query-log", "Duration-threshold slow query classifier", ("observability", "performance"),
                 '''def slow_queries(entries: list[dict[str, object]], threshold_ms: float = 200.0) -> list[dict[str, object]]:
    """Filter query log entries exceeding the latency threshold, sorted slowest first."""
    flagged = [e for e in entries if float(e.get("duration_ms", 0.0)) > threshold_ms]
    return sorted(flagged, key=lambda e: float(e.get("duration_ms", 0.0)), reverse=True)
'''),
                ("db-beg-nullable-constraint", "NOT NULL constraint violation pre-flight checker", ("constraints", "validation"),
                 '''def check_not_null(rows: list[dict[str, object]], required: list[str]) -> list[dict[str, object]]:
    """Return offending rows with the list of missing required columns."""
    problems: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        missing = [c for c in required if row.get(c) is None]
        if missing:
            problems.append({"row_index": idx, "missing": missing})
    return problems
'''),
            ],
            "intermediate": [
                ("db-int-batch-inserter", "Chunked multi-row INSERT batching respecting parameter limits", ("sql", "batching"),
                 '''def chunk_inserts(rows: list[dict[str, object]], columns: list[str], max_params: int = 1000) -> list[list[dict[str, object]]]:
    """Split rows so each statement stays under the driver's parameter count limit."""
    per_row = len(columns)
    if per_row == 0:
        raise ValueError("columns must not be empty")
    batch_size = max(1, max_params // per_row)
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
'''),
                ("db-int-optimistic-version", "Optimistic concurrency control with version column compare-and-set", ("concurrency", "optimistic_locking"),
                 '''def update_with_version(current_version: int, expected_version: int, changes: dict[str, object]) -> dict[str, object]:
    """Produce a compare-and-set UPDATE, raising on version mismatch."""
    if current_version != expected_version:
        raise ValueError(f"stale write: row is at version {current_version}, expected {expected_version}")
    payload = dict(changes)
    payload["version"] = expected_version + 1
    return payload
'''),
                ("db-int-query-plan-analyzer", "EXPLAIN plan analyzer flagging sequential scans and bad row estimates", ("query_planner", "performance"),
                 '''def analyze_plan(plan: dict[str, object], min_rows: int = 10_000) -> list[str]:
    """Report planner red flags: seq scans on big tables and estimate misprediction."""
    warnings: list[str] = []
    node_type = str(plan.get("Node Type", ""))
    plan_rows = int(plan.get("Plan Rows", 0) or 0)
    actual = int(plan.get("Actual Rows", 0) or 0)
    if node_type == "Seq Scan" and plan_rows >= min_rows:
        warnings.append(f"sequential scan on {plan.get('Relation Name')} with {plan_rows} rows")
    if plan_rows and actual and (actual > plan_rows * 10 or actual * 10 < plan_rows):
        warnings.append(f"row estimate off by >10x: planned {plan_rows}, actual {actual}")
    for child in plan.get("Plans", []) or []:
        warnings.extend(analyze_plan(child, min_rows))
    return warnings
'''),
                ("db-int-connection-pool", "Bounded connection pool with wait-for-availability accounting", ("pooling", "concurrency"),
                 '''import threading


class ConnectionPool:
    """Fixed-size pool blocking callers when all connections are checked out."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("pool size must be positive")
        self.size = size
        self._available = list(range(size))
        self._cond = threading.Condition()
        self.waits = 0

    def acquire(self, timeout: float | None = None) -> int:
        with self._cond:
            while not self._available:
                self.waits += 1
                if not self._cond.wait(timeout):
                    raise TimeoutError("timed out waiting for a pooled connection")
            return self._available.pop()

    def release(self, conn_id: int) -> None:
        with self._cond:
            if conn_id in self._available:
                raise ValueError("connection released twice")
            self._available.append(conn_id)
            self._cond.notify()

    def in_use(self) -> int:
        return self.size - len(self._available)
'''),
                ("db-int-index-advisor", "Composite index advisor ranking candidate column sets by selectivity", ("indexing", "performance"),
                 '''def rank_index_candidates(
    workloads: list[tuple[str, tuple[str, ...]]],
    selectivity: dict[str, float],
) -> list[tuple[tuple[str, ...], float]]:
    """Score candidate leading-column sets by summed query weight * column selectivity."""
    scores: dict[tuple[str, ...], float] = {}
    for _query, columns in workloads:
        key = tuple(columns)
        if not key:
            continue
        scores[key] = scores.get(key, 0.0) + max(selectivity.get(c, 0.0) for c in key)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
'''),
                ("db-int-deadlock-retry", "Advisory-lock acquisition with deadlock-aware retry loop", ("locking", "retries"),
                 '''import time


def with_deadlock_retry(fn, attempts: int = 5, base_delay: float = 0.05, sleep=time.sleep):
    """Retry a transaction function on serialization/deadlock failures."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - driver-specific error types
            if "deadlock" not in str(exc).lower() and "serialization" not in str(exc).lower():
                raise
            last = exc
            sleep(base_delay * (2 ** attempt))
    assert last is not None
    raise last
'''),
                ("db-int-mvcc-vacuum", "Dead-tuple vacuum threshold calculation (PostgreSQL-style)", ("vacuum", "maintenance"),
                 '''def needs_vacuum(
    live_tuples: int,
    dead_tuples: int,
    scale_factor: float = 0.2,
    base_threshold: int = 50,
) -> bool:
    """PostgreSQL autovacuum: vacuum when dead tuples exceed threshold + scale * live."""
    if live_tuples < 0 or dead_tuples < 0:
        raise ValueError("tuple counts must be non-negative")
    return dead_tuples >= base_threshold + scale_factor * live_tuples
'''),
                ("db-int-partition-pruner", "Range-partitioned table pruner selecting overlapping partitions", ("partitioning", "query_planner"),
                 '''def prune_partitions(
    partitions: list[tuple[str, int, int]],
    lo: int | None,
    hi: int | None,
) -> list[str]:
    """Return partitions whose [start, end) range overlaps the query bounds."""
    names: list[str] = []
    for name, start, end in partitions:
        if hi is not None and start >= hi:
            continue
        if lo is not None and end <= lo:
            continue
        names.append(name)
    return names
'''),
                ("db-int-replica-lag-router", "Read/write splitter routing reads to replicas under lag budget", ("replication", "routing"),
                 '''def choose_route(
    statement: str,
    replica_lags_s: dict[str, float],
    max_lag_s: float = 1.0,
    sticky_after_write_s: float = 5.0,
    since_write_s: float = 999.0,
) -> tuple[str, str | None]:
    """Route writes to the primary and reads to the freshest in-budget replica."""
    if statement.lstrip().lower().startswith(("select", "with")):
        if since_write_s < sticky_after_write_s:
            return "primary", None
        eligible = [(lag, name) for name, lag in replica_lags_s.items() if lag <= max_lag_s]
        if eligible:
            return "replica", min(eligible)[1]
    return "primary", None
'''),
                ("db-int-bulk-copy-format", "PostgreSQL COPY text-format encoder with escaping", ("bulk_load", "serialization"),
                 '''def to_copy_text(rows: list[list[object]], null_token: str = "\\\\N") -> str:
    """Encode rows in PostgreSQL COPY ... FROM STDIN text format."""
    out: list[str] = []
    for row in rows:
        cells: list[str] = []
        for value in row:
            if value is None:
                cells.append(null_token)
                continue
            text = str(value)
            escaped = (text.replace("\\\\", "\\\\\\\\").replace("\\t", "\\\\t").replace("\\n", "\\\\n").replace("\\r", "\\\\r"))
            cells.append(escaped)
        out.append("\\t".join(cells))
    return "\\n".join(out) + "\\n"
'''),
            ],
            "advanced": [
                ("db-adv-pg-mvcc-snapshot", "PostgreSQL-style xmin/xmax tuple visibility rules", ("mvcc", "visibility"),
                 '''def tuple_visible(xmin: int, xmax: int | None, snapshot_xmin: int, snapshot_xmax: int, in_progress: set[int]) -> bool:
    """Evaluate MVCC visibility for a tuple given a snapshot's transaction horizon."""
    if xmin in in_progress or xmin >= snapshot_xmax:
        return False
    if xmax is None or xmax in in_progress or xmax >= snapshot_xmax:
        return True
    return xmax < snapshot_xmin
'''),
                ("db-adv-two-phase-commit", "Two-phase commit coordinator with participant freeze and abort recovery", ("distributed_transactions", "2pc"),
                 '''class TwoPhaseCommitCoordinator:
    """Drives prepare/commit across participants with abort-on-any-vote-no semantics."""

    def __init__(self, participants: list[str]) -> None:
        self.participants = participants
        self.prepared: set[str] = set()
        self.decision: str | None = None

    def prepare(self, votes: dict[str, bool]) -> str:
        for p in self.participants:
            if not votes.get(p, False):
                self.decision = "abort"
                self.prepared.clear()
                return self.decision
            self.prepared.add(p)
        self.decision = "commit"
        return self.decision

    def commit(self) -> list[str]:
        if self.decision != "commit":
            raise RuntimeError("cannot commit without unanimous prepare")
        committed = sorted(self.prepared)
        self.prepared.clear()
        return committed
'''),
                ("db-adv-b-tree-split", "B+tree leaf split and median promotion for ordered index maintenance", ("btree", "indexing"),
                 '''def split_leaf(keys: list[int], values: list[object], order: int) -> dict[str, object]:
    """Split an overfull leaf node, promoting the median key to the parent."""
    if len(keys) != len(values):
        raise ValueError("keys and values must align")
    if len(keys) <= order:
        raise ValueError("node is not overfull")
    mid = len(keys) // 2
    return {
        "left": (keys[:mid], values[:mid]),
        "promote": keys[mid],
        "right": (keys[mid + 1 :], values[mid + 1 :]),
    }
'''),
                ("db-adv-write-skew-detection", "Serializable snapshot isolation write-skew dependency detector", ("ssi", "isolation"),
                 '''def detect_write_skew(
    read_sets: dict[int, set[str]],
    write_sets: dict[int, set[str]],
    committed_first: set[int],
) -> set[int]:
    """Flag transactions whose writes invalidate a concurrent transaction's reads."""
    doomed: set[int] = set()
    for tx, writes in write_sets.items():
        for other, reads in read_sets.items():
            if other == tx:
                continue
            if writes & reads and (tx in committed_first) != (other in committed_first):
                doomed.add(tx)
    return doomed
'''),
                ("db-adv-logical-replication-decoder", "WAL logical decoding with LSN-ordered change stream assembly", ("replication", "wal"),
                 '''def decode_change_stream(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Order WAL records by LSN and merge begin/commit into logical change events."""
    pending: dict[int, list[dict[str, object]]] = {}
    committed: list[dict[str, object]] = []
    for rec in sorted(records, key=lambda r: int(r["lsn"])):
        kind = rec.get("kind")
        xid = int(rec.get("xid", 0))
        if kind == "begin":
            pending[xid] = []
        elif kind in {"insert", "update", "delete"}:
            pending.setdefault(xid, []).append(rec)
        elif kind == "commit" and xid in pending:
            committed.append({"xid": xid, "commit_lsn": rec["lsn"], "changes": pending.pop(xid)})
    return committed
'''),
                ("db-adv-query-rewrite-optimizer", "Predicate pushdown and constant-folding rewrite pass", ("optimizer", "query_rewrite"),
                 '''def pushdown_predicates(
    filters: list[tuple[str, str, object]],
    partition_columns: set[str],
) -> tuple[list[tuple[str, str, object]], list[tuple[str, str, object]]]:
    """Split filters into pushable (partition keyed) and residual sets."""
    pushable: list[tuple[str, str, object]] = []
    residual: list[tuple[str, str, object]] = []
    for predicate in filters:
        col, op, value = predicate
        if col in partition_columns and op in {"=", "<", "<=", ">", ">="}:
            pushable.append(predicate)
        else:
            residual.append(predicate)
    return pushable, residual
'''),
                ("db-adv-cost-model-join-order", "Dynamic-programming join order selection with cardinality costs", ("optimizer", "join_order"),
                 '''from functools import lru_cache


def best_join_order(
    table_rows: dict[str, int],
    selectivity: dict[frozenset[str], float],
) -> tuple[float, list[str]]:
    """Selinger-style DP over subsets selecting the cheapest left-deep join order."""
    tables = sorted(table_rows)

    @lru_cache(maxsize=None)
    def best(subset: frozenset[str]) -> tuple[float, list[str]]:
        if len(subset) == 1:
            (only,) = tuple(subset)
            return float(table_rows[only]), [only]
        cheapest = (float("inf"), [])
        for name in sorted(subset):
            rest = subset - {name}
            cost, order = best(frozenset(rest))
            rows_left = max(1.0, cost)
            cost_join = rows_left * max(1.0, table_rows[name]) * selectivity.get(subset, 0.01)
            if cost + cost_join < cheapest[0]:
                cheapest = (cost + cost_join, order + [name])
        return cheapest

    cost, order = best(frozenset(tables))
    return cost, order
'''),
                ("db-adv-skew-aware-sharding", "Consistent-hash shard rebalancer minimizing cross-shard moves", ("sharding", "rebalancing"),
                 '''def plan_rebalance(
    shard_load: dict[str, float],
    shard_of_key: dict[str, str],
    target_utilization: float = 0.9,
) -> list[tuple[str, str, str]]:
    """Move keys from overloaded shards to underloaded ones, largest keys first."""
    if not shard_load:
        return []
    mean = sum(shard_load.values()) / len(shard_load)
    ceiling = mean * target_utilization
    over = [s for s, load in shard_load.items() if load > ceiling]
    under = [s for s, load in shard_load.items() if load < ceiling]
    moves: list[tuple[str, str, str]] = []
    load = dict(shard_load)
    for src in over:
        candidates = sorted((k for k, s in shard_of_key.items() if s == src), reverse=True)
        for key in candidates:
            if load[src] <= ceiling or not under:
                break
            dst = min(under, key=lambda s: load[s])
            weight = load[src] / max(1, len(candidates))
            load[src] -= weight
            load[dst] += weight
            moves.append((key, src, dst))
            if load[dst] >= ceiling:
                under.remove(dst)
    return moves
'''),
                ("db-adv-serializable-conflict-graph", "Serialization conflict graph with dangerous structure detection", ("ssi", "conflict_graph"),
                 '''def has_dangerous_structure(
    reads_from: dict[int, set[int]],
    writes_to: dict[int, set[int]],
) -> list[tuple[int, int, int]]:
    """Detect rw-antidependency chains T1 ->rw T2 ->rw T3 (PostgreSQL SSI pivot)."""
    pivots: list[tuple[int, int, int]] = []
    for t2, incoming in reads_from.items():
        if not incoming:
            continue
        for t3, outgoing in reads_from.items():
            if t3 == t2:
                continue
            if t2 in writes_to.get(t3, set()) or t3 in writes_to.get(t2, set()):
                for t1 in incoming:
                    if t1 != t2 and t1 != t3:
                        pivots.append((t1, t2, t3))
    return pivots
'''),
                ("db-adv-column-store-compression", "Run-length + dictionary columnar block encoder with min/max pruning", ("columnar", "compression"),
                 '''def encode_column_block(values: list[object]) -> dict[str, object]:
    """Dictionary-encode a column block and record run boundaries for RLE."""
    dictionary: list[object] = []
    index_of: dict[object, int] = {}
    codes: list[int] = []
    for value in values:
        if value not in index_of:
            index_of[value] = len(dictionary)
            dictionary.append(value)
        codes.append(index_of[value])
    runs: list[tuple[int, int]] = []
    for code in codes:
        if runs and runs[-1][0] == code:
            runs[-1] = (code, runs[-1][1] + 1)
        else:
            runs.append((code, 1))
    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {
        "dictionary": dictionary,
        "runs": runs,
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
    }
'''),
            ],
            "expert": [
                ("db-exp-aries-redo-recovery", "ARIES three-pass redo/undo crash recovery with compensation log records", ("aries", "recovery"),
                 '''def aries_recover(
    log: list[dict[str, object]],
    checkpoint_lsn: int,
    dirty_page_table: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Replay from the checkpoint (redo), then roll back losers (undo) in reverse LSN order."""
    redo = [r for r in log if int(r["lsn"]) >= checkpoint_lsn]
    page_state = dict(dirty_page_table)
    for rec in sorted(redo, key=lambda r: int(r["lsn"])):
        if rec["kind"] == "update":
            page_state[str(rec["page"])] = int(rec["after"])
    committed = {int(r["xid"]) for r in log if r["kind"] == "commit"}
    undone: list[dict[str, object]] = []
    for rec in sorted(log, key=lambda r: int(r["lsn"]), reverse=True):
        if rec["kind"] == "update" and int(rec["xid"]) not in committed:
            undone.append({"clr_for": rec["lsn"], "page": rec["page"], "before": rec["before"]})
            page_state[str(rec["page"])] = int(rec["before"])
    return page_state, undone
'''),
                ("db-exp-distributed-txn-raft-commit", "Raft-replicated transaction commit index with leader-lease reads", ("consensus", "transactions"),
                 '''from dataclasses import dataclass


@dataclass
class ReplicatedCommitLog:
    """Tracks quorum-committed transaction entries under leader leases."""

    quorum: int
    lease_until_ns: int = 0
    entries: list[tuple[int, str]] = None  # type: ignore[assignment]
    acks: dict[int, set[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.entries = []
        self.acks = {}

    def propose(self, index: int, txn_id: str) -> None:
        self.entries.append((index, txn_id))
        self.acks.setdefault(index, set())

    def ack(self, index: int, node: str) -> bool:
        self.acks.setdefault(index, set()).add(node)
        return len(self.acks[index]) >= self.quorum

    def lease_read_ok(self, now_ns: int) -> bool:
        return now_ns < self.lease_until_ns
'''),
                ("db-exp-vectorized-hash-join", "Vectorized hash join with build-side bloom pre-filtering", ("execution_engine", "hash_join"),
                 '''def hash_join(
    build: list[tuple[int, str]],
    probe: list[tuple[int, int]],
    bloom_bits: int = 1024,
) -> list[tuple[int, str, int]]:
    """Join on integer keys using a build hash table plus a bloom filter guard."""
    bloom = bytearray(bloom_bits // 8)

    def set_bit(k: int) -> None:
        idx = k % bloom_bits
        bloom[idx >> 3] |= 1 << (idx & 7)

    def test_bit(k: int) -> bool:
        idx = k % bloom_bits
        return bool(bloom[idx >> 3] & (1 << (idx & 7)))

    table: dict[int, list[str]] = {}
    for key, payload in build:
        table.setdefault(key, []).append(payload)
        set_bit(key)
    out: list[tuple[int, str, int]] = []
    for key, extra in probe:
        if not test_bit(key):
            continue
        for payload in table.get(key, ()):
            out.append((key, payload, extra))
    return out
'''),
                ("db-exp-cardinality-estimator", "Multi-column independence-adjusted selectivity estimator", ("optimizer", "statistics"),
                 '''def estimate_join_rows(
    table_rows: list[int],
    column_ndv: list[int],
    filters: list[tuple[int, float]],
    correlation: float = 1.0,
) -> float:
    """Estimate join cardinality with exponential backoff on filter independence."""
    if not table_rows:
        return 0.0
    base = 1.0
    for rows in table_rows:
        base *= rows
    denom = 1.0
    for i in range(len(table_rows)):
        for j in range(i + 1, len(table_rows)):
            pairs = max(1, max(column_ndv[i], column_ndv[j]))
            denom *= pairs
    selectivity = 1.0
    for idx, sel in filters:
        exponent = 1.0 / correlation
        selectivity *= min(1.0, max(0.0, sel) ** exponent)
        del idx
    return base / denom * selectivity
'''),
                ("db-exp-lsm-bloom-partition", "Per-SSTable bloom filter + fence-pointer index for range-seek lookups", ("lsm_tree", "indexing"),
                 '''class SSTableIndex:
    """Keeps first/last key per block and a bloom filter to skip non-resident keys."""

    def __init__(self, blocks: list[tuple[str, str]]) -> None:
        self.fence_keys = [first for first, _last in blocks]
        self.block_bounds = list(blocks)
        self.bits = bytearray(256)

    def _hashes(self, key: str) -> tuple[int, int]:
        h1 = hash(key) & 0xFFFFFFFF
        h2 = (hash(key[::-1]) | 1) & 0xFFFFFFFF
        return h1 % 2048, h2 % 2048

    def add(self, key: str) -> None:
        for bit in self._hashes(key):
            self.bits[bit >> 3] |= 1 << (bit & 7)

    def maybe_contains(self, key: str) -> bool:
        return all(self.bits[b >> 3] & (1 << (b & 7)) for b in self._hashes(key))

    def candidate_block(self, key: str) -> int | None:
        if not self.maybe_contains(key):
            return None
        lo, hi = 0, len(self.fence_keys) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            first, last = self.block_bounds[mid]
            if key < first:
                hi = mid - 1
            elif key > last:
                lo = mid + 1
            else:
                return mid
        return None
'''),
                ("db-exp-optimistic-replication-crdt", "Multi-leader last-writer-wins register with HLC timestamps", ("multi_leader", "crdt", "replication"),
                 '''from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class HybridTimestamp:
    wall_ms: int
    counter: int
    node: str = ""


class HLC:
    """Hybrid logical clock producing monotonic, causally consistent timestamps."""

    def __init__(self, node: str) -> None:
        self.node = node
        self.last = HybridTimestamp(0, 0, node)

    def tick(self, wall_ms: int) -> HybridTimestamp:
        if wall_ms > self.last.wall_ms:
            self.last = HybridTimestamp(wall_ms, 0, self.node)
        else:
            self.last = HybridTimestamp(self.last.wall_ms, self.last.counter + 1, self.node)
        return self.last

    def merge(self, remote: HybridTimestamp, wall_ms: int) -> HybridTimestamp:
        if remote.wall_ms > self.last.wall_ms:
            self.last = HybridTimestamp(remote.wall_ms, remote.counter, self.node)
        elif remote.wall_ms == self.last.wall_ms:
            self.last = HybridTimestamp(self.last.wall_ms, max(self.last.counter, remote.counter), self.node)
        return self.tick(wall_ms)


class LWWRegister:
    """Resolves conflicting writes deterministically by (timestamp, value) ordering."""

    def __init__(self) -> None:
        self.value: str | None = None
        self.ts: HybridTimestamp | None = None

    def assign(self, value: str, ts: HybridTimestamp) -> bool:
        if self.ts is None or (ts, value) > (self.ts, self.value or ""):
            self.value = value
            self.ts = ts
            return True
        return False
'''),
                ("db-exp-garbage-collector-horizon", "MVCC garbage collection horizon with active-transaction watermark", ("vacuum", "mvcc"),
                 '''def gc_horizon(active_xids: set[int], oldest_xmin: int, replica_feedback_xmin: int) -> int:
    """Compute the safe xmin below which no transaction can still observe tuples."""
    if active_xids:
        candidates = [min(active_xids), oldest_xmin, replica_feedback_xmin]
    else:
        candidates = [oldest_xmin, replica_feedback_xmin]
    return max(0, min(candidates))


def reclaimable_tuples(
    tuples_xmax: dict[int, int],
    horizon: int,
) -> list[int]:
    """Return tuple ids whose deleting transaction is strictly below the horizon."""
    return sorted(tid for tid, xmax in tuples_xmax.items() if xmax < horizon)
'''),
                ("db-exp-adaptive-compaction", "Leveled vs tiered LSM compaction strategy selector", ("lsm_tree", "compaction", "tuning"),
                 '''def choose_compaction_strategy(
    level_sizes: list[int],
    read_amplification: float,
    write_amplification: float,
    target_read_amp: float = 10.0,
) -> dict[str, object]:
    """Pick leveled (read-optimized) or tiered (write-optimized) per level overflow."""
    overfull = [i for i, size in enumerate(level_sizes) if i > 0 and size > level_sizes[i - 1] * 10]
    strategy = "leveled" if read_amplification > target_read_amp else "tiered"
    if write_amplification > 20.0 and read_amplification <= target_read_amp:
        strategy = "tiered"
    return {
        "strategy": strategy,
        "overfull_levels": overfull,
        "next_level": overfull[0] if overfull else None,
    }
'''),
                ("db-exp-plan-cache-invalidation", "Prepared-plan cache keyed by normalized SQL + stats generation", ("plan_cache", "statistics"),
                 '''import re

_WS_RE = re.compile(r"\\s+")
_LITERAL_RE = re.compile(r"'[^']*'|\\b\\d+\\b")


def plan_cache_key(sql: str, stats_generation: int) -> tuple[str, int]:
    """Normalize literals to placeholders and pair with the statistics generation."""
    normalized = _WS_RE.sub(" ", sql.strip())
    normalized = _LITERAL_RE.sub("?", normalized)
    return normalized.lower(), stats_generation


class PlanCache:
    """Invalidates cached generic plans whenever table statistics change."""

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._entries: dict[tuple[str, int], str] = {}
        self._generation = 0

    def bump_stats(self) -> None:
        self._generation += 1

    def get_plan(self, sql: str) -> str | None:
        return self._entries.get(plan_cache_key(sql, self._generation))

    def put_plan(self, sql: str, plan: str) -> None:
        if len(self._entries) >= self.capacity:
            self._entries.pop(next(iter(self._entries)))
        self._entries[plan_cache_key(sql, self._generation)] = plan
'''),
                ("db-exp-transaction-isolation-suite", "Isolation-level anomaly detector (dirty read, lost update, phantom)", ("isolation", "testing"),
                 '''def detect_anomalies(trace: list[tuple[str, str, object]], isolation: str = "read-committed") -> list[str]:
    """Classify anomalies observed in a (tx, op, value) trace under a given level."""
    seen: dict[str, object] = {}
    found: list[str] = []
    for tx, op, value in trace:
        if op == "dirty_read":
            found.append("dirty_read")
        elif op == "write":
            seen[tx] = value
        elif op == "lost_update":
            found.append("lost_update")
        elif op == "phantom":
            found.append("phantom")
    allowed = {
        "read-uncommitted": {"dirty_read", "lost_update", "phantom"},
        "read-committed": {"lost_update", "phantom"},
        "repeatable-read": {"phantom"},
        "serializable": set(),
    }[isolation]
    return sorted(set(found) - allowed)
'''),
            ],
        },
    )


def _build_cli_patterns() -> list:
    return _domain_specs(
        "cli",
        {
            "beginner": [
                ("cli-beg-arg-subset", "Sub-command parser validating required flags", ("argparse", "parsing"),
                 '''def split_command(argv: list[str], known: set[str]) -> tuple[str, dict[str, str]]:
    """Extract the sub-command name and --flag value pairs from argv."""
    if not argv:
        raise ValueError("no sub-command provided")
    command, rest = argv[0], argv[1:]
    if command not in known:
        raise ValueError(f"unknown sub-command {command!r}")
    flags: dict[str, str] = {}
    idx = 0
    while idx < len(rest):
        token = rest[idx]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argument {token!r}")
        if "=" in token:
            key, value = token[2:].split("=", 1)
            flags[key] = value
            idx += 1
            continue
        if idx + 1 >= len(rest):
            raise ValueError(f"missing value for {token}")
        flags[token[2:]] = rest[idx + 1]
        idx += 2
    return command, flags
'''),
                ("cli-beg-progress-bar", "Carriage-return text progress bar renderer", ("progress", "terminal"),
                 '''def render_progress(done: int, total: int, width: int = 30) -> str:
    """Render a single-line progress bar; caller writes it with a trailing \\\\r."""
    if total <= 0:
        raise ValueError("total must be positive")
    ratio = min(1.0, max(0.0, done / total))
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({done}/{total})"
'''),
                ("cli-beg-ansi-color", "ANSI color wrapper honoring NO_COLOR", ("terminal", "ansi"),
                 '''import os

_CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "bold": "1"}


def colorize(text: str, style: str, stream_is_tty: bool = True) -> str:
    """Wrap text in ANSI codes unless NO_COLOR is set or the stream is not a TTY."""
    if not stream_is_tty or "NO_COLOR" in os.environ or style not in _CODES:
        return text
    return f"\\x1b[{_CODES[style]}m{text}\\x1b[0m"
'''),
                ("cli-beg-env-precedence", "CLI flag over env var over default precedence resolver", ("config", "precedence"),
                 '''def resolve_option(flag: str | None, env: str | None, default: str) -> tuple[str, str]:
    """Resolve a configuration value and report which layer supplied it."""
    if flag is not None:
        return flag, "flag"
    if env is not None:
        return env, "env"
    return default, "default"
'''),
                ("cli-beg-human-duration", "Human-readable duration formatter (1h 5m 3s)", ("formatting", "time"),
                 '''def format_duration(total_seconds: float) -> str:
    """Format seconds as compact d/h/m/s units, omitting leading zero components."""
    seconds = int(round(total_seconds))
    if seconds < 0:
        raise ValueError("duration must be non-negative")
    units = [(86400, "d"), (3600, "h"), (60, "m"), (1, "s")]
    parts: list[str] = []
    for size, label in units:
        if seconds >= size or (parts and len(parts) < 2):
            parts.append(f"{seconds // size}{label}")
            seconds %= size
    return " ".join(parts) or "0s"
'''),
                ("cli-beg-exit-codes", "POSIX exit-code taxonomy mapping outcomes to sysexits values", ("exit_codes", "posix"),
                 '''EXIT_OK = 0
EXIT_USAGE = 64
EXIT_DATAERR = 65
EXIT_NOINPUT = 66
EXIT_SOFTWARE = 70
EXIT_CONFIG = 78


def exit_code_for(outcome: str) -> int:
    """Map a symbolic outcome to a sysexits.h compatible process exit code."""
    return {
        "ok": EXIT_OK,
        "usage": EXIT_USAGE,
        "bad_input": EXIT_DATAERR,
        "missing_file": EXIT_NOINPUT,
        "crash": EXIT_SOFTWARE,
        "config": EXIT_CONFIG,
    }.get(outcome, EXIT_SOFTWARE)
'''),
                ("cli-beg-confirm-prompt", "Y/n confirmation prompt with non-interactive default", ("interaction", "safety"),
                 '''def confirm(prompt: str, default: bool = False, interactive: bool = True, answer: str = "") -> bool:
    """Interpret a confirmation answer, using the default when non-interactive."""
    if not interactive:
        return default
    normalized = answer.strip().lower()
    if not normalized:
        return default
    if normalized in {"y", "yes"}:
        return True
    if normalized in {"n", "no"}:
        return False
    raise ValueError(f"unrecognized answer {answer!r} for prompt {prompt!r}")
'''),
                ("cli-beg-table-print", "Fixed-width column table renderer", ("formatting", "tables"),
                 '''def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a left-aligned fixed-width text table sized by the widest cell."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\\n".join(lines)
'''),
                ("cli-beg-stdin-lines", "Streaming stdin line reader with trailing-newline tolerance", ("io", "streaming"),
                 '''from collections.abc import Iterator


def iter_nonblank_lines(text: str, strip: bool = True) -> Iterator[str]:
    """Yield non-blank lines from stdin-like text, optionally whitespace-stripped."""
    for line in text.splitlines():
        value = line.strip() if strip else line.rstrip("\\n")
        if value.strip():
            yield value
'''),
            ],
            "intermediate": [
                ("cli-int-shell-escape", "POSIX-safe shell argument quoting", ("shell", "escaping"),
                 '''def shell_quote(arg: str) -> str:
    """Quote an argument so POSIX sh receives it verbatim as a single word."""
    if arg and all(c.isalnum() or c in "@%_+=:,./-" for c in arg):
        return arg
    return "'" + arg.replace("'", "'\\"'\\"'") + "'"
'''),
                ("cli-int-config-merge", "Layered config merge (defaults <- file <- env <- flags)", ("config", "layers"),
                 '''def merge_config_layers(*layers: dict[str, object]) -> dict[str, object]:
    """Merge configuration layers left to right, skipping None values."""
    merged: dict[str, object] = {}
    for layer in layers:
        for key, value in layer.items():
            if value is not None:
                merged[key] = value
    return merged
'''),
                ("cli-int-glob-expand", "Recursive glob expansion with exclusion patterns", ("filesystem", "globs"),
                 '''from pathlib import Path


def expand_patterns(root: Path, includes: list[str], excludes: list[str]) -> list[Path]:
    """Collect files matching include globs minus exclude globs, sorted for determinism."""
    found: set[Path] = set()
    for pattern in includes:
        found.update(p for p in root.glob(pattern) if p.is_file())
    for pattern in excludes:
        found.difference_update(p for p in root.glob(pattern))
    return sorted(found)
'''),
                ("cli-int-signal-handler", "Graceful SIGINT/SIGTERM handler with cleanup ordering", ("signals", "shutdown"),
                 '''import signal


class GracefulShutdown:
    """Installs signal handlers setting a cooperative stop flag."""

    def __init__(self) -> None:
        self.requested = False
        self.received: list[int] = []
        self._previous: dict[int, object] = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def _handle(self, signum: int, _frame: object) -> None:
        self.requested = True
        self.received.append(signum)

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()
'''),
                ("cli-int-diff-exit-status", "diff-compatible exit status computation (0/1/2)", ("diff", "exit_codes"),
                 '''def diff_exit_status(left: str, right: str, error: bool = False) -> int:
    """Return 0 when identical, 1 when differing, 2 on trouble (diff semantics)."""
    if error:
        return 2
    return 0 if left == right else 1
'''),
                ("cli-int-template-render", "Environment-substituting template renderer with strict mode", ("templating", "config"),
                 '''import re

_VAR_RE = re.compile(r"\\{\\{\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*\\}\\}")


def render_template(template: str, variables: dict[str, str], strict: bool = True) -> str:
    """Replace {{ var }} placeholders, raising KeyError for unknown names in strict mode."""
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return variables[name]
        if strict:
            raise KeyError(f"undefined template variable {name!r}")
        return match.group(0)

    return _VAR_RE.sub(substitute, template)
'''),
                ("cli-int-subcommand-registry", "Decorator-free sub-command dispatch registry", ("dispatch", "commands"),
                 '''from collections.abc import Callable


class CommandRegistry:
    """Maps sub-command names to handlers with aliases and usage text."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[list[str]], int]] = {}
        self._alias: dict[str, str] = {}

    def register(self, name: str, handler: Callable[[list[str]], int], aliases: tuple[str, ...] = ()) -> None:
        self._handlers[name] = handler
        for alias in aliases:
            self._alias[alias] = name

    def dispatch(self, argv: list[str]) -> int:
        if not argv:
            raise ValueError("missing command")
        name = self._alias.get(argv[0], argv[0])
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"unknown command {argv[0]!r}")
        return handler(argv[1:])
'''),
                ("cli-int-json-output", "Machine-readable JSON output mode with stable key ordering", ("json", "output"),
                 '''import json


def emit(payload: dict[str, object], as_json: bool, stream_write=print) -> None:
    """Print a payload as stable, indented JSON or a compact human summary."""
    if as_json:
        stream_write(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key in sorted(payload):
        stream_write(f"{key}: {payload[key]}")
'''),
                ("cli-int-file-lock-single-instance", "Single-instance CLI guard using an advisory file lock", ("locking", "processes"),
                 '''import fcntl
from pathlib import Path


def acquire_single_instance_lock(lock_path: Path) -> object | None:
    """Try to take an exclusive non-blocking lock; return the fd or None if held."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle
'''),
                ("cli-int-retry-with-backoff", "CLI command retry with jittered exponential backoff and logging", ("retries", "resilience"),
                 '''import random
import time


def retry_command(fn, attempts: int = 3, base_delay: float = 0.5, log=None, sleep=time.sleep):
    """Retry a command function, logging each failed attempt with its delay."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - CLI surfaces final error
            last = exc
            if attempt + 1 == attempts:
                break
            delay = base_delay * (2 ** attempt) * (0.5 + random.random())
            if log is not None:
                log(f"attempt {attempt + 1} failed: {exc}; retrying in {delay:.2f}s")
            sleep(delay)
    assert last is not None
    raise last
'''),
            ],
            "advanced": [
                ("cli-adv-shell-completion", "Bash/Zsh completion script generator from command spec", ("completion", "shell"),
                 '''def render_bash_completion(program: str, subcommands: list[str], flags: dict[str, list[str]]) -> str:
    """Emit a bash completion function covering sub-commands and their flags."""
    subs = " ".join(subcommands)
    cases = "\\n".join(
        f'        {sub}) opts="{ " ".join(flags.get(sub, [])) }" ;;' for sub in subcommands
    )
    return f"""_{program}_complete() {{
    local cur prev sub opts
    COMPREPLY=()
    cur="{{COMP_WORDS[COMP_CWORD]}}"
    prev="{{COMP_WORDS[COMP_CWORD-1]}}"
    sub="{{COMP_WORDS[1]}}"
    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "{subs}" -- "$cur") )
        return 0
    fi
    case "$sub" in
{cases}
    esac
    COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
}}
complete -F _{program}_complete {program}
"""
'''),
                ("cli-adv-pty-terminal-size", "POSIX ioctl terminal-size query with piped-stdout fallback", ("pty", "terminal"),
                 '''import os
import shutil


def terminal_size(default: tuple[int, int] = (24, 80)) -> tuple[int, int]:
    """Return (rows, cols) from the controlling TTY, falling back to defaults."""
    try:
        size = os.get_terminal_size(0)
        return size.lines, size.columns
    except OSError:
        return default


def effective_width(default: int = 80, max_width: int = 120) -> int:
    """Clamp the detected terminal width into a readable range."""
    _rows, cols = terminal_size((24, default))
    return max(40, min(cols, max_width)) if shutil.which("tput") or cols else default
'''),
                ("cli-adv-yaml-config-schema", "Nested config schema validator with JSON-pointer error paths", ("config", "validation"),
                 '''def validate_config(config: dict[str, object], schema: dict[str, object], path: str = "") -> list[str]:
    """Validate a nested mapping against a type-only schema, emitting JSON pointers."""
    errors: list[str] = []
    types = {"str": str, "int": int, "bool": bool, "list": list, "dict": dict, "float": float}
    for key, spec in schema.items():
        pointer = f"{path}/{key}"
        if key not in config:
            if isinstance(spec, dict) and spec.get("required", True):
                errors.append(f"{pointer}: required")
            continue
        value = config[key]
        if isinstance(spec, str):
            expected = types.get(spec)
            if expected is not None and not isinstance(value, expected):
                errors.append(f"{pointer}: expected {spec}, got {type(value).__name__}")
        elif isinstance(spec, dict) and isinstance(value, dict):
            errors.extend(validate_config(value, spec.get("properties", {}), pointer))
    return errors
'''),
                ("cli-adv-incremental-checksum", "Incremental file checksum with mtime+size cache bypass", ("hashing", "caching"),
                 '''import hashlib
import json
from pathlib import Path


def file_digest_cached(path: Path, cache_path: Path, algorithm: str = "sha256") -> str:
    """Return a digest, reusing the cached value when size and mtime are unchanged."""
    stat = path.stat()
    cache: dict[str, dict[str, float | str]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    key = str(path)
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size and entry.get("mtime") == stat.st_mtime:
        return str(entry["digest"])
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    cache[key] = {"size": stat.st_size, "mtime": stat.st_mtime, "digest": digest}
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return digest
'''),
                ("cli-adv-parallel-job-pool", "Bounded parallel CLI job pool preserving output order", ("parallelism", "process_pool"),
                 '''from concurrent.futures import ThreadPoolExecutor


def run_ordered(fn, items: list[object], workers: int = 4) -> list[object]:
    """Execute jobs concurrently but return results in the original input order."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        return [f.result() for f in futures]
'''),
                ("cli-adv-manpage-generator", "roff man-page generator from command metadata", ("documentation", "manpages"),
                 '''def render_manpage(name: str, synopsis: str, description: str, options: dict[str, str], section: int = 1) -> str:
    """Render a minimal roff-formatted man page."""
    esc = description.replace("-", "\\\\-")
    lines = [
        f'.TH {name.upper()} {section}',
        f'.SH NAME',
        f"{name} \\\\- {esc}",
        '.SH SYNOPSIS',
        f'.B {synopsis}',
        '.SH OPTIONS',
    ]
    for flag, help_text in sorted(options.items()):
        lines.append(f'.TP')
        lines.append(f'.B {flag}')
        lines.append(help_text)
    return "\\n".join(lines) + "\\n"
'''),
                ("cli-adv-output-stream-broken-pipe", "SIGPIPE-safe streaming writer for piped commands", ("pipes", "robustness"),
                 '''import sys


def write_lines_safely(lines, out=None, on_broken_pipe=None) -> int:
    """Write lines to stdout, treating EPIPE (e.g. `| head`) as a clean early exit."""
    out = out or sys.stdout
    written = 0
    try:
        for line in lines:
            out.write(f"{line}\\n")
            written += 1
    except BrokenPipeError:
        if on_broken_pipe is not None:
            on_broken_pipe()
        return written
    return written
'''),
                ("cli-adv-dry-run-planner", "Dry-run execution planner diffing intended vs applied operations", ("dry_run", "safety"),
                 '''def plan_operations(desired: dict[str, str], current: dict[str, str]) -> list[tuple[str, str, str | None, str]]:
    """Compute create/update/delete operations without mutating state."""
    ops: list[tuple[str, str, str | None, str]] = []
    for key in sorted(desired):
        want = desired[key]
        have = current.get(key)
        if have is None:
            ops.append(("create", key, None, want))
        elif have != want:
            ops.append(("update", key, have, want))
    for key in sorted(set(current) - set(desired)):
        ops.append(("delete", key, current[key], ""))
    return ops
'''),
                ("cli-adv-shell-injection-audit", "Argument vector audit rejecting shell metacharacter injection", ("security", "audit"),
                 '''_RISKY = {"|", "&&", ";", "$(", "`", ">", "<", "\\n"}


def audit_argv(argv: list[str]) -> list[str]:
    """Flag arguments containing shell metacharacters when a shell may be invoked."""
    findings: list[str] = []
    for idx, arg in enumerate(argv):
        hits = [tok for tok in _RISKY if tok in arg]
        if hits:
            findings.append(f"argv[{idx}] contains {hits}: {arg!r}")
    return findings
'''),
                ("cli-adv-config-migration", "Config schema version migration chain with idempotent steps", ("migrations", "config"),
                 '''def migrate_config(config: dict[str, object], steps: list[tuple[int, str]]) -> dict[str, object]:
    """Apply ordered migration steps (version, key rename) until the latest version."""
    current = dict(config)
    version = int(current.get("version", 0))
    for step_version, _label in steps:
        if step_version <= version:
            continue
        if "legacy_name" in current and "name" not in current:
            current["name"] = current.pop("legacy_name")
        current["version"] = step_version
        version = step_version
    return current
'''),
            ],
            "expert": [
                ("cli-exp-lexer-state-machine", "Hand-written argv lexer with quoting and combined short-flag expansion", ("lexer", "argv"),
                 '''def lex_argv(tokens: list[str], value_flags: set[str]) -> tuple[list[str], dict[str, str], dict[str, bool]]:
    """Parse positionals, --key=value, --key value, and -abc combined booleans."""
    positional: list[str] = []
    values: dict[str, str] = {}
    switches: dict[str, bool] = {}
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "--":
            positional.extend(tokens[idx + 1 :])
            break
        if tok.startswith("--"):
            name, sep, inline = tok[2:].partition("=")
            if sep:
                values[name] = inline
            elif name in value_flags:
                if idx + 1 >= len(tokens):
                    raise ValueError(f"flag --{name} requires a value")
                values[name] = tokens[idx + 1]
                idx += 1
            else:
                switches[name] = True
        elif tok.startswith("-") and len(tok) > 1:
            for ch in tok[1:]:
                if ch in value_flags:
                    values[ch] = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                    idx += 1
                    break
                switches[ch] = True
        else:
            positional.append(tok)
        idx += 1
    return positional, values, switches
'''),
                ("cli-exp-pty-session-driver", "PTY-backed child session driver with stdout capture and resize", ("pty", "process_control"),
                 '''import os
import pty
import select


def run_in_pty(argv: list[str], input_bytes: bytes = b"", timeout_s: float = 10.0) -> tuple[bytes, int]:
    """Spawn a child on a pseudo-terminal, feed input, and capture combined output."""
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - child branch
        os.execvp(argv[0], argv)
    chunks: list[bytes] = []
    if input_bytes:
        os.write(fd, input_bytes)
    while True:
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            break
        try:
            data = os.read(fd, 4096)
        except OSError:
            break
        if not data:
            break
        chunks.append(data)
    _, status = os.waitpid(pid, 0)
    os.close(fd)
    return b"".join(chunks), status
'''),
                ("cli-exp-plugin-discovery", "Entry-point-style plugin discovery with version compatibility gating", ("plugins", "extension"),
                 '''import importlib


def load_plugins(specs: list[tuple[str, str, str]], core_version: tuple[int, int]) -> tuple[dict[str, object], list[str]]:
    """Import (name, module_path, min_core_version) plugins, reporting incompatibles."""
    loaded: dict[str, object] = {}
    skipped: list[str] = []
    for name, module_path, min_version in specs:
        parts = tuple(int(p) for p in min_version.split(".") if p.isdigit())
        if parts > core_version:
            skipped.append(f"{name}: requires core >= {min_version}")
            continue
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            skipped.append(f"{name}: import failed ({exc})")
            continue
        loaded[name] = module
    return loaded, skipped
'''),
                ("cli-exp-command-transaction", "Compensating-action transaction for multi-step CLI operations", ("rollback", "compensation"),
                 '''class CommandTransaction:
    """Runs steps and unwinds completed ones in reverse order on failure."""

    def __init__(self) -> None:
        self._compensations: list[tuple[str, object]] = []
        self.log: list[str] = []

    def run(self, steps: list[tuple[str, object, object]]) -> None:
        for name, action, rollback in steps:
            try:
                action()
            except Exception:
                for cname, undo in reversed(self._compensations):
                    try:
                        undo()
                        self.log.append(f"rolled back {cname}")
                    except Exception as exc:  # noqa: BLE001 - compensation must not mask
                        self.log.append(f"rollback of {cname} failed: {exc}")
                self._compensations.clear()
                raise
            self._compensations.append((name, rollback))
            self.log.append(f"applied {name}")
        self._compensations.clear()
'''),
                ("cli-exp-terminal-table-virtualized", "Windowed terminal table renderer for large result sets", ("tables", "pagination"),
                 '''def render_viewport(
    rows: list[list[str]],
    headers: list[str],
    offset: int,
    height: int,
) -> tuple[list[str], dict[str, int]]:
    """Render only the visible slice of a large table plus scroll metadata."""
    total = len(rows)
    offset = max(0, min(offset, max(0, total - height)))
    window = rows[offset : offset + height]
    widths = [max([len(h)] + [len(r[i]) for r in window if i < len(r)]) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    for row in window:
        lines.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    meta = {"offset": offset, "total": total, "visible": len(window), "height": height}
    return lines, meta
'''),
                ("cli-exp-progress-eta-model", "EWMA throughput progress reporter with ETA estimation", ("progress", "eta"),
                 '''class EtaEstimator:
    """Smooths observed throughput to project remaining work completion time."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.rate: float | None = None
        self.last_done = 0
        self.last_ts = 0.0

    def update(self, done: int, timestamp_s: float) -> float | None:
        delta_done = done - self.last_done
        delta_t = timestamp_s - self.last_ts
        self.last_done = done
        self.last_ts = timestamp_s
        if delta_done <= 0 or delta_t <= 0:
            return None
        instant = delta_done / delta_t
        self.rate = instant if self.rate is None else self.alpha * instant + (1 - self.alpha) * self.rate
        return self.rate

    def eta_seconds(self, done: int, total: int) -> float | None:
        if not self.rate:
            return None
        remaining = max(0, total - done)
        return remaining / self.rate
'''),
                ("cli-exp-config-encryption", "Sealed config secrets using authenticated encryption envelope", ("secrets", "encryption"),
                 '''import hashlib
import hmac
import os


def seal_secret(plaintext: bytes, key: bytes, nonce: bytes | None = None) -> bytes:
    """Build a nonce-prefixed keystream XOR sealed with an HMAC-SHA256 tag."""
    nonce = nonce or os.urandom(16)
    stream = b"".join(
        hashlib.sha256(key + nonce + i.to_bytes(4, "big")).digest() for i in range((len(plaintext) + 31) // 32)
    )
    ciphertext = bytes(p ^ k for p, k in zip(plaintext, stream))
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    return nonce + ciphertext + tag


def open_secret(sealed: bytes, key: bytes) -> bytes:
    """Verify the authentication tag, then decrypt the sealed envelope."""
    if len(sealed) < 48:
        raise ValueError("sealed payload too short")
    nonce, ciphertext, tag = sealed[:16], sealed[16:-32], sealed[-32:]
    expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, tag):
        raise ValueError("authentication tag mismatch")
    stream = b"".join(
        hashlib.sha256(key + nonce + i.to_bytes(4, "big")).digest() for i in range((len(ciphertext) + 31) // 32)
    )
    return bytes(c ^ k for c, k in zip(ciphertext, stream))
'''),
                ("cli-exp-recursive-watch", "Recursive filesystem watcher with debounce coalescing", ("watching", "debounce"),
                 '''import time


class DebouncedWatcher:
    """Coalesces rapid filesystem events into a single actionable batch."""

    def __init__(self, debounce_s: float = 0.25) -> None:
        self.debounce_s = debounce_s
        self._pending: dict[str, float] = {}

    def observe(self, path: str, now: float) -> None:
        self._pending[path] = now

    def drain(self, now: float) -> list[str]:
        ready = sorted(p for p, ts in self._pending.items() if now - ts >= self.debounce_s)
        for p in ready:
            del self._pending[p]
        return ready

    def sleep_until_ready(self, now: float) -> float:
        if not self._pending:
            return self.debounce_s
        earliest = min(self._pending.values())
        return max(0.0, self.debounce_s - (now - earliest))
'''),
                ("cli-exp-output-format-adapter", "Pluggable human/json/ndjson/csv output formatter registry", ("output", "formats"),
                 '''import csv
import io
import json


def format_records(records: list[dict[str, object]], fmt: str) -> str:
    """Render records in the requested machine- or human-readable format."""
    if fmt == "json":
        return json.dumps(records, indent=2, sort_keys=True)
    if fmt == "ndjson":
        return "\\n".join(json.dumps(r, sort_keys=True) for r in records)
    if fmt == "csv":
        if not records:
            return ""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=sorted(records[0]), lineterminator="\\n")
        writer.writeheader()
        writer.writerows(records)
        return buffer.getvalue()
    if fmt == "human":
        return "\\n".join("  ".join(f"{k}={v}" for k, v in sorted(r.items())) for r in records)
    raise ValueError(f"unsupported output format {fmt!r}")
'''),
            ],
        },
    )

def _build_security_patterns() -> list:
    return _domain_specs(
        "security",
        {
            "beginner": [
                ("sec-beg-constant-time-compare", "Constant-time secret comparison resisting timing side channels", ("timing_attack", "comparison"),
                 '''import hmac


def secrets_equal(candidate: str, expected: str) -> bool:
    """Compare secrets in constant time relative to the shared prefix length."""
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))
'''),
                ("sec-beg-password-strength", "Password policy strength meter with entropy estimate", ("passwords", "policy"),
                 '''import math
import re


def password_entropy_bits(password: str) -> float:
    """Estimate Shannon-style search-space entropy from the character classes used."""
    if not password:
        return 0.0
    classes = 0
    classes += 26 if re.search(r"[a-z]", password) else 0
    classes += 26 if re.search(r"[A-Z]", password) else 0
    classes += 10 if re.search(r"[0-9]", password) else 0
    classes += 33 if re.search(r"[^A-Za-z0-9]", password) else 0
    if classes == 0:
        return 0.0
    return len(password) * math.log2(classes)


def meets_policy(password: str, min_bits: float = 60.0, min_len: int = 12) -> bool:
    return len(password) >= min_len and password_entropy_bits(password) >= min_bits
'''),
                ("sec-beg-redact-secrets", "Log redaction for tokens, keys, and card-like numbers", ("redaction", "logging"),
                 '''import re

_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|token|secret|password)\\s*[=:]\\s*\\S+"), r"\\1=<redacted>"),
    (re.compile(r"\\b[A-Za-z0-9_\\-]{20,}\\b"), "<redacted-token>"),
    (re.compile(r"\\b\\d{13,19}\\b"), "<redacted-number>"),
]


def redact(text: str) -> str:
    """Scrub common credential and PII shapes from log lines."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
'''),
                ("sec-beg-path-traversal", "Path traversal guard confining joins to a base directory", ("path_traversal", "filesystem"),
                 '''import os


def safe_join(base_dir: str, user_path: str) -> str:
    """Join user input under base_dir, rejecting escapes and absolute paths."""
    if not user_path or user_path.startswith(("/", "\\\\")) or ".." in user_path.replace("\\\\", "/").split("/"):
        raise ValueError(f"unsafe path component: {user_path!r}")
    target = os.path.realpath(os.path.join(base_dir, user_path))
    root = os.path.realpath(base_dir)
    if os.path.commonpath([root, target]) != root:
        raise ValueError("path escapes base directory")
    return target
'''),
                ("sec-beg-html-escape", "Context-aware HTML escaping for untrusted interpolation", ("xss", "escaping"),
                 '''_ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#x27;",
    "/": "&#x2F;",
}


def escape_html(value: str) -> str:
    """Escape characters that could break out of HTML text or attribute context."""
    return "".join(_ESCAPES.get(ch, ch) for ch in value)
'''),
                ("sec-beg-hash-chain", "SHA-256 hash chain for tamper-evident audit log entries", ("hashing", "audit"),
                 '''import hashlib


def append_hash_chain(previous_digest: str, entry: bytes) -> str:
    """Chain an audit entry onto the previous digest for append-only integrity."""
    h = hashlib.sha256()
    h.update(bytes.fromhex(previous_digest))
    h.update(b"|")
    h.update(entry)
    return h.hexdigest()


def verify_chain(entries: list[bytes], digests: list[str], genesis: str) -> bool:
    if len(entries) != len(digests):
        return False
    prev = genesis
    for entry, digest in zip(entries, digests):
        prev = append_hash_chain(prev, entry)
        if prev != digest:
            return False
    return True
'''),
                ("sec-beg-rate-limit-login", "Login throttling counter with lockout after repeated failures", ("authentication", "brute_force"),
                 '''class LoginThrottle:
    """Tracks failed attempts per identity and locks out past a threshold."""

    def __init__(self, max_failures: int = 5) -> None:
        self.max_failures = max_failures
        self.failures: dict[str, int] = {}

    def record_failure(self, identity: str) -> bool:
        self.failures[identity] = self.failures.get(identity, 0) + 1
        return self.failures[identity] >= self.max_failures

    def is_locked(self, identity: str) -> bool:
        return self.failures.get(identity, 0) >= self.max_failures

    def record_success(self, identity: str) -> None:
        self.failures.pop(identity, None)
'''),
                ("sec-beg-injection-anomaly", "SQL injection anomaly scorer over user input heuristics", ("sqli", "detection"),
                 '''import re

_SUSPICIOUS = [
    re.compile(r"(?i)\\bunion\\b\\s+\\bselect\\b"),
    re.compile(r"(?i)\\bor\\b\\s+1\\s*=\\s*1"),
    re.compile(r"--\\s|/\\*.*\\*/", re.DOTALL),
    re.compile(r"(?i)\\b(sleep|pg_sleep|benchmark)\\s*\\("),
    re.compile(r"(?i)\\b(xp_cmdshell|load_file|into\\s+outfile)\\b"),
]


def injection_score(user_input: str) -> int:
    """Count independent injection indicators in a string (0 means no signals)."""
    return sum(1 for pattern in _SUSPICIOUS if pattern.search(user_input))
'''),
                ("sec-beg-csp-builder", "Content-Security-Policy header builder defaulting to deny", ("csp", "headers"),
                 '''def build_csp(
    default_src: tuple[str, ...] = ("'none'",),
    script_src: tuple[str, ...] = ("'none'",),
    extra: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Compose a restrictive CSP string with no inline script allowances."""
    directives: dict[str, tuple[str, ...]] = {
        "default-src": default_src,
        "script-src": script_src,
        "object-src": ("'none'",),
        "base-uri": ("'none'",),
        "frame-ancestors": ("'none'",),
    }
    for key, value in (extra or {}).items():
        directives[key] = value
    return "; ".join(f"{k} {' '.join(v)}" for k, v in directives.items())
'''),
                ("sec-beg-ip-allowlist", "CIDR-based client IP allowlist matcher", ("network", "access_control"),
                 '''import ipaddress


def ip_allowed(client_ip: str, allowed_cidrs: list[str]) -> bool:
    """Return True only when the client address falls inside an allowlisted range."""
    try:
        addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    for cidr in allowed_cidrs:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            continue
        if addr.version == network.version and addr in network:
            return True
    return False
'''),
            ],
            "intermediate": [
                ("sec-int-jwt-verify", "JWT HS256 verifier with algorithm confusion and claim validation", ("jwt", "verification"),
                 '''import base64
import hashlib
import hmac
import json


def _b64d(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def verify_hs256(token: str, secret: bytes, now: int, leeway_s: int = 60) -> dict[str, object]:
    """Verify an HS256 JWT, pinning the algorithm and checking exp/nbf claims."""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("malformed token") from exc
    header = json.loads(_b64d(header_b64))
    if header.get("alg") != "HS256":
        raise ValueError("unexpected algorithm; only HS256 is accepted")
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = hmac.new(secret, signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64d(signature_b64)):
        raise ValueError("signature mismatch")
    claims = json.loads(_b64d(payload_b64))
    exp = claims.get("exp")
    if exp is not None and now > int(exp) + leeway_s:
        raise ValueError("token expired")
    nbf = claims.get("nbf")
    if nbf is not None and now + leeway_s < int(nbf):
        raise ValueError("token not yet valid")
    return claims
'''),
                ("sec-int-argon2-parameters", "Password hashing parameter selector for target verification latency", ("passwords", "hashing"),
                 '''def choose_kdf_parameters(
    target_ms: float = 250.0,
    measured_ms_per_iter: float = 1.1,
    min_memory_kib: int = 19_456,
    parallelism: int = 2,
) -> dict[str, int]:
    """Derive Argon2id cost parameters meeting a verification latency budget."""
    if measured_ms_per_iter <= 0:
        raise ValueError("measured cost must be positive")
    iterations = max(2, int(round(target_ms / measured_ms_per_iter)))
    return {
        "type": 2,  # Argon2id
        "time_cost": iterations,
        "memory_cost": max(min_memory_kib, 65_536),
        "parallelism": max(1, parallelism),
        "hash_len": 32,
    }
'''),
                ("sec-int-csrf-token", "Synchronizer CSRF token issue/verify with per-session binding", ("csrf", "session"),
                 '''import hmac
import os


class CsrfTokens:
    """Issues per-session tokens and validates them against the session binding."""

    def __init__(self, session_secret: bytes) -> None:
        self.secret = session_secret
        self._issued: dict[str, str] = {}

    def issue(self, session_id: str) -> str:
        nonce = os.urandom(16).hex()
        mac = hmac.new(self.secret, f"{session_id}:{nonce}".encode(), "sha256").hexdigest()
        token = f"{nonce}.{mac}"
        self._issued[session_id] = token
        return token

    def verify(self, session_id: str, token: str) -> bool:
        stored = self._issued.get(session_id)
        if stored is None or stored != token:
            return False
        nonce, _, mac = token.partition(".")
        expected = hmac.new(self.secret, f"{session_id}:{nonce}".encode(), "sha256").hexdigest()
        return hmac.compare_digest(expected, mac)
'''),
                ("sec-int-tls-config-audit", "TLS configuration auditor flagging weak protocols and ciphers", ("tls", "configuration"),
                 '''_WEAK_VERSIONS = {"SSLv2", "SSLv3", "TLSv1", "TLSv1.1"}
_WEAK_CIPHER_MARKERS = ("RC4", "3DES", "DES", "NULL", "EXPORT", "MD5")


def audit_tls_config(versions: set[str], ciphers: list[str], has_forward_secrecy: bool) -> list[str]:
    """Report deprecated protocol versions, weak ciphers, and missing PFS."""
    findings: list[str] = []
    for version in sorted(versions & _WEAK_VERSIONS):
        findings.append(f"deprecated protocol enabled: {version}")
    for cipher in ciphers:
        if any(marker in cipher.upper() for marker in _WEAK_CIPHER_MARKERS):
            findings.append(f"weak cipher suite: {cipher}")
    if not has_forward_secrecy:
        findings.append("no forward-secret (ECDHE/DHE) key exchange")
    return findings
'''),
                ("sec-int-secret-scan", "Repository secret scanner with high-entropy string detection", ("secrets", "scanning"),
                 '''import math
import re

_KNOWN = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def scan_for_secrets(text: str, min_entropy: float = 4.2, min_len: int = 24) -> list[str]:
    """Return secret-like findings from known token formats and entropy outliers."""
    findings: list[str] = []
    for pattern in _KNOWN:
        findings.extend(m.group(0)[:12] + "..." for m in pattern.finditer(text))
    for candidate in re.findall(r"[A-Za-z0-9+/=_\\-]{%d,}" % min_len, text):
        if _entropy(candidate) >= min_entropy and not candidate.isalpha():
            findings.append(candidate[:12] + "...")
    return findings
'''),
                ("sec-int-signed-cookie", "Encrypted, authenticated session cookie using AES-CTR-style keystream + HMAC", ("cookies", "session"),
                 '''import base64
import hashlib
import hmac
import json


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks = b"".join(hashlib.sha256(key + nonce + i.to_bytes(4, "big")).digest() for i in range((length + 31) // 32))
    return blocks[:length]


def seal_session(payload: dict[str, object], key: bytes, nonce: bytes) -> str:
    """Encrypt-then-MAC a session payload into a URL-safe cookie value."""
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    cipher = bytes(a ^ b for a, b in zip(raw, _keystream(key, nonce, len(raw))))
    tag = hmac.new(key, nonce + cipher, hashlib.sha256).digest()[:16]
    return base64.urlsafe_b64encode(nonce + cipher + tag).decode("ascii")


def open_session(cookie: str, key: bytes) -> dict[str, object]:
    """Verify the MAC then decrypt a sealed session cookie."""
    blob = base64.urlsafe_b64decode(cookie.encode("ascii"))
    nonce, cipher, tag = blob[:16], blob[16:-16], blob[-16:]
    if not hmac.compare_digest(hmac.new(key, nonce + cipher, hashlib.sha256).digest()[:16], tag):
        raise ValueError("cookie authentication failed")
    raw = bytes(a ^ b for a, b in zip(cipher, _keystream(key, nonce, len(cipher))))
    return json.loads(raw)
'''),
                ("sec-int-ssrf-guard", "SSRF guard resolving URLs and rejecting private address ranges", ("ssrf", "egress"),
                 '''import ipaddress
import socket
from urllib.parse import urlsplit

_BLOCKED = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]


def is_public_http_url(url: str, resolver=socket.getaddrinfo) -> bool:
    """Reject non-HTTP schemes and hosts resolving to internal or loopback ranges."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    try:
        infos = resolver(parts.hostname, None)
    except OSError:
        return False
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if any(addr in net for net in _BLOCKED if net.version == addr.version):
            return False
    return True
'''),
                ("sec-int-audit-structured", "Tamper-evident structured audit event with redaction and severity", ("audit", "logging"),
                 '''import json
import time


def build_audit_event(
    actor: str,
    action: str,
    target: str,
    outcome: str,
    severity: str = "info",
    redacted_fields: tuple[str, ...] = ("password", "token", "secret"),
    extra: dict[str, object] | None = None,
    now: float | None = None,
) -> str:
    """Serialize an audit event with sensitive keys redacted and stable ordering."""
    event: dict[str, object] = {
        "ts": int(now if now is not None else time.time()),
        "actor": actor,
        "action": action,
        "target": target,
        "outcome": outcome,
        "severity": severity,
    }
    for key, value in (extra or {}).items():
        event[key] = "<redacted>" if key.lower() in redacted_fields else value
    return json.dumps(event, sort_keys=True, separators=(",", ":"))
'''),
            ],
            "advanced": [
                ("sec-adv-constant-time-mac", "Length-independent constant-time comparison for variable-size buffers", ("timing_attack", "mac"),
                 '''def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Compare digests without leaking length or first-difference position."""
    diff = len(a) ^ len(b)
    padded = max(len(a), len(b))
    for i in range(padded):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        diff |= x ^ y
    return diff == 0
'''),
                ("sec-adv-abac-policy-engine", "Attribute-based access control engine with deny-overrides", ("authorization", "abac"),
                 '''from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    effect: str
    action: str
    resource: str
    conditions: dict[str, object]


def evaluate_policies(
    policies: list[Policy],
    action: str,
    resource: str,
    attributes: dict[str, object],
) -> str:
    """Evaluate policies with deny-overrides and default-deny semantics."""
    decision = "deny"
    for policy in policies:
        if policy.action not in (action, "*") or policy.resource not in (resource, "*"):
            continue
        if any(attributes.get(k) != v for k, v in policy.conditions.items()):
            continue
        if policy.effect == "deny":
            return "deny"
        decision = "allow"
    return decision
'''),
                ("sec-adv-oauth-scope-escalation", "OAuth token scope escalation detector comparing granted vs requested", ("oauth2", "authorization"),
                 '''def detect_scope_escalation(
    granted: set[str],
    requested: set[str],
    hierarchy: dict[str, set[str]],
) -> list[str]:
    """Report requested scopes not implied by any granted scope through hierarchy."""
    implied = set(granted)
    changed = True
    while changed:
        changed = False
        for scope in list(implied):
            for child in hierarchy.get(scope, ()):  # type: ignore[union-attr]
                if child not in implied:
                    implied.add(child)
                    changed = True
    return sorted(requested - implied)
'''),
                ("sec-adv-sql-parser-allowlist", "SQL allowlisted statement-shape validator (SELECT-only gateway)", ("sqli", "gateway"),
                 '''import re


def allow_only_select(sql: str) -> tuple[bool, str]:
    """Permit a single SELECT/CTE statement with no mutation keywords or comments."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty statement"
    if ";" in stripped:
        return False, "multiple statements are not allowed"
    if "--" in stripped or "/*" in stripped:
        return False, "comments are not allowed"
    lowered = stripped.lower()
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        return False, "only SELECT/CTE statements are allowed"
    for keyword in ("insert ", "update ", "delete ", "drop ", "alter ", "grant ", "copy ", "truncate "):
        if re.search(rf"(?<![a-z_]){keyword}", lowered):
            return False, f"disallowed keyword: {keyword.strip()}"
    return True, "ok"
'''),
                ("sec-adv-sandbox-syscall-policy", "Seccomp-style syscall allowlist policy evaluator", ("sandboxing", "syscalls"),
                 '''def evaluate_syscalls(
    request: list[str],
    allowed: set[str],
    deny_list: set[str] = frozenset(),
    errno: str = "EPERM",
) -> dict[str, object]:
    """Simulate a syscall filter verdict, preferring explicit denies."""
    denied: list[tuple[str, str]] = []
    for call in request:
        if call in deny_list:
            denied.append((call, errno))
        elif call not in allowed:
            denied.append((call, errno))
    return {"allowed": not denied, "violations": denied, "sandboxed": True}
'''),
                ("sec-adv-supply-chain-pin", "Dependency pin auditor checking hashes, ranges, and typosquat distance", ("supply_chain", "dependencies"),
                 '''import re


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def audit_requirements(
    requirements: list[str],
    popular_packages: set[str],
    max_distance: int = 1,
) -> list[str]:
    """Flag unpinned specs, missing hashes, and names one edit away from popular ones."""
    findings: list[str] = []
    for raw in requirements:
        spec = raw.strip()
        if not spec or spec.startswith("#"):
            continue
        name = re.split(r"[<>=!~\\[\\s]", spec, maxsplit=1)[0].lower()
        if "==" not in spec:
            findings.append(f"{name}: not pinned to an exact version")
        if "sha256:" not in spec:
            findings.append(f"{name}: no hash pin")
        for popular in popular_packages:
            if popular != name and _edit_distance(popular, name) <= max_distance:
                findings.append(f"{name}: suspiciously similar to {popular}")
                break
    return findings
'''),
                ("sec-adv-deserialization-guard", "Safe unpickler restricted to an explicit class allowlist", ("deserialization", "pickle"),
                 '''import io
import pickle


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses any class outside an explicit allowlist."""

    ALLOWED: dict[str, set[str]] = {}

    def find_class(self, module: str, name: str):
        if name not in self.ALLOWED.get(module, set()):
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        return super().find_class(module, name)


def safe_loads(data: bytes, allowed: dict[str, set[str]]) -> object:
    """Load a pickle only when every referenced global is allowlisted."""
    RestrictedUnpickler.ALLOWED = allowed
    return RestrictedUnpickler(io.BytesIO(data)).load()
'''),
                ("sec-adv-hsts-preload", "HSTS header builder with preload and includeSubDomains validation", ("hsts", "headers"),
                 '''def build_hsts(max_age: int = 63_072_000, include_subdomains: bool = True, preload: bool = True) -> str:
    """Render a Strict-Transport-Security value, enforcing preload minimums."""
    if preload and max_age < 31_536_000:
        raise ValueError("preload requires max-age of at least one year")
    if preload and not include_subdomains:
        raise ValueError("preload requires includeSubDomains")
    parts = [f"max-age={max_age}"]
    if include_subdomains:
        parts.append("includeSubDomains")
    if preload:
        parts.append("preload")
    return "; ".join(parts)
'''),
                ("sec-adv-cors-credentials-audit", "CORS configuration auditor rejecting wildcard-plus-credentials", ("cors", "audit"),
                 '''def audit_cors(
    allow_origin: str,
    allow_credentials: bool,
    allow_methods: list[str],
    allow_headers: list[str],
) -> list[str]:
    """Detect dangerous CORS combinations in a response configuration."""
    findings: list[str] = []
    if allow_origin == "*" and allow_credentials:
        findings.append("wildcard origin combined with credentials")
    if allow_origin == "*" and "Authorization" in allow_headers:
        findings.append("wildcard origin with Authorization header reflection")
    if "null" in allow_origin.lower():
        findings.append("'null' origin is spoofable by sandboxed frames")
    if allow_methods and "*" in allow_methods:
        findings.append("wildcard methods expose unsafe verbs")
    if not allow_headers:
        findings.append("empty allowed header list forces preflight failures")
    return findings
'''),
                ("sec-adv-rate-limit-distributed", "Distributed sliding-window limiter with clock-skew tolerance", ("rate_limiting", "distributed"),
                 '''class SkewTolerantLimiter:
    """Coordinates counters across nodes while tolerating bounded clock skew."""

    def __init__(self, limit: int, window_s: int, max_skew_s: float = 2.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self.max_skew_s = max_skew_s
        self.counters: dict[tuple[str, int], int] = {}

    def admit(self, key: str, now: float, peer_nows: list[float] | None = None) -> bool:
        if peer_nows:
            spread = max(peer_nows) - min(peer_nows)
            if spread > self.max_skew_s:
                raise RuntimeError(f"clock skew {spread:.2f}s exceeds tolerance")
        bucket = int(now // self.window_s)
        total = sum(self.counters.get((key, bucket - i), 0) for i in (0, 1))
        if total >= self.limit:
            return False
        self.counters[(key, bucket)] = self.counters.get((key, bucket), 0) + 1
        return True
'''),
                ("sec-adv-timing-leak-detector", "Statistical timing side-channel detector over two input classes", ("side_channel", "statistics"),
                 '''import statistics


def detect_timing_leak(
    fast_samples_ns: list[int],
    slow_samples_ns: list[int],
    threshold_sigma: float = 3.0,
) -> dict[str, object]:
    """Compare mean timings of two input classes and report a leak on large gaps."""
    if len(fast_samples_ns) < 2 or len(slow_samples_ns) < 2:
        raise ValueError("need at least two samples per class")
    m1 = statistics.mean(fast_samples_ns)
    m2 = statistics.mean(slow_samples_ns)
    sd = max(statistics.pstdev(fast_samples_ns), statistics.pstdev(slow_samples_ns), 1.0)
    delta = abs(m2 - m1) / sd
    return {"leak": delta > threshold_sigma, "sigma_delta": delta, "means_ns": (m1, m2)}
'''),
            ],
            "expert": [
                ("sec-exp-injection-taint", "Intraprocedural taint tracker propagating untrusted sources to sinks", ("taint_analysis", "static_analysis"),
                 '''import ast


class TaintTracker(ast.NodeVisitor):
    """Marks parameters as tainted and reports calls that reach dangerous sinks."""

    SINKS = {"execute", "exec", "eval", "system", "popen"}

    def __init__(self) -> None:
        self.tainted: set[str] = set()
        self.findings: list[tuple[str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args:
            self.tainted.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.tainted:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.tainted.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if func_name in self.SINKS:
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self.tainted:
                    self.findings.append((func_name, getattr(node, "lineno", 0)))
                elif isinstance(arg, ast.JoinedStr):
                    for value in arg.values:
                        inner = getattr(value, "value", None)
                        if isinstance(inner, ast.Name) and inner.id in self.tainted:
                            self.findings.append((func_name, getattr(node, "lineno", 0)))
        self.generic_visit(node)


def find_tainted_sinks(source: str) -> list[tuple[str, int]]:
    tree = ast.parse(source)
    tracker = TaintTracker()
    tracker.visit(tree)
    return tracker.findings
'''),
                ("sec-exp-cert-chain-verify", "X.509 chain path validator with basic-constraint and expiry checks", ("pki", "certificates"),
                 '''from dataclasses import dataclass


@dataclass(frozen=True)
class CertInfo:
    subject: str
    issuer: str
    is_ca: bool
    path_length: int | None
    not_after: int
    fingerprints: tuple[str, ...] = ()


def verify_chain(certs: list[CertInfo], trust_anchors: set[str], now: int, max_depth: int = 5) -> list[str]:
    """Validate signature linkage, CA bits, path length, and validity windows."""
    problems: list[str] = []
    if not certs:
        return ["empty chain"]
    if len(certs) > max_depth:
        problems.append(f"chain depth {len(certs)} exceeds {max_depth}")
    for idx, cert in enumerate(certs):
        if cert.not_after < now:
            problems.append(f"{cert.subject}: expired")
        if idx == 0:
            continue
        parent = certs[idx - 1]
        if cert.issuer != parent.subject:
            problems.append(f"{cert.subject}: issuer mismatch with {parent.subject}")
        if not parent.is_ca:
            problems.append(f"{parent.subject}: not a CA but issued a certificate")
        if parent.path_length is not None and idx - 1 > parent.path_length:
            problems.append(f"{parent.subject}: path length constraint violated")
    if certs[-1].subject not in trust_anchors:
        problems.append(f"root {certs[-1].subject} is not a trust anchor")
    return problems
'''),
                ("sec-exp-differential-privacy-budget", "Epsilon-delta privacy budget accountant with composition", ("privacy", "differential_privacy"),
                 '''import math


class PrivacyBudget:
    """Tracks epsilon/delta consumption under basic and advanced composition."""

    def __init__(self, epsilon: float, delta: float) -> None:
        if epsilon <= 0 or delta <= 0:
            raise ValueError("budget must be positive")
        self.max_epsilon = epsilon
        self.max_delta = delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0

    def spend(self, epsilon: float, delta: float = 0.0, queries: int = 1) -> bool:
        """Sequential composition: budgets add; reject when either is exhausted."""
        self.spent_epsilon += epsilon * queries
        self.spent_delta += delta * queries
        return self.spent_epsilon <= self.max_epsilon and self.spent_delta <= self.max_delta

    def advanced_composition_epsilon(self, epsilon: float, queries: int, delta_prime: float) -> float:
        """Advanced composition bound for k adaptive queries."""
        return math.sqrt(2 * queries * math.log(1 / delta_prime)) * epsilon + queries * epsilon * (math.e - 1)
'''),
                ("sec-exp-crypto-key-rotation", "Key rotation scheduler with overlap windows and re-encryption plan", ("key_management", "rotation"),
                 '''def plan_key_rotation(
    keys: list[dict[str, object]],
    now: int,
    rotation_interval_s: int = 90 * 86400,
    overlap_s: int = 7 * 86400,
) -> dict[str, object]:
    """Decide whether to mint a new key and list records needing re-encryption."""
    active = [k for k in keys if k.get("status") == "active"]
    if len(active) > 1:
        raise ValueError("more than one active key")
    rotate = not active or now - int(active[0]["created_at"]) >= rotation_interval_s
    stale = [
        str(k["id"])
        for k in keys
        if now - int(k["created_at"]) >= rotation_interval_s + overlap_s
    ]
    return {"rotate": rotate, "retire": stale, "active": active[0]["id"] if active else None}
'''),
                ("sec-exp-memory-zeroization", "Best-effort secret buffer zeroization helper", ("secrets", "memory"),
                 '''import ctypes


def zeroize(buffer: bytearray) -> None:
    """Overwrite a mutable byte buffer in place so secrets are not left in memory."""
    if not isinstance(buffer, bytearray):
        raise TypeError("zeroize requires a mutable bytearray")
    for i in range(len(buffer)):
        buffer[i] = 0
    if buffer:
        # Prevent the compiler/interpreter from eliding the writes.
        ctypes.memset(id(buffer) + 0, 0, 0)


class SecretBuffer:
    """Context manager guaranteeing zeroization of sensitive bytes."""

    def __init__(self, data: bytes) -> None:
        self._buf = bytearray(data)

    def __enter__(self) -> bytearray:
        return self._buf

    def __exit__(self, *_exc: object) -> None:
        zeroize(self._buf)
'''),
                ("sec-exp-policy-diff-analyzer", "IAM policy diff analyzer detecting newly granted privileges", ("iam", "policy"),
                 '''def analyze_policy_diff(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    """Report added actions/resources and escalate findings that grant '*'."""
    def expand(policy: dict[str, object]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for stmt in policy.get("Statement", []) or []:  # type: ignore[union-attr]
            actions = stmt.get("Action", [])
            resources = stmt.get("Resource", [])
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resources, str):
                resources = [resources]
            for action in actions:
                for resource in resources:
                    pairs.add((str(action), str(resource)))
        return pairs

    old, new = expand(before), expand(after)
    added = sorted(new - old)
    return {
        "added": added,
        "escalations": [p for p in added if p[0] == "*" or p[0].endswith(":*") or p[1] == "*"],
        "removed": sorted(old - new),
    }
'''),
                ("sec-exp-attack-path-graph", "Privilege escalation path search over an authorization graph", ("graph", "attack_paths"),
                 '''from collections import deque


def shortest_escalation_path(
    edges: dict[str, set[str]],
    start: str,
    targets: set[str],
    max_depth: int = 8,
) -> list[str] | None:
    """BFS for the shortest privilege-escalation path from a principal to a target."""
    queue: deque[list[str]] = deque([[start]])
    seen = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node in targets and len(path) > 1:
            return path
        if len(path) >= max_depth:
            continue
        for nxt in sorted(edges.get(node, ())):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(path + [nxt])
    return None
'''),
                ("sec-exp-encrypted-index-search", "Deterministic encrypted keyword index with keyed HMAC bucketing", ("searchable_encryption", "privacy"),
                 '''import hashlib
import hmac


def tokenize_document(doc_id: str, text: str, key: bytes, bucket_bits: int = 16) -> dict[int, list[str]]:
    """Build a keyed token bucket index that leaks only bucket co-occurrence."""
    index: dict[int, list[str]] = {}
    for word in {w.strip(".,;:!?()[]").lower() for w in text.split()}:
        if not word:
            continue
        mac = hmac.new(key, word.encode("utf-8"), hashlib.sha256).digest()
        bucket = int.from_bytes(mac[:4], "big") >> (32 - bucket_bits)
        index.setdefault(bucket, []).append(doc_id)
    return index


def merge_indexes(indexes: list[dict[int, list[str]]]) -> dict[int, list[str]]:
    merged: dict[int, list[str]] = {}
    for idx in indexes:
        for bucket, docs in idx.items():
            merged.setdefault(bucket, []).extend(doc_id for doc_id in docs if doc_id not in merged.get(bucket, []))
    return merged
'''),
                ("sec-exp-fuzzing-harness-sanitizer", "Bounds-checked parser harness surfacing memory-safety style violations", ("fuzzing", "memory_safety"),
                 '''def safe_read(buf: bytes, offset: int, length: int) -> bytes:
    """Slice with explicit bounds validation, mirroring sanitizer checks in Python."""
    if offset < 0 or length < 0:
        raise ValueError("negative offset or length")
    if offset + length > len(buf):
        raise BufferError(f"out-of-bounds read: want {length} at {offset}, buffer is {len(buf)}")
    return buf[offset : offset + length]


def fuzz_bounds(reader, samples: list[bytes]) -> list[str]:
    """Run a reader over adversarial buffers, collecting contract violations."""
    failures: list[str] = []
    for payload in samples:
        try:
            reader(payload)
        except BufferError as exc:
            failures.append(f"{payload[:16]!r}: {exc}")
        except (ValueError, IndexError):
            continue
    return failures
'''),
                ("sec-exp-zero-trust-mesh", "Zero-trust service mesh authorizer with identity, path, and SPIFFE checks", ("zero_trust", "service_mesh"),
                 '''def authorize_service_call(
    caller_spiffe: str,
    callee_spiffe: str,
    method: str,
    path: str,
    policy: dict[str, set[str]],
    mtls_verified: bool,
) -> tuple[bool, str]:
    """Enforce mutual TLS, SPIFFE identity binding, and per-path method allowlists."""
    if not mtls_verified:
        return False, "mtls_required"
    if not caller_spiffe.startswith("spiffe://") or not callee_spiffe.startswith("spiffe://"):
        return False, "invalid_spiffe_identity"
    allowed = policy.get(f"{caller_spiffe}->{callee_spiffe}")
    if allowed is None:
        return False, "no_policy_for_pair"
    key = f"{method.upper()} {path}"
    if key in allowed or f"{method.upper()} *" in allowed:
        return True, "allowed"
    return False, "method_path_not_allowed"
'''),
            ],
        },
    )


def _build_async_io_patterns() -> list:
    return _domain_specs(
        "async_io",
        {
            "beginner": [
                ("async-beg-gather-map", "Bounded-concurrency asyncio gather with result ordering", ("asyncio", "concurrency"),
                 '''import asyncio
from collections.abc import Awaitable, Callable


async def bounded_map(
    fn: Callable[[object], Awaitable[object]],
    items: list[object],
    limit: int = 8,
) -> list[object]:
    """Run an async function over items with at most `limit` in flight, preserving order."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    semaphore = asyncio.Semaphore(limit)

    async def run(item: object) -> object:
        async with semaphore:
            return await fn(item)

    return await asyncio.gather(*(run(item) for item in items))
'''),
                ("async-beg-timeout-wrap", "asyncio timeout wrapper with fallback value", ("asyncio", "timeouts"),
                 '''import asyncio
from collections.abc import Awaitable


async def with_timeout(awaitable: Awaitable[object], seconds: float, default: object = None) -> object:
    """Await a coroutine but return `default` if it exceeds the deadline."""
    if seconds <= 0:
        raise ValueError("timeout must be positive")
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError:
        return default
'''),
                ("async-beg-task-cancel", "Cancellation-safe task wrapper propagating CancelledError", ("asyncio", "cancellation"),
                 '''import asyncio


async def run_cancellable(coro) -> object:
    """Run a coroutine, cancelling cleanly and never swallowing cancellation."""
    task = asyncio.ensure_future(coro)
    try:
        return await task
    except asyncio.CancelledError:
        task.cancel()
        raise
'''),
                ("async-beg-async-iterator", "Async generator batching a paginated API", ("async_iterators", "pagination"),
                 '''async def batched(fetch_page, page_size: int = 100, max_pages: int = 1000):
    """Yield items from a paginated async fetch_page(page) until exhaustion."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    for page in range(max_pages):
        items = await fetch_page(page, page_size)
        if not items:
            return
        for item in items:
            yield item
'''),
                ("async-beg-shield-cleanup", "Shielded cleanup that survives caller cancellation", ("asyncio", "cleanup"),
                 '''import asyncio


async def cleanup_before_exit(resource) -> None:
    """Ensure release runs even when the surrounding task is cancelled."""
    try:
        await asyncio.shield(resource.release())
    except asyncio.CancelledError:
        await resource.release()
        raise
'''),
                ("async-beg-rate-limiter-async", "Async token-bucket rate limiter without busy waiting", ("rate_limiting", "asyncio"),
                 '''import asyncio


class AsyncRateLimiter:
    """Spaces acquisitions at a fixed rate using monotonic time and sleeps."""

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / rate_per_sec
        self._next_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, loop=None) -> float:
        loop = loop or asyncio.get_event_loop()
        async with self._lock:
            now = loop.time()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait:
            await asyncio.sleep(wait)
        return wait
'''),
                ("async-beg-queue-worker", "Bounded asyncio queue worker pool with sentinel shutdown", ("queues", "workers"),
                 '''import asyncio


async def worker(name: str, queue: asyncio.Queue, handle, results: list) -> None:
    """Consume items until the sentinel None arrives, storing handler output."""
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            results.append((name, await handle(item)))
        finally:
            queue.task_done()


async def run_pool(items: list[object], handle, workers: int = 4, queue_size: int = 32) -> list:
    """Feed items through a fixed worker pool and return completed results."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    results: list = []
    tasks = [asyncio.create_task(worker(f"w{i}", queue, handle, results)) for i in range(workers)]
    for item in items:
        await queue.put(item)
    for _ in tasks:
        await queue.put(None)
    await asyncio.gather(*tasks)
    return results
'''),
                ("async-beg-semaphore-bulkhead", "Bulkhead isolation with per-dependency semaphores", ("bulkheads", "asyncio"),
                 '''import asyncio


class Bulkheads:
    """Caps concurrent calls per downstream dependency to contain failures."""

    def __init__(self, limits: dict[str, int]) -> None:
        if any(v <= 0 for v in limits.values()):
            raise ValueError("bulkhead limits must be positive")
        self._sems = {name: asyncio.Semaphore(limit) for name, limit in limits.items()}
        self.in_flight: dict[str, int] = {name: 0 for name in limits}

    async def run(self, dependency: str, coro):
        sem = self._sems.get(dependency)
        if sem is None:
            raise KeyError(f"unknown dependency {dependency!r}")
        async with sem:
            self.in_flight[dependency] += 1
            try:
                return await coro
            finally:
                self.in_flight[dependency] -= 1
'''),
                ("async-beg-clock-injection", "Injectable async clock for deterministic timer tests", ("testing", "determinism"),
                 '''import asyncio


class FakeAsyncClock:
    """Advanceable clock letting tests fast-forward sleeps without real delay."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot move backwards")
        self.now += seconds
'''),
            ],
            "intermediate": [
                ("async-int-backoff-retry", "Full-jitter async retry with retryable error classification", ("retries", "resilience"),
                 '''import asyncio
import random


async def retry_async(
    fn,
    attempts: int = 4,
    base_s: float = 0.1,
    cap_s: float = 5.0,
    retryable: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError),
    sleep=asyncio.sleep,
    rng: random.Random | None = None,
):
    """Retry the coroutine function with exponential backoff and full jitter."""
    rng = rng or random.Random(0)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except retryable as exc:
            last = exc
            if attempt + 1 == attempts:
                break
            await sleep(rng.random() * min(cap_s, base_s * (2 ** attempt)))
    assert last is not None
    raise last
'''),
                ("async-int-async-pool-lifecycle", "Async resource pool with health checks and recycling", ("pooling", "lifecycle"),
                 '''import asyncio


class AsyncResourcePool:
    """Reuses connections, discarding those that fail the health check."""

    def __init__(self, factory, health_check, size: int = 4) -> None:
        self.factory = factory
        self.health_check = health_check
        self.size = size
        self._idle: list[object] = []
        self._sem = asyncio.Semaphore(size)
        self._closed = False

    async def acquire(self):
        if self._closed:
            raise RuntimeError("pool is closed")
        await self._sem.acquire()
        while self._idle:
            resource = self._idle.pop()
            if await self.health_check(resource):
                return resource
        return await self.factory()

    async def release(self, resource) -> None:
        if not self._closed and await self.health_check(resource):
            self._idle.append(resource)
        self._sem.release()

    async def close(self) -> None:
        self._closed = True
        self._idle.clear()
'''),
                ("async-int-circuit-breaker-async", "Async circuit breaker with half-open single probe", ("circuit_breaker", "asyncio"),
                 '''import asyncio


class AsyncCircuitBreaker:
    """Prevents cascading failures by short-circuiting when the dependency is down."""

    def __init__(self, threshold: int = 5, reset_after_s: float = 15.0) -> None:
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self.failures = 0
        self.state = "closed"
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    async def call(self, fn, clock=None):
        clock = clock or asyncio.get_event_loop()
        async with self._lock:
            if self.state == "open":
                if clock.time() - self._opened_at >= self.reset_after_s:
                    self.state = "half-open"
                elif self._probe_in_flight:
                    raise RuntimeError("circuit open")
                self._probe_in_flight = True
        try:
            result = await fn()
        except BaseException:
            self.failures += 1
            if self.failures >= self.threshold or self.state == "half-open":
                self.state = "open"
                self._opened_at = clock.time()
            self._probe_in_flight = False
            raise
        self.failures = 0
        self.state = "closed"
        self._probe_in_flight = False
        return result
'''),
                ("async-int-stream-backpressure", "Async stream bridge applying cooperative backpressure", ("streams", "backpressure"),
                 '''import asyncio


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, chunk: int = 65536) -> int:
    """Copy a stream honouring writer drain backpressure; return bytes moved."""
    total = 0
    while True:
        data = await reader.read(chunk)
        if not data:
            break
        writer.write(data)
        await writer.drain()
        total += len(data)
    return total
'''),
                ("async-int-taskgroup-supervisor", "TaskGroup structured concurrency with failure aggregation", ("structured_concurrency", "asyncio"),
                 '''import asyncio


async def run_structured(coros: list, fail_fast: bool = False) -> dict[str, object]:
    """Run coroutines as a structured group, aggregating errors instead of leaking tasks."""
    results: list[object] = []
    errors: list[BaseException] = []
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(coro) for coro in coros]
        for task in tasks:
            task.add_done_callback(lambda t: errors.append(t.exception()) if t.exception() else None)
    results = [t.result() for t in tasks if not t.cancelled() and t.exception() is None]
    if errors and fail_fast:
        raise errors[0]
    return {"results": results, "errors": [repr(e) for e in errors]}
'''),
                ("async-int-cache-stampede-lock", "Per-key asyncio lock preventing cache stampede", ("caching", "stampede"),
                 '''import asyncio


class AsyncSingleFlightCache:
    """Collapses concurrent misses for the same key into a single loader call."""

    def __init__(self, loader) -> None:
        self.loader = loader
        self._cache: dict[str, object] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.loads = 0

    async def get(self, key: str) -> object:
        if key in self._cache:
            return self._cache[key]
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self._cache:
                return self._cache[key]
            self.loads += 1
            value = await self.loader(key)
            self._cache[key] = value
            self._locks.pop(key, None)
            return value
'''),
                ("async-int-websocket-heartbeat", "WebSocket ping/pong heartbeat with dead-peer eviction", ("websockets", "heartbeat"),
                 '''import asyncio


class HeartbeatMonitor:
    """Tracks last-seen time per peer and reports peers that missed deadlines."""

    def __init__(self, interval_s: float = 15.0, misses_allowed: int = 2) -> None:
        self.interval_s = interval_s
        self.misses_allowed = misses_allowed
        self._last_seen: dict[str, float] = {}

    def seen(self, peer: str, now: float) -> None:
        self._last_seen[peer] = now

    def dead_peers(self, now: float) -> list[str]:
        deadline = self.interval_s * (self.misses_allowed + 1)
        dead = [p for p, ts in self._last_seen.items() if now - ts > deadline]
        for peer in dead:
            del self._last_seen[peer]
        return sorted(dead)

    async def run(self, peers, send_ping) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            for peer in list(peers):
                await send_ping(peer)
'''),
                ("async-int-batch-aggregator", "Time- and size-triggered async request batcher", ("batching", "asyncio"),
                 '''import asyncio


class RequestBatcher:
    """Aggregates individual calls into batches flushed on size or timer."""

    def __init__(self, handler, max_batch: int = 32, max_delay_s: float = 0.01) -> None:
        self.handler = handler
        self.max_batch = max_batch
        self.max_delay_s = max_delay_s
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._timer: asyncio.Task | None = None

    async def submit(self, key: str) -> object:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending.append((key, fut))
        if len(self._pending) >= self.max_batch:
            await self._flush()
        elif self._timer is None:
            self._timer = asyncio.create_task(self._delayed_flush())
        return await fut

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.max_delay_s)
        await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        results = await self.handler([k for k, _ in batch])
        for (_, fut), value in zip(batch, results):
            if not fut.done():
                fut.set_result(value)
'''),
                ("async-int-cancel-scope", "Cancellation scope with deadline and shielded finalizers", ("cancellation", "deadlines"),
                 '''import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def deadline_scope(seconds: float):
    """Cancel the enclosed block after `seconds`, always running teardown."""
    if seconds <= 0:
        raise ValueError("deadline must be positive")
    task = asyncio.current_task()
    assert task is not None
    handle = asyncio.get_running_loop().call_later(seconds, task.cancel)
    try:
        yield handle
    finally:
        handle.cancel()
        await asyncio.shield(asyncio.sleep(0))
'''),
            ],
            "advanced": [
                ("async-adv-backpressure-token-bucket", "Adaptive concurrency limit tuned by latency gradient (AIMD)", ("concurrency_control", "adaptive"),
                 '''import asyncio


class AdaptiveConcurrency:
    """Adjusts concurrency from observed latency using additive-increase/multiplicative-decrease."""

    def __init__(self, initial: int = 8, min_limit: int = 2, max_limit: int = 128) -> None:
        self.limit = initial
        self.min_limit = min_limit
        self.max_limit = max_limit
        self._sem = asyncio.Semaphore(initial)
        self.min_latency = float("inf")

    async def run(self, coro_fn):
        await self._sem.acquire()
        try:
            started = asyncio.get_running_loop().time()
            result = await coro_fn()
            elapsed = asyncio.get_running_loop().time() - started
        finally:
            self._sem.release()
        self._tune(elapsed)
        return result

    def _tune(self, latency: float) -> None:
        if latency <= 0:
            return
        self.min_latency = min(self.min_latency, latency)
        gradient = self.min_latency / latency
        target = self.limit + (1 if gradient > 0.9 else -1)
        target = max(self.min_limit, min(self.max_limit, target))
        while self.limit < target:
            self.limit += 1
            self._sem.release()
        while self.limit > target:
            self.limit -= 1
            self._sem._value = max(0, self._sem._value - 1)
'''),
                ("async-adv-selector-event-loop", "Minimal epoll-based event loop with timers", ("selectors", "event_loop"),
                 '''import selectors
import time


class MiniEventLoop:
    """Single-threaded readiness loop dispatching callbacks and deadline timers."""

    def __init__(self) -> None:
        self.selector = selectors.DefaultSelector()
        self._timers: list[tuple[float, object]] = []
        self._running = False

    def register(self, fileobj, callback) -> None:
        self.selector.register(fileobj, selectors.EVENT_READ, callback)

    def call_later(self, delay_s: float, callback) -> None:
        self._timers.append((time.monotonic() + delay_s, callback))

    def stop(self) -> None:
        self._running = False

    def run_once(self, timeout: float = 0.1) -> int:
        now = time.monotonic()
        due = [t for t in self._timers if t[0] <= now]
        self._timers = [t for t in self._timers if t[0] > now]
        for _, cb in due:
            cb()
        events = self.selector.select(timeout)
        for key, _mask in events:
            key.data(key.fileobj)
        return len(events) + len(due)

    def run(self) -> None:
        self._running = True
        while self._running:
            self.run_once()
'''),
                ("async-adv-actor-mailbox", "Bounded async actor mailbox with ordered per-actor processing", ("actors", "mailbox"),
                 '''import asyncio


class Actor:
    """Serializes message handling per instance while running actors concurrently."""

    def __init__(self, handler, capacity: int = 64) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=capacity)
        self._handler = handler
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def send(self, message: object) -> None:
        await self._queue.put(message)

    async def _loop(self) -> None:
        while True:
            message = await self._queue.get()
            try:
                if message is None:
                    return
                await self._handler(message)
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._task is not None:
            await self._task
            self._task = None
'''),
                ("async-adv-async-context-propagation", "Context-variable propagation across spawned tasks", ("contextvars", "tracing"),
                 '''import asyncio
import contextvars

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


async def run_with_context(value: str, coro_fn):
    """Bind a context variable for a task without leaking into siblings."""
    token = request_id.set(value)
    try:
        ctx = contextvars.copy_context()
        return await ctx.run(coro_fn)
    finally:
        request_id.reset(token)


async def fan_out(coro_fn, values: list[str]) -> list[str]:
    """Spawn children that each observe only their own context value."""
    async def child(v: str) -> str:
        return f"{v}:{request_id.get()}"

    return list(await asyncio.gather(*(run_with_context(v, child(v)) for v in values)))
'''),
                ("async-adv-async-generator-finalization", "Async generator cleanup with aclose and shield guarantees", ("async_generators", "cleanup"),
                 '''import asyncio


async def consume_with_cleanup(agen) -> list[object]:
    """Drain an async generator, guaranteeing aclose even on early exit."""
    collected: list[object] = []
    try:
        async for item in agen:
            collected.append(item)
            if len(collected) >= 1000:
                break
    finally:
        await asyncio.shield(agen.aclose())
    return collected


async def safe_aclose_all(generators: list) -> list[str]:
    """Close every generator, collecting errors instead of aborting early."""
    errors: list[str] = []
    for agen in generators:
        try:
            await agen.aclose()
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    return errors
'''),
                ("async-adv-io-uring-like-batch", "Batched syscall submission with completion reaping", ("io_uring", "batching"),
                 '''class SubmissionQueue:
    """Models io_uring style submit/reap with bounded in-flight completions."""

    def __init__(self, depth: int = 64) -> None:
        self.depth = depth
        self.submitted: list[tuple[int, str]] = []
        self.completed: list[tuple[int, str, int]] = []
        self._next_id = 1

    def submit(self, op: str) -> int | None:
        if len(self.submitted) >= self.depth:
            return None
        token = self._next_id
        self._next_id += 1
        self.submitted.append((token, op))
        return token

    def reap(self, token: int, result: int) -> str:
        for idx, (tid, op) in enumerate(self.submitted):
            if tid == token:
                del self.submitted[idx]
                self.completed.append((tid, op, result))
                return op
        raise KeyError(f"unknown completion token {token}")

    def has_room(self) -> bool:
        return len(self.submitted) < self.depth
'''),
                ("async-adv-async-scheduler-edf", "Earliest-deadline-first async task scheduler", ("scheduling", "edf"),
                 '''import asyncio
import heapq


class EdfScheduler:
    """Runs tasks in earliest-deadline-first order, reporting missed deadlines."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, object]] = []
        self._seq = 0
        self.missed: list[int] = []

    def add(self, deadline: float, coro_fn) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (deadline, self._seq, coro_fn))

    async def run(self, clock=None) -> list[object]:
        loop = asyncio.get_running_loop()
        out: list[object] = []
        while self._heap:
            deadline, seq, fn = heapq.heappop(self._heap)
            if loop.time() > deadline:
                self.missed.append(seq)
            out.append(await fn())
        return out
'''),
                ("async-adv-async-bulkhead-queue-timeout", "Queue-time budget enforcement with admission rejection", ("queueing", "slo"),
                 '''import asyncio


class QueueBudget:
    """Rejects work whose expected queue wait would exceed the caller's budget."""

    def __init__(self, service_rate_per_s: float, workers: int = 4) -> None:
        if service_rate_per_s <= 0:
            raise ValueError("service rate must be positive")
        self.service_rate = service_rate_per_s
        self.workers = workers
        self.depth = 0

    def predicted_wait_s(self) -> float:
        return self.depth / (self.service_rate * self.workers)

    def admit(self, budget_s: float) -> bool:
        if self.predicted_wait_s() > budget_s:
            return False
        self.depth += 1
        return True

    def complete(self) -> None:
        self.depth = max(0, self.depth - 1)

    async def guard(self, budget_s: float, coro_fn):
        if not self.admit(budget_s):
            raise TimeoutError("queue-time budget exceeded")
        try:
            return await coro_fn()
        finally:
            self.complete()
'''),
                ("async-adv-async-mutex-fair", "FIFO-fair async mutex preventing starvation", ("locks", "fairness"),
                 '''import asyncio
from collections import deque


class FairAsyncMutex:
    """Grants the lock to waiters in arrival order."""

    def __init__(self) -> None:
        self._locked = False
        self._waiters: deque[asyncio.Future] = deque()

    async def acquire(self) -> None:
        if not self._locked:
            self._locked = True
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut in self._waiters:
                self._waiters.remove(fut)
            raise

    def release(self) -> None:
        if self._waiters:
            nxt = self._waiters.popleft()
            if not nxt.done():
                nxt.set_result(None)
        else:
            self._locked = False
'''),
            ],
            "expert": [
                ("async-exp-async-priority-deadline-heap", "Deadline-aware async work queue with starvation mitigation", ("scheduling", "starvation"),
                 '''import asyncio
import heapq


class AgingWorkQueue:
    """Prioritizes by deadline while aging low-priority work to avoid starvation."""

    def __init__(self, aging_per_s: float = 0.5) -> None:
        self._heap: list[tuple[float, int, float, object]] = []
        self._seq = 0
        self.aging = aging_per_s

    def push(self, priority: float, enqueued_at: float, item: object) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (priority, self._seq, enqueued_at, item))

    def pop(self, now: float) -> object | None:
        if not self._heap:
            return None
        rescored = [
            (priority - (now - enq) * self.aging, seq, enq, item)
            for priority, seq, enq, item in self._heap
        ]
        rescored.sort()
        priority, seq, enq, item = rescored[0]
        self._heap.remove((priority + (now - enq) * self.aging, seq, enq, item))
        return item
'''),
                ("async-exp-reactive-streams-demand", "Reactive-streams style demand accounting with backpressure signals", ("reactive", "backpressure"),
                 '''class DemandTracker:
    """Tracks subscriber demand and flags operators that emit without demand."""

    def __init__(self, buffer_size: int = 16) -> None:
        self.demand = 0
        self.buffer: list[object] = []
        self.limit = buffer_size
        self.violations = 0

    def request(self, n: int) -> None:
        if n <= 0:
            raise ValueError("demand must be positive")
        self.demand += n
        while self.buffer and self.demand > 0:
            self.buffer.pop(0)
            self.demand -= 1

    def emit(self, item: object) -> bool:
        if self.demand > 0:
            self.demand -= 1
            return True
        if len(self.buffer) >= self.limit:
            self.violations += 1
            return False
        self.buffer.append(item)
        return True

    def outstanding(self) -> int:
        return len(self.buffer)
'''),
                ("async-exp-async-tracer-span", "Async distributed tracing span with parent propagation and error tagging", ("tracing", "observability"),
                 '''from __future__ import annotations

import asyncio
import itertools
import time
from contextlib import asynccontextmanager

_span_ids = itertools.count(1)


class Span:
    """Records start/end timestamps, parent linkage, and error status."""

    def __init__(self, name: str, parent: Span | None = None) -> None:
        self.name = name
        self.parent = parent
        self.span_id = next(_span_ids)
        self.start_ns = time.monotonic_ns()
        self.end_ns: int | None = None
        self.error: str | None = None

    def finish(self, error: str | None = None) -> None:
        self.end_ns = time.monotonic_ns()
        self.error = error

    def duration_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.monotonic_ns()
        return (end - self.start_ns) / 1e6


@asynccontextmanager
async def span(name: str, parent: Span | None = None, sink: list | None = None):
    s = Span(name, parent)
    try:
        yield s
    except BaseException as exc:
        s.finish(error=type(exc).__name__)
        if sink is not None:
            sink.append(s)
        raise
    else:
        s.finish()
        if sink is not None:
            sink.append(s)
'''),
                ("async-exp-async-supervisor-restart", "Supervision tree with restart intensity and ordered shutdown", ("supervision", "fault_tolerance"),
                 '''import asyncio
import time


class Supervisor:
    """Restarts failed children, escalating when the restart budget is exhausted."""

    def __init__(self, max_restarts: int = 3, window_s: float = 30.0) -> None:
        self.max_restarts = max_restarts
        self.window_s = window_s
        self.restart_times: list[float] = []
        self.children: dict[str, object] = {}

    def _can_restart(self, now: float) -> bool:
        self.restart_times = [t for t in self.restart_times if now - t < self.window_s]
        return len(self.restart_times) < self.max_restarts

    async def supervise(self, name: str, factory, clock=time.monotonic) -> None:
        while True:
            task = asyncio.create_task(factory())
            self.children[name] = task
            try:
                await task
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                now = clock()
                if not self._can_restart(now):
                    raise RuntimeError(f"supervisor escalation for {name}")
                self.restart_times.append(now)
'''),
                ("async-exp-async-ring-buffer-zerocopy", "Memoryview-based async ring buffer with wrap-around reads", ("zero_copy", "buffers"),
                 '''import asyncio


class AsyncRingBuffer:
    """Single-producer/single-consumer byte ring using memoryview slices."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf = bytearray(capacity)
        self._view = memoryview(self._buf)
        self._head = 0
        self._tail = 0
        self._size = 0
        self._readable = asyncio.Event()
        self._writable = asyncio.Event()
        self._writable.set()

    async def write(self, data: bytes) -> None:
        while self._size == self.capacity:
            self._writable.clear()
            await self._writable.wait()
        first = min(len(data), self.capacity - self._tail)
        self._view[self._tail : self._tail + first] = data[:first]
        if len(data) > first:
            self._view[: len(data) - first] = data[first:]
        self._tail = (self._tail + len(data)) % self.capacity
        self._size += len(data)
        self._readable.set()

    async def read(self, count: int) -> bytes:
        while self._size == 0:
            self._readable.clear()
            await self._readable.wait()
        n = min(count, self._size)
        first = min(n, self.capacity - self._head)
        out = bytes(self._view[self._head : self._head + first])
        if n > first:
            out += bytes(self._view[: n - first])
        self._head = (self._head + n) % self.capacity
        self._size -= n
        self._writable.set()
        return out
'''),
                ("async-exp-async-shard-dispatcher", "Consistent-hash async shard dispatcher with in-flight limits", ("sharding", "dispatch"),
                 '''import asyncio
import hashlib


class ShardDispatcher:
    """Routes keys to shard workers while capping per-shard concurrency."""

    def __init__(self, shards: list[str], per_shard_limit: int = 16) -> None:
        self.shards = sorted(shards)
        self._sems = {s: asyncio.Semaphore(per_shard_limit) for s in self.shards}

    def shard_for(self, key: str) -> str:
        if not self.shards:
            raise RuntimeError("no shards configured")
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        return self.shards[int.from_bytes(digest, "little") % len(self.shards)]

    async def dispatch(self, key: str, fn):
        shard = self.shard_for(key)
        async with self._sems[shard]:
            return shard, await fn(shard, key)
'''),
                ("async-exp-async-checkpoint-resume", "Resumable async job runner with checkpointed progress", ("checkpoints", "resume"),
                 '''import asyncio
import json
from pathlib import Path


class ResumableJob:
    """Persists progress after each unit so interrupted runs resume cleanly."""

    def __init__(self, name: str, total: int, checkpoint_path: Path) -> None:
        self.name = name
        self.total = total
        self.path = checkpoint_path
        self.done = 0
        if checkpoint_path.is_file():
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if state.get("name") == name:
                self.done = int(state.get("done", 0))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"name": self.name, "done": self.done}), encoding="utf-8")

    async def run(self, process_unit) -> int:
        processed = 0
        while self.done < self.total:
            await process_unit(self.done)
            self.done += 1
            processed += 1
            self._save()
            await asyncio.sleep(0)
        return processed
'''),
                ("async-exp-async-quorum-replication", "Async quorum write with straggler cancellation", ("quorum", "replication"),
                 '''import asyncio


async def quorum_write(replicas: list, payload: bytes, quorum: int, straggler_grace_s: float = 0.05) -> dict[str, object]:
    """Succeed once quorum acks land; cancel remaining replicas after a grace window."""
    if not 1 <= quorum <= len(replicas):
        raise ValueError("invalid quorum size")
    tasks = {asyncio.create_task(r.write(payload)): i for i, r in enumerate(replicas)}
    acks: list[int] = []
    errors: list[tuple[int, str]] = []
    pending = set(tasks)
    while pending and len(acks) < quorum:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=straggler_grace_s + 1.0)
        if not done:
            break
        for task in done:
            idx = tasks[task]
            if task.exception() is None:
                acks.append(idx)
            else:
                errors.append((idx, repr(task.exception())))
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return {"ok": len(acks) >= quorum, "acks": sorted(acks), "errors": errors}
'''),
                ("async-exp-async-retry-budget", "Retry budget with per-request token accounting", ("retries", "budget"),
                 '''class RetryBudget:
    """Limits retries globally so failures cannot amplify load (Enovy-style)."""

    def __init__(self, min_retry_concurrency: float = 3.0, retry_percent: float = 0.2, ttl_s: float = 10.0) -> None:
        self.min_retry_concurrency = min_retry_concurrency
        self.retry_percent = retry_percent
        self.ttl_s = ttl_s
        self.active_requests = 0
        self.retries = 0
        self._window_start = 0.0
        self._request_window = 0

    def on_request(self, now: float) -> None:
        if now - self._window_start > self.ttl_s:
            self._window_start = now
            self._request_window = 0
            self.retries = 0
        self._request_window += 1
        self.active_requests += 1

    def try_retry(self) -> bool:
        allowance = max(self.min_retry_concurrency, self._request_window * self.retry_percent)
        if self.retries + 1 > allowance:
            return False
        self.retries += 1
        return True
'''),
                ("async-exp-async-latency-histogram", "Async-aware latency histogram with p99 and SLO burn rate", ("metrics", "slo"),
                 '''import bisect


class LatencyHistogram:
    """Fixed-bucket histogram exposing quantiles and error-budget burn rate."""

    def __init__(self, buckets_ms: list[float], slo_ms: float, slo_objective: float = 0.999) -> None:
        self.buckets = sorted(buckets_ms)
        self.counts = [0] * (len(self.buckets) + 1)
        self.total = 0
        self.slo_ms = slo_ms
        self.slo_objective = slo_objective
        self.violations = 0

    def observe(self, value_ms: float) -> None:
        idx = bisect.bisect_left(self.buckets, value_ms)
        self.counts[idx] += 1
        self.total += 1
        if value_ms > self.slo_ms:
            self.violations += 1

    def quantile(self, q: float) -> float:
        if not 0 < q <= 1 or self.total == 0:
            return 0.0
        target = q * self.total
        cumulative = 0
        for idx, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target:
                return self.buckets[idx] if idx < len(self.buckets) else float("inf")
        return float("inf")

    def burn_rate(self) -> float:
        if self.total == 0:
            return 0.0
        allowed = 1.0 - self.slo_objective
        observed = self.violations / self.total
        return observed / allowed if allowed else 0.0
'''),
                ("async-exp-async-abortable-wait", "Abortable multi-source wait with deterministic tie-break", ("asyncio", "race"),
                 '''import asyncio


async def first_success_or_timeout(sources: list, timeout_s: float) -> tuple[object, int]:
    """Return the first successful result, cancelling losers and honouring a timeout."""
    if not sources:
        raise ValueError("no sources supplied")
    tasks = {asyncio.create_task(src()): i for i, src in enumerate(sources)}
    pending = set(tasks)
    errors: list[tuple[int, str]] = []
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED, timeout=timeout_s)
            if not done:
                raise TimeoutError("all sources timed out")
            for task in sorted(done, key=lambda t: tasks[t]):
                if task.exception() is None:
                    return task.result(), tasks[task]
                errors.append((tasks[task], repr(task.exception())))
        raise RuntimeError(f"all sources failed: {errors}")
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
'''),
            ],
        },
    )


def _build_ml_patterns() -> list:
    return _domain_specs(
        "ml",
        {
            "beginner": [
                ("ml-beg-train-test-split", "Deterministic train/validation split with stratification", ("splitting", "determinism"),
                 '''import random


def stratified_split(rows: list[dict[str, object]], label_key: str, valid_fraction: float = 0.2, seed: int = 42):
    """Split rows per label class so each class keeps its validation proportion."""
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be in (0, 1)")
    rng = random.Random(seed)
    by_label: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(row.get(label_key), []).append(row)
    train: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    for label in sorted(by_label, key=str):
        group = by_label[label]
        rng.shuffle(group)
        cut = max(1, int(len(group) * valid_fraction)) if len(group) > 1 else 0
        valid.extend(group[:cut])
        train.extend(group[cut:])
    return train, valid
'''),
                ("ml-beg-normalize-minmax", "Min-max feature normalizer with fitted statistics", ("preprocessing", "scaling"),
                 '''class MinMaxScaler:
    """Scales features into [0, 1] using statistics fitted on training data only."""

    def __init__(self) -> None:
        self.min: list[float] = []
        self.max: list[float] = []

    def fit(self, rows: list[list[float]]) -> "MinMaxScaler":
        if not rows:
            raise ValueError("cannot fit on empty data")
        width = len(rows[0])
        self.min = [min(r[i] for r in rows) for i in range(width)]
        self.max = [max(r[i] for r in rows) for i in range(width)]
        return self

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for row in rows:
            scaled: list[float] = []
            for i, value in enumerate(row):
                span = self.max[i] - self.min[i]
                scaled.append(0.0 if span == 0 else (value - self.min[i]) / span)
            out.append(scaled)
        return out
'''),
                ("ml-beg-accuracy-confusion", "Confusion matrix and per-class accuracy report", ("metrics", "classification"),
                 '''def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    """Build a class x class count matrix of predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("label vectors must align")
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if not (0 <= t < num_classes and 0 <= p < num_classes):
            raise ValueError("label outside class range")
        matrix[t][p] += 1
    return matrix


def accuracy(matrix: list[list[int]]) -> float:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return 0.0
    return sum(matrix[i][i] for i in range(len(matrix))) / total
'''),
                ("ml-beg-tokenizer-vocab", "Byte-pair style vocabulary builder from whitespace tokens", ("tokenization", "vocab"),
                 '''from collections import Counter


def build_vocab(texts: list[str], min_freq: int = 2, max_size: int = 30_000) -> dict[str, int]:
    """Count word tokens and assign stable ids, reserving 0 for unknown."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<unk>": 0, "<pad>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab
'''),
                ("ml-beg-train-loop", "Minimal gradient-descent training loop with loss logging", ("training_loop", "optimization"),
                 '''def train_linear(
    xs: list[float],
    ys: list[float],
    lr: float = 0.01,
    epochs: int = 100,
) -> tuple[float, float, list[float]]:
    """Fit y = w*x + b with batch gradient descent, returning weights and loss curve."""
    if not xs or len(xs) != len(ys):
        raise ValueError("xs and ys must be non-empty and aligned")
    w = 0.0
    b = 0.0
    losses: list[float] = []
    n = len(xs)
    for _ in range(epochs):
        preds = [w * x + b for x in xs]
        errors = [p - y for p, y in zip(preds, ys)]
        loss = sum(e * e for e in errors) / n
        losses.append(loss)
        gw = 2 * sum(e * x for e, x in zip(errors, xs)) / n
        gb = 2 * sum(errors) / n
        w -= lr * gw
        b -= lr * gb
    return w, b, losses
'''),
                ("ml-beg-embedding-lookup", "Index-safe embedding table lookup with padding masking", ("embeddings", "padding"),
                 '''def embedding_lookup(table: list[list[float]], ids: list[int], mask_id: int = 0) -> list[list[float]]:
    """Fetch embeddings for ids, returning zero vectors for padding tokens."""
    out: list[list[float]] = []
    for token_id in ids:
        if token_id == mask_id:
            out.append([0.0] * (len(table[0]) if table else 0))
        elif 0 <= token_id < len(table):
            out.append(list(table[token_id]))
        else:
            raise IndexError(f"token id {token_id} outside vocabulary")
    return out
'''),
                ("ml-beg-knn", "k-nearest neighbours classifier with majority vote", ("knn", "classification"),
                 '''import math
from collections import Counter


def knn_predict(
    train_x: list[list[float]],
    train_y: list[int],
    query: list[float],
    k: int = 3,
) -> int:
    """Classify a point by plurality vote among the k nearest Euclidean neighbours."""
    if not train_x or k <= 0:
        raise ValueError("need training data and positive k")
    distances = []
    for idx, row in enumerate(train_x):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, query)))
        distances.append((dist, idx))
    distances.sort()
    votes = Counter(train_y[idx] for _dist, idx in distances[:k])
    return votes.most_common(1)[0][0]
'''),
                ("ml-beg-seed-settable-rng", "Reproducible per-experiment RNG factory", ("reproducibility", "seeding"),
                 '''import random


def make_rng(seed: int, stream: str = "train") -> random.Random:
    """Derive an independent RNG per named stream from a single base seed."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    mixed = (seed * 1_000_003) ^ (sum(ord(c) * (i + 1) for i, c in enumerate(stream)) & 0xFFFF)
    return random.Random(mixed)


def split_streams(seed: int, names: tuple[str, ...]) -> dict[str, random.Random]:
    return {name: make_rng(seed, name) for name in names}
'''),
                ("ml-beg-early-stop", "Early stopping tracker on validation loss with patience", ("training", "early_stopping"),
                 '''class EarlyStopping:
    """Signals when validation loss has not improved for `patience` epochs."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
'''),
            ],
            "intermediate": [
                ("ml-int-warmup-cosine-scheduler", "Linear warmup + cosine decay learning-rate schedule", ("schedules", "training"),
                 '''import math


def warmup_cosine_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup to peak_lr then cosine decay down to min_lr."""
    if total_steps <= 0 or peak_lr <= 0:
        raise ValueError("total_steps and peak_lr must be positive")
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
'''),
                ("ml-int-bpe-merge", "Greedy byte-pair encoding merge trainer", ("tokenization", "bpe"),
                 '''from collections import Counter


def train_bpe_merges(corpus: list[list[str]], num_merges: int = 10) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Learn the most frequent adjacent symbol merges over a tokenized corpus."""
    vocab: Counter[str] = Counter()
    for doc in corpus:
        for token in doc:
            vocab[token] += 1
    symbols = [list(doc) for doc in corpus]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for doc in symbols:
            for a, b in zip(doc, doc[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (best, _count) = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append(best)
        merged = best[0] + best[1]
        new_symbols: list[list[str]] = []
        for doc in symbols:
            out: list[str] = []
            i = 0
            while i < len(doc):
                if i + 1 < len(doc) and doc[i] == best[0] and doc[i + 1] == best[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(doc[i])
                    i += 1
            new_symbols.append(out)
        symbols = new_symbols
    vocab_ids = {token: idx for idx, token in enumerate(sorted(vocab))}
    return vocab_ids, merges
'''),
                ("ml-int-grad-clip", "Gradient clipping by global norm with reporting", ("optimization", "stability"),
                 '''import math


def clip_by_global_norm(grads: list[float], max_norm: float) -> tuple[list[float], float]:
    """Rescale a gradient vector so its L2 norm does not exceed max_norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = math.sqrt(sum(g * g for g in grads))
    if norm <= max_norm or norm == 0:
        return list(grads), norm
    scale = max_norm / (norm + 1e-12)
    return [g * scale for g in grads], norm
'''),
                ("ml-int-auc-roc", "ROC-AUC computation with tie-aware ranking", ("metrics", "ranking"),
                 '''def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC-AUC via the Mann-Whitney U statistic with tie handling."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranked = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    idx = 0
    rank = 1
    while idx < len(ranked):
        j = idx
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[idx][0]:
            j += 1
        avg_rank = (rank + (rank + (j - idx))) / 2
        for k in range(idx, j + 1):
            if ranked[k][1] == 1:
                rank_sum += avg_rank
        rank += j - idx + 1
        idx = j + 1
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
'''),
                ("ml-int-dedup-minhash", "MinHash-based near-duplicate detection for training documents", ("deduplication", "minhash"),
                 '''import hashlib


def _shingles(text: str, k: int = 5) -> set[str]:
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def minhash_signature(text: str, num_hashes: int = 64, k: int = 5, seed: int = 7) -> tuple[int, ...]:
    """Compute a MinHash signature over k-shingles of the document."""
    shingles = _shingles(text, k)
    if not shingles:
        return tuple([0] * num_hashes)
    sig: list[int] = []
    for i in range(num_hashes):
        best = None
        for shingle in shingles:
            digest = hashlib.blake2b(f"{seed}:{i}:{shingle}".encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            if best is None or value < best:
                best = value
        sig.append(best if best is not None else 0)
    return tuple(sig)


def jaccard_estimate(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("signatures must be non-empty and equal length")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)
'''),
                ("ml-int-mixed-precision-loss-scale", "Dynamic loss-scaling policy for bf16/fp16 training", ("mixed_precision", "training"),
                 '''class DynamicLossScaler:
    """Doubles the scale after healthy windows and halves it on overflow."""

    def __init__(self, init_scale: float = 65_536.0, growth_interval: int = 2000) -> None:
        if init_scale <= 1:
            raise ValueError("initial scale must exceed 1")
        self.scale = init_scale
        self.growth_interval = growth_interval
        self._good = 0
        self.skipped_steps = 0

    def on_overflow(self) -> float:
        self.scale = max(1.0, self.scale * 0.5)
        self._good = 0
        self.skipped_steps += 1
        return self.scale

    def on_step(self, found_inf: bool) -> float:
        if found_inf:
            return self.on_overflow()
        self._good += 1
        if self._good >= self.growth_interval:
            self.scale = min(2.0 ** 24, self.scale * 2)
            self._good = 0
        return self.scale
'''),
                ("ml-int-data-loader-shuffle", "Deterministic epoch-shuffled minibatch loader", ("data_loading", "batching"),
                 '''import random


def epoch_batches(
    n_examples: int,
    batch_size: int,
    epoch: int,
    seed: int = 42,
    drop_last: bool = False,
) -> list[list[int]]:
    """Produce per-epoch shuffled index batches reproducible from (seed, epoch)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rng = random.Random(seed * 1_000_003 + epoch)
    indices = list(range(n_examples))
    rng.shuffle(indices)
    batches = [indices[i : i + batch_size] for i in range(0, n_examples, batch_size)]
    if drop_last and batches and len(batches[-1]) < batch_size:
        batches.pop()
    return batches
'''),
                ("ml-int-fim-transform", "Fill-in-the-middle training example builder with sentinel tokens", ("fim", "pretraining"),
                 '''def build_fim_example(
    prefix: str,
    middle: str,
    suffix: str,
    sentinel_prefix: str = "<fim_prefix>",
    sentinel_middle: str = "<fim_middle>",
    sentinel_suffix: str = "<fim_suffix>",
) -> str:
    """Compose a PSM-ordered FIM sequence for infilling pretraining."""
    if not middle.strip():
        raise ValueError("middle span must not be empty")
    return f"{sentinel_prefix}{prefix}{sentinel_suffix}{suffix}{sentinel_middle}{middle}"


def split_for_fim(tokens: list[str], rng, min_rate: float = 0.1, max_rate: float = 0.5) -> tuple[list[str], list[str], list[str]] | None:
    """Randomly split a token sequence into (prefix, middle, suffix) for FIM."""
    if len(tokens) < 8:
        return None
    rate = rng.uniform(min_rate, max_rate)
    mid_len = max(1, int(len(tokens) * rate))
    start = rng.randrange(1, len(tokens) - mid_len)
    return tokens[:start], tokens[start : start + mid_len], tokens[start + mid_len :]
'''),
                ("ml-int-class-weight", "Inverse-frequency class weighting for imbalanced targets", ("imbalance", "training"),
                 '''def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Balance classes by assigning weight inversely proportional to frequency."""
    if not labels:
        return {}
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    classes = len(counts)
    return {label: total / (classes * count) for label, count in counts.items()}


def apply_weights(labels: list[int], weights: dict[int, float]) -> list[float]:
    return [weights.get(label, 1.0) for label in labels]
'''),
            ],
            "advanced": [
                ("ml-adv-attention-mask", "Scaled dot-product attention with causal masking", ("transformers", "attention"),
                 '''import math


def scaled_dot_product_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
    causal: bool = True,
) -> list[list[float]]:
    """Softmax(QK^T / sqrt(d)) V with an optional lower-triangular causal mask."""
    n, d = len(queries), len(queries[0]) if queries else 0
    scale = 1.0 / math.sqrt(max(1, d))
    out: list[list[float]] = []
    for i, q in enumerate(queries):
        logits: list[float] = []
        for j, k in enumerate(keys):
            if causal and j > i:
                logits.append(float("-inf"))
            else:
                logits.append(sum(a * b for a, b in zip(q, k)) * scale)
        finite = [v for v in logits if v != float("-inf")]
        peak = max(finite) if finite else 0.0
        exps = [0.0 if v == float("-inf") else math.exp(v - peak) for v in logits]
        denom = sum(exps) or 1.0
        weights = [e / denom for e in exps]
        out.append([sum(w * v[dim] for w, v in zip(weights, values)) for dim in range(len(values[0]) if values else 0)])
    return out
'''),
                ("ml-adv-layer-norm", "Layer normalization with learnable affine parameters", ("transformers", "normalization"),
                 '''import math


def layer_norm(
    row: list[float],
    gamma: list[float] | None = None,
    beta: list[float] | None = None,
    eps: float = 1e-5,
) -> list[float]:
    """Normalize a feature vector to zero mean / unit variance then apply affine."""
    if not row:
        return []
    mean = sum(row) / len(row)
    var = sum((x - mean) ** 2 for x in row) / len(row)
    denom = math.sqrt(var + eps)
    out = [(x - mean) / denom for x in row]
    if gamma is not None:
        if len(gamma) != len(out):
            raise ValueError("gamma shape mismatch")
        out = [g * o for g, o in zip(gamma, out)]
    if beta is not None:
        if len(beta) != len(out):
            raise ValueError("beta shape mismatch")
        out = [b + o for b, o in zip(beta, out)]
    return out
'''),
                ("ml-adv-adamw-update", "AdamW optimizer step with decoupled weight decay", ("optimizers", "adamw"),
                 '''import math


class AdamW:
    """Adam with decoupled weight decay and bias-corrected moments."""

    def __init__(self, params: list[float], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0

    def step(self, grads: list[float]) -> list[float]:
        if len(grads) != len(self.params):
            raise ValueError("gradient length mismatch")
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            self.params[i] -= self.lr * (m_hat / (math.sqrt(v_hat) + self.eps) + self.wd * self.params[i])
        return self.params
'''),
                ("ml-adv-quantization-int8", "Symmetric int8 quantization with per-channel scales", ("quantization", "inference"),
                 '''def quantize_symmetric(values: list[float]) -> tuple[list[int], float]:
    """Map floats to int8 with a single symmetric scale, clipping outliers."""
    if not values:
        return [], 1.0
    max_abs = max(abs(v) for v in values)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    codes = [max(-127, min(127, int(round(v / scale)))) for v in values]
    return codes, scale


def dequantize(codes: list[int], scale: float) -> list[float]:
    return [c * scale for c in codes]


def quantization_error(values: list[float]) -> float:
    codes, scale = quantize_symmetric(values)
    restored = dequantize(codes, scale)
    return max((abs(a - b) for a, b in zip(values, restored)), default=0.0)
'''),
                ("ml-adv-dataset-contamination", "Benchmark contamination detector via n-gram overlap", ("evaluation", "contamination"),
                 '''def ngram_overlap(train_text: str, benchmark_text: str, n: int = 13) -> float:
    """Fraction of benchmark n-grams that also appear in the training text."""
    bench_tokens = benchmark_text.split()
    if len(bench_tokens) < n:
        return 0.0
    bench = {" ".join(bench_tokens[i : i + n]) for i in range(len(bench_tokens) - n + 1)}
    train_tokens = train_text.split()
    train = {" ".join(train_tokens[i : i + n]) for i in range(max(0, len(train_tokens) - n + 1))}
    if not bench:
        return 0.0
    return len(bench & train) / len(bench)


def is_contaminated(train_text: str, benchmark_text: str, threshold: float = 0.05) -> bool:
    return ngram_overlap(train_text, benchmark_text) > threshold
'''),
                ("ml-adv-gradual-warmup-resume", "Checkpoint resume with optimizer-state validation", ("checkpoints", "resume"),
                 '''def validate_resume_state(
    checkpoint: dict[str, object],
    expected_step: int,
    expected_lr: float,
    tolerance: float = 1e-9,
) -> list[str]:
    """Verify a checkpoint is safe to resume: step continuity and LR consistency."""
    problems: list[str] = []
    step = checkpoint.get("step")
    if not isinstance(step, int):
        problems.append("checkpoint missing integer step")
    elif step != expected_step:
        problems.append(f"step mismatch: checkpoint={step} expected={expected_step}")
    lr = checkpoint.get("lr")
    if not isinstance(lr, (int, float)):
        problems.append("checkpoint missing learning rate")
    elif abs(float(lr) - expected_lr) > tolerance:
        problems.append(f"lr mismatch: checkpoint={lr} expected={expected_lr}")
    for key in ("model_state", "optimizer_state", "rng_state"):
        if key not in checkpoint:
            problems.append(f"missing {key}")
    return problems
'''),
                ("ml-adv-speculative-decoding", "Draft-model speculative decoding acceptance loop", ("inference", "speculative_decoding"),
                 '''import random


def speculative_decode(
    draft_sample,
    target_prob,
    draft_prob,
    max_new_tokens: int = 8,
    rng: random.Random | None = None,
) -> list[str]:
    """Accept draft tokens by probability ratio; resample on first rejection."""
    rng = rng or random.Random(0)
    accepted: list[str] = []
    for _ in range(max_new_tokens):
        token = draft_sample()
        p_target = target_prob(token)
        p_draft = draft_prob(token)
        ratio = 1.0 if p_draft <= 0 else min(1.0, p_target / p_draft)
        if rng.random() <= ratio:
            accepted.append(token)
            continue
        accepted.append(f"<resample:{token}>")
        break
    return accepted
'''),
                ("ml-adv-distributed-allreduce", "Ring all-reduce gradient averaging across workers", ("distributed", "allreduce"),
                 '''def ring_allreduce(buffers: list[list[float]]) -> list[list[float]]:
    """Average identical-shaped buffers across workers via a ring exchange model."""
    if not buffers:
        raise ValueError("no worker buffers")
    width = len(buffers[0])
    if any(len(b) != width for b in buffers):
        raise ValueError("all buffers must share the same length")
    workers = len(buffers)
    partial = [list(b) for b in buffers]
    for step in range(workers - 1):
        for w in range(workers):
            send = (w - step) % workers
            recv = (w - step - 1) % workers
            partial[recv] = [a + b for a, b in zip(partial[recv], partial[send])]
    return [[v / workers for v in buf] for buf in partial]
'''),
                ("ml-adv-eval-decontamination-split", "Difficulty-stratified held-out evaluation split builder", ("evaluation", "splits"),
                 '''def stratified_holdout(
    examples: list[dict[str, object]],
    difficulty_key: str = "difficulty",
    per_bucket: int = 50,
    seed: int = 13,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Hold out a fixed number of examples per difficulty bucket."""
    import random

    rng = random.Random(seed)
    buckets: dict[object, list[dict[str, object]]] = {}
    for ex in examples:
        buckets.setdefault(ex.get(difficulty_key), []).append(ex)
    held: list[dict[str, object]] = []
    keep: list[dict[str, object]] = []
    for key in sorted(buckets, key=str):
        group = buckets[key]
        rng.shuffle(group)
        held.extend(group[:per_bucket])
        keep.extend(group[per_bucket:])
    return keep, held
'''),
            ],
            "expert": [
                ("ml-exp-flash-attention-tiling", "Tiled (flash-attention style) online-softmax attention", ("transformers", "flash_attention", "memory"),
                 '''import math


def tiled_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
    tile: int = 2,
) -> list[list[float]]:
    """Streaming softmax attention accumulating running max and normalizer per tile."""
    d = len(queries[0]) if queries else 0
    scale = 1.0 / math.sqrt(max(1, d))
    outputs: list[list[float]] = []
    for q in queries:
        running_max = -float("inf")
        running_sum = 0.0
        acc = [0.0] * (len(values[0]) if values else 0)
        for start in range(0, len(keys), tile):
            tile_keys = keys[start : start + tile]
            tile_values = values[start : start + tile]
            scores = [sum(a * b for a, b in zip(q, k)) * scale for k in tile_keys]
            tile_max = max(scores) if scores else -float("inf")
            new_max = max(running_max, tile_max)
            correction = math.exp(running_max - new_max) if running_max != -float("inf") else 0.0
            exp_scores = [math.exp(s - new_max) for s in scores]
            running_sum = running_sum * correction + sum(exp_scores)
            acc = [a * correction for a in acc]
            for weight, vec in zip(exp_scores, tile_values):
                for i, val in enumerate(vec):
                    acc[i] += weight * val
            running_max = new_max
        outputs.append([a / running_sum if running_sum else 0.0 for a in acc])
    return outputs
'''),
                ("ml-exp-mixture-of-experts-router", "Top-k mixture-of-experts router with load-balance loss", ("transformers", "moe"),
                 '''def top_k_router(
    logits: list[float],
    k: int = 2,
    experts: int = 8,
) -> tuple[list[int], list[float], float]:
    """Select top-k experts, normalize gate weights, and compute load-balance loss."""
    if not 0 < k <= min(len(logits), experts):
        raise ValueError("invalid top-k for expert count")
    import math

    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    ranked = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:k]
    chosen = [probs[i] for i in ranked]
    denom = sum(chosen) or 1.0
    gates = [c / denom for c in chosen]
    counts = [0] * experts
    for i in ranked:
        counts[i] += 1
    mean_count = sum(counts) / experts
    fraction = [c / k for c in counts]
    load_balance = experts * sum(f * p for f, p in zip(fraction, probs)) / max(1e-9, mean_count)
    return ranked, gates, load_balance
'''),
                ("ml-exp-grad-accumulation-schedule", "Gradient accumulation with correct loss scaling and step counting", ("training", "gradient_accumulation"),
                 '''class GradientAccumulator:
    """Accumulates micro-batch gradients, stepping once per effective batch."""

    def __init__(self, accumulation_steps: int, base_lr: float) -> None:
        if accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.steps = accumulation_steps
        self.base_lr = base_lr
        self.buffer: list[float] = []
        self.micro_batches = 0
        self.optimizer_steps = 0

    def accumulate(self, grads: list[float]) -> list[float] | None:
        if self.buffer and len(self.buffer) != len(grads):
            raise ValueError("gradient shape changed mid-accumulation")
        self.buffer = [a + b for a, b in zip(self.buffer, grads)] if self.buffer else list(grads)
        self.micro_batches += 1
        if self.micro_batches % self.steps != 0:
            return None
        mean = [g / self.steps for g in self.buffer]
        self.buffer = []
        self.optimizer_steps += 1
        return mean

    def effective_lr(self) -> float:
        return self.base_lr * self.steps
'''),
                ("ml-exp-contrastive-loss", "InfoNCE contrastive loss with temperature and in-batch negatives", ("losses", "contrastive"),
                 '''import math


def info_nce_loss(
    anchor: list[float],
    positives: list[list[float]],
    negatives: list[list[float]],
    temperature: float = 0.07,
) -> float:
    """Compute InfoNCE loss treating in-batch negatives as the denominator."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    pos_sims = [cosine(anchor, p) / temperature for p in positives]
    neg_sims = [cosine(anchor, n) / temperature for n in negatives]
    peak = max(pos_sims) if pos_sims else 0.0
    denom = sum(math.exp(s - peak) for s in pos_sims + neg_sims) if (pos_sims or neg_sims) else 1.0
    numer = sum(math.exp(s - peak) for s in pos_sims) if pos_sims else 1e-12
    return -math.log(max(numer, 1e-12) / max(denom, 1e-12))
'''),
                ("ml-exp-tokenizer-fertility", "Tokenizer fertility and compression ratio auditor", ("tokenization", "evaluation"),
                 '''def fertility_report(texts: list[str], tokenize) -> dict[str, float]:
    """Report mean tokens-per-word and characters-per-token for a tokenizer."""
    total_words = 0
    total_tokens = 0
    total_chars = 0
    for text in texts:
        words = text.split()
        tokens = tokenize(text)
        total_words += len(words)
        total_tokens += len(tokens)
        total_chars += sum(len(t) for t in tokens)
    if total_words == 0 or total_tokens == 0:
        return {"fertility": 0.0, "chars_per_token": 0.0, "tokens": 0.0}
    return {
        "fertility": total_tokens / total_words,
        "chars_per_token": total_chars / total_tokens,
        "tokens": float(total_tokens),
    }
'''),
                ("ml-exp-lr-transfer-scaling", "Hyperparameter transfer via muP-style width scaling", ("scaling_laws", "hyperparameter_transfer"),
                 '''def muP_scaled_lr(
    base_lr: float,
    base_width: int,
    target_width: int,
    embedding_lr_scale: float = 0.1,
) -> dict[str, float]:
    """Scale per-layer learning rates for muP width transfer."""
    if base_width <= 0 or target_width <= 0:
        raise ValueError("widths must be positive")
    ratio = base_width / target_width
    return {
        "hidden_lr": base_lr * ratio,
        "embedding_lr": base_lr * embedding_lr_scale,
        "output_lr": base_lr * ratio,
    }
'''),
                ("ml-exp-dataloader-prefetch-pipeline", "Deterministic prefetch pipeline with shard-aware sampling", ("data_loading", "throughput"),
                 '''import heapq
import random


class PrefetchPipeline:
    """Interleaves shard readers deterministically while keeping a bounded prefetch."""

    def __init__(self, shards: list[list[int]], depth: int = 4, seed: int = 0) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        rng = random.Random(seed)
        order = list(range(len(shards)))
        rng.shuffle(order)
        self.streams = [list(shards[i]) for i in order]
        self.depth = depth
        self.ready: list[tuple[int, int, int]] = []
        for idx, stream in enumerate(self.streams):
            if stream:
                heapq.heappush(self.ready, (stream[0], idx, 0))

    def next(self) -> int | None:
        if not self.ready:
            return None
        value, stream_idx, pos = heapq.heappop(self.ready)
        nxt = pos + 1
        if nxt < len(self.streams[stream_idx]):
            heapq.heappush(self.ready, (self.streams[stream_idx][nxt], stream_idx, nxt))
        return value

    def batch(self, size: int) -> list[int]:
        out: list[int] = []
        for _ in range(size):
            item = self.next()
            if item is None:
                break
            out.append(item)
        return out
'''),
                ("ml-exp-eval-pass-at-k", "Unbiased pass@k estimator with combinatorial correction", ("evaluation", "pass_at_k"),
                 '''import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k from n samples with c correct, avoiding naive bias."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
'''),
                ("ml-exp-checkpoint-sharding", "Sharded checkpoint writer with manifest verification", ("checkpoints", "sharding"),
                 '''import hashlib


def shard_tensors(tensors: dict[str, list[float]], max_shard_bytes: int = 1 << 20) -> list[dict[str, object]]:
    """Partition a tensor dict into size-bounded shards with digests."""
    shards: list[dict[str, object]] = []
    current: dict[str, list[float]] = {}
    current_bytes = 0
    for name, values in tensors.items():
        estimate = len(values) * 4
        if current and current_bytes + estimate > max_shard_bytes:
            shards.append({"tensors": current, "bytes": current_bytes})
            current, current_bytes = {}, 0
        current[name] = values
        current_bytes += estimate
    if current:
        shards.append({"tensors": current, "bytes": current_bytes})
    return [
        {
            "index": idx,
            "names": sorted(shard["tensors"]),  # type: ignore[arg-type]
            "bytes": shard["bytes"],
            "digest": hashlib.sha256(repr(sorted(shard["tensors"])).encode()).hexdigest()[:16],  # type: ignore[arg-type]
        }
        for idx, shard in enumerate(shards)
    ]
'''),
                ("ml-exp-synthetic-task-verifier", "Synthetic instruction task verifier with executable unit checks", ("synthetic_data", "verification"),
                 '''def verify_generated_task(task: dict[str, object]) -> list[str]:
    """Check a synthetic code task for required fields and runnable assertions."""
    problems: list[str] = []
    for key in ("prompt", "tests", "reference"):
        if not str(task.get(key, "")).strip():
            problems.append(f"missing field: {key}")
    tests = str(task.get("tests", ""))
    if "assert" not in tests:
        problems.append("tests contain no assertions")
    namespace: dict[str, object] = {}
    try:
        exec(str(task.get("reference", "")), {"__builtins__": {}}, namespace)  # noqa: S102 - sandboxed check
    except SyntaxError as exc:
        problems.append(f"reference does not compile: {exc}")
    return problems
'''),
            ],
        },
    )

_DOMAIN_BUILDERS: dict[str, Callable[[], list[PatternSpec]]] = {
    "algorithms": _build_algorithms_patterns,
    "data": _build_data_patterns,
    "web": _build_web_patterns,
    "systems": _build_systems_patterns,
    "testing": _build_testing_patterns,
    "database": _build_database_patterns,
    "cli": _build_cli_patterns,
    "security": _build_security_patterns,
    "async_io": _build_async_io_patterns,
    "ml": _build_ml_patterns,
}


def _build_all_patterns() -> list[PatternSpec]:
    """Materialize every pattern for every implemented domain builder."""
    out: list[PatternSpec] = []
    for domain in DOMAINS:
        builder = _DOMAIN_BUILDERS.get(domain)
        if builder is None:
            continue
        out.extend(builder())
    return out


ALL_PATTERNS: tuple[PatternSpec, ...] = tuple(_build_all_patterns())


PATTERNS_BY_CELL: dict[tuple[str, str], tuple[PatternSpec, ...]] = {
    (complexity, domain): tuple(
        spec for spec in ALL_PATTERNS if spec.complexity == complexity and spec.domain == domain
    )
    for complexity in COMPLEXITIES
    for domain in DOMAINS
}


def patterns_for(complexity: str, domain: str) -> tuple[PatternSpec, ...]:
    """Return patterns for an exact (complexity, domain) cell.

    Raises ``KeyError`` for unknown complexity tiers, and ``ValueError`` when the
    requested cell has no patterns, so callers never silently fall back to a
    cross-domain or cross-complexity sample.
    """
    comp = (complexity or "").strip().lower()
    dom = normalize_domain(domain)
    if comp not in COMPLEXITIES:
        raise KeyError(f"unknown complexity tier: {complexity!r}")
    cell = PATTERNS_BY_CELL.get((comp, dom))
    if not cell:
        raise ValueError(f"no code patterns for complexity={comp!r} domain={dom!r}")
    return cell


def pattern_count() -> int:
    return len(ALL_PATTERNS)


def validate_catalog(min_per_cell: int = 1) -> list[str]:
    """Return human-readable problems with the catalog (empty list means OK)."""
    problems: list[str] = []
    seen: dict[str, str] = {}
    for spec in ALL_PATTERNS:
        if spec.pattern_id in seen:
            problems.append(f"duplicate pattern_id {spec.pattern_id!r}")
        seen[spec.pattern_id] = spec.pattern_id
        if spec.complexity not in COMPLEXITIES:
            problems.append(f"{spec.pattern_id}: unknown complexity {spec.complexity!r}")
        if spec.domain not in DOMAINS:
            problems.append(f"{spec.pattern_id}: unknown domain {spec.domain!r}")
        try:
            compile(spec.code, f"<{spec.pattern_id}>", "exec")
        except SyntaxError as exc:
            problems.append(f"{spec.pattern_id}: does not compile: {exc}")
    for comp in COMPLEXITIES:
        for dom in DOMAINS:
            n = len(PATTERNS_BY_CELL[(comp, dom)])
            if n < min_per_cell:
                problems.append(f"cell ({comp}, {dom}) has {n} patterns, need >= {min_per_cell}")
    return problems
