"""
Code Template Engine - Generates Code Across 20+ Programming Languages

This engine creates syntactically valid, semantically meaningful code samples
across many programming languages. It uses:
- Language-specific AST patterns
- Realistic variable/function naming conventions
- Progressive complexity levels (beginner to expert)
- Domain-specific patterns (web, data science, systems, etc.)
"""







import random



import hashlib



from typing import List, Dict, Optional, Tuple



from dataclasses import dataclass











@dataclass



class CodeSample:



    """A generated code sample with metadata."""



    language: str



    code: str



    description: str



    complexity: str                                                    



    domain: str                                                           



    tags: List[str]











                                                              



                                   



                                                              







class NameGenerator:



    """Generate realistic variable, function, and class names."""







                                      



    VARIABLE_NAMES = {



        "general": ["result", "value", "data", "item", "element", "temp", "output", "input",



                    "count", "index", "total", "current", "previous", "next_val", "buffer",



                    "cache", "state", "config", "options", "params", "args", "context"],



        "web": ["request", "response", "session", "user", "token", "cookie", "headers",



                "body", "query", "params", "route", "endpoint", "middleware", "handler"],



        "data": ["dataset", "dataframe", "series", "array", "matrix", "tensor", "batch",



                 "features", "labels", "predictions", "embeddings", "weights", "gradients"],



        "algorithms": ["node", "edge", "graph", "tree", "heap", "stack", "queue",



                      "visited", "distance", "parent", "left", "right", "root", "leaf"],



        "systems": ["buffer", "socket", "connection", "process", "thread", "mutex",



                   "semaphore", "channel", "pipe", "stream", "file_descriptor", "handle"],



    }







    FUNCTION_PREFIXES = [



        "get", "set", "create", "delete", "update", "find", "search", "sort",



        "filter", "map", "reduce", "process", "handle", "parse", "format",



        "validate", "check", "compute", "calculate", "generate", "build",



        "initialize", "cleanup", "serialize", "deserialize", "encode", "decode",



        "transform", "merge", "split", "join", "connect", "disconnect",



    ]







    FUNCTION_SUFFIXES = [



        "data", "value", "item", "list", "map", "result", "output", "input",



        "config", "state", "record", "entry", "node", "element", "object",



        "user", "session", "token", "request", "response", "buffer",



    ]







    CLASS_NAMES = {



        "web": ["Server", "Router", "Controller", "Middleware", "Handler", "Validator",



                "Authenticator", "RateLimiter", "CacheManager", "SessionStore"],



        "data": ["DataLoader", "Preprocessor", "Pipeline", "Transformer", "Model",



                "Trainer", "Evaluator", "Metrics", "DataStore", "FeatureExtractor"],



        "algorithms": ["Graph", "Tree", "Heap", "HashTable", "LinkedList", "Stack",



                      "Queue", "Trie", "BloomFilter", "DisjointSet"],



        "systems": ["BufferPool", "ConnectionPool", "ThreadPool", "EventLoop",



                   "MessageQueue", "FileSystem", "MemoryManager", "Scheduler"],



        "general": ["Manager", "Factory", "Builder", "Strategy", "Observer", "Adapter",



                   "Decorator", "Proxy", "Facade", "Command", "Iterator"],



    }







    @classmethod



    def variable(cls, domain: str = "general", style: str = "snake_case") -> str:



        names = cls.VARIABLE_NAMES.get(domain, cls.VARIABLE_NAMES["general"])



        name = random.choice(names)



        return cls._apply_style(name, style)







    @classmethod



    def function(cls, domain: str = "general", style: str = "snake_case") -> str:



        prefix = random.choice(cls.FUNCTION_PREFIXES)



        suffix = random.choice(cls.FUNCTION_SUFFIXES)



        name = f"{prefix}_{suffix}"



        return cls._apply_style(name, style)







    @classmethod



    def class_name(cls, domain: str = "general") -> str:



        names = cls.CLASS_NAMES.get(domain, cls.CLASS_NAMES["general"])



        return random.choice(names)







    @classmethod



    def _apply_style(cls, name: str, style: str) -> str:



        if style == "snake_case":



            return name.lower().replace("-", "_")



        elif style == "camelCase":



            parts = name.split("_")



            return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])



        elif style == "PascalCase":



            return "".join(p.capitalize() for p in name.split("_"))



        elif style == "SCREAMING_SNAKE":



            return name.upper().replace("-", "_")



        return name











                                                              



                        



                                                              







class PythonCodeGenerator:



    """Generate Python code samples at various complexity levels."""







    def __init__(self):



        self.names = NameGenerator()







    def generate(self, complexity: str, domain: str) -> CodeSample:



        """Generate a code sample for the given complexity and domain.
        
        Falls back to any available generator if the exact combination
        is not implemented.
        """



                                       



        generators = {



            ("beginner", "algorithms"): self._beginner_algorithms,



            ("intermediate", "algorithms"): self._intermediate_algorithms,



            ("intermediate", "data"): self._intermediate_data,



            ("intermediate", "web"): self._intermediate_web,



            ("intermediate", "testing"): self._intermediate_testing,



            ("advanced", "algorithms"): self._advanced_algorithms,



            ("advanced", "data"): self._advanced_data,



            ("advanced", "web"): self._advanced_web,



            ("advanced", "systems"): self._advanced_systems,



            ("expert", "algorithms"): self._expert_algorithms,



            ("expert", "systems"): self._expert_systems,



            ("expert", "data"): self._expert_data,



        }







        gen_func = generators.get((complexity, domain))



        if gen_func is None:



                                                                  



            all_gens = list(generators.values())



            gen_func = random.choice(all_gens)







        return gen_func()







    def _beginner_algorithms(self) -> CodeSample:



        variants = [



            self._gen_binary_search,



            self._gen_bubble_sort,



            self._gen_fibonacci,



            self._gen_factorial,



            self._gen_palindrome_check,



            self._gen_list_operations,



            self._gen_string_manipulation,



            self._gen_matrix_operations,



        ]



        return random.choice(variants)()







    def _gen_binary_search(self) -> CodeSample:



        code = '''def binary_search(arr: list[int], target: int) -> int:
    """
    Perform binary search on a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


# Example usage
if __name__ == "__main__":
    numbers = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    result = binary_search(numbers, 23)
    print(f"Found at index: {result}")  # Output: Found at index: 5'''







        return CodeSample(



            language="python",



            code=code,



            description="Binary search implementation with type hints and docstring",



            complexity="beginner",



            domain="algorithms",



            tags=["binary_search", "search", "arrays", "type_hints"],



        )







    def _gen_bubble_sort(self) -> CodeSample:



        code = '''def bubble_sort(arr: list[int]) -> list[int]:
    """
    Sort an array using the bubble sort algorithm.
    
    Bubble sort repeatedly steps through the list, compares adjacent elements
    and swaps them if they are in the wrong order.
    
    Args:
        arr: List of integers to sort
        
    Returns:
        Sorted list (in-place modification)
        
    Time Complexity: O(n²) worst/average, O(n) best (already sorted)
    Space Complexity: O(1)
    """
    n = len(arr)
    
    for i in range(n):
        # Track if any swaps were made in this pass
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # If no swaps were made, array is already sorted
        if not swapped:
            break
    
    return arr


def bubble_sort_descending(arr: list[int]) -> list[int]:
    """Sort in descending order."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# Test the implementation
if __name__ == "__main__":
    test_cases = [
        [64, 34, 25, 12, 22, 11, 90],
        [1, 2, 3, 4, 5],  # Already sorted
        [5, 4, 3, 2, 1],  # Reverse sorted
        [1],               # Single element
        [],                # Empty list
    ]
    
    for test in test_cases:
        original = test.copy()
        result = bubble_sort(test)
        print(f"{original} -> {result}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Bubble sort with optimization (early termination)",



            complexity="beginner",



            domain="algorithms",



            tags=["bubble_sort", "sorting", "arrays", "optimization"],



        )







    def _gen_fibonacci(self) -> CodeSample:



        code = '''from functools import lru_cache
from typing import Generator


def fibonacci_iterative(n: int) -> int:
    """
    Calculate the nth Fibonacci number using iteration.
    
    The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    Each number is the sum of the two preceding ones.
    
    Args:
        n: Position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr


@lru_cache(maxsize=None)
def fibonacci_recursive(n: int) -> int:
    """
    Calculate the nth Fibonacci number using memoized recursion.
    
    Time Complexity: O(n) with memoization
    Space Complexity: O(n) for the call stack
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_generator(limit: int) -> Generator[int, None, None]:
    """
    Generate Fibonacci numbers up to a limit.
    
    Args:
        limit: Maximum value to generate (inclusive)
        
    Yields:
        Fibonacci numbers up to the limit
    """
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b


def fibonacci_matrix(n: int) -> int:
    """
    Calculate the nth Fibonacci number using matrix exponentiation.
    
    Uses the identity: [[F(n+1), F(n)], [F(n), F(n-1)]] = [[1,1],[1,0]]^n
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n
    
    def matrix_multiply(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0],
             a[0][0] * b[0][1] + a[0][1] * b[1][1]],
            [a[1][0] * b[0][0] + a[1][1] * b[1][0],
             a[1][0] * b[0][1] + a[1][1] * b[1][1]],
        ]
    
    def matrix_power(m: list[list[int]], p: int) -> list[list[int]]:
        result = [[1, 0], [0, 1]]  # Identity matrix
        base = m
        while p > 0:
            if p % 2 == 1:
                result = matrix_multiply(result, base)
            base = matrix_multiply(base, base)
            p //= 2
        return result
    
    base_matrix = [[1, 1], [1, 0]]
    result = matrix_power(base_matrix, n)
    return result[0][1]


# Comprehensive testing
if __name__ == "__main__":
    # Test all implementations agree
    for n in range(20):
        iter_result = fibonacci_iterative(n)
        rec_result = fibonacci_recursive(n)
        mat_result = fibonacci_matrix(n)
        assert iter_result == rec_result == mat_result, f"Mismatch at n={n}"
        print(f"F({n}) = {iter_result}")
    
    print("\\nFirst Fibonacci numbers up to 1000:")
    for fib in fibonacci_generator(1000):
        print(fib, end=" ")'''







        return CodeSample(



            language="python",



            code=code,



            description="Multiple Fibonacci implementations with different time complexities",



            complexity="beginner",



            domain="algorithms",



            tags=["fibonacci", "recursion", "memoization", "matrix_exponentiation", "generators"],



        )







    def _gen_factorial(self) -> CodeSample:



        code = '''import math
from functools import reduce
from typing import Union


def factorial_iterative(n: int) -> int:
    """
    Calculate n! using iteration.
    
    Args:
        n: Non-negative integer
        
    Returns:
        n factorial
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError(f"Factorial not defined for negative numbers: {n}")
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def factorial_recursive(n: int) -> int:
    """Calculate n! using recursion."""
    if n < 0:
        raise ValueError(f"Factorial not defined for negative numbers: {n}")
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


def factorial_reduce(n: int) -> int:
    """Calculate n! using reduce."""
    if n < 0:
        raise ValueError(f"Factorial not defined for negative numbers: {n}")
    if n <= 1:
        return 1
    return reduce(lambda x, y: x * y, range(1, n + 1))


def trailing_zeros_in_factorial(n: int) -> int:
    """
    Count trailing zeros in n! without computing n!.
    
    Trailing zeros come from factors of 10, which come from 2 × 5.
    Since there are always more 2s than 5s, count the 5s.
    
    Time Complexity: O(log n)
    """
    count = 0
    power_of_5 = 5
    while power_of_5 <= n:
        count += n // power_of_5
        power_of_5 *= 5
    return count


# Verify all implementations match
if __name__ == "__main__":
    for n in range(15):
        i = factorial_iterative(n)
        r = factorial_recursive(n)
        rd = factorial_reduce(n)
        m = math.factorial(n)
        assert i == r == rd == m
        print(f"{n}! = {i}")
    
    print(f"\\nTrailing zeros in 100! = {trailing_zeros_in_factorial(100)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Factorial implementations and trailing zeros counting",



            complexity="beginner",



            domain="algorithms",



            tags=["factorial", "recursion", "math", "number_theory"],



        )







    def _gen_palindrome_check(self) -> CodeSample:



        code = '''def is_palindrome(s: str) -> bool:
    """
    Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric).
    
    Args:
        s: Input string
        
    Returns:
        True if the string is a palindrome
        
    Examples:
        >>> is_palindrome("A man, a plan, a canal: Panama")
        True
        >>> is_palindrome("race a car")
        False
    """
    # Clean the string
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def is_palindrome_two_pointers(s: str) -> bool:
    """
    Check palindrome using two pointers (O(1) space).
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True


def is_palindrome_number(n: int) -> bool:
    """
    Check if an integer is a palindrome without converting to string.
    
    Time Complexity: O(log₁₀(n))
    Space Complexity: O(1)
    """
    if n < 0:
        return False
    if n < 10:
        return True
    if n % 10 == 0:
        return False
    
    reversed_half = 0
    while n > reversed_half:
        reversed_half = reversed_half * 10 + n % 10
        n //= 10
    
    return n == reversed_half or n == reversed_half // 10


def longest_palindromic_substring(s: str) -> str:
    """
    Find the longest palindromic substring using expand-around-center.
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    if not s:
        return ""
    
    def expand_around_center(left: int, right: int) -> str:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        # Odd length palindromes
        odd = expand_around_center(i, i)
        # Even length palindromes
        even = expand_around_center(i, i + 1)
        
        longer = odd if len(odd) > len(even) else even
        if len(longer) > len(longest):
            longest = longer
    
    return longest


# Tests
if __name__ == "__main__":
    test_strings = [
        "A man, a plan, a canal: Panama",
        "race a car",
        "Was it a car or a cat I saw?",
        "No lemon, no melon",
        "hello",
    ]
    
    for s in test_strings:
        r1 = is_palindrome(s)
        r2 = is_palindrome_two_pointers(s)
        print(f\'"{s[:40]}..." -> {r1} (two-pointer: {r2})\')
    
    print(f"\\nLongest palindromic substring of \\"babad\\": {longest_palindromic_substring(\\"babad\\")}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Palindrome checking with multiple approaches and longest palindromic substring",



            complexity="beginner",



            domain="algorithms",



            tags=["palindrome", "two_pointers", "strings", "dynamic_programming"],



        )







    def _gen_list_operations(self) -> CodeSample:



        code = '''from typing import List, Optional, Any
from collections import deque


class LinkedList:
    """Singly linked list implementation with common operations."""
    
    class Node:
        def __init__(self, data: Any, next_node: Optional['LinkedList.Node'] = None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self.head: Optional[self.Node] = None
        self._size: int = 0
    
    def append(self, data: Any) -> None:
        """Add element to the end of the list. O(n) time."""
        new_node = self.Node(data)
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self._size += 1
    
    def prepend(self, data: Any) -> None:
        """Add element to the beginning. O(1) time."""
        new_node = self.Node(data, self.head)
        self.head = new_node
        self._size += 1
    
    def delete(self, data: Any) -> bool:
        """Delete first occurrence of data. Returns True if found."""
        if self.head is None:
            return False
        
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        
        return False
    
    def find(self, data: Any) -> Optional[int]:
        """Find index of first occurrence. Returns None if not found."""
        current = self.head
        index = 0
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        return None
    
    def reverse(self) -> None:
        """Reverse the linked list in-place. O(n) time, O(1) space."""
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev
    
    def has_cycle(self) -> bool:
        """Detect if the linked list has a cycle using Floyd's algorithm."""
        slow = fast = self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
    
    def merge_sorted(self, other: 'LinkedList') -> 'LinkedList':
        """Merge two sorted linked lists into a new sorted list."""
        result = LinkedList()
        p1, p2 = self.head, other.head
        
        while p1 and p2:
            if p1.data <= p2.data:
                result.append(p1.data)
                p1 = p1.next
            else:
                result.append(p2.data)
                p2 = p2.next
        
        while p1:
            result.append(p1.data)
            p1 = p1.next
        while p2:
            result.append(p2.data)
            p2 = p2.next
        
        return result
    
    def to_list(self) -> List[Any]:
        """Convert to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result
    
    def __len__(self) -> int:
        return self._size
    
    def __repr__(self) -> str:
        return " -> ".join(str(x) for x in self.to_list()) or "Empty"


# Utility functions for Python lists
def remove_duplicates_preserve_order(lst: List[Any]) -> List[Any]:
    """Remove duplicates while preserving the original order."""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def flatten_nested_list(nested: List) -> List:
    """Flatten a deeply nested list structure."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result


def rotate_list(lst: List[Any], k: int) -> List[Any]:
    """Rotate list to the right by k positions."""
    if not lst:
        return lst
    k = k % len(lst)
    return lst[-k:] + lst[:-k]


# Demo
if __name__ == "__main__":
    ll = LinkedList()
    for val in [1, 2, 3, 4, 5]:
        ll.append(val)
    print(f"Original: {ll}")
    ll.reverse()
    print(f"Reversed: {ll}")
    print(f"Has cycle: {ll.has_cycle()}")
    
    print(f"\\nFlatten: {flatten_nested_list([1, [2, [3, 4], 5], [6]])}")
    print(f"Rotate: {rotate_list([1, 2, 3, 4, 5], 2)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Linked list implementation with cycle detection and list utilities",



            complexity="beginner",



            domain="algorithms",



            tags=["linked_list", "data_structures", "cycle_detection", "floyd_algorithm"],



        )







    def _gen_string_manipulation(self) -> CodeSample:



        code = '''from collections import Counter, defaultdict
from typing import List, Dict


def anagram_groups(words: List[str]) -> List[List[str]]:
    """
    Group a list of words into anagram groups.
    
    Args:
        words: List of words to group
        
    Returns:
        List of anagram groups
        
    Time Complexity: O(n * k) where n = number of words, k = max word length
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    
    for word in words:
        # Use sorted characters as the key
        key = "".join(sorted(word.lower()))
        groups[key].append(word)
    
    return [group for group in groups.values() if len(group) > 1]


def compress_string(s: str) -> str:
    """
    Compress a string using run-length encoding.
    
    Example: "aabcccccaaa" -> "a2b1c5a3"
    Returns original if compressed version is not shorter.
    """
    if not s:
        return s
    
    compressed = []
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            compressed.append(f"{s[i-1]}{count}")
            count = 1
    
    compressed.append(f"{s[-1]}{count}")
    result = "".join(compressed)
    
    return result if len(result) < len(s) else s


def find_longest_unique_substring(s: str) -> str:
    """
    Find the longest substring without repeating characters.
    
    Uses sliding window technique.
    Time Complexity: O(n)
    Space Complexity: O(min(n, m)) where m = character set size
    """
    if not s:
        return ""
    
    char_index: Dict[str, int] = {}
    max_length = 0
    start = 0
    best_start = 0
    
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        
        char_index[char] = end
        current_length = end - start + 1
        
        if current_length > max_length:
            max_length = current_length
            best_start = start
    
    return s[best_start:best_start + max_length]


def word_frequency(text: str, top_n: int = 10) -> List[tuple]:
    """
    Count word frequencies in text and return top N most common words.
    
    Args:
        text: Input text
        top_n: Number of top words to return
        
    Returns:
        List of (word, count) tuples
    """
    # Simple tokenization
    words = text.lower().split()
    # Remove punctuation
    words = ["".join(c for c in w if c.isalnum()) for w in words]
    words = [w for w in words if w]
    
    counter = Counter(words)
    return counter.most_common(top_n)


def caesar_cipher(text: str, shift: int, decrypt: bool = False) -> str:
    """
    Encrypt or decrypt text using Caesar cipher.
    
    Args:
        text: Input text
        shift: Number of positions to shift
        decrypt: If True, shift backwards
    """
    if decrypt:
        shift = -shift
    
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)
    
    return "".join(result)


if __name__ == "__main__":
    # Anagram groups
    words = ["eat", "tea", "tan", "ate", "nat", "bat", "tab"]
    print(f"Anagram groups: {anagram_groups(words)}")
    
    # String compression
    print(f"Compress: {compress_string('aabcccccaaa')}")
    
    # Longest unique substring
    print(f"Longest unique: {find_longest_unique_substring('abcabcbb')}")
    
    # Caesar cipher
    encrypted = caesar_cipher("Hello World", 3)
    decrypted = caesar_cipher(encrypted, 3, decrypt=True)
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")'''







        return CodeSample(



            language="python",



            code=code,



            description="String manipulation algorithms: anagrams, compression, sliding window",



            complexity="beginner",



            domain="algorithms",



            tags=["strings", "anagrams", "compression", "sliding_window", "cipher"],



        )







    def _gen_matrix_operations(self) -> CodeSample:



        code = '''from typing import List, Tuple


def matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices.
    
    Args:
        a: Matrix of shape (m, n)
        b: Matrix of shape (n, p)
        
    Returns:
        Result matrix of shape (m, p)
        
    Raises:
        ValueError: If matrices cannot be multiplied
    """
    m, n = len(a), len(a[0])
    n2, p = len(b), len(b[0])
    
    if n != n2:
        raise ValueError(f"Incompatible dimensions: ({m},{n}) × ({n2},{p})")
    
    result = [[0.0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            total = 0.0
            for k in range(n):
                total += a[i][k] * b[k][j]
            result[i][j] = total
    
    return result


def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """Transpose a matrix (swap rows and columns)."""
    if not matrix:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def spiral_order(matrix: List[List[int]]) -> List[int]:
    """
    Return elements of matrix in spiral order.
    
    Example: [[1,2,3],[4,5,6],[7,8,9]] -> [1,2,3,6,9,8,7,4,5]
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        # Traverse left
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        # Traverse up
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return result


def rotate_matrix_90(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate a square matrix 90 degrees clockwise in-place.
    
    Strategy: Transpose then reverse each row.
    """
    n = len(matrix)
    
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for row in matrix:
        row.reverse()
    
    return matrix


def search_sorted_matrix(matrix: List[List[int]], target: int) -> Tuple[int, int]:
    """
    Search for target in a row-wise and column-wise sorted matrix.
    
    Time Complexity: O(m + n)
    """
    if not matrix or not matrix[0]:
        return (-1, -1)
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return (row, col)
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return (-1, -1)


if __name__ == "__main__":
    m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"Spiral order: {spiral_order(m)}")
    print(f"Search 5: {search_sorted_matrix(m, 5)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Matrix operations: multiplication, transpose, spiral traversal, rotation",



            complexity="beginner",



            domain="algorithms",



            tags=["matrix", "2d_arrays", "spiral", "rotation", "search"],



        )







                                                







    def _intermediate_algorithms(self) -> CodeSample:



        variants = [



            self._gen_merge_sort,



            self._gen_dijkstra,



            self._gen_tree_traversals,



            self._gen_dynamic_programming,



            self._gen_hash_map_design,



        ]



        return random.choice(variants)()







    def _gen_merge_sort(self) -> CodeSample:



        code = '''from typing import List, TypeVar, Callable
import heapq

T = TypeVar('T')


def merge_sort(arr: List[T], key: Callable[[T], any] = None) -> List[T]:
    """
    Stable merge sort implementation.
    
    Merge sort is a divide-and-conquer algorithm that divides the input
    into halves, recursively sorts them, then merges the sorted halves.
    
    Time Complexity: O(n log n) in all cases
    Space Complexity: O(n)
    Stability: Yes (equal elements maintain relative order)
    
    Args:
        arr: List to sort
        key: Optional key function for comparison
        
    Returns:
        New sorted list
    """
    if key is None:
        key = lambda x: x
    
    if len(arr) <= 1:
        return arr[:]
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)
    
    return _merge(left, right, key)


def _merge(left: List[T], right: List[T], key: Callable) -> List[T]:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort_inplace(arr: List[T], low: int = 0, high: int = None) -> None:
    """
    In-place merge sort variant (still uses O(n) auxiliary space for merging).
    Modifies the array directly.
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        mid = (low + high) // 2
        merge_sort_inplace(arr, low, mid)
        merge_sort_inplace(arr, mid + 1, high)
        _merge_inplace(arr, low, mid, high)


def _merge_inplace(arr: List, low: int, mid: int, high: int) -> None:
    """Merge two sorted subarrays arr[low:mid+1] and arr[mid+1:high+1]."""
    left = arr[low:mid + 1]
    right = arr[mid + 1:high + 1]
    
    i = j = 0
    k = low
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
    
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1


def count_inversions(arr: List[int]) -> int:
    """
    Count the number of inversions in an array using merge sort.
    
    An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
    
    Time Complexity: O(n log n)
    """
    def sort_and_count(arr: List[int]) -> tuple:
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, left_inv = sort_and_count(arr[:mid])
        right, right_inv = sort_and_count(arr[mid:])
        merged, split_inv = _merge_and_count(left, right)
        
        return merged, left_inv + right_inv + split_inv
    
    def _merge_and_count(left: List[int], right: List[int]) -> tuple:
        merged = []
        inversions = 0
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inversions += len(left) - i  # All remaining in left are inversions
                j += 1
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inversions
    
    _, count = sort_and_count(arr)
    return count


def k_way_merge(lists: List[List[int]]) -> List[int]:
    """
    Merge k sorted lists efficiently using a min-heap.
    
    Time Complexity: O(N log k) where N = total elements, k = number of lists
    """
    heap = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result


if __name__ == "__main__":
    import random
    
    # Test merge sort
    test = [random.randint(0, 100) for _ in range(20)]
    print(f"Original: {test}")
    print(f"Sorted:   {merge_sort(test)}")
    
    # Count inversions
    arr = [2, 4, 1, 3, 5]
    print(f"\\nInversions in {arr}: {count_inversions(arr)}")
    
    # K-way merge
    lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    print(f"\\nK-way merge: {k_way_merge(lists)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Merge sort with inversion counting and k-way merge using heap",



            complexity="intermediate",



            domain="algorithms",



            tags=["merge_sort", "divide_and_conquer", "inversions", "heap", "k_way_merge"],



        )







    def _gen_dijkstra(self) -> CodeSample:



        code = '''import heapq
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Graph:
    """Weighted directed graph with adjacency list representation."""
    
    adjacency: Dict[str, List[Tuple[str, float]]] = field(default_factory=lambda: defaultdict(list))
    
    def add_edge(self, source: str, target: str, weight: float) -> None:
        """Add a directed edge with given weight."""
        self.adjacency[source].append((target, weight))
        # Ensure target exists in adjacency list
        if target not in self.adjacency:
            self.adjacency[target] = []
    
    def add_undirected_edge(self, u: str, v: str, weight: float) -> None:
        """Add an undirected edge (two directed edges)."""
        self.add_edge(u, v, weight)
        self.add_edge(v, u, weight)
    
    @property
    def vertices(self) -> Set[str]:
        return set(self.adjacency.keys())


def dijkstra(
    graph: Graph,
    source: str,
    target: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Dijkstra's shortest path algorithm.
    
    Finds the shortest path from source to all other vertices in a weighted
    graph with non-negative edge weights.
    
    Args:
        graph: Weighted graph
        source: Starting vertex
        target: Optional target vertex (early termination)
        
    Returns:
        Tuple of (distances dict, predecessors dict)
        
    Time Complexity: O((V + E) log V) with binary heap
    Space Complexity: O(V)
    
    Raises:
        ValueError: If graph has negative edge weights
    """
    # Validate no negative weights
    for vertex in graph.adjacency:
        for neighbor, weight in graph.adjacency[vertex]:
            if weight < 0:
                raise ValueError(
                    f"Dijkstra requires non-negative weights. "
                    f"Found negative weight {weight} on edge {vertex}->{neighbor}"
                )
    
    # Initialize distances and predecessors
    distances: Dict[str, float] = {v: float('inf') for v in graph.vertices}
    predecessors: Dict[str, Optional[str]] = {v: None for v in graph.vertices}
    distances[source] = 0.0
    
    # Priority queue: (distance, vertex)
    pq = [(0.0, source)]
    visited: Set[str] = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        # Skip if already visited (lazy deletion)
        if current in visited:
            continue
        visited.add(current)
        
        # Early termination if target found
        if target is not None and current == target:
            break
        
        # Skip if we found a better path already
        if current_dist > distances[current]:
            continue
        
        # Relax neighbors
        for neighbor, weight in graph.adjacency[current]:
            if neighbor in visited:
                continue
            
            new_dist = distances[current] + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                predecessors[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distances, predecessors


def reconstruct_path(
    predecessors: Dict[str, Optional[str]],
    source: str,
    target: str,
) -> Optional[List[str]]:
    """
    Reconstruct the shortest path from predecessors map.
    
    Returns:
        List of vertices in the path, or None if no path exists
    """
    if predecessors.get(target) is None and target != source:
        return None
    
    path = []
    current = target
    
    while current is not None:
        path.append(current)
        current = predecessors[current]
    
    path.reverse()
    return path if path[0] == source else None


def dijkstra_all_paths(graph: Graph, source: str) -> Dict[str, List[str]]:
    """Find shortest paths from source to all reachable vertices."""
    distances, predecessors = dijkstra(graph, source)
    
    paths = {}
    for target in graph.vertices:
        if distances[target] < float('inf'):
            path = reconstruct_path(predecessors, source, target)
            if path:
                paths[target] = path
    
    return paths


# Bellman-Ford for comparison (handles negative weights)
def bellman_ford(
    graph: Graph,
    source: str,
) -> Tuple[Dict[str, float], Dict[str, Optional[str]], bool]:
    """
    Bellman-Ford algorithm - handles negative weights.
    
    Returns:
        (distances, predecessors, has_negative_cycle)
        
    Time Complexity: O(V × E)
    """
    distances = {v: float('inf') for v in graph.vertices}
    predecessors = {v: None for v in graph.vertices}
    distances[source] = 0.0
    
    # Collect all edges
    edges = []
    for u in graph.adjacency:
        for v, w in graph.adjacency[u]:
            edges.append((u, v, w))
    
    # Relax edges V-1 times
    n = len(graph.vertices)
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                predecessors[v] = u
                updated = True
        if not updated:
            break
    
    # Check for negative cycles
    has_negative_cycle = False
    for u, v, w in edges:
        if distances[u] + w < distances[v]:
            has_negative_cycle = True
            break
    
    return distances, predecessors, has_negative_cycle


if __name__ == "__main__":
    # Build a sample graph
    g = Graph()
    edges = [
        ("A", "B", 4), ("A", "C", 2), ("B", "D", 3), ("B", "C", 1),
        ("C", "B", 1), ("C", "D", 5), ("C", "E", 7), ("D", "E", 2),
        ("E", "A", 6),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    # Find shortest paths from A
    distances, predecessors = dijkstra(g, "A")
    
    print("Shortest distances from A:")
    for vertex in sorted(distances):
        dist = distances[vertex]
        path = reconstruct_path(predecessors, "A", vertex)
        print(f"  A -> {vertex}: {dist} (path: {' -> '.join(path) if path else 'N/A'})")'''







        return CodeSample(



            language="python",



            code=code,



            description="Dijkstra's algorithm with path reconstruction and Bellman-Ford comparison",



            complexity="intermediate",



            domain="algorithms",



            tags=["dijkstra", "shortest_path", "graph", "bellman_ford", "priority_queue"],



        )







    def _gen_tree_traversals(self) -> CodeSample:



        code = '''from typing import Optional, List, Generator
from collections import deque
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Binary tree node."""
    val: int
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


class BinaryTree:
    """Binary tree with comprehensive traversal methods."""
    
    def __init__(self, root: Optional[TreeNode] = None):
        self.root = root
    
    @classmethod
    def from_list(cls, values: List[Optional[int]]) -> 'BinaryTree':
        """Build tree from level-order list representation. None = missing node."""
        if not values:
            return cls()
        
        root = TreeNode(values[0])
        queue = deque([root])
        i = 1
        
        while queue and i < len(values):
            node = queue.popleft()
            
            if i < len(values) and values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
        
        return cls(root)
    
    # ========== DEPTH-FIRST TRAVERSALS ==========
    
    def inorder(self) -> List[int]:
        """Left -> Root -> Right (gives sorted order for BST)."""
        result = []
        self._inorder_helper(self.root, result)
        return result
    
    def _inorder_helper(self, node: Optional[TreeNode], result: List[int]):
        if node:
            self._inorder_helper(node.left, result)
            result.append(node.val)
            self._inorder_helper(node.right, result)
    
    def inorder_iterative(self) -> List[int]:
        """Iterative inorder traversal using a stack."""
        result = []
        stack = []
        current = self.root
        
        while current or stack:
            while current:
                stack.append(current)
                current = current.left
            
            current = stack.pop()
            result.append(current.val)
            current = current.right
        
        return result
    
    def preorder(self) -> List[int]:
        """Root -> Left -> Right."""
        result = []
        self._preorder_helper(self.root, result)
        return result
    
    def _preorder_helper(self, node: Optional[TreeNode], result: List[int]):
        if node:
            result.append(node.val)
            self._preorder_helper(node.left, result)
            self._preorder_helper(node.right, result)
    
    def postorder(self) -> List[int]:
        """Left -> Right -> Root."""
        result = []
        self._postorder_helper(self.root, result)
        return result
    
    def _postorder_helper(self, node: Optional[TreeNode], result: List[int]):
        if node:
            self._postorder_helper(node.left, result)
            self._postorder_helper(node.right, result)
            result.append(node.val)
    
    # ========== BREADTH-FIRST TRAVERSAL ==========
    
    def level_order(self) -> List[List[int]]:
        """Level-order traversal, returning nodes grouped by level."""
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            level_size = len(queue)
            level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level)
        
        return result
    
    def zigzag_level_order(self) -> List[List[int]]:
        """Zigzag level-order traversal (alternating directions)."""
        if not self.root:
            return []
        
        result = []
        queue = deque([self.root])
        left_to_right = True
        
        while queue:
            level_size = len(queue)
            level = deque()
            
            for _ in range(level_size):
                node = queue.popleft()
                
                if left_to_right:
                    level.append(node.val)
                else:
                    level.appendleft(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(list(level))
            left_to_right = not left_to_right
        
        return result
    
    # ========== TREE PROPERTIES ==========
    
    def height(self) -> int:
        """Calculate the height of the tree."""
        return self._height(self.root)
    
    def _height(self, node: Optional[TreeNode]) -> int:
        if not node:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def is_balanced(self) -> bool:
        """Check if the tree is height-balanced."""
        def check(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = check(node.left)
            right = check(node.right)
            if left == -1 or right == -1 or abs(left - right) > 1:
                return -1
            return 1 + max(left, right)
        
        return check(self.root) != -1
    
    def is_bst(self) -> bool:
        """Check if the tree is a valid binary search tree."""
        def validate(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
            if not (low < node.val < high):
                return False
            return (validate(node.left, low, node.val) and
                    validate(node.right, node.val, high))
        
        return validate(self.root, float('-inf'), float('inf'))
    
    def diameter(self) -> int:
        """
        Calculate the diameter of the tree.
        The diameter is the longest path between any two nodes.
        """
        max_diameter = 0
        
        def depth(node: Optional[TreeNode]) -> int:
            nonlocal max_diameter
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            max_diameter = max(max_diameter, left + right)
            return 1 + max(left, right)
        
        depth(self.root)
        return max_diameter


if __name__ == "__main__":
    # Build tree: [3, 9, 20, None, None, 15, 7]
    tree = BinaryTree.from_list([3, 9, 20, None, None, 15, 7])
    
    print(f"Inorder:        {tree.inorder()}")
    print(f"Inorder (iter): {tree.inorder_iterative()}")
    print(f"Preorder:       {tree.preorder()}")
    print(f"Postorder:      {tree.postorder()}")
    print(f"Level order:    {tree.level_order()}")
    print(f"Zigzag:         {tree.zigzag_level_order()}")
    print(f"Height:         {tree.height()}")
    print(f"Balanced:       {tree.is_balanced()}")
    print(f"Is BST:         {tree.is_bst()}")
    print(f"Diameter:       {tree.diameter()}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Binary tree with all traversal methods and property checks",



            complexity="intermediate",



            domain="algorithms",



            tags=["binary_tree", "traversal", "dfs", "bfs", "bst", "data_structures"],



        )







    def _gen_dynamic_programming(self) -> CodeSample:



        code = '''from typing import List, Tuple, Optional
from functools import lru_cache


# ============================================================
# 1. KNAPSACK PROBLEM (0/1)
# ============================================================

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
    """
    0/1 Knapsack Problem - Dynamic Programming Solution.
    
    Given items with weights and values, find the maximum value that can
    be carried in a knapsack of given capacity.
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        Tuple of (maximum value, list of selected item indices)
        
    Time Complexity: O(n × W) where n = items, W = capacity
    Space Complexity: O(n × W)
    """
    n = len(weights)
    
    # DP table: dp[i][w] = max value using items 0..i-1 with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Take item i-1 if it fits
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)
            w -= weights[i-1]
    
    return dp[n][capacity], selected[::-1]


# ============================================================
# 2. LONGEST COMMON SUBSEQUENCE (LCS)
# ============================================================

def longest_common_subsequence(s1: str, s2: str) -> Tuple[int, str]:
    """
    Find the longest common subsequence of two strings.
    
    Time Complexity: O(m × n)
    Space Complexity: O(m × n)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Backtrack to find the actual subsequence
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], "".join(reversed(lcs))


# ============================================================
# 3. EDIT DISTANCE (LEVENSHTEIN)
# ============================================================

def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate the minimum number of operations to convert s1 to s2.
    Operations: insert, delete, replace (each costs 1).
    
    Time Complexity: O(m × n)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1],  # Replace
                )
    
    return dp[m][n]


# ============================================================
# 4. COIN CHANGE
# ============================================================

def coin_change(coins: List[int], amount: int) -> int:
    """
    Find minimum number of coins needed to make the given amount.
    
    Args:
        coins: Available coin denominations
        amount: Target amount
        
    Returns:
        Minimum number of coins, or -1 if impossible
        
    Time Complexity: O(amount × len(coins))
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
    
    return dp[amount] if dp[amount] != float('inf') else -1


# ============================================================
# 5. LONGEST INCREASING SUBSEQUENCE
# ============================================================

def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Find the length of the longest strictly increasing subsequence.
    
    O(n log n) approach using binary search with patience sorting.
    """
    import bisect
    
    if not nums:
        return 0
    
    # tails[i] = smallest tail of all increasing subsequences of length i+1
    tails = []
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


# ============================================================
# 6. MATRIX CHAIN MULTIPLICATION
# ============================================================

def matrix_chain_order(dimensions: List[int]) -> Tuple[int, List[List[int]]]:
    """
    Find the optimal order for multiplying a chain of matrices.
    
    Args:
        dimensions: List where matrix i has dimensions dimensions[i] × dimensions[i+1]
        
    Returns:
        (minimum multiplications, split table for reconstruction)
    """
    n = len(dimensions) - 1  # Number of matrices
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):  # Chain length
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    
    return dp[0][n-1], split


if __name__ == "__main__":
    # Knapsack
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    max_val, items = knapsack_01(weights, values, 8)
    print(f"Knapsack: max value = {max_val}, items = {items}")
    
    # LCS
    length, lcs = longest_common_subsequence("ABCBDAB", "BDCAB")
    print(f"LCS: length = {length}, subsequence = '{lcs}'")
    
    # Edit Distance
    print(f"Edit distance 'kitten' -> 'sitting': {edit_distance('kitten', 'sitting')}")
    
    # Coin Change
    print(f"Coin change (amount=11): {coin_change([1, 5, 10, 25], 11)}")
    
    # LIS
    print(f"LIS of [10,9,2,5,3,7,101,18]: {longest_increasing_subsequence([10,9,2,5,3,7,101,18])}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Dynamic programming: knapsack, LCS, edit distance, coin change, LIS, matrix chain",



            complexity="intermediate",



            domain="algorithms",



            tags=["dynamic_programming", "knapsack", "lcs", "edit_distance", "coin_change", "lis"],



        )







    def _gen_hash_map_design(self) -> CodeSample:



        code = '''from typing import Optional, Any, List, Tuple, Iterator
from dataclasses import dataclass
import hashlib


@dataclass
class Entry:
    """Hash map entry with chaining support."""
    key: Any
    value: Any
    next_entry: Optional['Entry'] = None


class HashMap:
    """
    Custom hash map implementation with separate chaining.
    
    Features:
    - Separate chaining for collision resolution
    - Dynamic resizing (load factor based)
    - O(1) average case for get/put/delete
    - Supports any hashable key type
    """
    
    INITIAL_CAPACITY = 16
    LOAD_FACTOR_THRESHOLD = 0.75
    RESIZE_FACTOR = 2
    
    def __init__(self, capacity: int = INITIAL_CAPACITY):
        self._capacity = capacity
        self._size = 0
        self._buckets: List[Optional[Entry]] = [None] * capacity
    
    def _hash(self, key: Any) -> int:
        """Compute bucket index for a key."""
        return hash(key) % self._capacity
    
    def put(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair.
        
        Time Complexity: O(1) average, O(n) worst case
        """
        # Check if resize is needed
        if (self._size + 1) / self._capacity > self.LOAD_FACTOR_THRESHOLD:
            self._resize()
        
        idx = self._hash(key)
        entry = self._buckets[idx]
        
        # Update existing key
        while entry:
            if entry.key == key:
                entry.value = value
                return
            entry = entry.next_entry
        
        # Insert new entry at head of chain
        new_entry = Entry(key, value, self._buckets[idx])
        self._buckets[idx] = new_entry
        self._size += 1
    
    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get value for key, or default if not found.
        
        Time Complexity: O(1) average, O(n) worst case
        """
        idx = self._hash(key)
        entry = self._buckets[idx]
        
        while entry:
            if entry.key == key:
                return entry.value
            entry = entry.next_entry
        
        return default
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair. Returns True if the key was found.
        
        Time Complexity: O(1) average, O(n) worst case
        """
        idx = self._hash(key)
        entry = self._buckets[idx]
        prev = None
        
        while entry:
            if entry.key == key:
                if prev:
                    prev.next_entry = entry.next_entry
                else:
                    self._buckets[idx] = entry.next_entry
                self._size -= 1
                return True
            prev = entry
            entry = entry.next_entry
        
        return False
    
    def contains(self, key: Any) -> bool:
        """Check if key exists in the map."""
        idx = self._hash(key)
        entry = self._buckets[idx]
        while entry:
            if entry.key == key:
                return True
            entry = entry.next_entry
        return False
    
    def _resize(self) -> None:
        """Double the capacity and rehash all entries."""
        old_buckets = self._buckets
        self._capacity *= self.RESIZE_FACTOR
        self._buckets = [None] * self._capacity
        self._size = 0
        
        for entry in old_buckets:
            while entry:
                self.put(entry.key, entry.value)
                entry = entry.next_entry
    
    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, key: Any) -> Any:
        value = self.get(key)
        if value is None and not self.contains(key):
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: Any, value: Any) -> None:
        self.put(key, value)
    
    def __delitem__(self, key: Any) -> None:
        if not self.delete(key):
            raise KeyError(key)
    
    def __iter__(self) -> Iterator[Any]:
        for entry in self._buckets:
            while entry:
                yield entry.key
                entry = entry.next_entry
    
    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Iterate over (key, value) pairs."""
        for entry in self._buckets:
            while entry:
                yield (entry.key, entry.value)
                entry = entry.next_entry
    
    def __repr__(self) -> str:
        pairs = [f"{k}: {v}" for k, v in self.items()]
        return "HashMap({" + ", ".join(pairs) + "})"


class LRUCache:
    """
    Least Recently Used Cache using HashMap + Doubly Linked List.
    
    O(1) get and put operations.
    """
    
    class Node:
        def __init__(self, key: Any = 0, value: Any = 0):
            self.key = key
            self.value = value
            self.prev: Optional['LRUCache.Node'] = None
            self.next: Optional['LRUCache.Node'] = None
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict = {}
        
        # Sentinel nodes
        self.head = self.Node()  # Most recently used
        self.tail = self.Node()  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_front(node)
        return node.value
    
    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                # Evict least recently used
                lru = self.tail.prev
                self._remove(lru)
                del self.cache[lru.key]
            
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_to_front(node)
    
    def _add_to_front(self, node: 'LRUCache.Node') -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove(self, node: 'LRUCache.Node') -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_front(self, node: 'LRUCache.Node') -> None:
        self._remove(node)
        self._add_to_front(node)


if __name__ == "__main__":
    # Test HashMap
    hm = HashMap()
    hm["name"] = "Alice"
    hm["age"] = 30
    hm["city"] = "Auckland"
    
    print(f"HashMap: {hm}")
    print(f"name = {hm['name']}")
    
    # Stress test
    for i in range(1000):
        hm[f"key_{i}"] = i * i
    print(f"After 1000 inserts, size = {len(hm)}")
    
    # Test LRU Cache
    cache = LRUCache(3)
    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    print(f"\\nLRU get(1): {cache.get(1)}")
    cache.put(4, "d")  # Evicts key 2
    print(f"LRU get(2): {cache.get(2)} (evicted)")'''







        return CodeSample(



            language="python",



            code=code,



            description="Custom HashMap with chaining and LRU Cache implementation",



            complexity="intermediate",



            domain="algorithms",



            tags=["hash_map", "lru_cache", "data_structures", "chaining", "linked_list"],



        )







    def _intermediate_data(self) -> CodeSample:



        code = '''import csv
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics


@dataclass
class DataPoint:
    """A single data point with features and optional label."""
    features: Dict[str, Any]
    label: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataFrame:
    """
    Lightweight DataFrame implementation for data manipulation.
    
    Supports basic operations like filtering, grouping, aggregation,
    and I/O without external dependencies.
    """
    
    def __init__(self, data: Optional[List[Dict[str, Any]]] = None):
        self._data = data or []
        self._columns: List[str] = []
        if self._data:
            self._columns = list(self._data[0].keys())
    
    @classmethod
    def from_csv(cls, filepath: str, delimiter: str = ',') -> 'DataFrame':
        """Load data from a CSV file."""
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Try to convert numeric values
                parsed = {}
                for k, v in row.items():
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                data.append(parsed)
        return cls(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DataFrame':
        """Load data from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(data)
    
    @property
    def shape(self) -> tuple:
        return (len(self._data), len(self._columns))
    
    @property
    def columns(self) -> List[str]:
        return self._columns[:]
    
    def head(self, n: int = 5) -> 'DataFrame':
        return DataFrame(self._data[:n])
    
    def tail(self, n: int = 5) -> 'DataFrame':
        return DataFrame(self._data[-n:])
    
    def select(self, columns: List[str]) -> 'DataFrame':
        """Select specific columns."""
        data = [{k: row[k] for k in columns if k in row} for row in self._data]
        return DataFrame(data)
    
    def filter(self, condition: Callable[[Dict], bool]) -> 'DataFrame':
        """Filter rows based on a condition function."""
        data = [row for row in self._data if condition(row)]
        return DataFrame(data)
    
    def sort_by(self, column: str, ascending: bool = True) -> 'DataFrame':
        """Sort by a column."""
        data = sorted(self._data, key=lambda x: x.get(column, 0), reverse=not ascending)
        return DataFrame(data)
    
    def group_by(self, column: str) -> Dict[Any, 'DataFrame']:
        """Group rows by a column value."""
        groups = defaultdict(list)
        for row in self._data:
            key = row.get(column)
            groups[key].append(row)
        return {k: DataFrame(v) for k, v in groups.items()}
    
    def aggregate(self, column: str, func: str = 'mean') -> Any:
        """Aggregate a numeric column."""
        values = [row[column] for row in self._data if column in row and isinstance(row[column], (int, float))]
        if not values:
            return None
        
        if func == 'mean':
            return statistics.mean(values)
        elif func == 'median':
            return statistics.median(values)
        elif func == 'sum':
            return sum(values)
        elif func == 'min':
            return min(values)
        elif func == 'max':
            return max(values)
        elif func == 'std':
            return statistics.stdev(values) if len(values) > 1 else 0
        elif func == 'count':
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation function: {func}")
    
    def describe(self) -> Dict[str, Dict[str, float]]:
        """Generate descriptive statistics for numeric columns."""
        result = {}
        for col in self._columns:
            values = [row[col] for row in self._data 
                     if col in row and isinstance(row[col], (int, float))]
            if values:
                result[col] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                }
        return result
    
    def add_column(self, name: str, values: List[Any]) -> 'DataFrame':
        """Add a new column."""
        for i, row in enumerate(self._data):
            row[name] = values[i] if i < len(values) else None
        if name not in self._columns:
            self._columns.append(name)
        return self
    
    def apply(self, column: str, func: Callable) -> 'DataFrame':
        """Apply a function to a column."""
        for row in self._data:
            if column in row:
                row[column] = func(row[column])
        return self
    
    def to_csv(self, filepath: str) -> None:
        """Save to CSV file."""
        if not self._data:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self._data)
    
    def to_json(self, filepath: str) -> None:
        """Save to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self._data, f, indent=2)
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __repr__(self) -> str:
        return f"DataFrame(shape={self.shape}, columns={self._columns})"


if __name__ == "__main__":
    # Create sample data
    data = [
        {"name": "Alice", "age": 30, "salary": 80000, "dept": "Engineering"},
        {"name": "Bob", "age": 25, "salary": 65000, "dept": "Marketing"},
        {"name": "Charlie", "age": 35, "salary": 95000, "dept": "Engineering"},
        {"name": "Diana", "age": 28, "salary": 72000, "dept": "Marketing"},
        {"name": "Eve", "age": 32, "salary": 88000, "dept": "Engineering"},
    ]
    
    df = DataFrame(data)
    print(f"Shape: {df.shape}")
    print(f"\\nDescribe: {json.dumps(df.describe(), indent=2)}")
    
    # Filter and aggregate
    engineers = df.filter(lambda r: r["dept"] == "Engineering")
    print(f"\\nEngineers avg salary: {engineers.aggregate('salary', 'mean')}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Custom DataFrame implementation with filtering, grouping, and aggregation",



            complexity="intermediate",



            domain="data",



            tags=["dataframe", "data_manipulation", "csv", "json", "aggregation"],



        )







    def _intermediate_web(self) -> CodeSample:



        code = '''"""
Lightweight Web Framework - A minimal Flask-like framework from scratch.

Demonstrates routing, middleware, request/response handling, and template rendering.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
import re
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import html


# ============================================================
# REQUEST & RESPONSE OBJECTS
# ============================================================

@dataclass
class Request:
    """HTTP Request object."""
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, List[str]]
    body: Optional[str] = None
    json_data: Optional[Dict] = None
    path_params: Dict[str, str] = field(default_factory=dict)
    
    @property
    def content_type(self) -> str:
        return self.headers.get('Content-Type', '')


@dataclass
class Response:
    """HTTP Response object."""
    status_code: int = 200
    body: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    content_type: str = "text/html"
    
    def json(self, data: Any, status_code: int = 200) -> 'Response':
        self.body = json.dumps(data, indent=2)
        self.content_type = "application/json"
        self.status_code = status_code
        return self
    
    def html(self, content: str, status_code: int = 200) -> 'Response':
        self.body = content
        self.content_type = "text/html"
        self.status_code = status_code
        return self
    
    def text(self, content: str, status_code: int = 200) -> 'Response':
        self.body = content
        self.content_type = "text/plain"
        self.status_code = status_code
        return self
    
    def redirect(self, url: str, status_code: int = 302) -> 'Response':
        self.headers['Location'] = url
        self.status_code = status_code
        return self


# ============================================================
# ROUTER
# ============================================================

class Router:
    """URL routing with path parameter support."""
    
    def __init__(self):
        self.routes: List[Tuple[str, str, Callable]] = []  # (method, pattern, handler)
    
    def add_route(self, method: str, path: str, handler: Callable) -> None:
        """Register a route handler."""
        # Convert path params like /users/<id> to regex
        pattern = re.sub(r'<(\\w+)>', r'(?P<\\1>[^/]+)', path)
        pattern = f'^{pattern}$'
        self.routes.append((method.upper(), pattern, handler))
    
    def match(self, method: str, path: str) -> Optional[Tuple[Callable, Dict[str, str]]]:
        """Find matching route for method and path."""
        for route_method, pattern, handler in self.routes:
            if route_method != method.upper():
                continue
            match = re.match(pattern, path)
            if match:
                return handler, match.groupdict()
        return None


# ============================================================
# MIDDLEWARE SYSTEM
# ============================================================

class Middleware:
    """Base middleware class."""
    
    def process_request(self, request: Request) -> Optional[Response]:
        """Process request before handler. Return Response to short-circuit."""
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """Process response after handler."""
        return response


class CORSMiddleware(Middleware):
    """Cross-Origin Resource Sharing middleware."""
    
    def __init__(self, allowed_origins: List[str] = None):
        self.allowed_origins = allowed_origins or ["*"]
    
    def process_response(self, request: Request, response: Response) -> Response:
        origin = request.headers.get('Origin', '*')
        if '*' in self.allowed_origins or origin in self.allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response


class AuthMiddleware(Middleware):
    """Simple token-based authentication middleware."""
    
    def __init__(self, protected_paths: List[str] = None, secret: str = "secret"):
        self.protected_paths = protected_paths or []
        self.secret = secret
    
    def process_request(self, request: Request) -> Optional[Response]:
        for path in self.protected_paths:
            if request.path.startswith(path):
                token = request.headers.get('Authorization', '')
                if not token.startswith('Bearer ') or token[7:] != self.secret:
                    return Response(
                        status_code=401,
                        body=json.dumps({"error": "Unauthorized"}),
                        content_type="application/json",
                    )
        return None


# ============================================================
# APPLICATION
# ============================================================

class WebApp:
    """Minimal web application framework."""
    
    def __init__(self):
        self.router = Router()
        self.middlewares: List[Middleware] = []
        self.error_handlers: Dict[int, Callable] = {}
    
    def route(self, path: str, methods: List[str] = None):
        """Decorator to register a route."""
        if methods is None:
            methods = ["GET"]
        
        def decorator(handler: Callable):
            for method in methods:
                self.router.add_route(method, path, handler)
            return handler
        
        return decorator
    
    def add_middleware(self, middleware: Middleware) -> None:
        self.middlewares.append(middleware)
    
    def handle_request(self, request: Request) -> Response:
        """Process a request through middleware and routing."""
        # Run request middleware
        for mw in self.middlewares:
            response = mw.process_request(request)
            if response:
                return response
        
        # Route matching
        result = self.router.match(request.method, request.path)
        
        if result is None:
            return Response(
                status_code=404,
                body=json.dumps({"error": "Not Found"}),
                content_type="application/json",
            )
        
        handler, path_params = result
        request.path_params = path_params
        
        try:
            response = handler(request)
        except Exception as e:
            return Response(
                status_code=500,
                body=json.dumps({"error": str(e)}),
                content_type="application/json",
            )
        
        # Run response middleware
        for mw in self.middlewares:
            response = mw.process_response(request, response)
        
        return response
    
    def run(self, host: str = '0.0.0.0', port: int = 8000) -> None:
        """Start the development server."""
        app = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self): self._handle('GET')
            def do_POST(self): self._handle('POST')
            def do_PUT(self): self._handle('PUT')
            def do_DELETE(self): self._handle('DELETE')
            
            def _handle(self, method: str):
                parsed = urlparse(self.path)
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode() if content_length else None
                
                request = Request(
                    method=method,
                    path=parsed.path,
                    headers=dict(self.headers),
                    query_params=parse_qs(parsed.query),
                    body=body,
                    json_data=json.loads(body) if body and 'application/json' in self.headers.get('Content-Type', '') else None,
                )
                
                response = app.handle_request(request)
                
                self.send_response(response.status_code)
                self.send_header('Content-Type', response.content_type)
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.body.encode())
            
            def log_message(self, format, *args):
                print(f"[{self.log_date_time_string()}] {format % args}")
        
        server = HTTPServer((host, port), Handler)
        print(f"Server running at http://{host}:{port}")
        server.serve_forever()


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    app = WebApp()
    app.add_middleware(CORSMiddleware())
    
    # In-memory database
    users_db = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ]
    
    @app.route("/")
    def index(req: Request) -> Response:
        return Response().html("<h1>Coding Agent API</h1><p>Welcome!</p>")
    
    @app.route("/api/users", methods=["GET"])
    def list_users(req: Request) -> Response:
        return Response().json({"users": users_db})
    
    @app.route("/api/users/<id>", methods=["GET"])
    def get_user(req: Request) -> Response:
        user_id = int(req.path_params["id"])
        user = next((u for u in users_db if u["id"] == user_id), None)
        if user:
            return Response().json(user)
        return Response().json({"error": "User not found"}, 404)
    
    @app.route("/api/users", methods=["POST"])
    def create_user(req: Request) -> Response:
        data = req.json_data
        new_user = {"id": len(users_db) + 1, **data}
        users_db.append(new_user)
        return Response().json(new_user, 201)
    
    print("Routes registered:")
    for method, pattern, handler in app.router.routes:
        print(f"  {method:6s} {pattern}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Minimal web framework from scratch with routing, middleware, and REST API",



            complexity="intermediate",



            domain="web",



            tags=["web_framework", "routing", "middleware", "rest_api", "http"],



        )







    def _intermediate_testing(self) -> CodeSample:



        code = '''"""
Comprehensive Testing Patterns and Utilities for Python Projects.

Demonstrates unit testing, integration testing, mocking, fixtures,
and test-driven development patterns.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import time


# ============================================================
# CODE UNDER TEST: User Service
# ============================================================

class UserRepository:
    """Database layer for user operations."""
    
    def __init__(self):
        self._users: Dict[int, Dict] = {}
        self._next_id = 1
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        return self._users.get(user_id)
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        for user in self._users.values():
            if user["email"] == email:
                return user
        return None
    
    def save(self, user: Dict) -> Dict:
        if "id" not in user:
            user["id"] = self._next_id
            self._next_id += 1
        self._users[user["id"]] = user
        return user
    
    def delete(self, user_id: int) -> bool:
        return self._users.pop(user_id, None) is not None


class EmailService:
    """Email sending service (external dependency)."""
    
    def send(self, to: str, subject: str, body: str) -> bool:
        # In production, this would call an actual email API
        raise NotImplementedError("Use mock in tests")


class UserService:
    """Business logic for user management."""
    
    def __init__(self, repo: UserRepository, email_service: EmailService):
        self.repo = repo
        self.email_service = email_service
    
    def register(self, name: str, email: str, password: str) -> Dict:
        """Register a new user with validation."""
        # Validation
        if not name or len(name) < 2:
            raise ValueError("Name must be at least 2 characters")
        if not email or "@" not in email:
            raise ValueError("Invalid email address")
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        # Check uniqueness
        existing = self.repo.find_by_email(email)
        if existing:
            raise ValueError(f"Email {email} is already registered")
        
        # Create user (in production, hash the password!)
        user = self.repo.save({
            "name": name,
            "email": email,
            "password_hash": hash(password),  # Simplified for demo
            "created_at": datetime.now().isoformat(),
            "is_active": True,
        })
        
        # Send welcome email
        self.email_service.send(
            to=email,
            subject="Welcome!",
            body=f"Hello {name}, welcome to our platform!",
        )
        
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate a user by email and password."""
        user = self.repo.find_by_email(email)
        if not user or not user.get("is_active"):
            return None
        if user["password_hash"] != hash(password):
            return None
        return user
    
    def deactivate(self, user_id: int) -> bool:
        """Deactivate a user account."""
        user = self.repo.find_by_id(user_id)
        if not user:
            return False
        user["is_active"] = False
        self.repo.save(user)
        self.email_service.send(
            to=user["email"],
            subject="Account Deactivated",
            body="Your account has been deactivated.",
        )
        return True


# ============================================================
# TESTS
# ============================================================

class TestUserService(unittest.TestCase):
    """Unit tests for UserService with mocking."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.repo = UserRepository()
        self.email_service = Mock(spec=EmailService)
        self.email_service.send.return_value = True
        self.service = UserService(self.repo, self.email_service)
    
    def test_register_success(self):
        """Test successful user registration."""
        user = self.service.register("Alice", "alice@test.com", "password123")
        
        self.assertEqual(user["name"], "Alice")
        self.assertEqual(user["email"], "alice@test.com")
        self.assertTrue(user["is_active"])
        self.assertIn("id", user)
        
        # Verify email was sent
        self.email_service.send.assert_called_once_with(
            to="alice@test.com",
            subject="Welcome!",
            body="Hello Alice, welcome to our platform!",
        )
    
    def test_register_short_name(self):
        """Test registration fails with short name."""
        with self.assertRaises(ValueError) as ctx:
            self.service.register("A", "a@test.com", "password123")
        self.assertIn("2 characters", str(ctx.exception))
    
    def test_register_invalid_email(self):
        """Test registration fails with invalid email."""
        with self.assertRaises(ValueError) as ctx:
            self.service.register("Alice", "not-an-email", "password123")
        self.assertIn("Invalid email", str(ctx.exception))
    
    def test_register_short_password(self):
        """Test registration fails with short password."""
        with self.assertRaises(ValueError) as ctx:
            self.service.register("Alice", "alice@test.com", "short")
        self.assertIn("8 characters", str(ctx.exception))
    
    def test_register_duplicate_email(self):
        """Test registration fails with duplicate email."""
        self.service.register("Alice", "alice@test.com", "password123")
        
        with self.assertRaises(ValueError) as ctx:
            self.service.register("Bob", "alice@test.com", "password456")
        self.assertIn("already registered", str(ctx.exception))
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        self.service.register("Alice", "alice@test.com", "password123")
        
        user = self.service.authenticate("alice@test.com", "password123")
        self.assertIsNotNone(user)
        self.assertEqual(user["name"], "Alice")
    
    def test_authenticate_wrong_password(self):
        """Test authentication fails with wrong password."""
        self.service.register("Alice", "alice@test.com", "password123")
        
        user = self.service.authenticate("alice@test.com", "wrongpassword")
        self.assertIsNone(user)
    
    def test_authenticate_nonexistent_user(self):
        """Test authentication fails for nonexistent user."""
        user = self.service.authenticate("nobody@test.com", "password123")
        self.assertIsNone(user)
    
    def test_deactivate_user(self):
        """Test user deactivation."""
        user = self.service.register("Alice", "alice@test.com", "password123")
        
        result = self.service.deactivate(user["id"])
        self.assertTrue(result)
        
        # Verify user can't authenticate after deactivation
        auth = self.service.authenticate("alice@test.com", "password123")
        self.assertIsNone(auth)
    
    def test_deactivate_nonexistent_user(self):
        """Test deactivation of nonexistent user returns False."""
        result = self.service.deactivate(999)
        self.assertFalse(result)


class TestUserRepository(unittest.TestCase):
    """Unit tests for UserRepository."""
    
    def setUp(self):
        self.repo = UserRepository()
    
    def test_save_and_find(self):
        user = self.repo.save({"name": "Alice", "email": "alice@test.com"})
        found = self.repo.find_by_id(user["id"])
        self.assertEqual(found["name"], "Alice")
    
    def test_find_by_email(self):
        self.repo.save({"name": "Alice", "email": "alice@test.com"})
        found = self.repo.find_by_email("alice@test.com")
        self.assertIsNotNone(found)
    
    def test_delete(self):
        user = self.repo.save({"name": "Alice", "email": "alice@test.com"})
        self.assertTrue(self.repo.delete(user["id"]))
        self.assertIsNone(self.repo.find_by_id(user["id"]))
    
    def test_auto_increment_id(self):
        u1 = self.repo.save({"name": "A"})
        u2 = self.repo.save({"name": "B"})
        self.assertEqual(u2["id"], u1["id"] + 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)'''







        return CodeSample(



            language="python",



            code=code,



            description="Comprehensive unit testing with mocking, fixtures, and TDD patterns",



            complexity="intermediate",



            domain="testing",



            tags=["unit_testing", "mocking", "tdd", "fixtures", "test_patterns"],



        )







                                            







    def _advanced_algorithms(self) -> CodeSample:



        variants = [



            self._gen_segment_tree,



            self._gen_red_black_tree,



            self._gen_a_star,



        ]



        return random.choice(variants)()







    def _gen_segment_tree(self) -> CodeSample:



        code = '''from typing import List, Callable, Optional, Any


class SegmentTree:
    """
    Segment Tree for efficient range queries and point updates.
    
    Supports:
    - Range sum queries
    - Range min/max queries
    - Point updates
    - Lazy propagation for range updates
    
    Time Complexity:
    - Build: O(n)
    - Query: O(log n)
    - Update: O(log n)
    """
    
    def __init__(self, data: List[int], merge: Callable = None, default: int = 0):
        """
        Initialize segment tree.
        
        Args:
            data: Input array
            merge: Merge function (default: sum)
            default: Identity element for merge
        """
        self.n = len(data)
        self.merge = merge or (lambda a, b: a + b)
        self.default = default
        
        # Tree size: 4 * n is always sufficient
        self.tree = [default] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.data = data
        
        if self.n > 0:
            self._build(1, 0, self.n - 1)
    
    def _build(self, node: int, start: int, end: int) -> None:
        """Build the segment tree recursively."""
        if start == end:
            self.tree[node] = self.data[start]
            return
        
        mid = (start + end) // 2
        self._build(2 * node, start, mid)
        self._build(2 * node + 1, mid + 1, end)
        self.tree[node] = self.merge(self.tree[2 * node], self.tree[2 * node + 1])
    
    def _push_down(self, node: int, start: int, end: int) -> None:
        """Push lazy values to children."""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            
            # Update left child
            self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
            self.lazy[2 * node] += self.lazy[node]
            
            # Update right child
            self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
            self.lazy[2 * node + 1] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def update(self, index: int, value: int) -> None:
        """Point update: set data[index] = value."""
        self._update(1, 0, self.n - 1, index, value)
    
    def _update(self, node: int, start: int, end: int, index: int, value: int) -> None:
        if start == end:
            self.tree[node] = value
            return
        
        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)
        
        self.tree[node] = self.merge(self.tree[2 * node], self.tree[2 * node + 1])
    
    def range_update(self, l: int, r: int, value: int) -> None:
        """Range update: add value to all elements in [l, r]."""
        self._range_update(1, 0, self.n - 1, l, r, value)
    
    def _range_update(self, node: int, start: int, end: int, l: int, r: int, value: int) -> None:
        if l > end or r < start:
            return
        
        if l <= start and end <= r:
            self.tree[node] += value * (end - start + 1)
            self.lazy[node] += value
            return
        
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, l, r, value)
        self._range_update(2 * node + 1, mid + 1, end, l, r, value)
        self.tree[node] = self.merge(self.tree[2 * node], self.tree[2 * node + 1])
    
    def query(self, l: int, r: int) -> int:
        """Range query: compute merge over [l, r]."""
        return self._query(1, 0, self.n - 1, l, r)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if l > end or r < start:
            return self.default
        
        if l <= start and end <= r:
            return self.tree[node]
        
        self._push_down(node, start, end)
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, l, r)
        right = self._query(2 * node + 1, mid + 1, end, l, r)
        return self.merge(left, right)


class FenwickTree:
    """
    Binary Indexed Tree (Fenwick Tree) for prefix sum queries.
    
    Simpler and faster than segment tree for prefix operations.
    
    Time Complexity:
    - Update: O(log n)
    - Prefix query: O(log n)
    - Range query: O(log n)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i: int, delta: int) -> None:
        """Add delta to element at index i (0-indexed)."""
        i += 1  # Convert to 1-indexed
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def prefix_sum(self, i: int) -> int:
        """Get sum of elements [0, i] (0-indexed)."""
        i += 1
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)
        return total
    
    def range_sum(self, l: int, r: int) -> int:
        """Get sum of elements [l, r] (0-indexed)."""
        return self.prefix_sum(r) - (self.prefix_sum(l - 1) if l > 0 else 0)
    
    @classmethod
    def from_array(cls, arr: List[int]) -> 'FenwickTree':
        """Build Fenwick tree from array."""
        ft = cls(len(arr))
        for i, val in enumerate(arr):
            ft.update(i, val)
        return ft


if __name__ == "__main__":
    data = [1, 3, 5, 7, 9, 11]
    
    # Segment Tree
    st = SegmentTree(data)
    print(f"Sum [1,3]: {st.query(1, 3)}")  # 3+5+7 = 15
    
    st.update(2, 10)  # data[2] = 10
    print(f"Sum [1,3] after update: {st.query(1, 3)}")  # 3+10+7 = 20
    
    st.range_update(1, 3, 5)  # Add 5 to [1,3]
    print(f"Sum [1,3] after range update: {st.query(1, 3)}")  # 8+15+12 = 35
    
    # Fenwick Tree
    ft = FenwickTree.from_array(data)
    print(f"\\nFenwick prefix_sum(3): {ft.prefix_sum(3)}")
    print(f"Fenwick range_sum(1,3): {ft.range_sum(1, 3)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Segment Tree with lazy propagation and Fenwick Tree for range queries",



            complexity="advanced",



            domain="algorithms",



            tags=["segment_tree", "fenwick_tree", "lazy_propagation", "range_queries"],



        )







    def _gen_red_black_tree(self) -> CodeSample:



        code = '''from typing import Optional, List, Any, Iterator
from enum import Enum


class Color(Enum):
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """Red-Black Tree node."""
    
    def __init__(self, key: Any, value: Any = None, color: Color = Color.RED):
        self.key = key
        self.value = value
        self.color = color
        self.left: Optional['RBNode'] = None
        self.right: Optional['RBNode'] = None
        self.parent: Optional['RBNode'] = None


class RedBlackTree:
    """
    Red-Black Tree implementation - a self-balancing binary search tree.
    
    Properties maintained:
    1. Every node is either red or black
    2. Root is black
    3. Every leaf (NIL) is black
    4. Red nodes have black children
    5. All paths from root to leaves have same number of black nodes
    
    Guarantees O(log n) for insert, delete, and search.
    """
    
    def __init__(self):
        self.NIL = RBNode(None, None, Color.BLACK)
        self.root = self.NIL
        self._size = 0
    
    def insert(self, key: Any, value: Any = None) -> None:
        """Insert a key-value pair into the tree."""
        node = RBNode(key, value)
        node.left = self.NIL
        node.right = self.NIL
        
        parent = None
        current = self.root
        
        while current != self.NIL:
            parent = current
            if node.key < current.key:
                current = current.left
            elif node.key > current.key:
                current = current.right
            else:
                current.value = value  # Update existing
                return
        
        node.parent = parent
        
        if parent is None:
            self.root = node
        elif node.key < parent.key:
            parent.left = node
        else:
            parent.right = node
        
        self._size += 1
        self._insert_fixup(node)
    
    def _insert_fixup(self, node: RBNode) -> None:
        """Fix Red-Black properties after insertion."""
        while node.parent and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                
                if uncle.color == Color.RED:
                    # Case 1: Uncle is red
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Uncle is black, node is right child
                        node = node.parent
                        self._left_rotate(node)
                    
                    # Case 3: Uncle is black, node is left child
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                # Mirror cases
                uncle = node.parent.parent.left
                
                if uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)
                    
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)
        
        self.root.color = Color.BLACK
    
    def delete(self, key: Any) -> bool:
        """Delete a node by key. Returns True if found and deleted."""
        node = self._search_node(self.root, key)
        if node == self.NIL:
            return False
        
        self._delete_node(node)
        self._size -= 1
        return True
    
    def _delete_node(self, node: RBNode) -> None:
        """Delete a node from the tree."""
        original_color = node.color
        
        if node.left == self.NIL:
            transplant_node = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            transplant_node = node.left
            self._transplant(node, node.left)
        else:
            successor = self._minimum(node.right)
            original_color = successor.color
            transplant_node = successor.right
            
            if successor.parent == node:
                transplant_node.parent = successor
            else:
                self._transplant(successor, successor.right)
                successor.right = node.right
                successor.right.parent = successor
            
            self._transplant(node, successor)
            successor.left = node.left
            successor.left.parent = successor
            successor.color = node.color
        
        if original_color == Color.BLACK:
            self._delete_fixup(transplant_node)
    
    def _delete_fixup(self, node: RBNode) -> None:
        """Fix Red-Black properties after deletion."""
        while node != self.root and node.color == Color.BLACK:
            if node == node.parent.left:
                sibling = node.parent.right
                
                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._left_rotate(node.parent)
                    sibling = node.parent.right
                
                if (sibling.left.color == Color.BLACK and
                    sibling.right.color == Color.BLACK):
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.right.color == Color.BLACK:
                        sibling.left.color = Color.BLACK
                        sibling.color = Color.RED
                        self._right_rotate(sibling)
                        sibling = node.parent.right
                    
                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.right.color = Color.BLACK
                    self._left_rotate(node.parent)
                    node = self.root
            else:
                # Mirror cases
                sibling = node.parent.left
                
                if sibling.color == Color.RED:
                    sibling.color = Color.BLACK
                    node.parent.color = Color.RED
                    self._right_rotate(node.parent)
                    sibling = node.parent.left
                
                if (sibling.right.color == Color.BLACK and
                    sibling.left.color == Color.BLACK):
                    sibling.color = Color.RED
                    node = node.parent
                else:
                    if sibling.left.color == Color.BLACK:
                        sibling.right.color = Color.BLACK
                        sibling.color = Color.RED
                        self._left_rotate(sibling)
                        sibling = node.parent.left
                    
                    sibling.color = node.parent.color
                    node.parent.color = Color.BLACK
                    sibling.left.color = Color.BLACK
                    self._right_rotate(node.parent)
                    node = self.root
        
        node.color = Color.BLACK
    
    def _left_rotate(self, x: RBNode) -> None:
        """Left rotation around node x."""
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
    
    def _right_rotate(self, y: RBNode) -> None:
        """Right rotation around node y."""
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is None:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x
    
    def _transplant(self, u: RBNode, v: RBNode) -> None:
        """Replace subtree rooted at u with subtree rooted at v."""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent
    
    def _minimum(self, node: RBNode) -> RBNode:
        while node.left != self.NIL:
            node = node.left
        return node
    
    def _search_node(self, node: RBNode, key: Any) -> RBNode:
        while node != self.NIL:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return self.NIL
    
    def search(self, key: Any) -> Optional[Any]:
        """Search for a key and return its value."""
        node = self._search_node(self.root, key)
        return node.value if node != self.NIL else None
    
    def inorder(self) -> List[Any]:
        """In-order traversal returning sorted keys."""
        result = []
        self._inorder(self.root, result)
        return result
    
    def _inorder(self, node: RBNode, result: List):
        if node != self.NIL:
            self._inorder(node.left, result)
            result.append(node.key)
            self._inorder(node.right, result)
    
    def height(self) -> int:
        """Calculate tree height."""
        return self._height(self.root)
    
    def _height(self, node: RBNode) -> int:
        if node == self.NIL:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))
    
    def black_height(self) -> int:
        """Calculate black height (number of black nodes on any root-to-leaf path)."""
        return self._black_height(self.root)
    
    def _black_height(self, node: RBNode) -> int:
        if node == self.NIL:
            return 1
        left_bh = self._black_height(node.left)
        if left_bh == 0:
            return 0
        if node.color == Color.BLACK:
            return left_bh + 1
        return left_bh
    
    def __len__(self) -> int:
        return self._size
    
    def __contains__(self, key: Any) -> bool:
        return self._search_node(self.root, key) != self.NIL


if __name__ == "__main__":
    import random
    
    tree = RedBlackTree()
    values = random.sample(range(100), 20)
    
    print(f"Inserting: {values}")
    for v in values:
        tree.insert(v, v * 10)
    
    print(f"\\nIn-order: {tree.inorder()}")
    print(f"Height: {tree.height()}")
    print(f"Black height: {tree.black_height()}")
    print(f"Size: {len(tree)}")
    
    # Search
    print(f"\\nSearch {values[0]}: {tree.search(values[0])}")
    print(f"Search 999: {tree.search(999)}")
    
    # Delete
    tree.delete(values[0])
    print(f"\\nAfter deleting {values[0]}: {tree.inorder()}")
    print(f"Size: {len(tree)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Complete Red-Black Tree with insert, delete, rotations, and property verification",



            complexity="advanced",



            domain="algorithms",



            tags=["red_black_tree", "balanced_bst", "data_structures", "rotations"],



        )







    def _gen_a_star(self) -> CodeSample:



        code = '''import heapq
from typing import Dict, List, Tuple, Set, Optional, Callable
from dataclasses import dataclass, field
import math


@dataclass(order=True)
class AStarNode:
    """Priority queue node for A* search."""
    f_score: float
    position: Tuple[int, int] = field(compare=False)
    g_score: float = field(compare=False, default=float('inf'))


class Grid:
    """2D grid for pathfinding with obstacles."""
    
    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
    
    def is_valid(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.obstacles)
    
    def neighbors(self, pos: Tuple[int, int], diagonal: bool = True) -> List[Tuple[Tuple[int, int], float]]:
        """Get valid neighbors with movement costs."""
        x, y = pos
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Cardinal
        if diagonal:
            moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        result = []
        for dx, dy in moves:
            next_pos = (x + dx, y + dy)
            if self.is_valid(next_pos):
                cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                result.append((next_pos, cost))
        
        return result
    
    def display(self, path: List[Tuple[int, int]] = None, start=None, goal=None) -> str:
        """Visualize the grid with optional path."""
        path_set = set(path) if path else set()
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                pos = (x, y)
                if pos == start:
                    row += "S "
                elif pos == goal:
                    row += "G "
                elif pos in path_set:
                    row += "* "
                elif pos in self.obstacles:
                    row += "# "
                else:
                    row += ". "
            lines.append(row)
        return "\\n".join(lines)


def heuristic_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic (admissible for 4-directional movement)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic (admissible for any movement)."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def heuristic_octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Octile distance heuristic (admissible for 8-directional movement)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def a_star(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable = heuristic_octile,
    diagonal: bool = True,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding algorithm.
    
    A* is an informed search algorithm that uses a heuristic to guide
    the search toward the goal, guaranteeing optimal paths when the
    heuristic is admissible (never overestimates).
    
    Args:
        grid: The grid to search on
        start: Starting position (x, y)
        goal: Goal position (x, y)
        heuristic: Heuristic function h(node, goal)
        diagonal: Whether diagonal movement is allowed
        
    Returns:
        List of positions forming the shortest path, or None if no path exists
        
    Time Complexity: O(E log V) where E = edges, V = vertices
    Space Complexity: O(V)
    """
    if not grid.is_valid(start) or not grid.is_valid(goal):
        return None
    
    # Open set (priority queue): ordered by f_score
    open_set = []
    heapq.heappush(open_set, AStarNode(
        f_score=heuristic(start, goal),
        position=start,
        g_score=0,
    ))
    
    # Track best known g_scores
    g_scores: Dict[Tuple[int, int], float] = {start: 0}
    
    # Track path for reconstruction
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    # Closed set
    closed_set: Set[Tuple[int, int]] = set()
    
    nodes_explored = 0
    
    while open_set:
        current = heapq.heappop(open_set)
        nodes_explored += 1
        
        if current.position == goal:
            print(f"A* explored {nodes_explored} nodes")
            return _reconstruct_path(came_from, goal)
        
        if current.position in closed_set:
            continue
        closed_set.add(current.position)
        
        # Skip if we found a better path already
        if current.g_score > g_scores.get(current.position, float('inf')):
            continue
        
        for neighbor_pos, move_cost in grid.neighbors(current.position, diagonal):
            if neighbor_pos in closed_set:
                continue
            
            tentative_g = current.g_score + move_cost
            
            if tentative_g < g_scores.get(neighbor_pos, float('inf')):
                came_from[neighbor_pos] = current.position
                g_scores[neighbor_pos] = tentative_g
                f_score = tentative_g + heuristic(neighbor_pos, goal)
                
                heapq.heappush(open_set, AStarNode(
                    f_score=f_score,
                    position=neighbor_pos,
                    g_score=tentative_g,
                ))
    
    print(f"A* explored {nodes_explored} nodes - no path found")
    return None


def _reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from map."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def ida_star(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    heuristic: Callable = heuristic_manhattan,
) -> Optional[List[Tuple[int, int]]]:
    """
    Iterative Deepening A* (IDA*) - memory-efficient variant of A*.
    
    Uses iterative deepening with f-cost threshold instead of open set.
    Space Complexity: O(d) where d = depth of solution
    """
    threshold = heuristic(start, goal)
    
    while True:
        result, new_threshold = _ida_star_search(
            grid, start, goal, 0, threshold, heuristic, []
        )
        if result is not None:
            return result
        if new_threshold == float('inf'):
            return None
        threshold = new_threshold


def _ida_star_search(
    grid: Grid, node: Tuple[int, int], goal: Tuple[int, int],
    g: float, threshold: float, heuristic: Callable,
    path: List[Tuple[int, int]],
) -> Tuple[Optional[List], float]:
    f = g + heuristic(node, goal)
    
    if f > threshold:
        return None, f
    
    if node == goal:
        return path + [node], float('inf')
    
    min_threshold = float('inf')
    
    for neighbor, cost in grid.neighbors(node, diagonal=False):
        if neighbor not in path:
            result, new_threshold = _ida_star_search(
                grid, neighbor, goal, g + cost, threshold, heuristic,
                path + [node]
            )
            if result is not None:
                return result, float('inf')
            min_threshold = min(min_threshold, new_threshold)
    
    return None, min_threshold


if __name__ == "__main__":
    # Create a grid with obstacles
    obstacles = {(3, i) for i in range(7)}  # Vertical wall
    obstacles |= {(i, 3) for i in range(3, 8)}  # Horizontal wall
    obstacles.discard((3, 3))  # Gap in the wall
    
    grid = Grid(10, 10, obstacles)
    start = (0, 0)
    goal = (9, 9)
    
    # A* search
    path = a_star(grid, start, goal)
    if path:
        print(f"Path length: {len(path)}")
        print(f"Path cost: {len(path) - 1}")
        print(grid.display(path, start, goal))'''







        return CodeSample(



            language="python",



            code=code,



            description="A* pathfinding with multiple heuristics and IDA* variant",



            complexity="advanced",



            domain="algorithms",



            tags=["a_star", "pathfinding", "heuristic", "ida_star", "grid_search"],



        )







    def _advanced_data(self) -> CodeSample:



        code = '''"""
Distributed Data Processing Pipeline with MapReduce Pattern.

Implements a complete data processing pipeline with:
- Map/Reduce operations
- Data partitioning and shuffling
- Fault tolerance concepts
- Streaming processing support
"""

import multiprocessing as mp
from typing import (
    List, Dict, Any, Callable, Tuple, Iterator, Optional, Generic, TypeVar
)
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import reduce

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class Record:
    """A data record with key and value."""
    key: str
    value: Any
    partition: int = 0
    timestamp: float = field(default_factory=time.time)


class Partitioner:
    """Hash-based partitioner for distributing data across workers."""
    
    def __init__(self, num_partitions: int):
        self.num_partitions = num_partitions
    
    def partition(self, key: str) -> int:
        """Compute partition index for a key."""
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % self.num_partitions


class Mapper:
    """Base mapper class for MapReduce operations."""
    
    def map(self, record: Record) -> Iterator[Tuple[str, Any]]:
        """Map a single record to key-value pairs."""
        raise NotImplementedError


class Reducer:
    """Base reducer class for MapReduce operations."""
    
    def reduce(self, key: str, values: Iterator[Any]) -> Tuple[str, Any]:
        """Reduce values for a given key."""
        raise NotImplementedError


# ============================================================
# CONCRETE MAP/REDUCE IMPLEMENTATIONS
# ============================================================

class WordCountMapper(Mapper):
    """Count words in text records."""
    
    def map(self, record: Record) -> Iterator[Tuple[str, int]]:
        text = str(record.value).lower()
        for word in text.split():
            # Clean punctuation
            word = "".join(c for c in word if c.isalnum())
            if word:
                yield (word, 1)


class WordCountReducer(Reducer):
    """Sum word counts."""
    
    def reduce(self, key: str, values: Iterator[int]) -> Tuple[str, int]:
        return (key, sum(values))


class LogParserMapper(Mapper):
    """Parse log entries and extract useful fields."""
    
    def map(self, record: Record) -> Iterator[Tuple[str, Dict]]:
        line = str(record.value)
        parts = line.split()
        
        if len(parts) >= 7:
            ip = parts[0]
            status = parts[5] if len(parts) > 5 else "unknown"
            path = parts[6] if len(parts) > 6 else "/"
            
            yield ("status_codes", {"code": status, "ip": ip, "path": path})
            yield (f"ip:{ip}", {"path": path, "status": status})
            yield (f"path:{path}", {"ip": ip, "status": status})


class LogAnalysisReducer(Reducer):
    """Aggregate log analysis results."""
    
    def reduce(self, key: str, values: Iterator[Dict]) -> Tuple[str, Dict]:
        items = list(values)
        result = {
            "count": len(items),
            "unique_ips": len(set(v.get("ip", "") for v in items)),
            "status_distribution": defaultdict(int),
        }
        for item in items:
            if "code" in item:
                result["status_distribution"][item["code"]] += 1
        
        return (key, dict(result))


# ============================================================
# MAPREDUCE ENGINE
# ============================================================

class MapReduceEngine:
    """
    Distributed MapReduce processing engine.
    
    Handles:
    - Parallel map execution
    - Shuffle and sort phase
    - Parallel reduce execution
    - Data partitioning
    """
    
    def __init__(
        self,
        num_mappers: int = 4,
        num_reducers: int = 4,
        num_partitions: int = 8,
    ):
        self.num_mappers = num_mappers
        self.num_reducers = num_reducers
        self.partitioner = Partitioner(num_partitions)
    
    def execute(
        self,
        records: List[Record],
        mapper: Mapper,
        reducer: Reducer,
    ) -> Dict[str, Any]:
        """
        Execute a complete MapReduce job.
        
        Args:
            records: Input records
            mapper: Mapper instance
            reducer: Reducer instance
            
        Returns:
            Dictionary of reduced results
        """
        print(f"Starting MapReduce with {len(records)} records")
        
        # === MAP PHASE ===
        print("  Phase 1: Mapping...")
        map_outputs = self._map_phase(records, mapper)
        print(f"    Generated {sum(len(v) for v in map_outputs.values())} intermediate pairs")
        
        # === SHUFFLE PHASE ===
        print("  Phase 2: Shuffling...")
        shuffled = self._shuffle_phase(map_outputs)
        print(f"    Created {len(shuffled)} partitions")
        
        # === SORT PHASE ===
        print("  Phase 3: Sorting...")
        sorted_data = self._sort_phase(shuffled)
        
        # === REDUCE PHASE ===
        print("  Phase 4: Reducing...")
        results = self._reduce_phase(sorted_data, reducer)
        print(f"    Produced {len(results)} output records")
        
        return results
    
    def _map_phase(
        self,
        records: List[Record],
        mapper: Mapper,
    ) -> Dict[int, List[Tuple[str, Any]]]:
        """Execute map phase with parallel processing."""
        # Split records across mappers
        chunks = self._split(records, self.num_mappers)
        
        # Map in parallel
        partitions: Dict[int, List[Tuple[str, Any]]] = defaultdict(list)
        
        with ThreadPoolExecutor(max_workers=self.num_mappers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(self._map_chunk, chunk, mapper))
            
            for future in futures:
                chunk_output = future.result()
                for key, value in chunk_output:
                    partition = self.partitioner.partition(key)
                    partitions[partition].append((key, value))
        
        return partitions
    
    def _map_chunk(
        self,
        records: List[Record],
        mapper: Mapper,
    ) -> List[Tuple[str, Any]]:
        """Map a chunk of records."""
        output = []
        for record in records:
            for key, value in mapper.map(record):
                output.append((key, value))
        return output
    
    def _shuffle_phase(
        self,
        map_outputs: Dict[int, List[Tuple[str, Any]]],
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Shuffle: group by key within each partition."""
        shuffled = {}
        
        for partition_id, pairs in map_outputs.items():
            grouped: Dict[str, List[Any]] = defaultdict(list)
            for key, value in pairs:
                grouped[key].append(value)
            shuffled[partition_id] = dict(grouped)
        
        return shuffled
    
    def _sort_phase(
        self,
        shuffled: Dict[int, Dict[str, List[Any]]],
    ) -> List[Tuple[str, List[Any]]]:
        """Sort keys within each partition."""
        sorted_data = []
        
        for partition_id, grouped in shuffled.items():
            for key in sorted(grouped.keys()):
                sorted_data.append((key, grouped[key]))
        
        return sorted_data
    
    def _reduce_phase(
        self,
        sorted_data: List[Tuple[str, List[Any]]],
        reducer: Reducer,
    ) -> Dict[str, Any]:
        """Execute reduce phase."""
        results = {}
        
        for key, values in sorted_data:
            reduced_key, reduced_value = reducer.reduce(key, iter(values))
            results[reduced_key] = reduced_value
        
        return results
    
    def _split(self, data: List, n: int) -> List[List]:
        """Split data into n roughly equal chunks."""
        k, m = divmod(len(data), n)
        return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


# ============================================================
# STREAMING PROCESSOR
# ============================================================

class StreamProcessor:
    """
    Real-time stream processing with windowed aggregations.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # seconds
        self.buffers: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)
        self.aggregations: Dict[str, List[float]] = defaultdict(list)
    
    def process_event(self, key: str, value: float, timestamp: float = None) -> None:
        """Process a streaming event."""
        if timestamp is None:
            timestamp = time.time()
        
        self.buffers[key].append((timestamp, value))
        
        # Evict old entries outside the window
        cutoff = timestamp - self.window_size
        self.buffers[key] = [
            (t, v) for t, v in self.buffers[key] if t >= cutoff
        ]
    
    def get_window_stats(self, key: str) -> Dict[str, float]:
        """Get statistics for the current window."""
        entries = self.buffers.get(key, [])
        if not entries:
            return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}
        
        values = [v for _, v in entries]
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }


if __name__ == "__main__":
    # Word Count Example
    texts = [
        Record("doc1", "the quick brown fox jumps over the lazy dog"),
        Record("doc2", "the dog barked at the fox in the garden"),
        Record("doc3", "a quick brown dog chased the cat around the garden"),
        Record("doc4", "the fox and the dog became friends in the garden"),
    ]
    
    engine = MapReduceEngine(num_mappers=2, num_reducers=2, num_partitions=4)
    results = engine.execute(texts, WordCountMapper(), WordCountReducer())
    
    # Sort by count
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\\nTop 10 words:")
    for word, count in sorted_results[:10]:
        print(f"  {word}: {count}")'''







        return CodeSample(



            language="python",



            code=code,



            description="MapReduce engine with streaming processor and windowed aggregations",



            complexity="advanced",



            domain="data",



            tags=["mapreduce", "distributed", "streaming", "data_pipeline", "parallel"],



        )







    def _advanced_web(self) -> CodeSample:



        code = '''"""
Production-Ready REST API with Authentication, Rate Limiting, and Caching.

Features:
- JWT-based authentication
- Role-based access control
- Rate limiting with sliding window
- Response caching
- Input validation
- Error handling middleware
"""

import json
import time
import hashlib
import hmac
import base64
import secrets
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from functools import wraps
from datetime import datetime, timedelta


# ============================================================
# JWT IMPLEMENTATION (Simplified for demonstration)
# ============================================================

class JWTHandler:
    """JSON Web Token handler with HMAC-SHA256 signing."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key.encode()
        self.algorithm = algorithm
    
    def encode(self, payload: Dict, expiration_minutes: int = 60) -> str:
        """Create a JWT token."""
        header = {"alg": self.algorithm, "typ": "JWT"}
        
        # Add expiration
        payload = {**payload}
        payload["exp"] = int((datetime.now() + timedelta(minutes=expiration_minutes)).timestamp())
        payload["iat"] = int(datetime.now().timestamp())
        
        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())
        
        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(self.secret_key, message.encode(), hashlib.sha256).digest()
        signature_b64 = self._base64url_encode(signature)
        
        return f"{header_b64}.{payload_b64}.{signature_b64}"
    
    def decode(self, token: str) -> Optional[Dict]:
        """Decode and validate a JWT token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(self.secret_key, message.encode(), hashlib.sha256).digest()
            actual_sig = self._base64url_decode(signature_b64)
            
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None
            
            # Decode payload
            payload = json.loads(self._base64url_decode(payload_b64))
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None
            
            return payload
        except Exception:
            return None
    
    def _base64url_encode(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode()
    
    def _base64url_decode(self, data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data)


# ============================================================
# RATE LIMITER
# ============================================================

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    
    More accurate than fixed window, prevents burst at window boundaries.
    """
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict]:
        """Check if request is allowed. Returns (allowed, info)."""
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Remove old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > cutoff
        ]
        
        current_count = len(self.requests[client_id])
        
        if current_count >= self.max_requests:
            retry_after = self.requests[client_id][0] - cutoff + 1
            return False, {
                "limit": self.max_requests,
                "remaining": 0,
                "retry_after": retry_after,
            }
        
        self.requests[client_id].append(now)
        return True, {
            "limit": self.max_requests,
            "remaining": self.max_requests - current_count - 1,
        }
    
    def cleanup(self) -> None:
        """Remove expired entries to free memory."""
        now = time.time()
        cutoff = now - self.window_seconds
        expired = [k for k, v in self.requests.items() 
                  if not v or v[-1] < cutoff]
        for k in expired:
            del self.requests[k]


# ============================================================
# RESPONSE CACHE
# ============================================================

class ResponseCache:
    """In-memory response cache with TTL."""
    
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            expiry, value = self.cache[key]
            if time.time() < expiry:
                self.hits += 1
                return value
            del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        expiry = time.time() + (ttl or self.default_ttl)
        self.cache[key] = (expiry, value)
    
    def invalidate(self, key: str) -> None:
        self.cache.pop(key, None)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a prefix pattern."""
        to_delete = [k for k in self.cache if k.startswith(pattern)]
        for k in to_delete:
            del self.cache[k]
        return len(to_delete)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============================================================
# INPUT VALIDATOR
# ============================================================

class Validator:
    """Request input validation."""
    
    @staticmethod
    def validate_schema(data: Dict, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate data against a schema.
        
        Schema format:
        {
            "field_name": {
                "type": "string" | "int" | "float" | "bool" | "list",
                "required": True | False,
                "min_length": int,
                "max_length": int,
                "pattern": str (regex),
                "choices": list,
            }
        }
        """
        errors = []
        
        for field, rules in schema.items():
            value = data.get(field)
            
            # Check required
            if rules.get("required", False) and (value is None or value == ""):
                errors.append(f"Field '{field}' is required")
                continue
            
            if value is None:
                continue
            
            # Type check
            expected_type = rules.get("type")
            type_map = {
                "string": str, "int": int, "float": (int, float),
                "bool": bool, "list": list,
            }
            if expected_type and expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    errors.append(f"Field '{field}' must be of type {expected_type}")
                    continue
            
            # String validations
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append(f"Field '{field}' must be at least {rules['min_length']} chars")
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append(f"Field '{field}' must be at most {rules['max_length']} chars")
            
            # Choices validation
            if "choices" in rules and value not in rules["choices"]:
                errors.append(f"Field '{field}' must be one of {rules['choices']}")
        
        return len(errors) == 0, errors


# ============================================================
# API ROUTER WITH MIDDLEWARE
# ============================================================

class APIRouter:
    """API Router with middleware support."""
    
    def __init__(self, jwt_handler: JWTHandler, rate_limiter: SlidingWindowRateLimiter):
        self.jwt = jwt_handler
        self.rate_limiter = rate_limiter
        self.cache = ResponseCache()
        self.validator = Validator()
        self.routes: Dict[str, Dict[str, Callable]] = defaultdict(dict)
    
    def route(self, path: str, methods: List[str] = None, auth_required: bool = False,
              roles: List[str] = None, cacheable: bool = False, rate_limit_group: str = "default"):
        """Route decorator with authentication and caching support."""
        if methods is None:
            methods = ["GET"]
        
        def decorator(handler: Callable):
            for method in methods:
                self.routes[path][method] = {
                    "handler": handler,
                    "auth_required": auth_required,
                    "roles": roles or [],
                    "cacheable": cacheable,
                    "rate_limit_group": rate_limit_group,
                }
            return handler
        return decorator
    
    def handle(self, method: str, path: str, headers: Dict = None,
               body: Dict = None, client_id: str = "anonymous") -> Dict:
        """Handle an API request."""
        headers = headers or {}
        
        # Rate limiting
        allowed, rate_info = self.rate_limiter.is_allowed(client_id)
        if not allowed:
            return {
                "status": 429,
                "body": {"error": "Rate limit exceeded", **rate_info},
            }
        
        # Find route
        route_info = self.routes.get(path, {}).get(method)
        if not route_info:
            return {"status": 404, "body": {"error": "Not found"}}
        
        # Authentication
        if route_info["auth_required"]:
            token = headers.get("Authorization", "").replace("Bearer ", "")
            payload = self.jwt.decode(token)
            if not payload:
                return {"status": 401, "body": {"error": "Invalid or expired token"}}
            
            # Role check
            if route_info["roles"]:
                user_role = payload.get("role", "")
                if user_role not in route_info["roles"]:
                    return {"status": 403, "body": {"error": "Insufficient permissions"}}
            
            body = body or {}
            body["_user"] = payload
        
        # Cache check
        cache_key = f"{method}:{path}:{json.dumps(body or {}, sort_keys=True)}"
        if route_info["cacheable"] and method == "GET":
            cached = self.cache.get(cache_key)
            if cached is not None:
                return {"status": 200, "body": cached, "cached": True}
        
        # Execute handler
        try:
            result = route_info["handler"](body or {}, headers)
            
            # Cache result
            if route_info["cacheable"] and method == "GET":
                self.cache.set(cache_key, result)
            
            return {"status": 200, "body": result}
        except ValueError as e:
            return {"status": 400, "body": {"error": str(e)}}
        except Exception as e:
            return {"status": 500, "body": {"error": "Internal server error"}}


# ============================================================
# EXAMPLE API
# ============================================================

if __name__ == "__main__":
    # Initialize components
    jwt = JWTHandler(secret_key="my-secret-key-change-in-production")
    rate_limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
    api = APIRouter(jwt, rate_limiter)
    
    # Simulated database
    users_db = {}
    products_db = {
        "1": {"id": "1", "name": "Widget", "price": 29.99},
        "2": {"id": "2", "name": "Gadget", "price": 49.99},
    }
    
    # Auth endpoints
    @api.route("/auth/register", methods=["POST"])
    def register(body, headers):
        valid, errors = api.validator.validate_schema(body, {
            "username": {"type": "string", "required": True, "min_length": 3, "max_length": 50},
            "email": {"type": "string", "required": True},
            "password": {"type": "string", "required": True, "min_length": 8},
        })
        if not valid:
            raise ValueError("; ".join(errors))
        
        if body["email"] in users_db:
            raise ValueError("Email already registered")
        
        user = {
            "id": str(len(users_db) + 1),
            "username": body["username"],
            "email": body["email"],
            "role": "user",
        }
        users_db[body["email"]] = user
        
        token = jwt.encode({"sub": user["id"], "role": user["role"]})
        return {"token": token, "user": user}
    
    @api.route("/auth/login", methods=["POST"])
    def login(body, headers):
        user = users_db.get(body.get("email"))
        if not user:
            raise ValueError("Invalid credentials")
        
        token = jwt.encode({"sub": user["id"], "role": user["role"]})
        return {"token": token, "user": user}
    
    # Product endpoints
    @api.route("/api/products", methods=["GET"], cacheable=True)
    def list_products(body, headers):
        return {"products": list(products_db.values())}
    
    @api.route("/api/products/<id>", methods=["GET"], cacheable=True)
    def get_product(body, headers):
        product_id = body.get("id", "")
        product = products_db.get(product_id)
        if not product:
            raise ValueError("Product not found")
        return product
    
    @api.route("/api/products", methods=["POST"], auth_required=True, roles=["admin"])
    def create_product(body, headers):
        valid, errors = api.validator.validate_schema(body, {
            "name": {"type": "string", "required": True},
            "price": {"type": "float", "required": True},
        })
        if not valid:
            raise ValueError("; ".join(errors))
        
        product_id = str(len(products_db) + 1)
        product = {"id": product_id, "name": body["name"], "price": body["price"]}
        products_db[product_id] = product
        api.cache.invalidate_pattern("GET:/api/products")
        return product
    
    # Demo
    print("=== API Demo ===")
    
    # Register
    result = api.handle("POST", "/auth/register", body={
        "username": "alice", "email": "alice@test.com", "password": "secret123"
    })
    print(f"Register: {result}")
    
    # List products (cached)
    result = api.handle("GET", "/api/products")
    print(f"Products: {result}")
    
    # Same request (should be cached)
    result = api.handle("GET", "/api/products")
    print(f"Cached: {result.get('cached', False)}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Production REST API with JWT auth, rate limiting, caching, and validation",



            complexity="advanced",



            domain="web",



            tags=["rest_api", "jwt", "rate_limiting", "caching", "authentication", "validation"],



        )







    def _advanced_systems(self) -> CodeSample:



        code = '''"""
Async Task Queue and Event-Driven Architecture.

Implements:
- Priority task queue with worker pools
- Event bus with pub/sub pattern
- Circuit breaker for fault tolerance
- Retry with exponential backoff
"""

import time
import threading
import queue
import uuid
from typing import (
    Callable, Dict, List, Any, Optional, Tuple, Set
)
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# TASK QUEUE
# ============================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass(order=True)
class Task:
    """A unit of work to be processed."""
    priority: int
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    name: str = field(default="", compare=False)
    payload: Dict = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)
    retry_count: int = field(default=0, compare=False)
    max_retries: int = field(default=3, compare=False)
    created_at: float = field(default_factory=time.time, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    completed_at: Optional[float] = field(default=None, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False, repr=False)


class TaskQueue:
    """
    Priority task queue with configurable worker pool.
    
    Features:
    - Priority-based processing
    - Configurable concurrency
    - Automatic retry with backoff
    - Task status tracking
    - Graceful shutdown
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        max_queue_size: int = 10000,
        retry_delay_base: float = 1.0,
    ):
        self.num_workers = num_workers
        self.retry_delay_base = retry_delay_base
        
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Callable] = {}
        self._workers: List[threading.Thread] = []
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Metrics
        self._processed = 0
        self._failed = 0
        self._lock = threading.Lock()
    
    def register_handler(self, task_name: str) -> Callable:
        """Decorator to register a task handler."""
        def decorator(func: Callable):
            self._handlers[task_name] = func
            return func
        return decorator
    
    def submit(
        self,
        name: str,
        payload: Dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        callback: Callable = None,
    ) -> str:
        """Submit a task to the queue. Returns task ID."""
        task = Task(
            priority=priority.value,
            name=name,
            payload=payload or {},
            max_retries=max_retries,
            callback=callback,
        )
        
        self._tasks[task.task_id] = task
        self._queue.put(task)
        
        logger.info(f"Task submitted: {task.task_id} ({name}) [priority={priority.name}]")
        return task.task_id
    
    def get_status(self, task_id: str) -> Optional[Task]:
        """Get the status of a task."""
        return self._tasks.get(task_id)
    
    def start(self) -> None:
        """Start worker threads."""
        self._running = True
        self._shutdown_event.clear()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        
        logger.info(f"Task queue started with {self.num_workers} workers")
    
    def stop(self, timeout: float = 30.0) -> None:
        """Gracefully stop all workers."""
        self._running = False
        self._shutdown_event.set()
        
        for worker in self._workers:
            worker.join(timeout=timeout)
        
        self._workers.clear()
        logger.info("Task queue stopped")
    
    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                task = self._queue.get(timeout=1.0)
                self._process_task(task)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_task(self, task: Task) -> None:
        """Process a single task."""
        handler = self._handlers.get(task.name)
        
        if handler is None:
            task.status = TaskStatus.FAILED
            task.error = f"No handler registered for task: {task.name}"
            self._failed += 1
            return
        
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        try:
            result = handler(task.payload)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            with self._lock:
                self._processed += 1
            
            # Execute callback
            if task.callback:
                task.callback(task)
            
            logger.info(f"Task completed: {task.task_id} ({task.name})")
            
        except Exception as e:
            task.error = str(e)
            
            if task.retry_count < task.max_retries:
                # Exponential backoff
                delay = self.retry_delay_base * (2 ** task.retry_count)
                delay += random.uniform(0, delay * 0.1)  # Jitter
                
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                logger.warning(
                    f"Task failed (retry {task.retry_count}/{task.max_retries}): "
                    f"{task.task_id} - {e}. Retrying in {delay:.1f}s"
                )
                
                # Re-queue with delay
                threading.Timer(delay, lambda t=task: self._queue.put(t)).start()
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                
                with self._lock:
                    self._failed += 1
                
                logger.error(f"Task failed permanently: {task.task_id} ({task.name}) - {e}")
    
    @property
    def metrics(self) -> Dict[str, int]:
        return {
            "queued": self._queue.qsize(),
            "processed": self._processed,
            "failed": self._failed,
            "total": len(self._tasks),
        }


# ============================================================
# EVENT BUS (PUB/SUB)
# ============================================================

class EventBus:
    """
    In-process event bus with publish/subscribe pattern.
    
    Supports:
    - Synchronous and asynchronous event handling
    - Event filtering
    - Wildcard subscriptions
    - Dead letter queue for unhandled events
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict] = []
        self._max_history = 1000
        self._dead_letters: List[Dict] = []
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, handler: Callable) -> Callable:
        """Subscribe to an event type. Supports wildcards (*)."""
        with self._lock:
            self._subscribers[event_type].append(handler)
        return handler
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    h for h in self._subscribers[event_type] if h != handler
                ]
    
    def publish(self, event_type: str, data: Any = None, source: str = "") -> int:
        """
        Publish an event. Returns number of handlers invoked.
        """
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time.time(),
        }
        
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        # Find matching handlers
        handlers = self._get_matching_handlers(event_type)
        
        if not handlers:
            self._dead_letters.append(event)
            return 0
        
        invoked = 0
        for handler in handlers:
            try:
                handler(event)
                invoked += 1
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        return invoked
    
    def _get_matching_handlers(self, event_type: str) -> List[Callable]:
        """Get all handlers matching an event type (including wildcards)."""
        handlers = []
        
        with self._lock:
            # Exact match
            handlers.extend(self._subscribers.get(event_type, []))
            
            # Wildcard matches
            for pattern, pattern_handlers in self._subscribers.items():
                if '*' in pattern:
                    prefix = pattern.replace('*', '')
                    if event_type.startswith(prefix):
                        handlers.extend(pattern_handlers)
            
            # Catch-all
            handlers.extend(self._subscribers.get('*', []))
        
        return handlers


# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered (limited requests)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if (self._last_failure_time and 
                    time.time() - self._last_failure_time > self.recovery_timeout):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through the circuit breaker."""
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            raise RuntimeError("Circuit breaker is OPEN - service unavailable")
        
        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise RuntimeError("Circuit breaker HALF_OPEN - max test calls reached")
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
    
    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN


if __name__ == "__main__":
    # Task Queue Demo
    tq = TaskQueue(num_workers=3)
    
    @tq.register_handler("send_email")
    def send_email(payload):
        print(f"  Sending email to {payload['to']}: {payload['subject']}")
        time.sleep(0.5)
        return {"sent": True}
    
    @tq.register_handler("process_data")
    def process_data(payload):
        print(f"  Processing {payload.get('items', 0)} items")
        if random.random() < 0.3:
            raise RuntimeError("Random processing failure")
        return {"processed": payload.get("items", 0)}
    
    tq.start()
    
    # Submit tasks
    for i in range(5):
        tq.submit("send_email", {
            "to": f"user{i}@example.com",
            "subject": f"Test email {i}",
        })
    
    tq.submit("process_data", {"items": 100}, priority=TaskPriority.HIGH)
    
    time.sleep(3)
    print(f"\\nMetrics: {tq.metrics}")
    tq.stop()
    
    # Event Bus Demo
    bus = EventBus()
    
    @bus.subscribe("user.created")
    def on_user_created(event):
        print(f"  User created: {event['data']}")
    
    @bus.subscribe("user.*")
    def on_any_user_event(event):
        print(f"  User event: {event['type']} -> {event['data']}")
    
    bus.publish("user.created", {"name": "Alice"}, source="auth")
    bus.publish("user.updated", {"name": "Alice", "role": "admin"})'''







        return CodeSample(



            language="python",



            code=code,



            description="Task queue, event bus, and circuit breaker for distributed systems",



            complexity="advanced",



            domain="systems",



            tags=["task_queue", "event_bus", "circuit_breaker", "pub_sub", "distributed"],



        )







                                          







    def _expert_algorithms(self) -> CodeSample:



        code = '''"""
Advanced Graph Algorithms: Network Flow, Matching, and Community Detection.

Implements:
- Ford-Fulkerson max flow with BFS (Edmonds-Karp)
- Hungarian algorithm for optimal assignment
- Louvain community detection
- Topological sort with cycle detection
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import math
import random


# ============================================================
# MAXIMUM FLOW (Edmonds-Karp / Ford-Fulkerson with BFS)
# ============================================================

class FlowNetwork:
    """
    Flow network for maximum flow problems.
    
    Implements Edmonds-Karp algorithm (Ford-Fulkerson with BFS).
    Time Complexity: O(V × E²)
    """
    
    def __init__(self, num_vertices: int):
        self.n = num_vertices
        self.capacity = [[0] * num_vertices for _ in range(num_vertices)]
        self.flow = [[0] * num_vertices for _ in range(num_vertices)]
        self.adj: List[List[int]] = [[] for _ in range(num_vertices)]
    
    def add_edge(self, u: int, v: int, cap: int) -> None:
        """Add a directed edge with given capacity."""
        self.capacity[u][v] += cap
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
    
    def max_flow(self, source: int, sink: int) -> int:
        """
        Compute maximum flow from source to sink using Edmonds-Karp.
        
        Returns the maximum flow value.
        """
        total_flow = 0
        
        while True:
            # Find augmenting path using BFS
            path = self._bfs(source, sink)
            if path is None:
                break
            
            # Find bottleneck
            bottleneck = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual = self.capacity[u][v] - self.flow[u][v]
                bottleneck = min(bottleneck, residual)
            
            # Update flow along path
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.flow[u][v] += bottleneck
                self.flow[v][u] -= bottleneck
            
            total_flow += bottleneck
        
        return total_flow
    
    def _bfs(self, source: int, sink: int) -> Optional[List[int]]:
        """BFS to find augmenting path in residual graph."""
        visited = [False] * self.n
        visited[source] = True
        parent = [-1] * self.n
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for v in self.adj[u]:
                if not visited[v] and self.capacity[u][v] - self.flow[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    if v == sink:
                        # Reconstruct path
                        path = []
                        curr = sink
                        while curr != -1:
                            path.append(curr)
                            curr = parent[curr]
                        return path[::-1]
                    queue.append(v)
        
        return None
    
    def min_cut(self, source: int, sink: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Find minimum cut (by max-flow min-cut theorem).
        
        Returns (cut_value, list_of_cut_edges).
        """
        flow_value = self.max_flow(source, sink)
        
        # Find reachable vertices from source in residual graph
        visited = [False] * self.n
        queue = deque([source])
        visited[source] = True
        
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if not visited[v] and self.capacity[u][v] - self.flow[u][v] > 0:
                    visited[v] = True
                    queue.append(v)
        
        # Cut edges: from visited to non-visited
        cut_edges = []
        for u in range(self.n):
            for v in range(self.n):
                if visited[u] and not visited[v] and self.capacity[u][v] > 0:
                    cut_edges.append((u, v))
        
        return flow_value, cut_edges


# ============================================================
# HUNGARIAN ALGORITHM (Optimal Assignment)
# ============================================================

def hungarian_algorithm(cost_matrix: List[List[float]]) -> Tuple[List[Tuple[int, int]], float]:
    """
    Hungarian algorithm for the assignment problem.
    
    Given an n×n cost matrix, find an assignment of rows to columns
    that minimizes the total cost.
    
    Time Complexity: O(n³)
    
    Args:
        cost_matrix: Square matrix where cost_matrix[i][j] = cost of assigning i to j
        
    Returns:
        (assignment list of (row, col) pairs, minimum total cost)
    """
    n = len(cost_matrix)
    if n == 0:
        return [], 0
    
    # Pad to square if needed
    m = len(cost_matrix[0])
    size = max(n, m)
    cost = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            cost[i][j] = cost_matrix[i][j]
    
    # Potential functions
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (size + 1)
        used = [False] * (size + 1)
        
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = -1
            
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
            if p[j0] == 0:
                break
        
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]
    
    # Extract assignment
    assignment = []
    total_cost = 0.0
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < n and j - 1 < m:
            assignment.append((p[j] - 1, j - 1))
            total_cost += cost_matrix[p[j] - 1][j - 1]
    
    return assignment, total_cost


# ============================================================
# TOPOLOGICAL SORT WITH CYCLE DETECTION
# ============================================================

def topological_sort(
    num_vertices: int,
    edges: List[Tuple[int, int]],
) -> Tuple[Optional[List[int]], bool]:
    """
    Topological sort using Kahn's algorithm with cycle detection.
    
    Args:
        num_vertices: Number of vertices
        edges: List of (from, to) directed edges
        
    Returns:
        (sorted order or None, has_cycle)
    """
    # Build adjacency list and in-degree count
    adj: List[List[int]] = [[] for _ in range(num_vertices)]
    in_degree = [0] * num_vertices
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    
    # Start with vertices having no incoming edges
    queue = deque([v for v in range(num_vertices) if in_degree[v] == 0])
    order = []
    
    while queue:
        u = queue.popleft()
        order.append(u)
        
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    has_cycle = len(order) != num_vertices
    return (None if has_cycle else order, has_cycle)


def topological_sort_all_orders(
    num_vertices: int,
    edges: List[Tuple[int, int]],
    max_orders: int = 100,
) -> List[List[int]]:
    """
    Find all possible topological orderings (up to max_orders).
    Uses backtracking.
    """
    adj: List[List[int]] = [[] for _ in range(num_vertices)]
    in_degree = [0] * num_vertices
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    
    results = []
    
    def backtrack(current_order: List[int], current_in_degree: List[int]):
        if len(results) >= max_orders:
            return
        
        if len(current_order) == num_vertices:
            results.append(current_order[:])
            return
        
        for v in range(num_vertices):
            if v not in current_order and current_in_degree[v] == 0:
                current_order.append(v)
                new_in_degree = current_in_degree[:]
                for w in adj[v]:
                    new_in_degree[w] -= 1
                
                backtrack(current_order, new_in_degree)
                
                current_order.pop()
    
    backtrack([], in_degree)
    return results


# ============================================================
# LOUVAIN COMMUNITY DETECTION
# ============================================================

class LouvainCommunityDetection:
    """
    Louvain method for community detection in networks.
    
    Optimizes modularity through iterative local moves and aggregation.
    """
    
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
    
    def detect(self, edges: List[Tuple[int, int, float]]) -> Dict[int, int]:
        """
        Detect communities in a weighted graph.
        
        Args:
            edges: List of (node1, node2, weight) tuples
            
        Returns:
            Dictionary mapping node -> community_id
        """
        # Build adjacency
        adj: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        nodes: Set[int] = set()
        
        for u, v, w in edges:
            adj[u][v] += w
            adj[v][u] += w
            nodes.add(u)
            nodes.add(v)
        
        # Initialize: each node in its own community
        community = {node: node for node in nodes}
        
        # Total edge weight
        m = sum(w for _, _, w in edges) / 2.0  # Each edge counted twice
        
        # Node strengths (sum of edge weights)
        strength = {node: sum(adj[node].values()) for node in nodes}
        
        improved = True
        iteration = 0
        
        while improved:
            improved = False
            iteration += 1
            
            # Phase 1: Local moving
            for node in random.sample(list(nodes), len(nodes)):
                current_comm = community[node]
                
                # Calculate modularity gain for each neighboring community
                neighbor_comms: Dict[int, float] = defaultdict(float)
                for neighbor, weight in adj[node].items():
                    neighbor_comms[community[neighbor]] += weight
                
                # Community strengths
                comm_strength: Dict[int, float] = defaultdict(float)
                for n in nodes:
                    comm_strength[community[n]] += strength[n]
                
                # Find best community
                best_comm = current_comm
                best_gain = 0.0
                
                ki = strength[node]
                
                for comm, ki_in in neighbor_comms.items():
                    if comm == current_comm:
                        continue
                    
                    sigma_tot = comm_strength[comm]
                    
                    # Modularity gain
                    gain = (ki_in / m - self.resolution * sigma_tot * ki / (2 * m * m))
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm
                
                if best_comm != current_comm:
                    community[node] = best_comm
                    improved = True
        
        # Renumber communities
        unique_comms = list(set(community.values()))
        comm_map = {old: new for new, old in enumerate(unique_comms)}
        return {node: comm_map[comm] for node, comm in community.items()}
    
    def modularity(self, edges: List[Tuple[int, int, float]], 
                   communities: Dict[int, int]) -> float:
        """Calculate modularity Q of the partition."""
        m = sum(w for _, _, w in edges) / 2.0
        if m == 0:
            return 0.0
        
        strength: Dict[int, float] = defaultdict(float)
        for u, v, w in edges:
            strength[u] += w
            strength[v] += w
        
        q = 0.0
        for u, v, w in edges:
            if communities.get(u) == communities.get(v):
                q += w - (strength[u] * strength[v]) / (2 * m)
        
        return q / m


if __name__ == "__main__":
    # Max Flow
    print("=== Maximum Flow ===")
    net = FlowNetwork(6)
    flow_edges = [(0,1,16),(0,2,13),(1,2,10),(1,3,12),(2,1,4),(2,4,14),(3,2,9),(3,5,20),(4,3,7),(4,5,4)]
    for u, v, c in flow_edges:
        net.add_edge(u, v, c)
    print(f"Max flow (0->5): {net.max_flow(0, 5)}")
    
    # Hungarian
    print("\\n=== Hungarian Algorithm ===")
    costs = [[4, 1, 3], [2, 0, 5], [3, 2, 1]]
    assignment, cost = hungarian_algorithm(costs)
    print(f"Optimal assignment: {assignment}, total cost: {cost}")
    
    # Topological Sort
    print("\\n=== Topological Sort ===")
    edges = [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]
    order, has_cycle = topological_sort(6, edges)
    print(f"Order: {order}, has_cycle: {has_cycle}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Advanced graph algorithms: max flow, Hungarian, topological sort, Louvain",



            complexity="expert",



            domain="algorithms",



            tags=["max_flow", "hungarian", "topological_sort", "community_detection", "graph"],



        )







    def _expert_systems(self) -> CodeSample:



        code = '''"""
Production-Grade Database Connection Pool and ORM.

Implements:
- Connection pooling with health checks
- Simple ORM with query builder
- Transaction management
- Migration system
- Connection retry with backoff
"""

import threading
import time
import queue
import uuid
import re
from typing import (
    Dict, List, Any, Optional, Tuple, Type, Callable, Set
)
from dataclasses import dataclass, field, fields
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from enum import Enum
import logging
import json
import copy

logger = logging.getLogger(__name__)


# ============================================================
# CONNECTION POOL
# ============================================================

class ConnectionState(Enum):
    IDLE = "idle"
    IN_USE = "in_use"
    BROKEN = "broken"
    CLOSED = "closed"


@dataclass
class PooledConnection:
    """A single database connection wrapper."""
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: ConnectionState = ConnectionState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    query_count: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if connection is still healthy."""
        return self.state != ConnectionState.BROKEN
    
    def mark_used(self) -> None:
        self.last_used_at = time.time()
        self.query_count += 1


class ConnectionPool:
    """
    Thread-safe database connection pool.
    
    Features:
    - Configurable min/max pool size
    - Connection health checking
    - Idle connection eviction
    - Connection timeout handling
    - Retry with exponential backoff
    """
    
    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 20,
        max_idle_time: float = 300.0,
        health_check_interval: float = 30.0,
        connection_timeout: float = 10.0,
        max_lifetime: float = 3600.0,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.connection_timeout = connection_timeout
        self.max_lifetime = max_lifetime
        
        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._all_connections: Set[PooledConnection] = set()
        self._lock = threading.Lock()
        self._initialized = False
        
        # Metrics
        self._total_acquired = 0
        self._total_released = 0
        self._total_created = 0
        self._total_failed = 0
    
    def initialize(self) -> None:
        """Initialize pool with minimum connections."""
        with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.min_size):
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn)
            
            self._initialized = True
            logger.info(f"Connection pool initialized: {self.min_size} connections")
    
    def acquire(self, timeout: float = None) -> Optional[PooledConnection]:
        """
        Acquire a connection from the pool.
        
        Args:
            timeout: Maximum time to wait (default: connection_timeout)
            
        Returns:
            PooledConnection or None if timeout
        """
        if not self._initialized:
            self.initialize()
        
        timeout = timeout or self.connection_timeout
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            # Try to get existing connection
            try:
                conn = self._pool.get(timeout=0.1)
                
                if conn.is_healthy():
                    conn.state = ConnectionState.IN_USE
                    conn.mark_used()
                    self._total_acquired += 1
                    return conn
                else:
                    # Remove broken connection
                    self._destroy_connection(conn)
            except queue.Empty:
                pass
            
            # Try to create new connection
            with self._lock:
                if len(self._all_connections) < self.max_size:
                    conn = self._create_connection()
                    if conn:
                        conn.state = ConnectionState.IN_USE
                        self._total_acquired += 1
                        return conn
            
            time.sleep(0.05)
        
        self._total_failed += 1
        logger.warning(f"Connection acquisition timed out after {timeout}s")
        return None
    
    def release(self, conn: PooledConnection) -> None:
        """Return a connection to the pool."""
        if conn is None:
            return
        
        conn.state = ConnectionState.IDLE
        conn.last_used_at = time.time()
        self._total_released += 1
        
        if conn.is_healthy():
            try:
                self._pool.put_nowait(conn)
            except queue.Full:
                self._destroy_connection(conn)
        else:
            self._destroy_connection(conn)
    
    @contextmanager
    def connection(self, timeout: float = None):
        """Context manager for automatic connection management."""
        conn = self.acquire(timeout)
        if conn is None:
            raise RuntimeError("Could not acquire database connection")
        try:
            yield conn
        finally:
            self.release(conn)
    
    def _create_connection(self) -> Optional[PooledConnection]:
        """Create a new database connection."""
        try:
            conn = PooledConnection()
            self._all_connections.add(conn)
            self._total_created += 1
            return conn
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self._total_failed += 1
            return None
    
    def _destroy_connection(self, conn: PooledConnection) -> None:
        """Destroy a connection."""
        conn.state = ConnectionState.CLOSED
        with self._lock:
            self._all_connections.discard(conn)
    
    def health_check(self) -> Dict[str, int]:
        """Run health checks on idle connections."""
        healthy = 0
        removed = 0
        now = time.time()
        
        remaining = []
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                
                # Check age and idle time
                age = now - conn.created_at
                idle = now - conn.last_used_at
                
                if age > self.max_lifetime or idle > self.max_idle_time:
                    if len(self._all_connections) > self.min_size:
                        self._destroy_connection(conn)
                        removed += 1
                        continue
                
                if conn.is_healthy():
                    conn.last_health_check = now
                    healthy += 1
                    remaining.append(conn)
                else:
                    self._destroy_connection(conn)
                    removed += 1
            except queue.Empty:
                break
        
        for conn in remaining:
            self._pool.put(conn)
        
        return {"healthy": healthy, "removed": removed}
    
    @property
    def metrics(self) -> Dict[str, Any]:
        return {
            "pool_size": len(self._all_connections),
            "available": self._pool.qsize(),
            "total_created": self._total_created,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "total_failed": self._total_failed,
        }
    
    def close_all(self) -> None:
        """Close all connections."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                self._destroy_connection(conn)
            except queue.Empty:
                break
        
        with self._lock:
            for conn in list(self._all_connections):
                self._destroy_connection(conn)


# ============================================================
# SIMPLE ORM
# ============================================================

class QueryBuilder:
    """Fluent SQL query builder."""
    
    def __init__(self, table: str):
        self._table = table
        self._select_cols: List[str] = ["*"]
        self._where_clauses: List[str] = []
        self._where_params: List[Any] = []
        self._order_by: Optional[str] = None
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._joins: List[str] = []
    
    def select(self, *columns: str) -> 'QueryBuilder':
        self._select_cols = list(columns) if columns else ["*"]
        return self
    
    def where(self, condition: str, *params: Any) -> 'QueryBuilder':
        self._where_clauses.append(condition)
        self._where_params.extend(params)
        return self
    
    def order_by(self, column: str, direction: str = "ASC") -> 'QueryBuilder':
        self._order_by = f"{column} {direction}"
        return self
    
    def limit(self, n: int) -> 'QueryBuilder':
        self._limit = n
        return self
    
    def offset(self, n: int) -> 'QueryBuilder':
        self._offset = n
        return self
    
    def join(self, table: str, on: str, join_type: str = "INNER") -> 'QueryBuilder':
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self
    
    def build(self) -> Tuple[str, List[Any]]:
        """Build the SQL query string."""
        parts = [f"SELECT {', '.join(self._select_cols)} FROM {self._table}"]
        
        for j in self._joins:
            parts.append(j)
        
        if self._where_clauses:
            parts.append("WHERE " + " AND ".join(self._where_clauses))
        
        if self._order_by:
            parts.append(f"ORDER BY {self._order_by}")
        
        if self._limit is not None:
            parts.append(f"LIMIT {self._limit}")
        
        if self._offset is not None:
            parts.append(f"OFFSET {self._offset}")
        
        return " ".join(parts), self._where_params


class Model:
    """Base ORM model class."""
    
    _table: str = ""
    _primary_key: str = "id"
    _pool: Optional[ConnectionPool] = None
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def query(cls) -> QueryBuilder:
        return QueryBuilder(cls._table)
    
    @classmethod
    def find(cls, id: Any) -> Optional['Model']:
        """Find a record by primary key."""
        qb = cls.query().where(f"{cls._primary_key} = ?", id).limit(1)
        sql, params = qb.build()
        # In a real implementation, this would execute the SQL
        return None
    
    @classmethod
    def find_all(cls, **filters) -> List['Model']:
        """Find all records matching filters."""
        qb = cls.query()
        for key, value in filters.items():
            qb.where(f"{key} = ?", value)
        return []
    
    @classmethod
    def create(cls, **kwargs) -> 'Model':
        """Create a new record."""
        instance = cls(**kwargs)
        return instance
    
    def save(self) -> 'Model':
        """Save (insert or update) the record."""
        return self
    
    def delete(self) -> bool:
        """Delete the record."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items())
        return f"{self.__class__.__name__}({attrs})"


# Example models
class User(Model):
    _table = "users"
    _primary_key = "id"


class Product(Model):
    _table = "products"
    _primary_key = "id"


class Order(Model):
    _table = "orders"
    _primary_key = "id"


# ============================================================
# MIGRATION SYSTEM
# ============================================================

class Migration:
    """Database migration."""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.up_statements: List[str] = []
        self.down_statements: List[str] = []
    
    def up(self, sql: str) -> 'Migration':
        self.up_statements.append(sql)
        return self
    
    def down(self, sql: str) -> 'Migration':
        self.down_statements.append(sql)
        return self


class MigrationRunner:
    """Runs database migrations in order."""
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self.applied: Set[str] = set()
    
    def register(self, migration: Migration) -> None:
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    def migrate_up(self, target: str = None) -> List[str]:
        """Run pending migrations up to target version."""
        applied = []
        for m in self.migrations:
            if m.version in self.applied:
                continue
            if target and m.version > target:
                break
            
            for sql in m.up_statements:
                applied.append(f"[{m.version}] {sql}")
            
            self.applied.add(m.version)
        
        return applied
    
    def migrate_down(self, target: str) -> List[str]:
        """Roll back migrations down to target version."""
        rolled_back = []
        for m in reversed(self.migrations):
            if m.version <= target:
                break
            if m.version not in self.applied:
                continue
            
            for sql in m.down_statements:
                rolled_back.append(f"[{m.version}] {sql}")
            
            self.applied.discard(m.version)
        
        return rolled_back


if __name__ == "__main__":
    # Connection Pool
    pool = ConnectionPool(min_size=3, max_size=10)
    pool.initialize()
    
    print("Pool metrics:", pool.metrics)
    
    with pool.connection() as conn:
        print(f"Acquired: {conn.connection_id[:8]}...")
        conn.mark_used()
    
    print("After release:", pool.metrics)
    print("Health check:", pool.health_check())
    
    # Query Builder
    sql, params = (
        QueryBuilder("users")
        .select("id", "name", "email")
        .where("age > ?", 18)
        .where("active = ?", True)
        .join("orders", "users.id = orders.user_id", "LEFT")
        .order_by("name", "ASC")
        .limit(10)
        .offset(20)
        .build()
    )
    print(f"\\nGenerated SQL:\\n{sql}")
    print(f"Params: {params}")
    
    # Migrations
    runner = MigrationRunner()
    runner.register(
        Migration("001", "Create users table")
        .up("CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT, email TEXT)")
        .down("DROP TABLE users")
    )
    runner.register(
        Migration("002", "Add age column")
        .up("ALTER TABLE users ADD COLUMN age INTEGER")
        .down("ALTER TABLE users DROP COLUMN age")
    )
    
    applied = runner.migrate_up()
    print(f"\\nApplied migrations:")
    for a in applied:
        print(f"  {a}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Production connection pool, ORM with query builder, and migration system",



            complexity="expert",



            domain="systems",



            tags=["connection_pool", "orm", "query_builder", "migrations", "database"],



        )







    def _expert_data(self) -> CodeSample:



        code = '''"""
Neural Network from Scratch - Complete Implementation with Autograd.

Builds a neural network framework entirely from NumPy:
- Automatic differentiation engine
- Tensor operations with gradient tracking
- Common layers (Linear, Conv2d, BatchNorm, Dropout)
- Activation functions
- Loss functions
- Optimizers (SGD, Adam, AdamW)
- Learning rate schedulers
- Training loop with validation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import math
import pickle
import os


# ============================================================
# TENSOR WITH AUTOGRAD
# ============================================================

class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.
    
    Tracks computation graph for backpropagation.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = '',
    ):
        self.data = np.array(data, dtype=np.float64) if not isinstance(data, np.ndarray) else data.astype(np.float64)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        
        # Autograd
        self._children = _children
        self._op = _op
        self._backward = lambda: None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, grad={self.grad is not None})"
    
    # ======== ARITHMETIC OPERATIONS ========
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data, 
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad, other.shape)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad += self._unbroadcast(out.grad * self.data, other.shape)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, power: float) -> 'Tensor':
        out = Tensor(self.data ** power,
                     requires_grad=self.requires_grad,
                     _children=(self,), _op=f'**{power}')
        
        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> 'Tensor':
        return self * Tensor(np.array(-1.0))
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        return self + (-other if isinstance(other, Tensor) else Tensor(np.array(-other)))
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        return self * (other ** -1 if isinstance(other, Tensor) else Tensor(np.array(1.0 / other)))
    
    def __radd__(self, other) -> 'Tensor': return self.__add__(other)
    def __rmul__(self, other) -> 'Tensor': return self.__mul__(other)
    def __rsub__(self, other) -> 'Tensor': return Tensor(np.array(other)) + (-self)
    
    # ======== ACTIVATION FUNCTIONS ========
    
    def relu(self) -> 'Tensor':
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(float) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> 'Tensor':
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op='sigmoid')
        
        def _backward():
            if self.requires_grad:
                self.grad += s * (1 - s) * out.grad
        
        out._backward = _backward
        return out
    
    def tanh(self) -> 'Tensor':
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad,
                     _children=(self,), _op='tanh')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def gelu(self) -> 'Tensor':
        """Gaussian Error Linear Unit."""
        x = self.data
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
        out_data = x * cdf
        out = Tensor(out_data, requires_grad=self.requires_grad,
                     _children=(self,), _op='gelu')
        
        def _backward():
            if self.requires_grad:
                pdf = np.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
                self.grad += (cdf + x * pdf) * out.grad
        
        out._backward = _backward
        return out
    
    def softmax(self, axis: int = -1) -> 'Tensor':
        """Numerically stable softmax."""
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        s = exp / np.sum(exp, axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad,
                     _children=(self,), _op='softmax')
        
        def _backward():
            if self.requires_grad:
                # Jacobian-vector product for softmax
                grad_input = s * (out.grad - np.sum(out.grad * s, axis=axis, keepdims=True))
                self.grad += grad_input
        
        out._backward = _backward
        return out
    
    def log(self) -> 'Tensor':
        out = Tensor(np.log(self.data + 1e-12),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='log')
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad / (self.data + 1e-12)
        
        out._backward = _backward
        return out
    
    def sum(self, axis: int = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad,
                     _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                if axis is not None and not keepdims:
                    grad = np.expand_dims(out.grad, axis)
                else:
                    grad = out.grad
                self.grad += np.broadcast_to(grad, self.shape)
        
        out._backward = _backward
        return out
    
    def mean(self, axis: int = None) -> 'Tensor':
        s = self.sum(axis=axis)
        n = self.data.size if axis is None else self.data.shape[axis]
        return s * Tensor(np.array(1.0 / n))
    
    # ======== BACKPROPAGATION ========
    
    def backward(self) -> None:
        """Compute gradients via backpropagation."""
        topo = []
        visited = set()
        
        def build_topo(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Set output gradient to 1
        self.grad = np.ones_like(self.data)
        
        # Backward pass in reverse topological order
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self) -> None:
        """Reset gradients to zero."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
    
    def _unbroadcast(self, grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """Handle gradient for broadcasted operations."""
        # Add dimensions
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        
        # Sum over broadcast dimensions
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    # ======== UTILITIES ========
    
    def numpy(self) -> np.ndarray:
        return self.data.copy()
    
    def item(self) -> float:
        return float(self.data)
    
    @staticmethod
    def zeros(*shape, requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape, requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape, requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


# ============================================================
# LAYERS
# ============================================================

class Linear:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int):
        # Kaiming initialization
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias
    
    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class BatchNorm:
    """Batch normalization layer."""
    
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = Tensor((x.data - mean) / np.sqrt(var + self.eps), requires_grad=x.requires_grad)
        return x_norm * self.gamma + self.beta
    
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Dropout:
    """Dropout regularization layer."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
    
    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.rand(*x.shape) > self.p).astype(float) / (1 - self.p)
        return Tensor(x.data * mask, requires_grad=x.requires_grad, _children=(x,), _op='dropout')
    
    def parameters(self) -> List[Tensor]:
        return []


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Cross-entropy loss for classification."""
    probs = logits.softmax(axis=-1)
    batch_size = logits.shape[0]
    
    # Negative log-likelihood
    log_probs = probs.log()
    target_probs = np.zeros_like(probs.data)
    target_probs[np.arange(batch_size), targets.astype(int)] = 1.0
    
    loss = Tensor(-np.sum(log_probs.data * target_probs) / batch_size,
                  requires_grad=True, _children=(log_probs,), _op='ce_loss')
    
    def _backward():
        grad = (probs.data - target_probs) / batch_size
        logits.grad = grad
    
    loss._backward = _backward
    return loss


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error loss."""
    diff = predictions - targets
    return (diff ** 2).mean()


# ============================================================
# OPTIMIZERS
# ============================================================

class SGD:
    """Stochastic Gradient Descent with momentum."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in parameters]
    
    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            
            self.velocities[i] = self.momentum * self.velocities[i] + grad
            p.data -= self.lr * self.velocities[i]
    
    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()


class Adam:
    """Adam optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
    
    def step(self) -> None:
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            
            # Update biased first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()


class AdamW(Adam):
    """AdamW optimizer with decoupled weight decay."""
    
    def step(self) -> None:
        self.t += 1
        
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # Decoupled weight decay
            if self.weight_decay > 0:
                p.data -= self.lr * self.weight_decay * p.data
            
            grad = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler with warm restarts."""
    
    def __init__(self, optimizer, max_lr: float, min_lr: float = 0.0,
                 warmup_steps: int = 100, total_steps: int = 10000):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_count = 0
    
    def step(self) -> float:
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.max_lr - self.min_lr) * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        self.optimizer.lr = lr
        return lr


# ============================================================
# NEURAL NETWORK MODEL
# ============================================================

class NeuralNetwork:
    """Multi-layer neural network."""
    
    def __init__(self, layer_sizes: List[int], dropout: float = 0.1):
        self.layers = []
        self.norms = []
        self.dropouts = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No norm/dropout on output layer
                self.norms.append(BatchNorm(layer_sizes[i + 1]))
                self.dropouts.append(Dropout(dropout))
            else:
                self.norms.append(None)
                self.dropouts.append(None)
    
    def __call__(self, x: Tensor) -> Tensor:
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            x = layer(x)
            if norm is not None:
                x = norm(x)
                x = x.gelu()
                if dropout is not None:
                    x = dropout(x)
        return x
    
    def parameters(self) -> List[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        for norm in self.norms:
            if norm is not None:
                params.extend(norm.parameters())
        return params
    
    def train(self):
        for norm in self.norms:
            if norm: norm.training = True
        for d in self.dropouts:
            if d: d.training = True
    
    def eval(self):
        for norm in self.norms:
            if norm: norm.training = False
        for d in self.dropouts:
            if d: d.training = False


if __name__ == "__main__":
    # Create a simple classification problem
    np.random.seed(42)
    
    # Generate spiral dataset
    n_samples = 500
    n_classes = 3
    
    X_data = np.zeros((n_samples * n_classes, 2))
    y_data = np.zeros(n_samples * n_classes, dtype=int)
    
    for c in range(n_classes):
        ix = range(n_samples * c, n_samples * (c + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(c * 4, (c + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X_data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y_data[ix] = c
    
    # Model
    model = NeuralNetwork([2, 64, 64, n_classes], dropout=0.1)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, max_lr=0.01, warmup_steps=50, total_steps=1000)
    
    # Training loop
    print("Training neural network on spiral dataset...")
    for epoch in range(100):
        model.train()
        
        # Forward pass
        x = Tensor(X_data, requires_grad=False)
        logits = model(x)
        loss = cross_entropy_loss(logits, y_data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        lr = scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            # Compute accuracy
            model.eval()
            logits = model(Tensor(X_data))
            predictions = np.argmax(logits.data, axis=1)
            accuracy = np.mean(predictions == y_data)
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {accuracy:.4f} | LR: {lr:.6f}")'''







        return CodeSample(



            language="python",



            code=code,



            description="Neural network from scratch with autograd, layers, optimizers, and training loop",



            complexity="expert",



            domain="data",



            tags=["neural_network", "autograd", "deep_learning", "adam", "backpropagation"],



        )



