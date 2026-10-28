"""
Main Dataset Generator - Orchestrates Creation of the Full Training Dataset.

Generates a massive, diverse code dataset from scratch using:
1. Code template engine (multi-language code samples)
2. Instruction-following data
3. Combinatorial expansion for scale
4. Fill-in-the-middle (FIM) training data
5. Repository-level context data

Target: 300B+ tokens for training a 20B parameter model.
"""















import os







import json







import random







import hashlib







import logging







from typing import List, Dict, Any, Optional







from pathlib import Path







from dataclasses import asdict















from .code_templates import (







    CodeSample,







    PythonCodeGenerator, NameGenerator,







)







from .instruction_data import InstructionGenerator, InstructionSample







from .processor import DatasetProcessor, DatasetConfig















logger = logging.getLogger(__name__)























class JavaScriptCodeGenerator:







    """Generate JavaScript/TypeScript code samples."""















    SAMPLES = [







        {







            "language": "javascript",







            "code": '''/**
 * Debounce function - delays execution until after wait milliseconds
 * have elapsed since the last invocation.
 */
function debounce(func, wait, options = {}) {
    let timeoutId = null;
    let lastCallTime = 0;
    let lastArgs = null;
    const { leading = false, trailing = true, maxWait } = options;
    
    function invoke() {
        const args = lastArgs;
        lastArgs = null;
        return func(...args);
    }
    
    function debounced(...args) {
        const now = Date.now();
        const timeSinceLastCall = now - lastCallTime;
        lastArgs = args;
        lastCallTime = now;
        
        const shouldCallLeading = leading && timeSinceLastCall >= wait;
        
        if (timeoutId !== null) {
            clearTimeout(timeoutId);
        }
        
        if (shouldCallLeading) {
            invoke();
        }
        
        if (trailing) {
            timeoutId = setTimeout(() => {
                timeoutId = null;
                if (lastArgs) invoke();
            }, wait);
        }
        
        if (maxWait && timeSinceLastCall >= maxWait) {
            invoke();
        }
    }
    
    debounced.cancel = () => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = null;
        lastArgs = null;
    };
    
    return debounced;
}''',







            "description": "Advanced debounce function with leading/trailing edge options",







            "complexity": "intermediate",







            "domain": "web",







        },







        {







            "language": "typescript",







            "code": '''/**
 * Type-safe event emitter with generics.
 */
type EventMap = Record<string, unknown[]>;
type EventCallback<T extends unknown[]> = (...args: T) => void;

class TypedEventEmitter<Events extends EventMap> {
    private listeners: Map<keyof Events, Set<EventCallback<any>>> = new Map();
    
    on<K extends keyof Events>(event: K, callback: EventCallback<Events[K]>): () => void {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event)!.add(callback);
        
        // Return unsubscribe function
        return () => this.off(event, callback);
    }
    
    once<K extends keyof Events>(event: K, callback: EventCallback<Events[K]>): void {
        const wrapper: EventCallback<Events[K]> = (...args) => {
            this.off(event, wrapper);
            callback(...args);
        };
        this.on(event, wrapper);
    }
    
    off<K extends keyof Events>(event: K, callback: EventCallback<Events[K]>): void {
        this.listeners.get(event)?.delete(callback);
    }
    
    emit<K extends keyof Events>(event: K, ...args: Events[K]): void {
        this.listeners.get(event)?.forEach(callback => {
            try {
                callback(...args);
            } catch (error) {
                console.error(`Error in event handler for ${String(event)}:`, error);
            }
        });
    }
    
    removeAllListeners(event?: keyof Events): void {
        if (event) {
            this.listeners.delete(event);
        } else {
            this.listeners.clear();
        }
    }
}

// Usage with full type safety
interface AppEvents {
    'user:login': [userId: string, timestamp: Date];
    'user:logout': [userId: string];
    'error': [error: Error, context: string];
}

const emitter = new TypedEventEmitter<AppEvents>();

// TypeScript enforces correct argument types
const unsub = emitter.on('user:login', (userId, timestamp) => {
    console.log(`User ${userId} logged in at ${timestamp.toISOString()}`);
});''',







            "description": "Type-safe event emitter with generics in TypeScript",







            "complexity": "advanced",







            "domain": "web",







        },







        {







            "language": "rust",







            "code": '''use std::collections::HashMap;
use std::hash::Hash;

/// A thread-safe LRU cache implementation using a doubly-linked list
/// and a hash map for O(1) operations.
pub struct LRUCache<K: Hash + Eq + Clone, V: Clone> {
    capacity: usize,
    map: HashMap<K, usize>,  // key -> node index
    nodes: Vec<LRUNode<K, V>>,
    head: usize,  // most recently used
    tail: usize,  // least recently used
    free: Vec<usize>,  // free node indices
}

struct LRUNode<K, V> {
    key: Option<K>,
    value: Option<V>,
    prev: usize,
    next: usize,
}

const NULL: usize = usize::MAX;

impl<K: Hash + Eq + Clone, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be > 0");
        
        let mut nodes = Vec::with_capacity(capacity + 2);
        
        // Sentinel head node
        nodes.push(LRUNode {
            key: None, value: None,
            prev: NULL, next: 1,
        });
        
        // Sentinel tail node
        nodes.push(LRUNode {
            key: None, value: None,
            prev: 0, next: NULL,
        });
        
        LRUCache {
            capacity,
            map: HashMap::with_capacity(capacity),
            nodes,
            head: 0,
            tail: 1,
            free: Vec::new(),
        }
    }
    
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            self.move_to_front(idx);
            self.nodes[idx].value.as_ref()
        } else {
            None
        }
    }
    
    pub fn put(&mut self, key: K, value: V) {
        if let Some(&idx) = self.map.get(&key) {
            // Update existing
            self.nodes[idx].value = Some(value);
            self.move_to_front(idx);
        } else {
            // Insert new
            if self.map.len() >= self.capacity {
                self.evict_lru();
            }
            
            let idx = self.alloc_node(key.clone(), value);
            self.map.insert(key, idx);
            self.link_after_head(idx);
        }
    }
    
    fn move_to_front(&mut self, idx: usize) {
        self.unlink(idx);
        self.link_after_head(idx);
    }
    
    fn evict_lru(&mut self) {
        let lru = self.nodes[self.tail].prev;
        if lru != self.head {
            self.unlink(lru);
            if let Some(key) = self.nodes[lru].key.take() {
                self.map.remove(&key);
            }
            self.nodes[lru].value = None;
            self.free.push(lru);
        }
    }
    
    fn alloc_node(&mut self, key: K, value: V) -> usize {
        if let Some(idx) = self.free.pop() {
            self.nodes[idx] = LRUNode {
                key: Some(key), value: Some(value),
                prev: NULL, next: NULL,
            };
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(LRUNode {
                key: Some(key), value: Some(value),
                prev: NULL, next: NULL,
            });
            idx
        }
    }
    
    fn link_after_head(&mut self, idx: usize) {
        let old_first = self.nodes[self.head].next;
        self.nodes[idx].next = old_first;
        self.nodes[idx].prev = self.head;
        self.nodes[old_first].prev = idx;
        self.nodes[self.head].next = idx;
    }
    
    fn unlink(&mut self, idx: usize) {
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;
        if prev != NULL { self.nodes[prev].next = next; }
        if next != NULL { self.nodes[next].prev = prev; }
    }
    
    pub fn len(&self) -> usize { self.map.len() }
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
}''',







            "description": "LRU Cache implementation in Rust with doubly-linked list",







            "complexity": "advanced",







            "domain": "systems",







        },







        {







            "language": "go",







            "code": '''package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// WorkerPool manages a pool of goroutines for processing jobs.
type WorkerPool struct {
	numWorkers int
	jobs       chan Job
	results    chan Result
	wg         sync.WaitGroup
	ctx        context.Context
	cancel     context.CancelFunc
}

// Job represents a unit of work.
type Job struct {
	ID      int
	Payload interface{}
	Handler func(interface{}) (interface{}, error)
}

// Result represents the outcome of a job.
type Result struct {
	JobID int
	Value interface{}
	Err   error
}

// NewWorkerPool creates a new worker pool.
func NewWorkerPool(numWorkers, bufferSize int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	return &WorkerPool{
		numWorkers: numWorkers,
		jobs:       make(chan Job, bufferSize),
		results:    make(chan Result, bufferSize),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start launches worker goroutines.
func (wp *WorkerPool) Start() {
	for i := 0; i < wp.numWorkers; i++ {
		wp.wg.Add(1)
		go wp.worker(i)
	}
}

func (wp *WorkerPool) worker(id int) {
	defer wp.wg.Done()

	for {
		select {
		case <-wp.ctx.Done():
			fmt.Printf("Worker %d shutting down\\n", id)
			return
		case job, ok := <-wp.jobs:
			if !ok {
				return
			}

			value, err := job.Handler(job.Payload)
			result := Result{
				JobID: job.ID,
				Value: value,
				Err:   err,
			}

			select {
			case wp.results <- result:
			case <-wp.ctx.Done():
				return
			}
		}
	}
}

// Submit adds a job to the pool.
func (wp *WorkerPool) Submit(job Job) {
	select {
	case wp.jobs <- job:
	case <-wp.ctx.Done():
	}
}

// Results returns the results channel.
func (wp *WorkerPool) Results() <-chan Result {
	return wp.results
}

// Stop gracefully shuts down the worker pool.
func (wp *WorkerPool) Stop() {
	close(wp.jobs)
	wp.wg.Wait()
	wp.cancel()
	close(wp.results)
}

// Example usage
func main() {
	pool := NewWorkerPool(4, 100)
	pool.Start()

	// Submit jobs
	for i := 0; i < 20; i++ {
		jobID := i
		pool.Submit(Job{
			ID: jobID,
			Payload: jobID * 10,
			Handler: func(payload interface{}) (interface{}, error) {
				val := payload.(int)
				time.Sleep(100 * time.Millisecond)
				return val * val, nil
			},
		})
	}

	// Collect results
	go func() {
		for result := range pool.Results() {
			if result.Err != nil {
				fmt.Printf("Job %d failed: %v\\n", result.JobID, result.Err)
			} else {
				fmt.Printf("Job %d result: %v\\n", result.JobID, result.Value)
			}
		}
	}()

	// Wait and stop
	time.Sleep(3 * time.Second)
	pool.Stop()
}''',







            "description": "Concurrent worker pool in Go with graceful shutdown",







            "complexity": "intermediate",







            "domain": "systems",







        },







        {







            "language": "cpp",







            "code": '''#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <variant>

/**
 * Compile-time type-safe Result type (similar to Rust's Result<T, E>)
 * Uses std::variant for zero-cost abstraction.
 */
template<typename T, typename E = std::string>
class Result {
    std::variant<T, E> data;
    bool is_ok;

public:
    static Result Ok(T value) {
        Result r;
        r.data = std::move(value);
        r.is_ok = true;
        return r;
    }

    static Result Err(E error) {
        Result r;
        r.data = std::move(error);
        r.is_ok = false;
        return r;
    }

    bool ok() const { return is_ok; }
    bool is_err() const { return !is_ok; }

    T unwrap() const {
        if (!is_ok) throw std::runtime_error("Called unwrap on Err");
        return std::get<T>(data);
    }

    T unwrap_or(T default_val) const {
        return is_ok ? std::get<T>(data) : default_val;
    }

    template<typename F>
    auto map(F&& f) const -> Result<decltype(f(std::declval<T>())), E> {
        using U = decltype(f(std::declval<T>()));
        if (is_ok) {
            return Result<U, E>::Ok(f(std::get<T>(data)));
        }
        return Result<U, E>::Err(std::get<E>(data));
    }

    template<typename F>
    auto and_then(F&& f) const -> decltype(f(std::declval<T>())) {
        if (is_ok) {
            return f(std::get<T>(data));
        }
        using ReturnType = decltype(f(std::declval<T>()));
        return ReturnType::Err(std::get<E>(data));
    }
};

// RAII-based resource management
template<typename T>
class ScopeGuard {
    T cleanup;
    bool active;

public:
    explicit ScopeGuard(T&& fn) : cleanup(std::move(fn)), active(true) {}
    
    ~ScopeGuard() {
        if (active) cleanup();
    }

    void dismiss() { active = false; }

    // Non-copyable, movable
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup(std::move(other.cleanup)), active(other.active) {
        other.active = false;
    }
};

// Smart pointer-based linked list with move semantics
template<typename T>
class LinkedList {
    struct Node {
        T data;
        std::unique_ptr<Node> next;
        
        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::unique_ptr<Node> head;
    size_t count = 0;

public:
    void push_front(T value) {
        auto node = std::make_unique<Node>(std::move(value));
        node->next = std::move(head);
        head = std::move(node);
        ++count;
    }

    std::optional<T> pop_front() {
        if (!head) return std::nullopt;
        
        T value = std::move(head->data);
        head = std::move(head->next);
        --count;
        return value;
    }

    void for_each(std::function<void(const T&)> fn) const {
        for (auto* node = head.get(); node; node = node->next.get()) {
            fn(node->data);
        }
    }

    [[nodiscard]] size_t size() const { return count; }
    [[nodiscard]] bool empty() const { return count == 0; }
};

int main() {
    // Result type usage
    auto divide = [](int a, int b) -> Result<double> {
        if (b == 0) return Result<double>::Err("Division by zero");
        return Result<double>::Ok(static_cast<double>(a) / b);
    };

    auto result = divide(10, 3)
        .map([](double v) { return v * 2; })
        .unwrap_or(0.0);
    
    std::cout << "Result: " << result << std::endl;

    // LinkedList usage
    LinkedList<std::string> list;
    list.push_front("World");
    list.push_front("Hello");

    list.for_each([](const std::string& s) {
        std::cout << s << " ";
    });
    std::cout << std::endl;

    return 0;
}''',







            "description": "Modern C++ with Result types, RAII, smart pointers, and move semantics",







            "complexity": "advanced",







            "domain": "systems",







        },







    ]















    def generate(self) -> List[CodeSample]:







        return [







            CodeSample(







                language=s["language"],







                code=s["code"],







                description=s["description"],







                complexity=s["complexity"],







                domain=s["domain"],







                tags=[],







            )







            for s in self.SAMPLES







        ]























class FIMGenerator:







    """
    Fill-in-the-Middle (FIM) data generator.
    
    Creates training examples where the model learns to complete
    code given a prefix and suffix, which is essential for code
    completion use cases.
    """















    def __init__(self):







        self.split_points = [0.3, 0.5, 0.7]                           















    def generate_fim(self, code: str, language: str = "python") -> List[Dict[str, str]]:







        """
        Generate FIM examples from a code sample.
        
        Split code at multiple points to create prefix/middle/suffix triples.
        """







        lines = code.split('\n')







        if len(lines) < 5:







            return []















        examples = []















        for split_ratio in self.split_points:







            split_idx = max(2, int(len(lines) * split_ratio))















            prefix = '\n'.join(lines[:split_idx])







            middle = '\n'.join(lines[split_idx:split_idx + 3])







            suffix = '\n'.join(lines[split_idx + 3:])















            if not middle.strip():







                continue















                        







            fim_text = (







                f"<|fim_prefix|>{prefix}"







                f"<|fim_suffix|>{suffix}"







                f"<|fim_middle|>{middle}"







            )















            examples.append({







                "text": fim_text,







                "category": "fim",







                "language": language,







            })















        return examples























class DatasetGenerator:







    """
    Main dataset generator that combines all sources and creates
    the full training dataset.
    """















    def __init__(self, output_dir: str = "data/processed"):







        self.output_dir = Path(output_dir)







        self.python_gen = PythonCodeGenerator()







        self.js_gen = JavaScriptCodeGenerator()







        self.instruction_gen = InstructionGenerator()







        self.fim_gen = FIMGenerator()















    def generate_full_dataset(







        self,







        target_samples: int = 1_000_000,







        output_dir: Optional[str] = None,







    ) -> Dict[str, Any]:







        """
        Generate the complete training dataset.
        
        Args:
            target_samples: Target number of training samples
            output_dir: Override output directory
            
        Returns:
            Generation statistics
        """







        output_dir = output_dir or str(self.output_dir)







        os.makedirs(output_dir, exist_ok=True)















        stats = {







            "target_samples": target_samples,







            "code_samples": 0,







            "instruction_samples": 0,







            "fim_samples": 0,







            "total_samples": 0,







        }















        all_documents = []















        logger.info("Phase 1: Generating code samples...")







        code_docs = self._generate_code_samples(target_samples // 3)







        all_documents.extend(code_docs)







        stats["code_samples"] = len(code_docs)







        logger.info(f"  Generated {len(code_docs)} code documents")















        logger.info("Phase 2: Generating instruction samples...")







        instruction_docs = self._generate_instruction_samples(target_samples // 3)







        all_documents.extend(instruction_docs)







        stats["instruction_samples"] = len(instruction_docs)







        logger.info(f"  Generated {len(instruction_docs)} instruction documents")















        logger.info("Phase 3: Generating FIM samples...")







        fim_docs = self._generate_fim_samples(target_samples // 3)







        all_documents.extend(fim_docs)







        stats["fim_samples"] = len(fim_docs)







        logger.info(f"  Generated {len(fim_docs)} FIM documents")















                 







        random.shuffle(all_documents)







        stats["total_samples"] = len(all_documents)















                            







        raw_path = os.path.join(output_dir, "raw_documents.jsonl")







        with open(raw_path, 'w') as f:







            for doc in all_documents:







                f.write(json.dumps(doc) + '\n')















        logger.info(f"Saved {len(all_documents)} documents to {raw_path}")















                    







        stats_path = os.path.join(output_dir, "generation_stats.json")







        with open(stats_path, 'w') as f:







            json.dump(stats, f, indent=2)















        return stats















    def _generate_code_samples(self, target: int) -> List[Dict[str, Any]]:







        """Generate code samples with combinatorial expansion."""







        documents = []















        complexities = ["beginner", "intermediate", "advanced", "expert"]







        domains = ["algorithms", "data", "web", "systems", "testing"]















        for i in range(target):







            complexity = random.choice(complexities)







            domain = random.choice(domains)















            try:







                sample = self.python_gen.generate(complexity, domain)







                doc = {







                    "text": self._format_code_document(sample),







                    "category": "code",







                    "language": sample.language,







                    "complexity": sample.complexity,







                    "domain": sample.domain,







                }







                documents.append(doc)







            except Exception:







                pass















                                                     







            if i % 5 == 0:







                for js_sample in self.js_gen.generate():







                    doc = {







                        "text": self._format_code_document(js_sample),







                        "category": "code",







                        "language": js_sample.language,







                        "complexity": js_sample.complexity,







                        "domain": js_sample.domain,







                    }







                    documents.append(doc)















        return documents[:target]















    def _generate_instruction_samples(self, target: int) -> List[Dict[str, Any]]:







        """Generate instruction-following samples."""







        documents = []















        samples = self.instruction_gen.generate_batch(target)







        for sample in samples:







            doc = {







                "text": self._format_instruction_document(sample),







                "category": sample.category,







                "language": sample.language,







            }







            documents.append(doc)















        return documents[:target]















    def _generate_fim_samples(self, target: int) -> List[Dict[str, Any]]:







        """Generate fill-in-the-middle samples from code."""







        documents = []















                                                      







        complexities = ["intermediate", "advanced", "expert"]







        domains = ["algorithms", "data", "web", "systems"]















        generated = 0







        attempts = 0















        while generated < target and attempts < target * 10:







            attempts += 1







            complexity = random.choice(complexities)







            domain = random.choice(domains)















            try:







                sample = self.python_gen.generate(complexity, domain)







                fim_examples = self.fim_gen.generate_fim(sample.code, sample.language)















                for fim in fim_examples:







                    documents.append(fim)







                    generated += 1







                    if generated >= target:







                        break







            except Exception:







                pass















        return documents[:target]















    def _format_code_document(self, sample: CodeSample) -> str:







        """Format a code sample into a training document."""







        parts = []















                      







        parts.append(f"<|{sample.language}|>")















                                







        if sample.description:







            parts.append(f"# {sample.description}")















              







        parts.append("<|code_start|>")







        parts.append(sample.code)







        parts.append("<|code_end|>")















        return '\n'.join(parts)















    def _format_instruction_document(self, sample: InstructionSample) -> str:







        """Format an instruction sample into a training document."""







        parts = [







            "<|system|>",







            "You are an expert coding assistant. Help the user with their programming task.",







            "<eos>",







            "<|user|>",







            sample.instruction,







        ]















        if sample.input_context:







            parts.append("")







            parts.append("```" + (sample.language or ""))







            parts.append(sample.input_context)







            parts.append("```")















        parts.extend([







            "<eos>",







            "<|assistant|>",







            sample.output,







            "<eos>",







        ])















        return '\n'.join(parts)















    def tokenize_and_shard(







        self,







        raw_dir: str,







        output_dir: str,







        tokenizer,







        config: Optional[DatasetConfig] = None,







    ) -> Dict[str, Any]:







        """
        Tokenize raw documents and create binary shards.
        
        Args:
            raw_dir: Directory containing raw_documents.jsonl
            output_dir: Directory for processed shards
            tokenizer: Tokenizer instance
            config: Dataset processing config
            
        Returns:
            Processing statistics
        """







        if config is None:







            config = DatasetConfig()















                            







        raw_path = os.path.join(raw_dir, "raw_documents.jsonl")







        documents = []















        logger.info(f"Loading documents from {raw_path}...")







        with open(raw_path) as f:







            for line in f:







                documents.append(json.loads(line))















        logger.info(f"Loaded {len(documents)} documents")















                 







        processor = DatasetProcessor(config, tokenizer)







        stats = processor.process(documents, output_dir)















        return stats







