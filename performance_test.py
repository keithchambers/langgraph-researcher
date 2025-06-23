#!/usr/bin/env python3
"""
Performance Testing Framework for Research Orchestrator

Measures timing characteristics of different research patterns and question types.
Supports parallel execution, dynamic question generation, and statistical analysis.
"""

import time
import random
import argparse
import concurrent.futures
import json
import re
from typing import List, Dict
from supervisor_ollama import research_orchestrator
from langchain_ollama import ChatOllama


def generate_test_questions() -> List[Dict]:
    """Static question bank for testing different research patterns."""
    questions = [
        # Simple questions
        {"q": "What is the capital of France?", "type": "simple", "expected_pattern": "direct_search"},
        {"q": "Who invented the telephone?", "type": "simple", "expected_pattern": "direct_search"},
        {"q": "What is machine learning?", "type": "simple", "expected_pattern": "parallel_research"},
        
        # Math questions
        {"q": "Calculate 25 plus 17", "type": "math", "expected_pattern": "math_processing"},
        {"q": "What is 144 divided by 12?", "type": "math", "expected_pattern": "math_processing"},
        {"q": "Calculate 8 times 12", "type": "math", "expected_pattern": "math_processing"},
        
        # Medium complexity questions
        {"q": "What are the benefits of renewable energy?", "type": "medium", "expected_pattern": "sequential_chain"},
        {"q": "How does exercise affect mental health?", "type": "medium", "expected_pattern": "sequential_chain"},
        {"q": "What causes climate change?", "type": "medium", "expected_pattern": "sequential_chain"},
        {"q": "How do vaccines work?", "type": "medium", "expected_pattern": "sequential_chain"},
        
        # Complex questions
        {"q": "Compare the advantages and disadvantages of solar vs wind energy", "type": "complex", "expected_pattern": "validation_chain"},
        {"q": "Analyze the impact of remote work on productivity", "type": "complex", "expected_pattern": "validation_chain"},
        {"q": "What are the economic effects of artificial intelligence on employment?", "type": "complex", "expected_pattern": "adversarial_loop"},
        {"q": "What are the environmental benefits of electric cars?", "type": "complex", "expected_pattern": "validation_chain"},
    ]
    
    return questions


def generate_dynamic_questions(llm: ChatOllama, num_questions: int = 14) -> List[Dict]:
    """Generate questions dynamically using LLM."""
    print(f"Generating {num_questions} dynamic questions...")
    
    # Question distribution
    simple_count = max(2, num_questions // 4)
    medium_count = max(3, num_questions // 3)  
    complex_count = max(2, num_questions - simple_count - medium_count - 2)
    math_count = max(2, num_questions // 5)
    
    all_questions = []
    
    # Generate simple questions
    simple_prompt = f"""
    Generate {simple_count} SIMPLE research questions that require only basic fact lookup.
    
    Examples: "What is the capital of Japan?", "Who wrote Romeo and Juliet?"
    
    Generate {simple_count} questions in JSON format:
    {{"questions": [{{"question": "What is...", "complexity": "simple"}}]}}
    """
    
    try:
        simple_response = llm.invoke(simple_prompt)
        simple_data = extract_json_from_response(simple_response.content)
        if simple_data and "questions" in simple_data:
            for q in simple_data["questions"]:
                all_questions.append({
                    "q": q["question"], 
                    "type": "simple", 
                    "expected_pattern": "direct_search",
                    "generated": True
                })
    except Exception as e:
        print(f"Error generating simple questions: {e}")
        all_questions.extend([
            {"q": "What is the capital of Canada?", "type": "simple", "expected_pattern": "direct_search", "generated": False},
            {"q": "Who invented the lightbulb?", "type": "simple", "expected_pattern": "direct_search", "generated": False}
        ])
    
    # Generate medium questions
    medium_prompt = f"""
    Generate {medium_count} MEDIUM complexity research questions requiring analysis.
    
    Examples: "How does exercise affect mental health?", "What causes deforestation?"
    
    Generate {medium_count} questions in JSON format:
    {{"questions": [{{"question": "How does...", "complexity": "medium"}}]}}
    """
    
    try:
        medium_response = llm.invoke(medium_prompt)
        medium_data = extract_json_from_response(medium_response.content)
        if medium_data and "questions" in medium_data:
            for q in medium_data["questions"]:
                all_questions.append({
                    "q": q["question"], 
                    "type": "medium", 
                    "expected_pattern": "sequential_chain",
                    "generated": True
                })
    except Exception as e:
        print(f"Error generating medium questions: {e}")
        all_questions.extend([
            {"q": "How does sleep affect learning?", "type": "medium", "expected_pattern": "sequential_chain", "generated": False},
            {"q": "What causes ocean pollution?", "type": "medium", "expected_pattern": "sequential_chain", "generated": False}
        ])
    
    # Generate complex questions
    complex_prompt = f"""
    Generate {complex_count} COMPLEX research questions requiring validation.
    
    Examples: "What are the economic effects of AI on employment?", "Compare nuclear vs renewable energy"
    
    Generate {complex_count} questions in JSON format:
    {{"questions": [{{"question": "What are the impacts of...", "complexity": "complex"}}]}}
    """
    
    try:
        complex_response = llm.invoke(complex_prompt)
        complex_data = extract_json_from_response(complex_response.content)
        if complex_data and "questions" in complex_data:
            for q in complex_data["questions"]:
                all_questions.append({
                    "q": q["question"], 
                    "type": "complex", 
                    "expected_pattern": "validation_chain",
                    "generated": True
                })
    except Exception as e:
        print(f"Error generating complex questions: {e}")
        all_questions.extend([
            {"q": "What are the societal impacts of genetic engineering?", "type": "complex", "expected_pattern": "validation_chain", "generated": False},
            {"q": "How effective are different climate change solutions?", "type": "complex", "expected_pattern": "validation_chain", "generated": False}
        ])
    
    # Generate math questions
    math_prompt = f"""
    Generate {math_count} basic MATH questions with numbers.
    
    Examples: "Calculate 25 plus 17", "What is 144 divided by 12?"
    
    Generate {math_count} questions in JSON format:
    {{"questions": [{{"question": "Calculate 15 plus 23", "complexity": "math"}}]}}
    """
    
    try:
        math_response = llm.invoke(math_prompt)
        math_data = extract_json_from_response(math_response.content)
        if math_data and "questions" in math_data:
            for q in math_data["questions"]:
                all_questions.append({
                    "q": q["question"], 
                    "type": "math", 
                    "expected_pattern": "math_processing",
                    "generated": True
                })
    except Exception as e:
        print(f"Error generating math questions: {e}")
        all_questions.extend([
            {"q": "Calculate 42 plus 28", "type": "math", "expected_pattern": "math_processing", "generated": False},
            {"q": "What is 72 divided by 8?", "type": "math", "expected_pattern": "math_processing", "generated": False}
        ])
    
    print(f"Generated {len(all_questions)} questions")
    return all_questions


def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from LLM response text."""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"JSON extraction failed: {e}")
    return {}


def run_single_question_test(args) -> Dict:
    """Execute single question test for parallel processing."""
    question_data, question_num, total_questions, llm = args
    
    question = question_data["q"]
    q_type = question_data["type"]
    expected_pattern = question_data["expected_pattern"]
    
    print(f"[Q{question_num}] {q_type.upper()}: {question}")
    print(f"[Q{question_num}] Expected pattern: {expected_pattern}")
    
    start_time = time.time()
    
    try:
        # Execute with timeout using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(research_orchestrator, question, llm)
            result = future.result(timeout=300)  # 5 minute timeout per question
        
        elapsed = time.time() - start_time
        
        result_data = {
            "question": question,
            "type": q_type,
            "expected_pattern": expected_pattern,
            "time": elapsed,
            "result_length": len(result),
            "generated": question_data.get("generated", False),
            "error": None
        }
        
        print(f"[Q{question_num}] Completed in {elapsed:.2f}s (result length: {len(result)})")
        
        return result_data
        
    except concurrent.futures.TimeoutError as e:
        elapsed = time.time() - start_time
        print(f"[Q{question_num}] TIMEOUT in {elapsed:.2f}s: Question timed out")
        return {
            "question": question,
            "type": q_type,
            "expected_pattern": expected_pattern,
            "time": elapsed,
            "result_length": 0,
            "generated": question_data.get("generated", False),
            "error": "TIMEOUT: Question timed out after 300s"
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Q{question_num}] ERROR in {elapsed:.2f}s: {str(e)}")
        
        return {
            "question": question,
            "type": q_type,
            "expected_pattern": expected_pattern,
            "time": elapsed,
            "result_length": 0,
            "generated": question_data.get("generated", False),
            "error": str(e)
        }


def run_performance_test(llm, num_questions: int = 10, use_static: bool = False, max_parallel: int = 5) -> Dict:
    """Execute performance test with parallel execution."""
    question_source = "static" if use_static else "dynamic"
    print(f"=== PERFORMANCE TEST: {num_questions} questions ({question_source}) ===")
    
    # Select questions
    if use_static:
        test_questions = generate_test_questions()
        selected_questions = random.sample(test_questions, min(num_questions, len(test_questions)))
    else:
        selected_questions = generate_dynamic_questions(llm, num_questions)
    
    results = {
        "questions": [],
        "total_time": 0,
        "average_time": 0,
        "question_source": question_source,
        "timing_stats": {}
    }
    
    start_total = time.time()
    
    # Prepare parallel execution arguments
    test_args = []
    for i, q_data in enumerate(selected_questions, 1):
        test_args.append((q_data, i, len(selected_questions), llm))
    
    print(f"\nRunning {len(selected_questions)} questions in parallel (max workers: {max_parallel})...")
    
    # Execute tests in parallel with timeout
    completed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_question = {executor.submit(run_single_question_test, args): args for args in test_args}
        
        # Set timeout for individual futures
        timeout_per_question = 300  # 5 minutes per question
        for future in concurrent.futures.as_completed(future_to_question, timeout=timeout_per_question * len(test_args)):
            try:
                result = future.result(timeout=timeout_per_question)
                completed_results.append(result)
            except concurrent.futures.TimeoutError:
                args = future_to_question[future]
                q_data, question_num, total_questions, _ = args
                print(f"[Q{question_num}] TIMEOUT: Question timed out after {timeout_per_question}s")
                completed_results.append({
                    "question": q_data["q"],
                    "type": q_data["type"],
                    "expected_pattern": q_data["expected_pattern"],
                    "time": timeout_per_question,
                    "result_length": 0,
                    "generated": q_data.get("generated", False),
                    "error": f"TIMEOUT: Question timed out after {timeout_per_question}s"
                })
            except Exception as e:
                args = future_to_question[future]
                q_data, question_num, total_questions, _ = args
                print(f"[Q{question_num}] PARALLEL ERROR: {str(e)}")
                completed_results.append({
                    "question": q_data["q"],
                    "type": q_data["type"],
                    "expected_pattern": q_data["expected_pattern"],
                    "time": 0,
                    "result_length": 0,
                    "generated": q_data.get("generated", False),
                    "error": f"Parallel execution error: {str(e)}"
                })
    
    total_elapsed = time.time() - start_total
    error_count = sum(1 for r in completed_results if r.get("error"))
    
    results["questions"] = completed_results
    results["total_time"] = total_elapsed
    results["average_time"] = sum(r["time"] for r in completed_results) / len(completed_results) if completed_results else 0
    
    # Calculate timing statistics
    question_times = [r["time"] for r in completed_results if not r.get("error")]
    if question_times:
        results["timing_stats"] = {
            "min_time": min(question_times),
            "max_time": max(question_times),
            "median_time": sorted(question_times)[len(question_times)//2],
            "error_count": error_count,
            "completed_count": len(completed_results) - error_count
        }
    else:
        results["timing_stats"] = {
            "min_time": 0,
            "max_time": 0,
            "median_time": 0,
            "error_count": error_count,
            "completed_count": 0
        }
    
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total parallel execution time: {total_elapsed:.2f}s")
    print(f"Average per question: {results['average_time']:.2f}s")
    print(f"Completed: {results['timing_stats']['completed_count']}/{len(completed_results)}")
    print(f"Errors: {results['timing_stats']['error_count']}")
    if results['timing_stats']['completed_count'] > 0:
        print(f"Min time: {results['timing_stats']['min_time']:.2f}s")
        print(f"Max time: {results['timing_stats']['max_time']:.2f}s")
        print(f"Median time: {results['timing_stats']['median_time']:.2f}s")
    
    # Timing breakdown by question type
    type_stats = {}
    for q in completed_results:
        qtype = q["type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"count": 0, "total_time": 0, "errors": 0, "generated": 0, "times": []}
        type_stats[qtype]["count"] += 1
        type_stats[qtype]["total_time"] += q["time"]
        type_stats[qtype]["times"].append(q["time"])
        if q.get("error"):
            type_stats[qtype]["errors"] += 1
        if q.get("generated", False):
            type_stats[qtype]["generated"] += 1
    
    print(f"\n=== TIMING BREAKDOWN BY TYPE ===")
    for qtype, stats in type_stats.items():
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        error_count = stats["errors"]
        generated_count = stats["generated"]
        
        if stats["times"]:
            min_time = min(stats["times"])
            max_time = max(stats["times"])
            print(f"{qtype.upper()}: {avg_time:.2f}s avg, {min_time:.2f}s min, {max_time:.2f}s max, {error_count} errors, {generated_count} generated")
        else:
            print(f"{qtype.upper()}: No completed questions")
    
    return results


def run_multiple_tests(llm, iterations: int = 10, use_static: bool = False):
    """Run multiple test iterations for consistency analysis."""
    question_mode = "static" if use_static else "dynamic"
    print(f"=== RUNNING {iterations} TEST ITERATIONS ({question_mode} questions) ===")
    
    all_results = []
    
    for iteration in range(1, iterations + 1):
        print(f"\nTEST {iteration}/{iterations}")
        result = run_performance_test(llm, num_questions=10, use_static=use_static)
        all_results.append(result)
        
        # Report iteration results
        error_count = result["timing_stats"]["error_count"]
        completed_count = result["timing_stats"]["completed_count"]
        print(f"TEST {iteration}: {result['average_time']:.2f}s avg, {completed_count} completed, {error_count} errors")
    
    print(f"\n=== TIMING CONSISTENCY ANALYSIS ===")
    avg_time = sum(r["average_time"] for r in all_results) / len(all_results)
    total_completed = sum(r["timing_stats"]["completed_count"] for r in all_results)
    total_errors = sum(r["timing_stats"]["error_count"] for r in all_results)
    total_questions = sum(len(r["questions"]) for r in all_results)
    
    print(f"Average time per question: {avg_time:.2f}s")
    print(f"Total completed: {total_completed}/{total_questions}")
    print(f"Total errors: {total_errors}")
    
    # Timing consistency analysis
    time_std = (sum((r["average_time"] - avg_time)**2 for r in all_results) / len(all_results))**0.5
    print(f"Timing consistency (std dev): {time_std:.2f}s")
    
    # Overall timing statistics
    all_times = []
    for result in all_results:
        for q in result["questions"]:
            if not q.get("error"):
                all_times.append(q["time"])
    
    if all_times:
        print(f"Overall min time: {min(all_times):.2f}s")
        print(f"Overall max time: {max(all_times):.2f}s")
        print(f"Overall median time: {sorted(all_times)[len(all_times)//2]:.2f}s")
    
    # Dynamic generation statistics
    if not use_static:
        total_generated = sum(sum(1 for q in r["questions"] if q.get("generated", False)) for r in all_results)
        print(f"Dynamic generation: {total_generated}/{total_questions} ({total_generated/total_questions:.1%})")
    
    return all_results


def main():
    """Main performance testing function."""
    parser = argparse.ArgumentParser(description="Performance Testing for Research Orchestrator")
    parser.add_argument("-s", "--static", action="store_true", 
                        help="Use static questions instead of dynamic generation")
    parser.add_argument("-q", "--questions", type=int, default=10,
                        help="Number of questions per test (default: 10)")
    parser.add_argument("-i", "--iterations", type=int, default=10,
                        help="Number of test iterations (default: 10)")
    parser.add_argument("-p", "--parallel", type=int, default=5,
                        help="Maximum parallel workers (default: 5)")
    
    args = parser.parse_args()
    
    question_mode = "static" if args.static else "dynamic"
    print(f"Starting Performance Testing for Research Orchestrator ({question_mode} questions)")
    
    # Initialize LLM
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)
    
    # Quick test
    print(f"\n=== QUICK TEST ({args.questions} questions) ===")
    quick_result = run_performance_test(llm, num_questions=args.questions, 
                                       use_static=args.static, max_parallel=args.parallel)
    
    print("Running full test suite...")
    
    # Multiple iteration testing
    all_results = run_multiple_tests(llm, iterations=args.iterations, use_static=args.static)
    
    # Final timing assessment
    print(f"\nFINAL TIMING ASSESSMENT")
    avg_time = sum(r["average_time"] for r in all_results) / len(all_results)
    total_completed = sum(r["timing_stats"]["completed_count"] for r in all_results)
    total_questions = sum(len(r["questions"]) for r in all_results)
    
    print(f"Average time per question across all tests: {avg_time:.2f}s")
    print(f"Overall completion rate: {total_completed}/{total_questions}")
    
    if not args.static:
        print(f"Dynamic question generation is operational")
    
    print(f"System timing profile complete - see details above")


if __name__ == "__main__":
    main() 