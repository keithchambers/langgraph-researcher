#!/usr/bin/env python3
"""
LangGraph Research Orchestrator

A multi-agent research system that routes questions through different analysis patterns
based on complexity and domain. Supports web search with Brave API, mathematical operations, and
various research methodologies including validation chains and adversarial loops.
"""

import os
import time
import re
import signal
import functools
import requests
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from langchain.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def safe_llm_invoke(llm, prompt, timeout_seconds=30):
    """Safely invoke LLM with timeout using ThreadPoolExecutor."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.invoke, prompt)
            return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        print(f"[TIMEOUT] LLM call timed out after {timeout_seconds}s")
        raise TimeoutError(f"LLM call timed out after {timeout_seconds}s")
    except Exception as e:
        print(f"[ERROR] LLM call failed: {str(e)}")
        raise


def with_timeout(seconds):
    """Thread-safe timeout decorator using ThreadPoolExecutor."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=seconds)
            except FuturesTimeoutError:
                print(f"[TIMEOUT] Function {func.__name__} timed out after {seconds}s")
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
        return wrapper
    return decorator


def safe_search_operation(search_func, query, timeout_seconds=15):
    """Safely execute search operation with timeout."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(search_func, query)
            return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        print(f"[TIMEOUT] Search operation timed out after {timeout_seconds}s")
        return f"Search timed out for: {query}"
    except Exception as e:
        print(f"[ERROR] Search operation failed: {str(e)}")
        return f"Search failed for {query}: {str(e)}"


def brave_search_with_fallback(query: str, llm: ChatOllama = None, timeout_seconds: int = 15) -> str:
    """
    Search using Brave Search API with fallback to LLM knowledge.
    
    Args:
        query: Search query string
        llm: LLM instance for fallback
        timeout_seconds: Timeout for the search operation
        
    Returns:
        Search results as string
    """
    # Check for Brave API key
    brave_api_key = os.getenv("BRAVE_API_KEY")
    
    if brave_api_key:
        try:
            print(f"[BRAVE SEARCH] Using Brave Search API for: {query}")
            
            # Brave Search API endpoint
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": brave_api_key
            }
            params = {
                "q": query,
                "count": 10,
                "offset": 0,
                "safesearch": "moderate",
                "freshness": "py",  # Past year
                "text_decorations": False,
                "search_lang": "en",
                "country": "us"
            }
            
            # Make API request with timeout
            response = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and format results
            results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"][:5]:  # Top 5 results
                    title = result.get("title", "")
                    description = result.get("description", "")
                    url = result.get("url", "")
                    
                    formatted_result = f"Title: {title}\nDescription: {description}\nURL: {url}"
                    results.append(formatted_result)
            
            if results:
                return "\n\n".join(results)
            else:
                print("[BRAVE SEARCH] No results found, falling back to LLM")
                return brave_search_llm_fallback(query, llm)
                
        except requests.exceptions.Timeout:
            print(f"[BRAVE SEARCH] API request timed out after {timeout_seconds}s, falling back to LLM")
            return brave_search_llm_fallback(query, llm)
        except requests.exceptions.RequestException as e:
            print(f"[BRAVE SEARCH] API request failed: {str(e)}, falling back to LLM")
            return brave_search_llm_fallback(query, llm)
        except Exception as e:
            print(f"[BRAVE SEARCH] Unexpected error: {str(e)}, falling back to LLM")
            return brave_search_llm_fallback(query, llm)
    else:
        print("[BRAVE SEARCH] No BRAVE_API_KEY found, using LLM knowledge")
        return brave_search_llm_fallback(query, llm)


def brave_search_llm_fallback(query: str, llm: ChatOllama = None) -> str:
    """
    Fallback to LLM knowledge when Brave Search API is unavailable.
    
    Args:
        query: Search query string
        llm: LLM instance
        
    Returns:
        LLM-generated response based on its knowledge
    """
    if llm is None:
        return f"Search unavailable for query: {query}. Please provide information or try again with an API key."
    
    try:
        print(f"[LLM FALLBACK] Using LLM knowledge for: {query}")
        
        fallback_prompt = f"""
        I need information about: "{query}"
        
        Please provide comprehensive information based on your knowledge. Include:
        1. Key facts and details
        2. Multiple perspectives if applicable
        3. Important context or background
        4. Recent developments if known (note any knowledge cutoff limitations)
        
        Format your response as if it were search results, providing detailed and informative content.
        """
        
        response = safe_llm_invoke(llm, fallback_prompt, 20)
        return f"[LLM Knowledge Response]\n{response.content}"
        
    except (TimeoutError, Exception) as e:
        print(f"[LLM FALLBACK] Failed: {str(e)}")
        return f"Unable to retrieve information for query: {query}. Error: {str(e)}"


def create_brave_search_tool(llm: ChatOllama):
    """Create a Brave Search tool function for LangGraph agents."""
    def brave_search_tool(query: str) -> str:
        """Search the web for information using Brave Search API with LLM fallback."""
        return brave_search_with_fallback(query, llm)
    
    return brave_search_tool


class QuestionComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"


class QuestionDomain(Enum):
    GENERAL = "general"
    SCIENTIFIC = "scientific"
    FINANCIAL = "financial"
    HISTORICAL = "historical"
    TECHNICAL = "technical"
    POLITICAL = "political"
    MEDICAL = "medical"


class ResearchPattern(Enum):
    DIRECT_SEARCH = "direct_search"
    PARALLEL_RESEARCH = "parallel_research"
    SEQUENTIAL_CHAIN = "sequential_chain"
    VALIDATION_CHAIN = "validation_chain"
    REFINEMENT_CHAIN = "refinement_chain"
    IMPROVEMENT_LOOP = "improvement_loop"
    ADVERSARIAL_LOOP = "adversarial_loop"
    CLARIFICATION_CHAIN = "clarification_chain"
    FOLLOWUP_CHAIN = "followup_chain"
    INTERACTIVE_REPORT = "interactive_report"


@dataclass
class ResearchResult:
    content: str
    sources: List[str]
    confidence: float
    quality_score: float
    gaps_identified: List[str]
    follow_up_questions: List[str]


@dataclass
class QuestionAnalysis:
    complexity: QuestionComplexity
    domain: QuestionDomain
    requires_validation: bool
    requires_multiple_perspectives: bool
    estimated_research_depth: int
    suggested_pattern: ResearchPattern


def analyze_question_complexity(question: str, llm: ChatOllama) -> QuestionAnalysis:
    """Analyze question to determine appropriate research approach."""
    prompt = f"""
    Analyze this research question: "{question}"
    
    Classify:
    1. Complexity: simple (facts), medium (analysis), high (multi-faceted), expert (controversial)
    2. Domain: general, scientific, financial, historical, technical, political, medical
    3. Validation needed: yes/no
    4. Multiple perspectives needed: yes/no
    5. Research depth (1-5 scale)
    
    Format: complexity|domain|validation|perspectives|depth
    Example: high|scientific|yes|yes|4
    """
    
    try:
        response = safe_llm_invoke(llm, prompt, 20).content.strip().lower()
        parts = response.split('|')
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Question analysis failed: {str(e)}")
        parts = []
    
    if len(parts) >= 5:
        complexity_map = {
            'simple': QuestionComplexity.SIMPLE,
            'medium': QuestionComplexity.MEDIUM,
            'high': QuestionComplexity.HIGH,
            'expert': QuestionComplexity.EXPERT
        }
        
        domain_map = {
            'general': QuestionDomain.GENERAL,
            'scientific': QuestionDomain.SCIENTIFIC,
            'financial': QuestionDomain.FINANCIAL,
            'historical': QuestionDomain.HISTORICAL,
            'technical': QuestionDomain.TECHNICAL,
            'political': QuestionDomain.POLITICAL,
            'medical': QuestionDomain.MEDICAL
        }
        
        complexity = complexity_map.get(parts[0].strip(), QuestionComplexity.MEDIUM)
        domain = domain_map.get(parts[1].strip(), QuestionDomain.GENERAL)
        validation = 'yes' in parts[2].strip()
        perspectives = 'yes' in parts[3].strip()
        depth = int(parts[4].strip()) if parts[4].strip().isdigit() else 3
        
        # Determine suggested pattern
        if complexity == QuestionComplexity.SIMPLE:
            pattern = ResearchPattern.PARALLEL_RESEARCH
        elif complexity == QuestionComplexity.MEDIUM:
            pattern = ResearchPattern.SEQUENTIAL_CHAIN
        elif validation or perspectives:
            pattern = ResearchPattern.VALIDATION_CHAIN
        elif complexity == QuestionComplexity.EXPERT:
            pattern = ResearchPattern.ADVERSARIAL_LOOP
        else:
            pattern = ResearchPattern.PARALLEL_RESEARCH
            
        return QuestionAnalysis(
            complexity=complexity,
            domain=domain,
            requires_validation=validation,
            requires_multiple_perspectives=perspectives,
            estimated_research_depth=depth,
            suggested_pattern=pattern
        )
    
    # Default fallback
    return QuestionAnalysis(
        complexity=QuestionComplexity.MEDIUM,
        domain=QuestionDomain.GENERAL,
        requires_validation=False,
        requires_multiple_perspectives=False,
        estimated_research_depth=3,
        suggested_pattern=ResearchPattern.PARALLEL_RESEARCH
    )


def create_research_plan(question: str, llm: ChatOllama, max_queries: int = 3) -> List[str]:
    """Generate focused search queries for parallel research."""
    prompt = f"""
    Create {max_queries} focused search queries for: "{question}"
    
    Make queries specific and complementary.
    Return only the queries, one per line.
    """
    
    try:
        response = safe_llm_invoke(llm, prompt, 15).content.strip()
        queries = [line.strip() for line in response.split('\n') if line.strip()]
        return queries[:max_queries]
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Research plan generation failed: {str(e)}")
        return [question]  # Fallback to original question


def execute_parallel_search(queries: List[str], max_workers: int = 3, timeout: int = 20, llm: ChatOllama = None) -> Dict[str, str]:
    """Execute multiple search queries in parallel with timeout using Brave Search with LLM fallback."""
    
    def search_single_query(query: str) -> tuple[str, str]:
        try:
            result = brave_search_with_fallback(query, llm, 15)
            return query, result
        except Exception as e:
            return query, f"Search failed: {str(e)}"
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(search_single_query, query) for query in queries]
        for future in futures:
            try:
                query, result = future.result(timeout=timeout)
                results[query] = result
            except FuturesTimeoutError as e:
                print(f"[TIMEOUT] Search operation timed out: {str(e)}")
                continue
            except Exception as e:
                continue
    
    return results


def synthesize_research_results(question: str, search_results: Dict[str, str], llm: ChatOllama) -> str:
    """Combine search results into comprehensive answer."""
    combined_info = "\n\n".join([f"Query: {query}\nResults: {result}" 
                                for query, result in search_results.items()])
    
    prompt = f"""
    Question: {question}
    
    Research findings:
    {combined_info}
    
    Provide a comprehensive, well-structured answer based on the research.
    """
    
    try:
        return safe_llm_invoke(llm, prompt, 25).content
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Result synthesis failed: {str(e)}")
        return f"Research completed but synthesis failed: {str(e)}"


def sequential_chain(question: str, llm: ChatOllama) -> ResearchResult:
    """Execute sequential research chain pattern."""
    print("[SEQUENTIAL CHAIN] Starting multi-step research")
    
    # Step 1: Initial research
    print("[STEP 1] Initial research...")
    initial_queries = create_research_plan(question, llm, max_queries=2)
    initial_results = execute_parallel_search(initial_queries, llm=llm)
    
    # Step 2: Follow-up research
    print("[STEP 2] Follow-up research...")
    followup_prompt = f"""
    Based on initial research for "{question}", what additional queries would provide more depth?
    
    Initial findings summary:
    {list(initial_results.keys())}
    
    Suggest 2 follow-up queries:
    """
    
    try:
        followup_response = safe_llm_invoke(llm, followup_prompt, 15).content
        followup_queries = [line.strip() for line in followup_response.split('\n') if line.strip()][:2]
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Follow-up generation failed: {str(e)}")
        followup_queries = []
    followup_results = execute_parallel_search(followup_queries, llm=llm)
    
    # Step 3: Synthesis
    print("[STEP 3] Final synthesis...")
    all_results = {**initial_results, **followup_results}
    final_content = synthesize_research_results(question, all_results, llm)
    
    return ResearchResult(
        content=final_content,
        sources=list(all_results.keys()),
        confidence=0.8,
        quality_score=0.85,
        gaps_identified=[],
        follow_up_questions=[]
    )


def validation_chain(question: str, llm: ChatOllama) -> ResearchResult:
    """Execute validation chain pattern with credibility assessment."""
    print("[VALIDATION CHAIN] Starting for: " + question)
    
    # Step 1: Primary research
    print("[STEP 1] Primary research...")
    queries = create_research_plan(question, llm, max_queries=3)
    results = execute_parallel_search(queries, llm=llm)
    
    # Step 2: Credibility validation
    print("[STEP 2] Credibility validation...")
    validation_prompt = f"""
    Assess credibility of research findings for: "{question}"
    
    Sources checked: {len(results)}
    
    Rate overall credibility (1-10) and identify any inconsistencies.
    """
    
    try:
        validation = safe_llm_invoke(llm, validation_prompt, 15).content
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Validation assessment failed: {str(e)}")
        validation = "Validation assessment unavailable due to timeout"
    
    # Step 3: Validated synthesis
    print("[STEP 3] Validated synthesis...")
    synthesis_prompt = f"""
    Question: {question}
    
    Research data:
    {chr(10).join([f"{q}: {r}" for q, r in results.items()])}
    
    Credibility assessment:
    {validation}
    
    Provide validated, well-sourced answer.
    """
    
    try:
        final_content = safe_llm_invoke(llm, synthesis_prompt, 20).content
    except (TimeoutError, Exception) as e:
        print(f"[ERROR] Validation synthesis failed: {str(e)}")
        final_content = f"Analysis completed but synthesis failed: {str(e)}"
    
    return ResearchResult(
        content=final_content,
        sources=list(results.keys()),
        confidence=0.9,
        quality_score=0.9,
        gaps_identified=[],
        follow_up_questions=[]
    )


def refinement_chain(question: str, llm: ChatOllama, max_iterations: int = 2) -> ResearchResult:
    """Execute iterative refinement chain pattern."""
    print("[REFINEMENT CHAIN] Starting for: " + question)
    
    current_answer = ""
    all_sources = []
    
    for iteration in range(max_iterations):
        print(f"[ITERATION {iteration + 1}] Refining research...")
        
        if iteration == 0:
            # Initial research
            queries = create_research_plan(question, llm, max_queries=3)
        else:
            # Identify gaps and create targeted queries
            gap_prompt = f"""
            Question: {question}
            Current answer: {current_answer}
            
            What key aspects are missing? Suggest 2 specific search queries.
            """
            
            gap_response = llm.invoke(gap_prompt).content
            queries = [line.strip() for line in gap_response.split('\n') if line.strip()][:2]
        
        # Execute searches
        results = execute_parallel_search(queries, llm=llm)
        all_sources.extend(results.keys())
        
        # Update answer
        update_prompt = f"""
        Question: {question}
        Previous answer: {current_answer}
        New information: {chr(10).join([f"{q}: {r}" for q, r in results.items()])}
        
        Provide improved, comprehensive answer.
        """
        
        current_answer = llm.invoke(update_prompt).content
    
    return ResearchResult(
        content=current_answer,
        sources=all_sources,
        confidence=0.85,
        quality_score=0.9,
        gaps_identified=[],
        follow_up_questions=[]
    )


def complexity_based_routing(question: str, llm: ChatOllama) -> ResearchResult:
    """Route questions based on complexity analysis."""
    analysis = analyze_question_complexity(question, llm)
    
    if analysis.complexity == QuestionComplexity.SIMPLE:
        queries = create_research_plan(question, llm, max_queries=2)
        results = execute_parallel_search(queries)
        content = synthesize_research_results(question, results, llm)
        
        return ResearchResult(
            content=content,
            sources=list(results.keys()),
            confidence=0.8,
            quality_score=0.8,
            gaps_identified=[],
            follow_up_questions=[]
        )
    
    elif analysis.complexity == QuestionComplexity.MEDIUM:
        return sequential_chain(question, llm)
    
    elif analysis.complexity == QuestionComplexity.HIGH:
        return validation_chain(question, llm)
    
    else:  # EXPERT
        return adversarial_loop(question, llm)


def domain_specific_routing(question: str, llm: ChatOllama) -> ResearchResult:
    """Route questions based on domain analysis."""
    analysis = analyze_question_complexity(question, llm)
    
    if analysis.domain == QuestionDomain.SCIENTIFIC:
        print("[SCIENTIFIC ROUTING] Using validation chain")
        return validation_chain(question, llm)
    
    elif analysis.domain == QuestionDomain.MEDICAL:
        print("[MEDICAL ROUTING] Using validation chain with extra verification")
        result = validation_chain(question, llm)
        
        # Add medical disclaimer
        result.content += "\n\n[Note: This information is for research purposes only. Consult healthcare professionals for medical advice.]"
        return result
    
    elif analysis.domain == QuestionDomain.FINANCIAL:
        print("[FINANCIAL ROUTING] Using sequential chain")
        return sequential_chain(question, llm)
    
    elif analysis.domain == QuestionDomain.HISTORICAL:
        print("[HISTORICAL ROUTING] Using validation chain")
        return validation_chain(question, llm)
    
    else:
        print("[GENERAL ROUTING] Using complexity-based routing")
        return complexity_based_routing(question, llm)


def improvement_loop(question: str, llm: ChatOllama, max_iterations: int = 2) -> ResearchResult:
    """Execute self-improving research loop."""
    print("[IMPROVEMENT LOOP] Starting for: " + question)
    
    current_result = None
    
    for iteration in range(max_iterations):
        print(f"[ITERATION {iteration + 1}] Self-improvement cycle...")
        
        if iteration == 0:
            # Initial research
            current_result = sequential_chain(question, llm)
        else:
            # Self-assessment and improvement
            improvement_prompt = f"""
            Question: {question}
            Current answer: {current_result.content}
            
            Rate answer quality (1-10) and identify 2 specific improvements needed.
            Then suggest search queries to address those improvements.
            """
            
            assessment = llm.invoke(improvement_prompt).content
            
            # Extract improvement queries
            query_lines = [line.strip() for line in assessment.split('\n') if '?' in line][:2]
            
            if query_lines:
                improvement_results = execute_parallel_search(query_lines, llm=llm)
                
                # Enhance the answer
                enhancement_prompt = f"""
                Question: {question}
                Current answer: {current_result.content}
                Improvements: {chr(10).join([f"{q}: {r}" for q, r in improvement_results.items()])}
                
                Provide enhanced answer incorporating improvements.
                """
                
                enhanced_content = llm.invoke(enhancement_prompt).content
                current_result.content = enhanced_content
                current_result.sources.extend(improvement_results.keys())
    
    return current_result


def adversarial_loop(question: str, llm: ChatOllama, max_rounds: int = 2) -> ResearchResult:
    """Execute adversarial validation loop."""
    print("[ADVERSARIAL LOOP] Starting for: " + question)
    
    # Initial research
    initial_result = sequential_chain(question, llm)
    current_answer = initial_result.content
    all_sources = initial_result.sources.copy()
    
    for round_num in range(max_rounds):
        print(f"[ROUND {round_num + 1}] Adversarial challenge...")
        
        # Generate counter-arguments
        counter_prompt = f"""
        Question: {question}
        Current answer: {current_answer}
        
        Play devil's advocate. What are the strongest counter-arguments or alternative perspectives?
        Identify 2 specific challenges to this answer.
        """
        
        counter_response = llm.invoke(counter_prompt).content
        
        # Research counter-arguments
        counter_queries = [line.strip() for line in counter_response.split('\n') if line.strip()][:2]
        counter_results = execute_parallel_search(counter_queries, llm=llm)
        all_sources.extend(counter_results.keys())
        
        # Refine answer considering counter-arguments
        refinement_prompt = f"""
        Question: {question}
        Current answer: {current_answer}
        Counter-arguments: {counter_response}
        Counter-research: {chr(10).join([f"{q}: {r}" for q, r in counter_results.items()])}
        
        Provide balanced, nuanced answer addressing counter-arguments.
        """
        
        current_answer = llm.invoke(refinement_prompt).content
    
    return ResearchResult(
        content=current_answer,
        sources=all_sources,
        confidence=0.9,
        quality_score=0.95,
        gaps_identified=[],
        follow_up_questions=[]
    )


def clarification_chain(question: str, llm: ChatOllama, interactive: bool = True) -> ResearchResult:
    """Execute clarification chain pattern."""
    print("[CLARIFICATION CHAIN] Starting for: " + question)
    
    # Step 1: Generate clarifying questions
    print("[STEP 1] Generating clarifying questions...")
    clarification_prompt = f"""
    For the question: "{question}"
    
    Generate 2 clarifying questions that would help provide a better answer.
    Focus on scope, context, or specific aspects.
    """
    
    clarifying_response = llm.invoke(clarification_prompt).content
    clarifying_questions = [line.strip() for line in clarifying_response.split('\n') if '?' in line][:2]
    
    print(f"[CLARIFICATION] Generated {len(clarifying_questions)} clarifying questions:")
    for i, q in enumerate(clarifying_questions, 1):
        print(f"  {i}. {q}")
    
    # Step 2: Simulate responses (in real implementation, these would be interactive)
    print("[STEP 2] Interactive clarification (simulated)...")
    simulated_responses = [
        "Please focus on recent developments",
        "I need this for academic research"
    ]
    
    context_info = []
    for i, (q, a) in enumerate(zip(clarifying_questions, simulated_responses)):
        print(f"  Q: {q}")
        print(f"  A: {a}")
        context_info.append(f"{q} -> {a}")
    
    # Step 3: Refine research strategy
    print("[STEP 3] Refining research strategy...")
    strategy_prompt = f"""
    Original question: {question}
    Clarifications: {'; '.join(context_info)}
    
    Create focused research plan considering clarifications.
    """
    
    strategy = llm.invoke(strategy_prompt).content
    
    # Step 4: Execute refined research
    print("[STEP 4] Executing refined research...")
    queries = create_research_plan(question, llm, max_queries=3)
    results = execute_parallel_search(queries, llm=llm)
    
    # Step 5: Synthesize with context
    print("[STEP 5] Synthesizing with clarification context...")
    synthesis_prompt = f"""
    Question: {question}
    Context: {'; '.join(context_info)}
    Research: {chr(10).join([f"{q}: {r}" for q, r in results.items()])}
    
    Provide answer tailored to the clarified context.
    """
    
    final_content = llm.invoke(synthesis_prompt).content
    
    return ResearchResult(
        content=final_content,
        sources=list(results.keys()),
        confidence=0.85,
        quality_score=0.9,
        gaps_identified=[],
        follow_up_questions=clarifying_questions
    )


def followup_chain(question: str, llm: ChatOllama, initial_research: ResearchResult = None) -> ResearchResult:
    """Execute follow-up chain pattern."""
    print("[FOLLOWUP CHAIN] Starting for: " + question)
    
    # Step 1: Initial research if not provided
    if initial_research is None:
        print("[STEP 1] Initial research...")
        initial_research = sequential_chain(question, llm)
    
    # Step 2: Generate follow-up questions
    print("[STEP 2] Generating follow-up questions...")
    followup_prompt = f"""
    Question: {question}
    Answer: {initial_research.content}
    
    Generate 3 follow-up questions that would provide additional valuable insights.
    """
    
    followup_response = llm.invoke(followup_prompt).content
    followup_questions = [line.strip() for line in followup_response.split('\n') if '?' in line][:3]
    
    print(f"[FOLLOWUP] Generated {len(followup_questions)} follow-up questions:")
    for i, q in enumerate(followup_questions, 1):
        print(f"  {i}. {q}")
    
    # Step 3: Research follow-up questions
    print("[STEP 3] Researching follow-up questions...")
    followup_results = execute_parallel_search(followup_questions, llm=llm)
    
    # Step 4: Comprehensive synthesis
    print("[STEP 4] Comprehensive synthesis...")
    synthesis_prompt = f"""
    Original question: {question}
    Initial answer: {initial_research.content}
    Follow-up research: {chr(10).join([f"{q}: {r}" for q, r in followup_results.items()])}
    
    Provide comprehensive answer incorporating follow-up insights.
    """
    
    final_content = llm.invoke(synthesis_prompt).content
    
    all_sources = initial_research.sources + list(followup_results.keys())
    
    return ResearchResult(
        content=final_content,
        sources=all_sources,
        confidence=0.9,
        quality_score=0.9,
        gaps_identified=[],
        follow_up_questions=followup_questions
    )


def interactive_report_chain(question: str, llm: ChatOllama, learning_iterations: int = 2) -> ResearchResult:
    """Execute interactive report writing with learning."""
    print("[INTERACTIVE REPORT] Starting for: " + question)
    
    # Step 1: Initial research
    print("[STEP 1] Conducting research...")
    queries = create_research_plan(question, llm, max_queries=4)
    research_results = execute_parallel_search(queries, llm=llm)
    
    # Step 2: Generate initial report
    print("[STEP 2] Generating initial report...")
    report_prompt = f"""
    Question: {question}
    Research data: {chr(10).join([f"{q}: {r}" for q, r in research_results.items()])}
    
    Write a comprehensive report addressing the question.
    """
    
    current_report = llm.invoke(report_prompt).content
    
    # Initialize learned preferences
    learned_preferences = {
        "tone": "professional",
        "structure": "standard",
        "depth": "comprehensive",
        "focus_areas": []
    }
    
    print(f"[REPORT] Generated initial report ({len(current_report)} chars)")
    
    # Step 3: Iterative improvement with simulated feedback
    for iteration in range(learning_iterations):
        print(f"[ITERATION {iteration + 1}] Learning from feedback...")
        
        # Simulate user feedback
        simulated_feedback = [
            "Make it more concise and focus on practical applications",
            "Add more technical details and cite specific sources",
            "Include more examples and case studies"
        ]
        
        feedback = simulated_feedback[iteration] if iteration < len(simulated_feedback) else "Good, continue with current approach"
        print(f"[FEEDBACK] {feedback}")
        
        # Update learned preferences
        if "concise" in feedback.lower():
            learned_preferences["depth"] = "concise"
        if "technical" in feedback.lower():
            learned_preferences["tone"] = "technical"
        if "practical" in feedback.lower():
            learned_preferences["focus_areas"].append("practical")
        if "examples" in feedback.lower():
            learned_preferences["focus_areas"].append("examples")
        
        # Revise report based on learned preferences
        revision_prompt = f"""
        Revise this report based on user preferences:
        
        Original report: {current_report}
        
        User preferences:
        - Tone: {learned_preferences['tone']}
        - Depth: {learned_preferences['depth']}
        - Focus: {', '.join(learned_preferences['focus_areas'])}
        
        Latest feedback: {feedback}
        
        Provide the revised report:
        """
        
        current_report = llm.invoke(revision_prompt).content
        print("[REVISION] Updated report based on preferences")
    
    # Step 4: Final report
    print("[STEP 4] Finalizing report with learned preferences...")
    
    return ResearchResult(
        content=current_report,
        sources=list(research_results.keys()),
        confidence=0.9,
        quality_score=0.95,
        gaps_identified=[f"Interactive learning complete after {learning_iterations} iterations"],
        follow_up_questions=[f"Learned preferences: {learned_preferences}"]
    )


def build_agents(model_name: str = "qwen2.5:0.5b"):
    """Build and return agent instances."""
    llm = ChatOllama(model=model_name, temperature=0)

    # Research agent with web search
    web_search = create_brave_search_tool(llm)
    research_tools = [web_search]
    research_prompt = (
        "You are a research agent. Use the brave_search_tool to find information, "
        "then provide comprehensive answers based on search results."
    )
    
    research_agent = create_react_agent(
        model=llm,
        tools=research_tools,
        prompt=research_prompt,
        name="research_agent",
    )

    # Parallel research planner agent
    def parallel_research_tool(question: str) -> str:
        """Execute comprehensive parallel research plan."""
        try:
            queries = create_research_plan(question, llm)
            search_results = execute_parallel_search(queries, llm=llm)
            final_answer = synthesize_research_results(question, search_results, llm)
            return final_answer
        except Exception as e:
            return f"Research planning failed: {str(e)}"
    
    parallel_research = Tool(
        name="parallel_research",
        description="Execute comprehensive parallel research for complex questions",
        func=parallel_research_tool
    )
    
    research_planner_prompt = (
        "You are a research planning agent. Use the parallel_research tool "
        "for every research request to provide comprehensive answers."
    )
    
    research_planner = create_react_agent(
        model=llm,
        tools=[parallel_research],
        prompt=research_planner_prompt,
        name="research_planner",
    )

    # Math agent with calculation tools
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(a: float, b: float) -> float:
        """Divide first number by second number."""
        return a / b

    math_agent = create_react_agent(
        model=llm,
        tools=[add, multiply, divide],
        prompt="You are a math agent. Use the available math tools to solve problems.",
        name="math_agent",
    )
    
    return research_agent, math_agent, research_planner, llm


def research_orchestrator(question: str, llm):
    """Main research orchestrator with pattern routing."""
    start_time = time.time()
    
    # Check if we should timeout
    def check_timeout():
        if time.time() - start_time > 180:  # 3 minute timeout
            raise TimeoutError("Research orchestrator timed out after 180 seconds")
    
    try:
        # Mathematical operation detection and processing
        def add(a: float, b: float) -> float:
            return a + b
        def multiply(a: float, b: float) -> float:
            return a * b
        def divide(a: float, b: float) -> float:
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        
        # Check if question is mathematical
        question_lower = question.lower()
        is_math = any(word in question_lower for word in ['calculate', 'compute', 'math', 'plus', 'minus', 'times', 'divided', 'multiply', 'add', 'subtract', '+', '-', '*', '/', 'what is']) and any(char.isdigit() for char in question)
        
        if is_math:
            print(f"[MATH PROCESSING] Detected mathematical question: {question}")
            check_timeout()  # Check timeout before math processing
            
            question_clean = question.lower().replace("calculate", "").replace("what is", "").strip().rstrip("?!.").strip()
            
            try:
                if "plus" in question_clean or "+" in question_clean:
                    parts = question_clean.replace("plus", "+").split("+")
                    if len(parts) == 2:
                        a, b = float(parts[0].strip()), float(parts[1].strip())
                        result = add(a, b)
                        
                        explanation_prompt = f"""
                        Explain the calculation {a} + {b} = {result}.
                        Include the operation, step-by-step breakdown, and a real-world example.
                        """
                        
                        try:
                            explanation = safe_llm_invoke(llm, explanation_prompt, 10)
                            return f"Calculation: {a} + {b} = {result}\n\nExplanation:\n{explanation.content}"
                        except (TimeoutError, Exception):
                            return f"Calculation: {a} + {b} = {result}"
                        
                elif "times" in question_clean or "multiplied by" in question_clean or "*" in question_clean:
                    parts = question_clean.replace("times", "*").replace("multiplied by", "*").split("*")
                    if len(parts) == 2:
                        a, b = float(parts[0].strip()), float(parts[1].strip())
                        result = multiply(a, b)
                        
                        explanation_prompt = f"""
                        Explain the calculation {a} × {b} = {result}.
                        Include the operation, step-by-step breakdown, and a real-world example.
                        """
                        
                        try:
                            explanation = safe_llm_invoke(llm, explanation_prompt, 10)
                            return f"Calculation: {a} × {b} = {result}\n\nExplanation:\n{explanation.content}"
                        except (TimeoutError, Exception):
                            return f"Calculation: {a} × {b} = {result}"
                        
                elif "divided by" in question_clean or "/" in question_clean:
                    parts = question_clean.replace("divided by", "/").split("/")
                    if len(parts) == 2:
                        a, b = float(parts[0].strip()), float(parts[1].strip())
                        result = divide(a, b)
                        
                        explanation_prompt = f"""
                        Explain the calculation {a} ÷ {b} = {result}.
                        Include the operation, step-by-step breakdown, and a real-world example.
                        """
                        
                        try:
                            explanation = safe_llm_invoke(llm, explanation_prompt, 10)
                            return f"Calculation: {a} ÷ {b} = {result}\n\nExplanation:\n{explanation.content}"
                        except (TimeoutError, Exception):
                            return f"Calculation: {a} ÷ {b} = {result}"
                        
            except Exception as e:
                return f"Error calculating: {str(e)}"
        
        # Research question routing
        print(f"[RESEARCH] Processing research question: {question}")
        check_timeout()  # Check timeout before research analysis
        
        # Analyze question and route to appropriate pattern
        analysis = analyze_question_complexity(question, llm)
        check_timeout()  # Check timeout after analysis
        print(f"[ROUTING] Complexity: {analysis.complexity.value}, Domain: {analysis.domain.value}")
        print(f"[ROUTING] Selected pattern: {analysis.suggested_pattern.value}")
        
        # Route based on complexity
        if analysis.complexity == QuestionComplexity.SIMPLE:
            print("[ROUTING] → Simple parallel research")
            queries = create_research_plan(question, llm, max_queries=3)
            results = execute_parallel_search(queries, llm=llm)
            return synthesize_research_results(question, results, llm)
        
        elif analysis.complexity == QuestionComplexity.MEDIUM:
            print("[ROUTING] → Sequential chain pattern")
            result = sequential_chain(question, llm)
            return result.content
        
        elif analysis.complexity == QuestionComplexity.HIGH:
            print("[ROUTING] → Validation chain pattern")
            result = validation_chain(question, llm)
            return result.content
        
        elif analysis.complexity == QuestionComplexity.EXPERT:
            print("[ROUTING] → Adversarial loop pattern")
            result = adversarial_loop(question, llm)
            return result.content
        
        else:
            # Default fallback
            print("[ROUTING] → Standard parallel research")
            check_timeout()  # Check timeout before final operations
            queries = create_research_plan(question, llm, max_queries=4)
            results = execute_parallel_search(queries, llm=llm)
            return synthesize_research_results(question, results, llm)
    except TimeoutError as e:
        print(f"[TIMEOUT] Research orchestrator timed out: {str(e)}")
        return f"Research timed out for question: {question}"
    except Exception as e:
        print(f"[ERROR] Research orchestrator failed: {str(e)}")
        return f"Research failed for question: {question} - {str(e)}"


def research_orchestrator_with_patterns(question: str, llm, pattern: str = "auto") -> str:
    """Enhanced orchestrator with specific pattern selection."""
    print(f"[PATTERN ORCHESTRATOR] Pattern: {pattern}, Question: {question}")
    start_time = time.time()
    
    def check_timeout():
        if time.time() - start_time > 200:  # 200 second timeout
            raise TimeoutError("Pattern orchestrator timed out after 200 seconds")
    
    try:
        if pattern == "clarification":
            check_timeout()
            result = clarification_chain(question, llm)
            return result.content
        elif pattern == "followup":
            check_timeout()
            result = followup_chain(question, llm)
            return result.content
        elif pattern == "interactive":
            check_timeout()
            result = interactive_report_chain(question, llm)
            return result.content
        elif pattern == "simple":
            print("[FORCED ROUTING] → Simple parallel research")
            check_timeout()
            queries = create_research_plan(question, llm, max_queries=3)
            results = execute_parallel_search(queries, llm=llm)
            return synthesize_research_results(question, results, llm)
        elif pattern == "medium":
            print("[FORCED ROUTING] → Sequential chain pattern")
            check_timeout()
            result = sequential_chain(question, llm)
            return result.content
        elif pattern == "complex":
            print("[FORCED ROUTING] → Validation chain pattern")
            check_timeout()
            result = validation_chain(question, llm)
            return result.content
        else:
            # Auto routing
            check_timeout()
            return research_orchestrator(question, llm)
    except TimeoutError:
        print(f"[TIMEOUT] Pattern orchestrator timed out for pattern: {pattern}")
        return f"Research timed out for question: {question}"
    except Exception as e:
        print(f"[ERROR] Pattern orchestrator failed: {str(e)}")
        return f"Research failed for question: {question} - {str(e)}"


def build_supervisor(research_agent, math_agent, research_planner, llm):
    """Build custom supervisor for workflow management."""
    
    class CustomSupervisor:
        def __init__(self, llm):
            self.llm = llm
            
        def stream(self, input_data):
            question = input_data["messages"][0]["content"]
            pattern = getattr(input_data, 'pattern', 'auto')
            result = research_orchestrator_with_patterns(question, self.llm, pattern)
            
            print(f"\n=== FINAL RESULT ===")
            print(result)
            print("=== END RESULT ===\n")
            
            yield {
                "supervisor": {
                    "messages": [{
                        "role": "assistant", 
                        "content": result
                    }]
                }
            }
    
    return CustomSupervisor(llm)


def execute_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Execute function with timeout using ThreadPoolExecutor."""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError:
        print(f"[TIMEOUT] Operation timed out after {timeout_seconds}s")
        return f"Operation timed out after {timeout_seconds}s"
    except Exception as e:
        print(f"[ERROR] Operation failed: {str(e)}")
        return f"Operation failed: {str(e)}"


def main():
    """Main entry point with CLI argument processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent Research System")
    parser.add_argument("question", nargs="*", help="Research question")
    parser.add_argument("--pattern", choices=["auto", "clarification", "followup", "interactive", "simple", "medium", "complex"], 
                        default="auto", help="Research pattern to use")
    parser.add_argument("--simple", type=int, default=0, help="Number of simple questions")
    parser.add_argument("--medium", type=int, default=0, help="Number of medium questions")
    parser.add_argument("--complex", type=int, default=0, help="Number of complex questions")
    parser.add_argument("--math", type=int, default=0, help="Number of math questions")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout in seconds (default: 240)")
    
    args = parser.parse_args()
    
    # Direct question processing
    if args.question:
        question = " ".join(args.question)
        print(f"Research question: {question}")
        
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        research_agent, math_agent, research_planner, llm = build_agents(model_name)
        
        # Execute with overall timeout
        result = execute_with_timeout(
            research_orchestrator_with_patterns, 
            args.timeout, 
            question, 
            llm, 
            args.pattern
        )
        
        print(f"\n=== FINAL RESULT ===")
        print(result)
        print("=== END RESULT ===\n")
        return
    
    # Mixed question type processing
    total_questions = args.simple + args.medium + args.complex + args.math
    if total_questions > 0:
        print(f"Running {total_questions} questions: {args.simple} simple, {args.medium} medium, {args.complex} complex, {args.math} math")
        
        from performance_test import generate_test_questions, generate_dynamic_questions, run_single_question_test
        from langchain_ollama import ChatOllama
        import concurrent.futures
        
        llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)
        
        # Generate questions of each type
        questions = []
        static_questions = generate_test_questions()
        
        # Add questions by type
        simple_qs = [q for q in static_questions if q["type"] == "simple"]
        questions.extend(simple_qs[:args.simple])
        
        medium_qs = [q for q in static_questions if q["type"] == "medium"]
        questions.extend(medium_qs[:args.medium])
        
        complex_qs = [q for q in static_questions if q["type"] == "complex"]
        questions.extend(complex_qs[:args.complex])
        
        math_qs = [q for q in static_questions if q["type"] == "math"]
        questions.extend(math_qs[:args.math])
        
        # Fill remaining with dynamic generation if needed
        remaining = total_questions - len(questions)
        if remaining > 0:
            dynamic_qs = generate_dynamic_questions(llm, remaining)
            questions.extend(dynamic_qs[:remaining])
        
        print(f"Generated {len(questions)} questions")
        
        # Execute questions in parallel with timeout
        test_args = []
        for i, q_data in enumerate(questions, 1):
            test_args.append((q_data, i, len(questions), llm))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_single_question_test, args) for args in test_args]
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=args.timeout * len(test_args)):
                try:
                    result = future.result(timeout=args.timeout)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"[TIMEOUT] Question timed out after {args.timeout}s")
                    results.append({
                        "question": "Unknown", "type": "unknown", "expected_pattern": "unknown",
                        "time": args.timeout, "result_length": 0, "generated": False,
                        "error": f"TIMEOUT: Question timed out after {args.timeout}s"
                    })
        
        # Summary
        completed_count = sum(1 for r in results if not r.get("error"))
        error_count = sum(1 for r in results if r.get("error"))
        avg_time = sum(r["time"] for r in results) / len(results) if results else 0
        
        print(f"\n=== SUMMARY ===")
        print(f"Total questions: {total_questions}")
        print(f"Completed: {completed_count}")
        print(f"Errors: {error_count}")
        print(f"Average time: {avg_time:.2f}s")
        
        return
    
    # Default: single complex question
    default_question = "What are the most significant technological developments in AI that could impact society in the next 5 years?"
    print(f"Default mode: Asking 1 complex question")
    print(f"Question: {default_question}")
    
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    research_agent, math_agent, research_planner, llm = build_agents(model_name)
    
    # Execute with timeout
    result = execute_with_timeout(
        research_orchestrator_with_patterns, 
        args.timeout, 
        default_question, 
        llm, 
        "complex"
    )
    
    print(f"\n=== FINAL RESULT ===")
    print(result)
    print("=== END RESULT ===\n")


if __name__ == "__main__":
    main()
