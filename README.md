# LangGraph Research Orchestrator

A sophisticated multi-agent research system that dynamically routes questions through different analysis patterns based on complexity and domain. Built with LangGraph and local Ollama models for privacy-focused AI research.

## Features

### Research Patterns
- **Sequential Chains**: Multi-step research with follow-up analysis
- **Validation Chains**: Credibility assessment and cross-verification
- **Adversarial Loops**: Counter-argument analysis for balanced perspectives
- **Clarification Chains**: Interactive question refinement
- **Follow-up Chains**: Deep-dive research with iterative questioning
- **Interactive Reports**: Adaptive report writing with learning

### Intelligent Routing
- **Complexity Analysis**: Automatic classification (simple, medium, high, expert)
- **Domain Detection**: Specialized handling for scientific, medical, financial, historical topics
- **Pattern Selection**: Optimal research strategy based on question characteristics

### Capabilities
- **Web Search Integration**: Brave Search API with LLM knowledge fallback
- **Mathematical Processing**: Built-in calculation engine
- **Parallel Research**: Concurrent query execution for efficiency
- **Performance Testing**: Comprehensive timing analysis framework

## Quick Start

```bash
# Install Ollama and pull model
brew install ollama
ollama serve &
ollama pull qwen2.5:0.5b

# Setup Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Set Brave Search API key for web search
export BRAVE_API_KEY="your_brave_api_key_here"
# Get your free API key from: https://api.search.brave.com/
# Note: If not set, the app will fall back to LLM knowledge

# Run with direct question
python supervisor_ollama.py "What are the environmental impacts of electric vehicles?"

# Run with specific pattern
python supervisor_ollama.py "What is machine learning?" --pattern clarification

# Run performance tests
python performance_test.py -q 5 -i 3 --static
```

## Usage Examples

### Direct Questions
```bash
# Simple factual question
python supervisor_ollama.py "What is the capital of Japan?"

# Complex analysis question
python supervisor_ollama.py "Compare renewable energy vs nuclear power"

# Mathematical calculation
python supervisor_ollama.py "Calculate 156 times 23"
```

### Pattern Selection
```bash
# Force validation chain for rigorous analysis
python supervisor_ollama.py "Climate change causes" --pattern complex

# Use clarification chain for unclear questions
python supervisor_ollama.py "AI impacts" --pattern clarification

# Generate follow-up questions
python supervisor_ollama.py "Quantum computing" --pattern followup
```

### Batch Testing
```bash
# Mixed question types
python supervisor_ollama.py --simple 2 --medium 2 --complex 1 --math 1

# Performance benchmarking
python performance_test.py -q 10 -i 5 -p 3
```

## Architecture

### Core Components

1. **Question Analysis Engine**
   - Complexity classification
   - Domain identification
   - Pattern recommendation

2. **Research Orchestrator**
   - Dynamic pattern routing
   - Parallel search execution
   - Result synthesis

3. **Agent Framework**
   - Research agents with web search
   - Mathematical calculation agents
   - Specialized domain handlers

4. **Performance Monitor**
   - Timing analysis
   - Pattern effectiveness
   - Scalability testing

### Research Patterns

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| Sequential Chain | Medium complexity | Step-by-step analysis |
| Validation Chain | High complexity | Credibility verification |
| Adversarial Loop | Expert level | Counter-argument analysis |
| Clarification Chain | Ambiguous questions | Interactive refinement |
| Follow-up Chain | Deep research | Iterative questioning |
| Interactive Report | Report generation | Adaptive writing |

## Configuration

### Environment Variables
```bash
export OLLAMA_MODEL=qwen2.5:0.5b       # Model selection
export BRAVE_API_KEY="your_api_key"    # Optional: Brave Search API key
```

### Search Configuration
- **With BRAVE_API_KEY**: Uses Brave Search API for current web results
- **Without API key**: Falls back to LLM knowledge (may have knowledge cutoff limitations)
- **API Key Setup**: Get free API key from https://api.search.brave.com/

### Model Options
- `qwen2.5:0.5b` - Fast, lightweight (default)
- `qwen2.5:1.5b` - Balanced performance
- `llama3:8b` - Higher quality, slower
- `mistral:7b` - Alternative high-quality option

## Performance Testing

The framework includes comprehensive performance analysis:

```bash
# Basic timing test
python performance_test.py -q 5 -i 1

# Parallel execution test
python performance_test.py -q 10 -p 5

# Static vs dynamic questions
python performance_test.py --static -q 8
python performance_test.py -q 8  # dynamic mode
```

### Test Results Analysis
- Question completion times
- Pattern effectiveness
- Error rates and handling
- Parallel execution efficiency
- Dynamic question generation success

## File Structure

```
├── supervisor_ollama.py      # Main orchestrator and agents
├── performance_test.py       # Testing framework
├── requirements.txt          # Python dependencies
├── setup.sh                  # Environment setup script
└── README.md                 # Documentation
```

## Requirements

- Python 3.8+
- Ollama installation
- qwen2.5:0.5b model (or compatible)
- Internet connection for web search

## Dependencies

```
langgraph>=0.0.35
langchain-ollama>=0.1.0
langchain-community>=0.2.0
langchain-core>=0.2.0
brave-search>=1.0.0
requests>=2.31.0
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details.

## Performance Notes

- **qwen2.5:0.5b**: ~2-4GB RAM, fast responses
- **Parallel search**: 3-5 concurrent queries optimal
- **Typical timing**: 10-60s per question depending on complexity
- **Web search**: 20s timeout per query for reliability

## Troubleshooting

### Common Issues

1. **Ollama not found**: Ensure Ollama is installed and running
2. **Model not available**: Run `ollama pull qwen2.5:0.5b`
3. **Search timeout**: Check internet connection
4. **Import errors**: Verify all dependencies installed

### Debug Mode

```bash
# Enable verbose logging
python supervisor_ollama.py "question" --pattern auto

# Test individual components
python performance_test.py -q 1 -i 1 --static
```

## Acknowledgments

Built on the LangGraph framework for multi-agent systems. Inspired by research in autonomous AI agents and dynamic workflow orchestration.
