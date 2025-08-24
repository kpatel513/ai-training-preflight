# **AI Training Pre-Flight Check**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Predict AI training failures through static analysis before wasting compute resources.**

Every commercial aircraft undergoes 115 mandatory checks before takeoff. Meanwhile, we launch expensive AI training runs with nothing more than a syntax check. This tool brings aviation-grade pre-flight validation to AI model training.

## **Motivation**

After watching multiple training runs fail after days or weeks of computation, I realized most failures were entirely predictable through static analysis. A script that will crash with out-of-memory errors at step 1,000 shows clear signs in its configuration at line 0. This tool identifies those signs before you waste time and money.

## **What It Does**

This tool performs static analysis on PyTorch training scripts to predict:

- **Memory overflow failures** with the exact step where OOM will occur
- **Training inefficiencies** that waste compute resources
- **Numerical instabilities** that cause gradient explosions or NaN losses
- **Configuration conflicts** that prevent successful training

## **Installation**

### **From Source**
```bash
git clone https://github.com/kpatel513/ai-training-preflight
cd ai-training-preflight
pip install -e .
```

### **Requirements**
- Python 3.8 or higher
- No GPU required (pure static analysis)
- Currently supports PyTorch training scripts

## **Usage**

### **Command Line**

Basic analysis:
```bash
preflight check your_training_script.py
```

Specify GPU memory:
```bash
preflight check train.py --gpu-memory 80
```

Generate JSON report:
```bash
preflight check train.py --format json -o analysis_report.json
```

Quick pass/fail check:
```bash
preflight quick train.py
```

### **Python API**

```python
from preflight import analyze

# Simple analysis
result = analyze("train.py")

if result.critical_errors:
    print("Training will fail:")
    for error in result.critical_errors:
        print(f"  - {error}")
```

Advanced usage:
```python
from preflight import TrainingPreflightAnalyzer

# Custom configuration
analyzer = TrainingPreflightAnalyzer(gpu_memory_gb=80.0)
result = analyzer.analyze("train.py")

# Access detailed breakdown
print("Memory breakdown:")
for component, size in result.breakdown.items():
    print(f"  {component}: {size}")
```

## **How It Works**

### **Static Analysis Pipeline**

1. **AST Parsing**: Extracts model architecture, training configuration, and data pipeline setup from your Python code

2. **Memory Calculation**: Computes exact memory requirements using deterministic formulas:
   ```
   Total Memory = Model Parameters + Optimizer States + Gradients + Activations
   ```

3. **Pattern Recognition**: Identifies common failure patterns like quadratic attention scaling, distributed training imbalances, and gradient accumulation overflows

4. **Predictive Modeling**: Simulates training execution to predict failures and bottlenecks

### **Key Formulas**

For Transformer models:
```python
# Attention memory (quadratic in sequence length)
attention_memory = seq_len^2 * num_heads * num_layers * batch_size * dtype_bytes

# Optimizer memory (Adam/AdamW)
optimizer_memory = model_parameters * 2 * dtype_bytes  # Momentum + variance
```

## **Example Analysis**

Running on a script with configuration issues:
```bash
preflight check examples/memory_overflow.py
```

Output:
```
AI TRAINING PRE-FLIGHT CHECK
================================================================

MEMORY ANALYSIS:
   Peak Usage: 206.4GB / 40GB
   Breakdown:
     - model_parameters: 6.24 GB
     - optimizer_states: 12.48 GB
     - gradients: 6.24 GB
     - activations: 181.44 GB
   
   MEMORY OVERFLOW PREDICTED
   Will fail at step 1001

EFFICIENCY ANALYSIS:
   Score: 42%
   Severe efficiency problems detected
   
WARNINGS:
   1. Variable sequence lengths detected - memory usage will fluctuate
   2. Batch size too small for model size - poor GPU utilization
   3. DataLoader using single process
   
CRITICAL ERRORS:
   1. MEMORY OVERFLOW: 206.4GB exceeds 40GB GPU memory
   2. Predicted OOM at step 1001
   
NO GO - Fix critical errors before training
================================================================
```

## **Examples**

The repository includes example training scripts demonstrating various failure modes:

- `examples/memory_overflow.py` - Demonstrates OOM failure from variable sequence lengths
- `examples/inefficient_distributed.py` - Shows distributed training bottlenecks
- `examples/gradient_explosion.py` - Illustrates numerical instability risks
- `examples/good_model.py` - Properly configured training script

## **Understanding the Output**

### **Memory Analysis**
Shows peak memory usage and breakdown by component:
- Model parameters: Weight storage
- Optimizer states: Adam/SGD momentum and variance
- Gradients: Backward pass storage
- Activations: Forward pass intermediate tensors

### **Efficiency Score**
- 80-100%: Well-optimized configuration
- 60-79%: Minor improvements possible
- 40-59%: Significant inefficiencies
- Below 40%: Severe problems requiring attention

### **Recommendations**
Each recommendation includes the specific change needed and expected impact on training performance.

## **Technical Details**

### **Supported Frameworks**
- PyTorch (full support)
- TensorFlow (planned)
- JAX/Flax (planned)

### **Analysis Capabilities**
- Transformer architectures (BERT, GPT, T5 variants)
- Convolutional networks (ResNet, EfficientNet families)
- Distributed training configurations (DDP, FSDP)
- Mixed precision training (AMP, FP16)
- Gradient accumulation strategies

### **Limitations**
- Requires relatively standard training loop structure
- Dynamic model architectures may not be fully analyzed
- Custom CUDA kernels are not evaluated
- Runtime data variations are estimated, not guaranteed

## **Project Structure**

```
ai-training-preflight/
├── preflight/           # Core analysis engine
│   ├── analyzers/       # Memory, efficiency, stability analyzers
│   ├── extractors/      # AST-based information extraction
│   └── utils/           # Reporting and utilities
├── examples/            # Example training scripts
├── tests/              # Test suite
└── docs/               # Documentation
```

## **Contributing**

This is a personal project, but contributions are welcome. Areas of interest:

- Extending support to other frameworks
- Improving accuracy of predictions
- Adding new analysis patterns
- Performance optimizations

Please open an issue to discuss potential changes before submitting pull requests.

## **Future Work**

Planned improvements:

- TensorFlow and JAX support
- Automatic fix generation for common issues
- Training time and cost estimation
- Integration with cloud platforms
- VSCode extension for real-time analysis

## **Citation**

If you find this tool useful in your work, please consider citing:

```bibtex
@software{preflight2025,
  title={AI Training Pre-Flight Check: Predicting Deep Learning Training Failures through Static Analysis},
  author={Krunal Patel},
  year={2025},
  url={https://github.com/kpatel513/ai-training-preflight}
}
```

## **License**

MIT License - see [LICENSE](LICENSE) for details.

## **Contact**

- **GitHub Issues**: [https://github.com/kpatel513/ai-training-preflight/issues](https://github.com/kpatel513/ai-training-preflight/issues)
- **Author**: Krunal Patel

---

*Built from the frustration of watching too many preventable training failures.*
