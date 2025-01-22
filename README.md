# Noisy Logic Bayesian Inference

A Bayesian Networks approach for inferring active Transcription Factors
using logic models of transcriptional regulation.

## Examples

The package includes several examples to help you get started:

### Jupyter Notebooks
- [TF-inference.ipynb](nlbayes/examples/TF-inference.ipynb) - Basic TF inference workflow
- [TF-inference-simulation.ipynb](nlbayes/examples/TF-inference-simulation.ipynb) - Simulation-based inference example

### Python Scripts
- [tf_inference_simulation.py](nlbayes/examples/tf_inference_simulation.py) - Command-line example demonstrating TF inference with simulated data, including:
  - Random network generation
  - Evidence simulation
  - OR-NOR model inference
  - Performance evaluation (ROC/PR curves)
  - Terminal-based visualization

## Installation Instructions

### Prerequisites

The package requires the GNU Scientific Library (GSL). Install it for your operating system:

#### Linux (Debian/Ubuntu)
```bash
sudo apt install -y libgsl-dev
```

#### macOS
```bash
brew install gsl
```

#### Windows
Download the GSL binaries from [MSYS2](https://www.msys2.org/)
1. Install MSYS2 and open the MSYS2 terminal
2. Install GSL:
```bash
pacman -S mingw-w64-x86_64-gsl
```
4. Add the MSYS2 binary path (typically `C:\msys64\mingw64\bin`) to your system's PATH environment variable

### Package Installation

Install nlbayes directly from GitHub:
```bash
pip install git+https://github.com/umbibio/nlbayes-python.git
```

## Usage

```python
from nlbayes import ORNOR

# Create and fit the model
model = ORNOR(network, evidence, n_graphs=5, uniform_prior=False)
model.fit(n_samples=2000, gelman_rubin=1.1, burnin=True)

# Get inference results
results = model.get_results()
print(results)
```

### Input Data Format

The input data for the `ORNOR` class should be formatted as follows:

- `network`: A dictionary where keys are TF IDs and values are dictionaries mapping target gene names to regulation modes (-1 for repression, 1 for activation)
- `evidence`: A dictionary mapping gene names to expression states (-1 for down-regulated, 1 for up-regulated)

For more detailed examples, check the example notebooks and scripts in the `nlbayes/examples` directory.
