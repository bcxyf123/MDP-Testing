# MDP Testing Framework

This is the repo for paper [**TBD**](https://arxiv.org/abs/placeholder).

The RL environments used in this project support gym interface, or custom environments can be defined.

## 🏗️ Project Structure

```
testing/
├── algs/                   # OOD algorithms for the baseline detectors
│   ├── pedm/               
│   └── ...                 
├── configs/               
│   ├── maze.yaml          # Maze config
│   └── gym.yaml           # Gym config
├── detector/               # Core detection modules
│   ├── detector.py        # detectors
│   ├── sampler.py         # sample_function for gym
│   └── agent.py           # Agent interfaces
├── envs/                   # Environment implementations
│   ├── maze/              # Custom maze environments
│   ├── carl/              # CARL (Contextual RL) environments
│   ├── bandit/            # Multi-armed bandit environments
├── examples/               # Example usage scripts
│   ├── maze.py            # Maze environment example
│   └── gym.py             # Gym environment example
├── utils/                  # Utility functions
└── ...
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
conda create -n testing
conda activate testing

pip install -r requirements.txt

# For CARL environments, please install CARL following the official instructions:
# https://automl.github.io/CARL/main/source/getting_started/installation.html
```


### Basic Usage

#### 1. Maze Environment Example

```bash
# Run maze environment MDP testing
python -m testing.examples.maze
```

modify the parameters in the `testing/configs/maze.yaml` or create a custom config file

#### 2. Gym Environment Example

```bash
# Run gym environment MDP testing
python -m testing.examples.gym
```
modify the parameters in the `testing/configs/gym.yaml` or create a custom config file


<!-- ## 📊 Statistical Tests

The framework supports various statistical tests for MDP comparison:

### 1. **Student's t-test** (`detector_type: "t"`)
- Tests for differences in means between two samples
- Assumes equal variances and normal distribution

### 2. **Welch's t-test** (`detector_type: "Welchs-t"`)  
- Modified t-test for unequal variances
- More robust than standard t-test

### 3. **Mann-Whitney U test** (`detector_type: "mann-whitney-u"`)
- Non-parametric test for distribution differences  
- Does not assume normality

### 4. **Rank t-test** (`detector_type: "rank-t"`)
- Combines ranking with t-test methodology

### 5. **Likelihood Ratio Test** (`detector_type: "lrt"`)
- Uses likelihood ratios for detection
- Requires `data_type: "likelihood-ratio"`

### 6. **Maximum Mean Discrepancy** (`detector_type: "mmd"`)
- Kernel-based test for distribution differences
- Supports various kernel functions
 -->

<!-- ## 🧪 Environments

### Custom Maze Environment
- Grid-world maze navigation
- Configurable obstacles, start/goal positions
- Stochastic transitions with adjustable probabilities

### OpenAI Gym Environments  
- Support for various Gym environments (Acrobot-v1, etc.)
- CARL framework integration for contextual variations
- Parameter perturbation for environment testing

### Multi-Armed Bandits
- Bernoulli bandit implementations
- Configurable arm probabilities
- Suitable for simple MDP testing scenarios


Or define your own environment. -->

## 📚 Acknowledgments

This project builds upon the following open-source project:

[gym-simplegrid](https://github.com/damat-le/gym-simplegrid.git)

[CARL](https://github.com/automl/CARL.git)
