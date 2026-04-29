# Contributing to T2I-RL

Thank you for your interest in contributing to T2I-RL! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- CUDA-capable GPU (for testing)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/T2I-RL-Project.git
   cd T2I-RL-Project
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/Inoriany/T2I-RL-Project.git
   ```

## Development Setup

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate t2i-rl

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_generators.py

# Run with coverage
pytest --cov=src tests/
```

## Making Contributions

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(generator): add support for FLUX model
fix(reward): handle empty image list in VLM reward
docs(readme): update installation instructions
```

### Pull Request Process

1. Update your fork with latest upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

3. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```

5. Open a Pull Request on GitHub

### PR Requirements

- [ ] Code passes all tests
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention
- [ ] PR description explains changes

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for function signatures
- Use docstrings for all public functions/classes

```python
def compute_reward(
    images: List[Image.Image],
    prompts: List[str],
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute rewards for image-prompt pairs.
    
    Args:
        images: List of PIL images
        prompts: List of text prompts
        normalize: Whether to normalize rewards
        
    Returns:
        Tensor of reward scores
    """
    pass
```

### Formatting Tools

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

The project uses pre-commit hooks for:
- black (formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

## Testing

### Test Structure

```
tests/
├── test_generators.py      # Generator tests
├── test_reward_models.py   # Reward model tests
├── test_trainers.py        # Trainer tests
├── test_evaluation.py      # Evaluation tests
└── conftest.py            # Pytest fixtures
```

### Writing Tests

```python
import pytest
from src.models.generators import JanusProGenerator

class TestJanusProGenerator:
    @pytest.fixture
    def generator(self):
        return JanusProGenerator(device="cpu")
    
    def test_generate_single_prompt(self, generator):
        images = generator.generate(["a red apple"])
        assert len(images) == 1
        assert images[0].size == (384, 384)
    
    def test_generate_with_logprobs(self, generator):
        images, logprobs, ids = generator.generate_with_logprobs(
            ["a red apple"]
        )
        assert logprobs.shape[1] == 576  # num visual tokens
```

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **Smoke tests**: Quick sanity checks

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> List[str]:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2. Defaults to 10.
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is empty
        
    Example:
        >>> function("test", 5)
        ['test', 'test', 'test', 'test', 'test']
    """
```

### Updating Documentation

1. API changes: Update `docs/api_reference.md`
2. Algorithm changes: Update `docs/methodology.md`
3. Architecture changes: Update `docs/architecture.md`
4. User-facing changes: Update `README.md`

## Areas for Contribution

### Good First Issues

- Add more evaluation metrics
- Improve documentation
- Add example notebooks
- Fix typos and bugs

### Advanced Contributions

- New generator implementations
- New reward model types
- Training algorithm improvements
- Benchmark integrations

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email the maintainers for private matters

Thank you for contributing!
