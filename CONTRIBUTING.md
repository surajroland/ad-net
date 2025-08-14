# Contributing to ADNet

Thank you for your interest in contributing to ADNet! 🚀 We welcome contributions from the community and are excited to work with you.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Dataset Contributions](#dataset-contributions)
- [Performance Considerations](#performance-considerations)

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and professional in all interactions.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of computer vision and 3D object detection
- Familiarity with PyTorch

### First Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ad-net.git
   cd ad-net
   ```
3. **Set up the development environment** (see below)
4. **Look for good first issues** labeled with `good-first-issue`

## 🛠️ Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install in development mode with all dependencies
pip install -e ".[dev,docs,training,visualization]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation
```bash
# Run basic tests
pytest tests/data/datasets/test_nuscenes_dataset.py::TestNuScenesDataset::test_dataset_initialization -v

# Check code style
pre-commit run --all-files
```

## 📝 Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **🐛 Bug fixes**
2. **✨ New features**
3. **📚 Documentation improvements**
4. **🧪 Test additions**
5. **⚡ Performance optimizations**
6. **🔧 Refactoring**
7. **🎯 Dataset support**

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss new features before implementing
3. **Follow the coding standards** outlined below
4. **Write tests** for new functionality

### Coding Standards

#### Python Style
- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [Google style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

#### Code Quality
- Use meaningful variable and function names
- Keep functions focused and small
- Add docstrings to all public functions and classes
- Include type hints where appropriate

#### Example Function Structure
```python
def load_dataset_sample(
    sample_id: str, 
    dataset_path: Path, 
    load_images: bool = True
) -> Sample:
    """Load a single dataset sample with optional image loading.
    
    Args:
        sample_id: Unique identifier for the sample
        dataset_path: Path to the dataset root directory
        load_images: Whether to load image data along with metadata
        
    Returns:
        Sample object containing all loaded data
        
    Raises:
        FileNotFoundError: If sample files are not found
        ValueError: If sample_id format is invalid
    """
    # Implementation here
    pass
```

## 🔄 Pull Request Process

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for Waymo dataset loading

- Implement WaymoDataset class with full API compatibility
- Add coordinate system conversion utilities
- Include comprehensive test suite
- Update documentation with usage examples

Closes #123"
```

#### Commit Message Format
We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `style`: Code style changes

### 4. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots/examples if applicable

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/data/datasets/test_nuscenes_dataset.py

# Run with coverage
pytest --cov=src/adnet --cov-report=html

# Run only fast tests (exclude slow/integration tests)
pytest -m "not slow"
```

### Writing Tests

#### Unit Tests
- Test individual functions/methods
- Use mocking for external dependencies
- Focus on edge cases and error conditions

```python
def test_sample_loading_with_invalid_id():
    """Test that invalid sample IDs raise appropriate errors."""
    dataset = NuScenesDataset(data_root="/mock/path")
    
    with pytest.raises(ValueError, match="Invalid sample ID format"):
        dataset.get_sample("invalid-id-format")
```

#### Integration Tests
- Test component interactions
- Use real data when possible
- Mark with `@pytest.mark.integration`

```python
@pytest.mark.integration
def test_multi_dataset_harmonization():
    """Test that multi-dataset loading harmonizes coordinates correctly."""
    # Integration test implementation
    pass
```

### Test Coverage
- Aim for >80% code coverage
- Focus on critical paths and error handling
- Don't sacrifice test quality for coverage percentage

## 📚 Documentation

### Documentation Requirements
- Update docstrings for all modified functions
- Add examples for new features
- Update README.md if needed
- Add to tutorials for major features

### Building Documentation
```bash
cd docs
sphinx-build -b html . _build/html
```

### Documentation Style
- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Add links to relevant papers/resources

## 🎯 Dataset Contributions

### Adding New Dataset Support

When adding support for a new dataset:

1. **Research the dataset structure**
2. **Implement the dataset loader class**
3. **Add coordinate system conversion**
4. **Create comprehensive tests**
5. **Update documentation**
6. **Add example usage**

#### Dataset Loader Template
```python
@register_dataset("your_dataset")
class YourDataset(TemporalDataset, MultiModalDataset):
    """Dataset loader for Your Dataset.
    
    Supports the Your Dataset format with multi-view cameras,
    3D annotations, and temporal sequences.
    """
    
    CAMERA_NAMES = ["cam_front", "cam_back"]  # Define camera setup
    CLASS_NAMES = ["car", "pedestrian"]       # Define object classes
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        version: str = "v1.0",
        **kwargs
    ):
        # Implementation
        pass
    
    def _load_dataset_info(self) -> None:
        # Load dataset metadata
        pass
    
    def _load_annotations(self) -> None:
        # Load annotation data
        pass
    
    # Implement other required methods...
```

### Dataset Testing
- Test with various data splits
- Verify coordinate transformations
- Check temporal sequence handling
- Validate cross-dataset compatibility

## ⚡ Performance Considerations

### Guidelines
- Profile code for performance bottlenecks
- Use appropriate data structures
- Consider memory usage for large datasets
- Optimize data loading pipelines

### Benchmarking
```python
@pytest.mark.benchmark
def test_dataset_loading_performance(benchmark):
    """Benchmark dataset loading performance."""
    dataset = NuScenesDataset(data_root="/path/to/data")
    
    def load_sample():
        return dataset[0]
    
    result = benchmark(load_sample)
    assert result is not None
```

## 🤝 Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** on GitHub
3. **Ask questions** in GitHub Discussions
4. **Join our community** (links in README)

## 🙏 Recognition

Contributors will be:
- Listed in our AUTHORS file
- Credited in release notes for significant contributions
- Given recognition in documentation

## 📞 Contact

For questions about contributing:
- Open an issue on GitHub
- Email: hello@surajit.de
- GitHub: [@surajroland](https://github.com/surajroland)

Thank you for contributing to ADNet! 🎉