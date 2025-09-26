# Contributing to SVD Image Compression System

We welcome contributions to the SVD Image Compression System! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of linear algebra and image processing
- Familiarity with NumPy, SciPy, and Streamlit (helpful but not required)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/svd-image-compression.git
   cd svd-image-compression
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce the problem
- Expected vs actual behavior
- System information (OS, Python version)
- Screenshots or error messages (if applicable)

### ✨ Feature Requests

For new features, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant research or references

### 🔧 Code Contributions

We welcome contributions in these areas:
- **Core Algorithms**: Improvements to SVD compression implementation
- **Web Interface**: Enhanced UI/UX features for the Streamlit app
- **Visualization**: New plotting capabilities and analysis tools
- **Performance**: Optimization and efficiency improvements
- **Documentation**: Improvements to guides, examples, and API docs
- **Testing**: Additional test cases and coverage improvements

## 🛠️ Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Write comprehensive docstrings for all functions and classes
- Keep functions focused and modular
- Use meaningful variable and function names

### Code Structure

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of the function.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
    # Implementation here
    pass
```

### Testing Requirements

- Write unit tests for all new functions
- Ensure integration tests pass for workflow changes
- Maintain or improve test coverage
- Test edge cases and error conditions
- Include performance tests for optimization changes

### Documentation

- Update relevant documentation for any changes
- Include docstrings for new functions and classes
- Add usage examples for new features
- Update README if adding major features
- Ensure all links and references are correct

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/performance/   # Performance tests

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place performance tests in `tests/performance/`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases

Example test structure:
```python
import pytest
import numpy as np
from src.compression.svd_compressor import SVDCompressor

class TestSVDCompressor:
    def test_compress_channel_svd_basic(self):
        """Test basic SVD compression functionality."""
        compressor = SVDCompressor()
        channel = np.random.rand(64, 64)
        k = 10
        
        U, S, Vt = compressor.compress_channel_svd(channel, k)
        
        assert U.shape == (64, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, 64)
    
    def test_compress_channel_svd_edge_cases(self):
        """Test SVD compression with edge cases."""
        # Test with k larger than matrix rank
        # Test with zero matrix
        # Test with single pixel
        pass
```

## 📋 Pull Request Process

### Before Submitting

1. **Ensure tests pass**: Run the full test suite
2. **Check code style**: Follow PEP 8 guidelines
3. **Update documentation**: Include relevant documentation updates
4. **Test manually**: Verify your changes work as expected
5. **Rebase if needed**: Ensure your branch is up to date with main

### Pull Request Template

When submitting a pull request, please include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests and style checks
2. **Code review**: Maintainers review code for quality and correctness
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, changes will be merged

## 🎯 Priority Areas

We're particularly interested in contributions in these areas:

### High Priority
- **Performance optimization**: Faster SVD computation and memory efficiency
- **Additional quality metrics**: New evaluation methods and metrics
- **Enhanced visualizations**: Interactive plots and analysis tools
- **Mobile responsiveness**: Improved web app mobile experience

### Medium Priority
- **Additional image formats**: Support for more file types
- **Batch processing improvements**: Better progress tracking and error handling
- **Educational content**: More tutorials and examples
- **API documentation**: Comprehensive API reference

### Low Priority
- **Code refactoring**: Improved code organization and structure
- **Additional tests**: Increased test coverage
- **Documentation improvements**: Better examples and guides

## 🤝 Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Be Collaborative
- Help others learn and grow
- Share knowledge and resources
- Provide constructive feedback
- Support fellow contributors

### Be Professional
- Maintain professional communication
- Follow project guidelines and standards
- Respect maintainer decisions
- Keep discussions focused and productive

## 📞 Getting Help

If you need help with contributing:

1. **Check existing documentation**: README, usage guides, and API docs
2. **Search existing issues**: Your question might already be answered
3. **Ask in discussions**: Use GitHub Discussions for questions
4. **Contact maintainers**: Reach out directly for complex issues

## 🏆 Recognition

Contributors will be recognized in:
- **README contributors section**: All contributors are listed
- **Release notes**: Significant contributions are highlighted
- **Academic citations**: Research contributions are properly credited

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the SVD Image Compression System! Your efforts help make this project better for everyone. 🎉