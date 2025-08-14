# Contributing to Research Brief Generator

We love your input! We want to make contributing to the Research Brief Generator as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Request Process

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Standards

### Python Code Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://github.com/PyCQA/isort) for import sorting
- Maximum line length: 100 characters

### Type Hints
- All functions must include type hints
- Use `typing` module for complex types
- Prefer `Optional[T]` over `Union[T, None]`

```python
def process_brief(request: BriefRequest) -> FinalBrief:
    """Process a research brief request."""
    pass
```

### Documentation
- All public functions and classes must have docstrings
- Follow Google-style docstrings
- Include examples for complex functions

```python
def search_sources(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    Search for sources using the given query.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
        
    Raises:
        SearchAPIError: If the search API fails
        
    Example:
        >>> results = search_sources("AI in healthcare", max_results=5)
        >>> len(results) <= 5
        True
    """
    pass
```

### Testing
- Minimum 90% test coverage
- Use pytest for testing
- Write both unit and integration tests
- Mock external dependencies

```python
@pytest.mark.asyncio
async def test_workflow_execution(mock_llm_manager):
    """Test complete workflow execution."""
    # Arrange
    request = BriefRequest(...)
    
    # Act
    result = await workflow.run_workflow(request)
    
    # Assert
    assert isinstance(result, FinalBrief)
```

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat(workflow): add retry logic for failed nodes
fix(api): handle validation errors gracefully
docs(readme): update installation instructions
```

## Setting Up Development Environment

1. **Clone the repository**
```bash
git clone https://github.com/username/research-brief-generator.git
cd research-brief-generator
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Set up pre-commit hooks**
```bash
pre-commit install
```

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API keys for testing
```

6. **Run tests**
```bash
pytest tests/ -v --cov=app
```

## Development Workflow

### Branch Naming
- Feature branches: `feature/description-of-feature`
- Bug fixes: `fix/description-of-bug`
- Documentation: `docs/description-of-changes`
- Hotfixes: `hotfix/description-of-fix`

### Development Process
1. Create an issue describing the bug or feature
2. Create a branch from `main`
3. Make your changes
4. Add tests for your changes
5. Update documentation if needed
6. Run tests and ensure they pass
7. Submit a pull request

### Code Review Process
1. All PRs require at least one review
2. All CI checks must pass
3. Maintainers will review for:
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Breaking changes

## Testing Guidelines

### Test Structure
```
tests/
├── conftest.py          # Shared fixtures
├── test_models.py       # Model validation tests
├── test_nodes.py        # Individual node tests
├── test_workflow.py     # Workflow integration tests
├── test_api.py          # API endpoint tests
└── test_llm_tools.py    # LLM and tool tests
```

### Writing Tests
- One test file per module
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common setup
- Mock external dependencies

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests in parallel
pytest -n auto

# Run only failed tests
pytest --lf
```

## Documentation

### API Documentation
- Use FastAPI automatic docs generation
- Add comprehensive examples
- Document all error responses

### Code Documentation
- Docstrings for all public APIs
- Type hints for all functions
- Inline comments for complex logic

### User Documentation
- Update README.md for user-facing changes
- Add examples to documentation
- Include troubleshooting guides

## Issue Reporting

### Bug Reports
Include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs
- Minimal code example

### Feature Requests
Include:
- Clear description of the feature
- Use case and motivation
- Possible implementation approach
- Breaking changes considerations

## Performance Guidelines

### Code Performance
- Use async/await for I/O operations
- Implement proper caching strategies
- Profile code for bottlenecks
- Use appropriate data structures

### Resource Management
- Close resources properly (files, connections)
- Use context managers
- Implement proper error handling
- Monitor memory usage

## Security Guidelines

### API Security
- Validate all inputs
- Use proper authentication
- Implement rate limiting
- Sanitize outputs

### Dependency Security
- Keep dependencies updated
- Run security audits
- Use pinned versions in production
- Monitor for vulnerabilities

## Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Steps
1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Merge to main
5. Tag release
6. Deploy to production

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers
- Follow the [Contributor Covenant](https://www.contributor-covenant.org/)

### Getting Help
- Check existing issues first
- Use GitHub Discussions for questions
- Join our community chat
- Read the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

Thank you for contributing to the Research Brief Generator! 🚀
