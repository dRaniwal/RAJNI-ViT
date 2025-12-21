# Contributing to RAJNI-ViT

Thank you for your interest in contributing to RAJNI-ViT! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/RAJNI-ViT.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Write descriptive docstrings for all functions and classes
- Keep functions focused and modular

### Adding New Features

When adding new features, please:

1. **Update documentation**: Add usage examples to README.md
2. **Add configuration options**: If applicable, add to config files
3. **Test your changes**: Ensure code runs without errors
4. **Add examples**: Demonstrate new functionality in scripts/

### Areas for Contribution

- **Importance Metrics**: Implement new token importance computation methods
- **Model Support**: Add support for different ViT architectures
- **Dataset Loaders**: Implement real dataset loading (ImageNet, CIFAR, etc.)
- **Optimization**: Improve pruning efficiency
- **Visualization**: Add tools for visualizing pruned tokens
- **Documentation**: Improve examples and tutorials

## Pull Request Process

1. Update README.md with details of changes if applicable
2. Ensure all Python files are syntactically correct
3. Update configuration files if you add new parameters
4. Write a clear description of your changes in the PR

## Code Structure

```
rajni/
├── pruning/      # Token pruning implementations
├── utils/        # Utilities (logging, metrics)
└── config/       # Configuration management

scripts/          # Executable scripts
configs/          # YAML configuration files
```

## Testing

Currently, the repository uses placeholder implementations. When contributing:

- Ensure your code doesn't break existing functionality
- Test with actual ViT models when possible
- Verify config loading works correctly

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the code
- Discussion about methods

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
