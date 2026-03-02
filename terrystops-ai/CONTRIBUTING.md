# Contributing to TerryStops-AI

Thank you for your interest in contributing! This document outlines the process and guidelines for contributing to the project.

---

## How to Contribute

1. **Fork** the repository.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit with clear, descriptive messages.
4. **Push** your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** against the `main` branch of this repository.

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/<your-username>/Terry--Traffic--Stops-Prediction.git
cd Terry--Traffic--Stops-Prediction/terrystops-ai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py

# Run tests
pytest --cov=app tests/
```

---

## Code Style

- Follow **[PEP 8](https://peps.python.org/pep-0008/)** for all Python code.
- Use **[flake8](https://flake8.pycqa.org/)** to lint your changes before submitting:
  ```bash
  flake8 app/ ml/ tests/
  ```
- Write **Google-style docstrings** for all public modules, classes, and functions:
  ```python
  def predict(features: dict) -> dict:
      """Generate an arrest-outcome prediction.

      Args:
          features: A dictionary of input features for the model.

      Returns:
          A dictionary containing the prediction and confidence score.
      """
  ```

---

## Testing Requirements

- **All existing tests must pass** before a PR will be reviewed.
- **Maintain ≥ 80 % code coverage.** Run the suite with:
  ```bash
  pytest --cov=app tests/
  ```
- Add tests for any new functionality. Place test files in the `tests/` directory following the existing naming convention (`test_*.py`).

---

## Pull Request Guidelines

When opening a PR, please ensure:

1. **Description** — Provide a clear summary of *what* changed and *why*.
2. **Tests** — Include new or updated tests that cover your changes.
3. **No breaking changes** — Existing functionality and API contracts must remain intact. If a breaking change is unavoidable, discuss it in an issue first.
4. **Small, focused PRs** — Keep pull requests scoped to a single feature or fix.
5. **Documentation** — Update the README or other docs if your change affects usage.

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold a welcoming, inclusive, and harassment-free environment.

---

## Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates.
2. **Open a new issue** with a descriptive title and as much context as possible:
   - Steps to reproduce (for bugs).
   - Expected vs. actual behavior.
   - Environment details (OS, Python version, etc.).
3. Use labels (`bug`, `enhancement`, `question`) to categorize your issue.

---

Thank you for helping make TerryStops-AI better! 🙌
