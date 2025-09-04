# TwisteRL Documentation

This directory contains the Sphinx documentation for TwisteRL.

## Quick Start

### 1. Install Dependencies

```bash
# From the project root directory
source .venv/bin/activate
cd docs
pip install -r requirements.txt
```

### 2. Build Documentation

```bash
# Build HTML documentation
make html

# Or using Sphinx directly
../.venv/bin/sphinx-build -b html . _build/html
```

### 3. View Documentation

```bash
# Open in browser (macOS)
open _build/html/index.html

# Or serve with Python HTTP server
python -m http.server 8000 -d _build/html
# Then visit: http://localhost:8000
```

## Development

For active development with auto-rebuild:

```bash
# Install sphinx-autobuild
pip install sphinx-autobuild

# Start auto-rebuilding server
sphinx-autobuild . _build/html --port 8000
```

Visit http://localhost:8000 and the docs will rebuild automatically when you make changes.

## Documentation Structure

- `index.rst` - Main documentation index
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start tutorial
- `api/` - API documentation
- `contributing.rst` - Contributing guidelines
- `docs-guide.rst` - Complete guide for running documentation
- `conf.py` - Sphinx configuration

## Troubleshooting

- **Import errors**: Make sure you're in the virtual environment and dependencies are installed
- **Torch warnings**: These are expected and can be ignored
- **Caching issues**: Run `make clean && make html`

## GitHub Pages Deployment

Documentation is automatically deployed to GitHub Pages:

- **Automatic**: Push to `main` branch with changes to `docs/` or `src/`
- **Manual**: Use "Run workflow" in GitHub Actions
- **URL**: `https://<username>.github.io/<repository-name>`

### Setup Steps:
1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Save - that's it!

### Troubleshooting:
- Check Actions tab for build errors
- Ensure GitHub Pages is enabled
- Allow a few minutes for changes to appear

For detailed information, see the complete [Documentation Guide](docs-guide.rst) in the built docs.