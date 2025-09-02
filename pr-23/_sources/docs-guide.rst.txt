Documentation Guide
==================

This guide explains how to build and view the TwisteRL documentation locally.

Prerequisites
-------------

Before building the documentation, ensure you have:

1. **Python 3.9+** installed
2. **Virtual environment** set up (recommended)
3. **Documentation dependencies** installed

Setup
-----

1. **Navigate to the project root**:

.. code-block:: bash

   cd twisteRL

2. **Activate your virtual environment**:

.. code-block:: bash

   source .venv/bin/activate

3. **Install documentation dependencies**:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt

Building the Documentation
--------------------------

There are several ways to build the documentation:

Method 1: Using Make (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the ``docs/`` directory:

.. code-block:: bash

   # Build HTML documentation
   make html
   
   # Clean previous builds
   make clean
   
   # View all available targets
   make help

Method 2: Using Sphinx directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   sphinx-build -b html . _build/html

Method 3: Using virtual environment's Sphinx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   ../.venv/bin/sphinx-build -b html . _build/html

Viewing the Documentation
-------------------------

Once built, you can view the documentation in several ways:

Option 1: Open in Browser (macOS/Linux)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # macOS
   open _build/html/index.html
   
   # Linux
   xdg-open _build/html/index.html

Option 2: Python HTTP Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serve the documentation locally with Python's built-in server:

.. code-block:: bash

   # From the docs directory
   python -m http.server 8000 -d _build/html

Then visit: http://localhost:8000

Option 3: Alternative HTTP Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use other simple HTTP servers:

.. code-block:: bash

   # Using Node.js http-server (if available)
   npx http-server _build/html -p 8000
   
   # Using PHP (if available)
   cd _build/html && php -S localhost:8000

Development Workflow
--------------------

For documentation development, use this workflow:

1. **Make changes** to ``.rst`` files
2. **Rebuild** the documentation:

.. code-block:: bash

   make html

3. **Refresh** your browser to see changes
4. **Repeat** as needed

Auto-rebuild (Optional)
~~~~~~~~~~~~~~~~~~~~~~

For automatic rebuilding during development, you can use ``sphinx-autobuild``:

.. code-block:: bash

   # Install sphinx-autobuild
   pip install sphinx-autobuild
   
   # Start auto-rebuilding server
   sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

This will:
- Watch for file changes
- Automatically rebuild the documentation
- Refresh your browser automatically
- Serve at http://localhost:8000

Troubleshooting
---------------

Missing Dependencies
~~~~~~~~~~~~~~~~~~~

If you see import errors during build:

.. code-block:: bash

   # Make sure you're in the virtual environment
   source .venv/bin/activate
   
   # Install missing dependencies
   pip install -r requirements.txt

Torch Import Warnings
~~~~~~~~~~~~~~~~~~~~~

The warnings about ``torch`` not being found are **expected** and can be ignored. The documentation builds successfully without PyTorch installed in the docs environment.

Clean Build
~~~~~~~~~~~

If you encounter caching issues:

.. code-block:: bash

   make clean
   make html

File Permissions
~~~~~~~~~~~~~~~~

On some systems, you might need to ensure proper permissions:

.. code-block:: bash

   chmod +x .venv/bin/sphinx-build

Output Formats
--------------

Sphinx can generate documentation in multiple formats:

HTML (Default)
~~~~~~~~~~~~~

.. code-block:: bash

   make html

PDF (requires LaTeX)
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make latexpdf

EPUB
~~~~

.. code-block:: bash

   make epub

Available Make Targets
----------------------

Run ``make help`` to see all available targets:

.. code-block:: text

   html        - Build HTML documentation
   dirhtml     - Build HTML pages in directories
   singlehtml  - Build single HTML page
   pickle      - Build pickle files
   json        - Build JSON files
   htmlhelp    - Build HTML help files
   qthelp      - Build QtHelp files
   devhelp     - Build DevHelp files
   epub        - Build EPUB files
   latex       - Build LaTeX files
   latexpdf    - Build PDF from LaTeX
   text        - Build text files
   man         - Build manual pages
   texinfo     - Build Texinfo files
   gettext     - Build gettext files
   changes     - Build changes overview
   xml         - Build Docutils-native XML files
   pseudoxml   - Build pseudoxml files
   linkcheck   - Check all external links

Configuration
-------------

The documentation is configured in ``conf.py``. Key settings include:

- **Project information**: Name, version, author
- **Extensions**: Sphinx extensions for enhanced features
- **Theme**: ReadTheDocs theme for professional appearance
- **Paths**: Source and build directories
- **Autodoc settings**: API documentation generation

For advanced customization, edit ``conf.py`` as needed.

Quick Reference
---------------

**Most common commands**:

.. code-block:: bash

   # Setup (one-time)
   cd docs && pip install -r requirements.txt
   
   # Build and view
   make html && python -m http.server 8000 -d _build/html
   
   # Clean rebuild
   make clean && make html

The documentation will be available at http://localhost:8000

GitHub Pages Deployment
========================

TwisteRL documentation is automatically deployed to GitHub Pages using GitHub Actions.

Automatic Deployment
--------------------

The documentation is automatically built and deployed when:

1. **Push to main branch** with changes to:
   - ``docs/**`` files
   - ``src/**`` files (affects API docs)
   - ``.github/workflows/docs.yml``

2. **Pull requests** create preview deployments:
   - Built docs deployed to ``pr-{number}/`` subdirectory
   - Bot comments with preview URL on the PR
   - Auto-cleanup when PR is closed

3. **Manual trigger** via GitHub Actions interface

Setup GitHub Pages
------------------

To enable GitHub Pages for your repository:

1. **Go to your repository settings** on GitHub
2. **Navigate to Pages** (in the left sidebar)
3. **Set Source** to "GitHub Actions"
4. **Save the configuration**

That's it! The workflow will handle the rest.

Accessing Your Documentation
----------------------------

Once deployed, your documentation will be available at:

.. code-block:: text

   # Production (main branch)
   https://ai4quantum.github.io/twisteRL
   
   # PR Previews
   https://ai4quantum.github.io/twisteRL/pr-<number>/

Workflow Details
----------------

The GitHub Actions workflow (`.github/workflows/docs.yml`) includes:

**Build Stage:**
- Sets up Python 3.11 and Rust toolchain
- Caches dependencies for faster builds
- Installs TwisteRL in development mode
- Builds documentation with Sphinx
- Handles warnings as errors (``-W --keep-going``)

**Deploy Stage (main branch only):**
- Configures GitHub Pages
- Uploads documentation artifacts
- Deploys to GitHub Pages

**Key Features:**
- Only deploys from main branch pushes
- Pull requests build docs for validation
- Manual workflow dispatch available
- Proper permissions and security settings

Troubleshooting Deployment
---------------------------

**Common Issues:**

1. **Workflow fails on build:**
   - Check the Actions tab in GitHub
   - Look for Python/Rust dependency issues
   - Verify all documentation files are valid

2. **Pages not updating:**
   - Ensure GitHub Pages is enabled in repository settings
   - Check that workflow completed successfully
   - May take a few minutes to reflect changes

3. **Import errors in documentation:**
   - The workflow installs TwisteRL in development mode
   - Missing dependencies should be added to ``docs/requirements.txt``

**Viewing Build Logs:**
- Go to Actions tab in your GitHub repository
- Click on the workflow run
- Expand the failed step to see detailed logs

**Manual Deployment:**
- Go to Actions tab
- Select "Build and Deploy Documentation"
- Click "Run workflow"
- Choose the branch and click "Run workflow"

The documentation will be available at http://localhost:8000

Custom Domain (Optional)
------------------------

To use a custom domain for your documentation:

1. **Create a CNAME file** in the ``docs/_build/html/`` directory:

.. code-block:: bash

   echo "your-domain.com" > docs/_static/CNAME

2. **Update Sphinx configuration** in ``docs/conf.py``:

.. code-block:: python

   html_extra_path = ['_static']

3. **Configure DNS** to point your domain to GitHub Pages
4. **Update repository settings** to use your custom domain

This ensures the CNAME file is included in every build.