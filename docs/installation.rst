Installation
============

This page covers the installation of HAAM and its dependencies.

Requirements
------------

* Python 3.8 or higher
* NumPy, SciPy, Pandas
* Scikit-learn
* Matplotlib, Seaborn, Plotly
* BERTopic for topic modeling
* UMAP-learn for dimensionality reduction

Quick Install
-------------

The easiest way to install HAAM is via pip:

.. code-block:: bash

   pip install haam

Development Installation
------------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/yourusername/haam.git
   cd haam
   pip install -e .

This installs HAAM in "editable" mode, allowing you to modify the source code.

Installing Dependencies
-----------------------

If you encounter dependency issues, install the core requirements first:

.. code-block:: bash

   pip install numpy scipy pandas scikit-learn
   pip install matplotlib seaborn plotly
   pip install bertopic umap-learn hdbscan

Optional Dependencies
---------------------

For additional features:

.. code-block:: bash

   # For Jupyter notebook support
   pip install notebook ipywidgets
   
   # For parallel processing
   pip install joblib
   
   # For advanced NLP features
   pip install sentence-transformers

Troubleshooting
---------------

Common installation issues:

**HDBSCAN Installation Fails**
   On some systems, you may need to install build tools first:
   
   .. code-block:: bash
   
      # macOS
      brew install gcc
      
      # Ubuntu/Debian
      sudo apt-get install build-essential

**Import Errors**
   Make sure all dependencies are installed:
   
   .. code-block:: bash
   
      pip install -r requirements.txt

**Memory Issues**
   For large datasets, ensure you have at least 8GB RAM available.

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import haam
   print(haam.__version__)
   
   # Run a simple test
   from haam import HAAMAnalysis
   print("HAAM successfully installed!")

Platform-Specific Notes
-----------------------

**Windows**
   Some dependencies may require Visual C++ build tools. Install from:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

**macOS**
   Ensure you have Xcode Command Line Tools:
   
   .. code-block:: bash
   
      xcode-select --install

**Linux**
   Most distributions work out of the box. For GPU acceleration with UMAP:
   
   .. code-block:: bash
   
      pip install cupy-cuda11x  # Adjust for your CUDA version