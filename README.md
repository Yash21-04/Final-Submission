# Driver Drowsiness Detection using AI, OpenCV & Streamlit



This repository contains materials for a lane / driver drowsiness detection study. It includes a Jupyter notebook that demonstrates data download, preprocessing, model training/evaluation, and a small Streamlit app for interactive demos.

Repository layout
- `lane_drowsiness_pipeline.ipynb` — Main Jupyter notebook with experiments, data download steps, and model code.
- `streamlit_app.py` — Streamlit demo app to run the pipeline or visualize results interactively.
- `a.py` — auxiliary script (inspect before running).

Highlights
- Combines classical computer-vision (Eye Aspect Ratio with facial landmarks) and deep-learning (CNN) approaches.
- Notebook contains example training/evaluation code and visualization cells.
- Streamlit app for local interactive demos.

Important dependencies
- Python 3.8+ recommended
- jupyterlab / notebook
- numpy
- opencv-python
- imutils
- scipy
- scikit-learn
- matplotlib
- tensorflow (if you use the Keras models in the notebook)
- streamlit (for `streamlit_app.py`)
- dlib (used for facial landmarks — see notes below)

Notes about installing `dlib` on Windows
- `dlib` frequently needs a C++ build toolchain or a prebuilt wheel on Windows. The easiest approach is to use conda and install from conda-forge:

	```powershell
	conda install -c conda-forge dlib
	```

- If you don't have conda, a pip wheel may exist for your Python version; otherwise pip install may fail and require Visual Studio Build Tools.

Recommended setup (PowerShell on Windows)

1) Create and activate a virtual environment (venv) or use conda (recommended for `dlib`):

```powershell
# Using venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install most packages
pip install -U pip
pip install jupyterlab numpy opencv-python imutils scipy scikit-learn matplotlib streamlit

# Optional: TensorFlow (CPU)
pip install tensorflow

# If you prefer conda (recommended when you need dlib):
# conda create -n lane_env python=3.10 -y
# conda activate lane_env
# conda install -c conda-forge dlib
# pip install jupyterlab numpy opencv-python imutils scipy scikit-learn matplotlib streamlit tensorflow
```

2) Run the notebook

```powershell
jupyter lab
# or
jupyter notebook
```

Notes for notebooks
- Use the IPython pip magic inside notebook cells when installing from a cell: `%pip install package` so the install targets the running kernel environment.
- Some notebook static analyzers flag `!` magics and `get_ipython()` calls; these are normal in Jupyter notebooks and work at runtime.

3) Run the Streamlit app

```powershell
# From the repository root
pip install streamlit
streamlit run streamlit_app.py
```

Troubleshooting
- If VS Code shows imports as "could not be resolved", ensure the editor/language server is using the same Python environment as your runtime kernel or virtualenv.
- If `dlib` pip install fails on Windows, prefer installing via conda-forge. If you must build from source, install Visual Studio Build Tools and CMake.

Next steps / suggestions
- Replace `!pip install` notebook magics with `%pip install` to avoid kernel-environment mismatches.
- Add a `requirements.txt` or `environment.yml` for reproducible setup.
- Add simple unit tests or an integration script to validate the Streamlit app and notebook examples.

License
- Add your preferred license here (e.g., MIT) if you plan to publish the repository.

Contact
