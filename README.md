# Generative-Economic-Modeling-Code

Example code for the analytical real-business-cycle model from *"Generative Economic Modeling"* by H. Kase, M. Rottner, and F. Stohler.

## Requirements
- Python 3.11
- Git

## Setup

To use the code open a shell and execute the following commands.

1. Clone the repository:

   **SSH**  
   `git clone git@github.com:Fabio-Stohler/Generative-Economic-Modeling.git`

   **HTTPS**  
   `git clone https://github.com/Fabio-Stohler/Generative-Economic-Modeling.git`

2. Change into the project directory:  
   `cd Generative-Economic-Modeling`

3. Create and activate a virtual environment:

   **Option A (recommended): built-in venv**
   - Create: `python -m venv venv`
   - Activate:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

   **Option B: virtualenv**
   - Install: `pip install virtualenv`
   - Create: `virtualenv venv`
   - Activate:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

4. Install the package in editable mode (brings deps):  
   ```bash
   pip install -e .
   ```

## Run
- Script version (recommended):  
  `python examples/analysis_BM.py`
- Notebook version:  
  - Generate once via Jupytext: `jupytext --to ipynb examples/analysis_BM.py`  
  - Then open/run `examples/analysis_BM.ipynb` in Jupyter/Lab.
- Colab version:  
  [Open in Colab](https://colab.research.google.com/github/Fabio-Stohler/Generative-Economic-Modeling/blob/main/examples/analysis_BM_colab.ipynb) (includes a setup cell to install the package).
- Docs site:  
  ```bash
  # generate API pages via quartodoc then render site
  quarto render docs
  ```

## Tested environments
- Windows 11 Pro (Dell Latitude 5330, Intel i5-1235U)

## References

If you use the code, please cite our paper:

- Kase, H., Rottner, M., & Stohler, F. (2025, December 2). *Generative economic modeling* (BIS Working Papers No. 1312). Bank for International Settlements. https://www.bis.org/publ/work1312.htm

```bibtex
@techreport{kase_rottner_stohler_2025_generative_economic_modeling,
  title       = {Generative economic modeling},
  author      = {Kase, Hanno and Rottner, Matthias and Stohler, Fabio},
  year        = {2025},
  month       = dec,
  number      = {1312},
  institution = {Bank for International Settlements},
  series      = {BIS Working Papers},
  url         = {https://www.bis.org/publ/work1312.htm}
}
