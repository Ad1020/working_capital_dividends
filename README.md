# Working Capital and Dividend Policy

This repository contains the code used for the project 
**“Working Capital and Dividend Policy: Evidence from U.S. Firms”**, 
conducted as part of the *Advanced Programming* course at HEC Lausanne (UNIL).

The objective of the project is to examine whether working capital dynamics
can explain or predict changes in dividend per share beyond standard financial
determinants, using both econometric panel regressions and machine learning
methods.

---

## Data Sources

All data used in this project are publicly available:

- **U.S. Securities and Exchange Commission (SEC)**  
  XBRL *Company Facts* API for firm-level accounting data  
  https://www.sec.gov/edgar.shtml
  
- **Yahoo Finance**  
  Annual dividend per share data  
  https://finance.yahoo.com

These sources were selected to ensure full transparency and reproducibility
without requiring paid subscriptions or proprietary databases.


## Repository Structure

**proposal.md**  
  Initial project proposal submitted for approval.

**main.py**  
  Entry point script used to verify that the project runs correctly.

**requirements.txt**  
  List of Python dependencies required to run the code.

 **working-capital-dividends.py**  
  Main Python script containing the full analysis pipeline:
  - data collection from SEC and Yahoo Finance  
  - data cleaning and variable construction  
  - fixed-effects panel regressions  
  - machine learning prediction models (Random Forest)

**output/** *(generated locally)*  
  Directory automatically created when running the notebook, containing
  generated Excel files with cleaned and merged datasets.  
  This folder is not included in the repository.

**figures/**  
  Final figures and LaTeX tables used in the final report.

**Working_Capital_Dividends_Report.pdf**  
  Final project report.




## Methodology Overview

The project follows a two-step empirical strategy:

1. **Econometric analysis (2015–2019)**  
   Firm fixed-effects panel regressions are used to assess the explanatory
   relationship between working capital changes and dividend per share.

2. **Predictive analysis (2021–2024)**  
   Random Forest models are estimated using lagged accounting variables to
   evaluate the out-of-sample predictive power of working capital dynamics.

To avoid look-ahead bias, all predictive models rely exclusively on information
available at time *t–1*.


## Entry Point

An entry point script (`main.py`) is provided to verify that the project runs
correctly in a standard Python environment


## How to Reproduce the Results

1. Install the required Python libraries listed in `requirements.txt`
2. Open the Python script `working-capital-dividends.py`
3. Run the Python script from top to bottom to execute the full workflow.  
   This step automatically downloads the raw data and generates the cleaned
   datasets stored locally in the `output/` directory.
4. All tables and figures are automatically exported to the `figures/` folders.

## Reproducibility

Random seeds (`random_state = 42`) are fixed in all machine learning models to
ensure that results are fully reproducible across executions.


## Environment

- **Operating system:** macOS  
- **Python version:** 3.14  
- **Main libraries:** pandas, numpy, statsmodels, linearmodels, scikit-learn  
- **Execution environment:** Jupyter Notebook


## Author

**Adis Arifi**  
HEC Lausanne – University of Lausanne  
Email: adis.arifi@unil.ch  
Academic year: 2025–2026
