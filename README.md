# Working Capital and Dividend Policy

This repository contains the code and data used for the project 
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

---

## Repository Structure

.
├── notebooks/            # Jupyter notebooks (data collection, analysis)
├── output/               # Generated Excel files, regression tables
├── figures/              # Figures used in the report
├── data/                 # Intermediate datasets (if applicable)
└── README.md



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


## How to Reproduce the Results

1. Install the required Python libraries listed at the top of each notebook
2. Run the data collection notebook to retrieve SEC and Yahoo Finance data
3. Execute the regression and machine learning notebooks sequentially
4. All tables and figures are automatically exported to the `output/` and
   `figures/` folders


## Environment

- **Operating system:** macOS  
- **Python version:** 3.14  
- **Main libraries:** pandas, numpy, statsmodels, linearmodels, scikit-learn  
- **Execution environment:** Jupyter Notebook


## Author
ﬁ
**Adis Arifi**  
HEC Lausanne – University of Lausanne  
Email: adis.arifi@unil.ch  
Academic year: 2025–2026