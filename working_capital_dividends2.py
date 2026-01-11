# =========================
# GLOBAL IMPORTS (one-time)
# =========================
import time
from pathlib import Path

import requests
import pandas as pd
import numpy as np

import yfinance as yf

import matplotlib.pyplot as plt
import statsmodels.api as sm

# PanelOLS (linearmodels)
from linearmodels.panel import PanelOLS

try:
    from IPython.display import display  # works if IPython exists
except Exception:
    display = print


def run(show_plots=True):
    # =========================
    # 0) INSTALL (notebook Python)
    # =========================
    

    # =========================
    # 1) IMPORTS
    # =========================
    import time
    import requests
    import pandas as pd
    import yfinance as yf
    from pathlib import Path

    # =========================
    # 2) PARAMETERS
    # =========================
    # ~100 staples / staples-adjacent candidates to maximize usable firms
    TICKERS = [
        # Core staples
        "PG","KO","PEP","PM","MO","MDLZ","CL","KMB","GIS","HSY","KHC","MKC","CPB","CAG","HRL","TSN","SJM","CLX",
        "K","TAP","STZ","KDP","MNST","CHD","NWL","EL","COTY",
        # Retail / grocery / distributors (staples-ish)
        "WMT","COST","TGT","KR","DG","DLTR","WBA","ACI","BJ","SFM","IMKTA","WMK","SPTN","UNFI","USFD","PFGC","SYY",
        # Agriculture / ingredients / food supply chain
        "ADM","BG","DAR","ANDE","FDP","INGR","CALM","LW","FLO","POST","THS","LANC","VITL","PPC",
        # Beverages / snack / food names (some may be borderline; included to increase pool)
        "SAM","FIZZ","CELH","UTZ","JJSF","CVGW","SENEA","SENEB","HAIN","NOMD","SMPL","GO","BRBR","NTRI","CHEF","USNA",
        # Tobacco / nicotine / related
        "UVV","VGR","TPB",
        # Extra food/retail candidates
        "SAFM","JBSS","FARM","FSTR","SPB","NUS","IPAR","EPC","NATR","KLG"
    ]

    SEC_IDENTITY = "unil-project adis.arifi@unil.ch"  
    SLEEP_BETWEEN_CALLS = 0.2

    # Periods
    REG_START, REG_END = 2015, 2019
    ML_START,  ML_END  = 2020, 2024
    START_ALL, END_ALL = REG_START, ML_END

    # Eligibility rules
    MIN_COMPLETE_YEARS_REG = 5   # 2015-2019 window => aim 5/5
    MIN_COMPLETE_YEARS_ML  = 3   # 2020-2024 window => set 4 if you want stricter

    # Required columns for "complete" firm-year
    # (OCF not required; kept if available)
    REQUIRED_BASE_COLS = [
        "DPS_Annual",
        "TotalAssets",
        "CurrentAssets",
        "CurrentLiabilities",
        "TotalLiabilities",
        "NetIncome",
    ]

    SEC_HEADERS = {"User-Agent": SEC_IDENTITY, "Accept-Encoding": "gzip, deflate"}

    # =========================
    # 3) SEC HELPERS
    # =========================
    def get_cik_map():
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers=SEC_HEADERS, timeout=30).json()
        df = pd.DataFrame.from_dict(data, orient="index")
        df["ticker"] = df["ticker"].str.upper()
        return dict(zip(df["ticker"], df["cik_str"].astype(int)))

    def fetch_companyfacts(cik_int: int):
        cik10 = str(cik_int).zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        return requests.get(url, headers=SEC_HEADERS, timeout=30).json()

    def extract_fy(companyfacts_json, tag, start_y, end_y):
        usgaap = companyfacts_json.get("facts", {}).get("us-gaap", {})
        if tag not in usgaap:
            return pd.DataFrame(columns=["Year", tag])

        units = usgaap[tag].get("units", {})
        if not units:
            return pd.DataFrame(columns=["Year", tag])

        unit_key = "USD" if "USD" in units else list(units.keys())[0]
        rows = []
        for item in units.get(unit_key, []):
            if item.get("fp") == "FY" and "fy" in item and "val" in item:
                rows.append({"Year": int(item["fy"]), tag: item["val"]})

        if not rows:
            return pd.DataFrame(columns=["Year", tag])

        out = pd.DataFrame(rows).drop_duplicates(subset=["Year"]).sort_values("Year")
        out = out[(out["Year"] >= start_y) & (out["Year"] <= end_y)]
        return out

    def extract_first_available(companyfacts_json, tags, out_name, start_y, end_y):
        for tag in tags:
            df = extract_fy(companyfacts_json, tag, start_y, end_y)
            if not df.empty:
                return df.rename(columns={tag: out_name})
        return pd.DataFrame(columns=["Year", out_name])

    # Robust tags
    TAGS_TOTAL_ASSETS   = ["Assets"]
    TAGS_CURRENT_ASSETS = ["AssetsCurrent"]
    TAGS_CURRENT_LIAB   = ["LiabilitiesCurrent"]
    TAGS_TOTAL_LIAB     = ["Liabilities"]
    TAGS_NET_INCOME     = ["NetIncomeLoss", "ProfitLoss"]
    TAGS_OCF            = [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]

    # =========================
    # 4) YAHOO (DPS annual)
    # =========================
    def yahoo_dividends_annual(ticker: str, start_y: int, end_y: int):
        t = yf.Ticker(ticker)
        s = t.dividends
        if s is None or len(s) == 0:
            return pd.DataFrame({"Year": list(range(start_y, end_y + 1)),
                                "DPS_Annual": [pd.NA]*(end_y-start_y+1)})

        div = s.to_frame("Dividend").reset_index()
        div["Year"] = div["Date"].dt.year
        dps = div.groupby("Year", as_index=False)["Dividend"].sum().rename(columns={"Dividend": "DPS_Annual"})
        dps = dps[(dps["Year"] >= start_y) & (dps["Year"] <= end_y)]
        return dps

    # =========================
    # 5) BUILD RAW PANEL (2015-2024)
    # =========================
    cik_map = get_cik_map()

    # de-duplicate tickers
    unique_tickers = []
    for t in TICKERS:
        u = str(t).upper().strip()
        if u and u not in unique_tickers:
            unique_tickers.append(u)

    all_rows = []
    for tk in unique_tickers:
        print(f"--- {tk} ---")
        base_years = pd.DataFrame({"Year": list(range(START_ALL, END_ALL + 1))})

        # Yahoo DPS
        dps = yahoo_dividends_annual(tk, START_ALL, END_ALL)

        # SEC facts
        if tk not in cik_map:
            sec_df = base_years.copy()
        else:
            time.sleep(SLEEP_BETWEEN_CALLS)
            facts = fetch_companyfacts(cik_map[tk])

            a   = extract_first_available(facts, TAGS_TOTAL_ASSETS,   "TotalAssets",       START_ALL, END_ALL)
            ca  = extract_first_available(facts, TAGS_CURRENT_ASSETS, "CurrentAssets",     START_ALL, END_ALL)
            cl  = extract_first_available(facts, TAGS_CURRENT_LIAB,   "CurrentLiabilities",START_ALL, END_ALL)
            tl  = extract_first_available(facts, TAGS_TOTAL_LIAB,     "TotalLiabilities",  START_ALL, END_ALL)
            ni  = extract_first_available(facts, TAGS_NET_INCOME,     "NetIncome",         START_ALL, END_ALL)
            ocf = extract_first_available(facts, TAGS_OCF,            "OperatingCashFlow", START_ALL, END_ALL)  # optional

            sec_df = base_years.merge(a, on="Year", how="left") \
                            .merge(ca, on="Year", how="left") \
                            .merge(cl, on="Year", how="left") \
                            .merge(tl, on="Year", how="left") \
                            .merge(ni, on="Year", how="left") \
                            .merge(ocf, on="Year", how="left")

        df = base_years.merge(dps, on="Year", how="left").merge(sec_df, on="Year", how="left")
        df.insert(0, "Ticker", tk)
        all_rows.append(df)

    panel_all = pd.concat(all_rows, ignore_index=True).sort_values(["Ticker","Year"])

    # =========================
    # 6) SPLIT + CLEAN (REG + ML)
    # =========================
    def make_clean_subset(panel: pd.DataFrame, start_y: int, end_y: int, min_years: int):
        sub = panel[(panel["Year"] >= start_y) & (panel["Year"] <= end_y)].copy()

        sub_clean = sub.dropna(subset=REQUIRED_BASE_COLS).copy()

        counts = sub_clean.groupby("Ticker")["Year"].nunique()
        keep = counts[counts >= min_years].index.tolist()
        sub_clean = sub_clean[sub_clean["Ticker"].isin(keep)].copy()

        # Useful constructed variables
        sub_clean["WorkingCapital"] = sub_clean["CurrentAssets"] - sub_clean["CurrentLiabilities"]
        sub_clean["WC_to_TA"] = sub_clean["WorkingCapital"] / sub_clean["TotalAssets"]
        sub_clean["ROA"] = sub_clean["NetIncome"] / sub_clean["TotalAssets"]
        sub_clean["Leverage"] = sub_clean["TotalLiabilities"] / sub_clean["TotalAssets"]

        return sub_clean, counts

    reg_clean, reg_counts = make_clean_subset(panel_all, REG_START, REG_END, MIN_COMPLETE_YEARS_REG)
    ml_clean,  ml_counts  = make_clean_subset(panel_all, ML_START,  ML_END,  MIN_COMPLETE_YEARS_ML)

    # Completeness report over full 2015-2024 (by REQUIRED_BASE_COLS)
    report_all = (panel_all.assign(is_complete=~panel_all[REQUIRED_BASE_COLS].isna().any(axis=1))
                        .groupby("Ticker")
                        .agg(total_years=("Year","nunique"),
                            complete_years=("is_complete","sum"))
                        .reset_index()
                        .sort_values(["complete_years","total_years"], ascending=[True, True]))

    print("\n=== RESULTS ===")
    print("RAW tickers:", panel_all["Ticker"].nunique(), "| RAW rows:", panel_all.shape[0])
    print("REG firms kept:", reg_clean["Ticker"].nunique(), "| REG rows:", reg_clean.shape[0])
    print("ML  firms kept:", ml_clean["Ticker"].nunique(),  "| ML  rows:", ml_clean.shape[0])

    # =========================
    # 7) EXPORT EXCEL (RAW + REG + ML + REPORT)
    # =========================
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / f"SECplusYahoo_RAW_REG_{REG_START}_{REG_END}_ML_{ML_START}_{ML_END}.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        panel_all.to_excel(writer, sheet_name="RAW_2015_2024", index=False)
        reg_clean.to_excel(writer, sheet_name="REG_2015_2019", index=False)
        ml_clean.to_excel(writer, sheet_name="ML_2020_2024", index=False)
        report_all.to_excel(writer, sheet_name="COMPLETENESS_REPORT", index=False)

    print("\n✅ Excel created:", out_file.resolve())
    print("Sheets: RAW_2015_2024 | REG_2015_2019 | ML_2020_2024 | COMPLETENESS_REPORT")
    reg_clean.head(10)
    # =====================================================
    # 0) INSTALL + IMPORTS
    # =====================================================
  

    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from pathlib import Path


    # =====================================================
    # 1) LOAD DATA (REG 2015–2019)
    # =====================================================
    FILE = Path("output") / "SECplusYahoo_RAW_REG_2015_2019_ML_2020_2024.xlsx"
    SHEET = "REG_2015_2019"

    df = pd.read_excel(FILE, sheet_name=SHEET)


    df = df.sort_values(["Ticker", "Year"]).reset_index(drop=True)

    # =====================================================
    # 2) CONSTRUCTION OF VARIABLES
    # =====================================================

    # Working Capital and ratio
    df["WorkingCapital"] = df["CurrentAssets"] - df["CurrentLiabilities"]
    df["WC_to_TA"] = df["WorkingCapital"] / df["TotalAssets"]

    # consecutive years 
    df["year_gap"] = df.groupby("Ticker")["Year"].diff()

    # log(DPS) only if DPS > 0
    df["ln_dps"] = np.where(df["DPS_Annual"] > 0, np.log(df["DPS_Annual"]), np.nan)

    # annual changes
    df["dlog_dps"] = df.groupby("Ticker")["ln_dps"].diff()
    df["dwc_ta"]   = df.groupby("Ticker")["WC_to_TA"].diff()

    # keeping only consecutive observations
    df_model = df[df["year_gap"] == 1].copy()

    # dropping observation with misssing values required for the regressions
    df_model = df_model.dropna(subset=["dlog_dps", "dwc_ta"]).copy()

    print("Nombre d'observations utilisées :", df_model.shape[0])
    print("Nombre d'entreprises :", df_model["Ticker"].nunique())

    # =====================================================
    # 3) REGRESSION OLS BASELINE
    # Δlog(DPS) = α + β Δ(WC/TA) + ε
    # clustering standard errors at the firm level
    # =====================================================
    Y = df_model["dlog_dps"]
    X = sm.add_constant(df_model["dwc_ta"])

    ols = sm.OLS(Y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df_model["Ticker"]}
    )

    print(ols.summary())

    # =====================================================
    # 4) TABLE OF RESULTS
    # =====================================================
    results_table = pd.DataFrame({
        "Coefficient": ols.params,
        "Std. Error (cluster firm)": ols.bse,
        "t-stat": ols.tvalues,
        "p-value": ols.pvalues
    })



    print("\n=== TABLE OF RESULTS ===")
    print(results_table)

    # =====================================================
    # 6) PANEL FIXED EFFECTS (BASELINE) - ULTRA ROBUSTE + TABLEAU
    # =====================================================
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # 1) X = const + dwc_ta + firm dummies
    fe_dummies = pd.get_dummies(df_model["Ticker"], prefix="FE", drop_first=True)
    X_fe = pd.concat([df_model[["dwc_ta"]], fe_dummies], axis=1)
    X_fe = sm.add_constant(X_fe)

    # 2) Force numeric types and drop missing values
    X_fe = X_fe.apply(pd.to_numeric, errors="coerce")
    Y_fe = pd.to_numeric(df_model["dlog_dps"], errors="coerce")

    mask = (~Y_fe.isna()) & (~X_fe.isna().any(axis=1))
    Y_fe_clean = Y_fe.loc[mask]
    X_fe_clean = X_fe.loc[mask]
    groups = df_model.loc[mask, "Ticker"]

    print("# Use observations included in the fixed-effects regressions :", len(Y_fe_clean), "| Firms :", groups.nunique())

    # 3)  Convert to NumPy float (critical to avoid dtype=object errors)
    X_np = X_fe_clean.to_numpy(dtype=float)
    Y_np = Y_fe_clean.to_numpy(dtype=float)
    groups_np = groups.to_numpy()

    # 4) Fit FE with SE cluster by firm
    fe_model = sm.OLS(Y_np, X_np).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups_np}
    )

    # 5) Table of coefficient
    fe_table = pd.DataFrame({
        "coef": fe_model.params,
        "std_err(cluster)": fe_model.bse,
        "z": fe_model.tvalues,
        "p_value": fe_model.pvalues
    }, index=X_fe_clean.columns)

    print("\n=== TABLEAU FIXED EFFECTS (baseline) ===")
  

    # 6) Resume
    print("\nResume :")
    print(fe_model.summary())
    
    import matplotlib.pyplot as plt
    import numpy as np

    from pathlib import Path
    FIGURES_DIR = Path("figures")
    FIGURES_DIR.mkdir(exist_ok=True)

    # Recréer df_model à partir des données REG
    df_plot = reg_clean.copy()

    # Construire les variables nécessaires si besoin
    df_plot = df_plot.sort_values(["Ticker", "Year"])
    df_plot["dwc_ta"] = df_plot.groupby("Ticker")["WC_to_TA"].diff()

    df_plot["ln_dps"] = np.where(df_plot["DPS_Annual"] > 0,
                                np.log(df_plot["DPS_Annual"]),
                                np.nan)
    df_plot["dlog_dps"] = df_plot.groupby("Ticker")["ln_dps"].diff()

    # Drop NA
    df_plot = df_plot.dropna(subset=["dwc_ta", "dlog_dps"])

    # Axes (en pourcentage)
    x_pp = df_plot["dwc_ta"] * 100
    y_percent = df_plot["dlog_dps"] * 100

    # Droite de tendance
    b1, b0 = np.polyfit(x_pp, y_percent, 1)

    # Plot
    plt.figure(figsize=(6,4))
    plt.scatter(x_pp, y_percent, alpha=0.4, s=15)
    plt.plot(np.sort(x_pp), b1*np.sort(x_pp) + b0, color="red", linewidth=2)

    plt.xlabel("Δ(WC / Total Assets) (percentage points)")
    plt.ylabel("Dividend growth (%)")
    plt.title("Dividend growth vs. changes in Working Capital")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_wc_dividend_regression.png",
                dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    # =====================================================
    # 7) COMPREHENSIVE MODEL — PANEL FIXED EFFECTS (WITHOUT WC)
    # Δlog(DPS) = ROA + Leverage + Size + firm FE + ε
    # =====================================================

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # -----------------------------
    # 1) Build variables
    # -----------------------------

    df_comp = df_model.copy()

    # ROA
    df_comp["ROA"] = df_comp["NetIncome"] / df_comp["TotalAssets"]

    # Leverage
    df_comp["Leverage"] = df_comp["TotalLiabilities"] / df_comp["TotalAssets"]

    # Size = log(Total Assets)
    df_comp["Size"] = np.log(df_comp["TotalAssets"])

    # -----------------------------
    # 2) Prepare X and Y
    # -----------------------------

    Y = pd.to_numeric(df_comp["dlog_dps"], errors="coerce")

    X_controls = df_comp[["ROA", "Leverage", "Size"]]

    # Firm fixed effects (dummies)
    fe_dummies = pd.get_dummies(df_comp["Ticker"], prefix="FE", drop_first=True)

    X = pd.concat([X_controls, fe_dummies], axis=1)
    X = sm.add_constant(X)

    # Enforce numeric types
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop missing values (safety)
    mask = (~Y.isna()) & (~X.isna().any(axis=1))
    Y_clean = Y.loc[mask]
    X_clean = X.loc[mask]
    groups = df_comp.loc[mask, "Ticker"]

    print("Obs utilisées :", len(Y_clean))
    print("Firms :", groups.nunique())

    # -----------------------------
    # 3) Estimation FE (numpy float)
    # -----------------------------

    X_np = X_clean.to_numpy(dtype=float)
    Y_np = Y_clean.to_numpy(dtype=float)
    groups_np = groups.to_numpy()

    fe_comp_no_wc = sm.OLS(Y_np, X_np).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups_np}
    )

    # -----------------------------
    # 4) TABLE OF RESULTS
    # -----------------------------

    results_no_wc = pd.DataFrame({
        "Coefficient": fe_comp_no_wc.params,
        "Std. Error (cluster firm)": fe_comp_no_wc.bse,
        "t-stat": fe_comp_no_wc.tvalues,
        "p-value": fe_comp_no_wc.pvalues
    }, index=X_clean.columns)

    print("\n=== COMPREHENSIVE FE (without WORKING CAPITAL) ===")


    # Resume
    print("\nResume :")
    print(fe_comp_no_wc.summary())
    # =====================================================
    # 8) COMPREHENSIVE MODEL — PANEL FIXED EFFECTS (WITH WC)
    # Δlog(DPS) = Δ(WC/TA) + ROA + Leverage + Size + firm FE + ε
    # =====================================================

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # -----------------------------
    # 1) Build WC variable
    # -----------------------------

    df_full = df_comp.copy()

    # Δ(WC/TA) already computed
    df_full["dwc_ta"] = df_model["dwc_ta"]

    # -----------------------------
    # 2) Prepare X and Y
    # -----------------------------

    Y = pd.to_numeric(df_full["dlog_dps"], errors="coerce")

    X_vars = df_full[["dwc_ta", "ROA", "Leverage", "Size"]]

    # Firm fixed effects
    fe_dummies = pd.get_dummies(df_full["Ticker"], prefix="FE", drop_first=True)

    X = pd.concat([X_vars, fe_dummies], axis=1)
    X = sm.add_constant(X)

    # Enforce numeric types
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop missing value (safety)
    mask = (~Y.isna()) & (~X.isna().any(axis=1))
    Y_clean = Y.loc[mask]
    X_clean = X.loc[mask]
    groups = df_full.loc[mask, "Ticker"]

    print("Obs utilisées :", len(Y_clean))
    print("Firmes :", groups.nunique())

    # -----------------------------
    # 3) Estimation FE (robust)
    # -----------------------------

    X_np = X_clean.to_numpy(dtype=float)
    Y_np = Y_clean.to_numpy(dtype=float)
    groups_np = groups.to_numpy()

    fe_comp_wc = sm.OLS(Y_np, X_np).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups_np}
    )

    # -----------------------------
    # 4) Table of results
    # -----------------------------

    results_wc = pd.DataFrame({
        "Coefficient": fe_comp_wc.params,
        "Std. Error (cluster firm)": fe_comp_wc.bse,
        "t-stat": fe_comp_wc.tvalues,
        "p-value": fe_comp_wc.pvalues
    }, index=X_clean.columns)

    print("\n=== COMPREHENSIVE FE (WITH WORKING CAPITAL) ===")
    display(results_wc.loc[["const", "dwc_ta", "ROA", "Leverage", "Size"]])

    # Resume
    print("\nResume :")
    print(fe_comp_wc.summary())
    # =====================================================
    # FE BASELINE — WC ALONE (for table)
    # =====================================================

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Y
    Y = pd.to_numeric(df_model["dlog_dps"], errors="coerce")

    # X = dwc_ta + firm FE
    X_vars = df_model[["dwc_ta"]]
    fe_dummies = pd.get_dummies(df_model["Ticker"], prefix="FE", drop_first=True)

    X = pd.concat([X_vars, fe_dummies], axis=1)
    X = sm.add_constant(X)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Drop missing value
    mask = (~Y.isna()) & (~X.isna().any(axis=1))
    Y_clean = Y.loc[mask]
    X_clean = X.loc[mask]
    groups = df_model.loc[mask, "Ticker"]

    # numpy arrays (robust)
    X_np = X_clean.to_numpy(dtype=float)
    Y_np = Y_clean.to_numpy(dtype=float)
    groups_np = groups.to_numpy()

    # Estimation FE baseline
    fe_baseline = sm.OLS(Y_np, X_np).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups_np}
    )

    print("FE baseline estimé.")
    print("Obs :", fe_baseline.nobs)
    

    import numpy as np
    import pandas as pd
    from linearmodels.panel import PanelOLS

  

    # =====================================================
    # 1) Load the file generated by the data-collection script
    # =====================================================
    path = "output/SECplusYahoo_RAW_REG_2015_2019_ML_2020_2024.xlsx"
    df = pd.read_excel(path, sheet_name="REG_2015_2019")

    # =====================================================
    # 2) Build variable (if needed)
    # =====================================================
    df = df.sort_values(["Ticker", "Year"])
    df["dwc_ta"] = df.groupby("Ticker")["WC_to_TA"].diff()
    df["Size"] = np.log(df["TotalAssets"].where(df["TotalAssets"] > 0))

    # =====================================================
    # 3) Panel index
    # =====================================================
    df = df.set_index(["Ticker", "Year"]).sort_index()

    # =====================================================
    # 4) FE regressions
    # =====================================================
    def fe(y, x):
        d = df[[y] + x].dropna()
        mod = PanelOLS(d[y], d[x], entity_effects=True)
        return mod.fit(cov_type="clustered", cluster_entity=True)

    Y = "DPS_Annual"

    m1 = fe(Y, ["dwc_ta"])
    m2 = fe(Y, ["ROA", "Leverage", "Size"])
    m3 = fe(Y, ["dwc_ta", "ROA", "Leverage", "Size"])

    # =====================================================
    # 5) Final table
    # =====================================================
    def coef(model, var):
        if var in model.params.index:
            return f"{model.params[var]:.3f} ({model.std_errors[var]:.3f})"
        return ""

    table = pd.DataFrame({
        "(1) Baseline FE\nWC only": {
            "Δ(WC/TA)": coef(m1, "dwc_ta"),
            "ROA": "",
            "Leverage": "",
            "Size": "",
        },
        "(2) Comprehensive FE\nNo WC": {
            "Δ(WC/TA)": "",
            "ROA": coef(m2, "ROA"),
            "Leverage": coef(m2, "Leverage"),
            "Size": coef(m2, "Size"),
        },
        "(3) Comprehensive FE\nWith WC": {
            "Δ(WC/TA)": coef(m3, "dwc_ta"),
            "ROA": coef(m3, "ROA"),
            "Leverage": coef(m3, "Leverage"),
            "Size": coef(m3, "Size"),
        }
    })

    info = pd.DataFrame({
        "(1) Baseline FE\nWC only": ["Yes", "Firm clustered", int(m1.nobs)],
        "(2) Comprehensive FE\nNo WC": ["Yes", "Firm clustered", int(m2.nobs)],
        "(3) Comprehensive FE\nWith WC": ["Yes", "Firm clustered", int(m3.nobs)],
    }, index=["Firm Fixed Effects", "SE", "Observations"])

    final_table = pd.concat([table, info])

  

    # Export table
    final_table.to_excel(FIGURES_DIR / "final_dividend_regression_table.xlsx")
    print("✅ Exported :", FIGURES_DIR / "final_dividend_regression_table.xlsx")

    with open(FIGURES_DIR / "table_fe.tex", "w") as f:
        f.write(final_table.to_latex(
            index=True,
            escape=False,
            column_format="lccc"
        ))

    print("✅ Exported :", FIGURES_DIR / "table_fe.tex")


    import pandas as pd
    import numpy as np
    from pathlib import Path

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import statsmodels.api as sm

    # -----------------------------
    # 1) LOAD RAW DATA
    # -----------------------------
    FILE = Path("output") / "SECplusYahoo_RAW_REG_2015_2019_ML_2020_2024.xlsx"  
    SHEET = "RAW_2015_2024"

    df = pd.read_excel(FILE, sheet_name=SHEET)
    df = df.sort_values(["Ticker", "Year"]).reset_index(drop=True)

    print("RAW shape:", df.shape, "| Firms:", df["Ticker"].nunique(), "| Years:", df["Year"].min(), "-", df["Year"].max())

    # -----------------------------
    # 2) BUILD BASE VARIABLES
    # -----------------------------
    # Working capital ratio
    df["WorkingCapital"] = df["CurrentAssets"] - df["CurrentLiabilities"]
    df["WC_to_TA"] = df["WorkingCapital"] / df["TotalAssets"]

    # Fundamentals
    df["ROA"] = df["NetIncome"] / df["TotalAssets"]
    df["Leverage"] = df["TotalLiabilities"] / df["TotalAssets"]
    df["Size"] = np.log(df["TotalAssets"])

    # Target: dlog_dps
    df["ln_dps"] = np.where(df["DPS_Annual"] > 0, np.log(df["DPS_Annual"]), np.nan)
    df["dlog_dps"] = df.groupby("Ticker")["ln_dps"].diff()

    # ΔWC/TA (contemporaneous change)
    df["dwc_ta"] = df.groupby("Ticker")["WC_to_TA"].diff()

    # consecutive year check
    df["year_gap"] = df.groupby("Ticker")["Year"].diff()

    # Keep only rows where change is year-to-year (no gaps)
    df = df[df["year_gap"] == 1].copy()

    # -----------------------------
    # 3) CREATE LAGGED FEATURES (use t-1 info to predict dlog_dps at t)
    # -----------------------------
    # For each firm-year t, use values from t-1 as predictors
    for col in ["ROA", "Leverage", "Size", "WC_to_TA", "dwc_ta"]:
        df[col + "_lag1"] = df.groupby("Ticker")[col].shift(1)

    # Our prediction target is dlog_dps at year t
    # We require lag features not missing
    needed = ["dlog_dps", "ROA_lag1", "Leverage_lag1", "Size_lag1", "dwc_ta_lag1"]
    df_model = df.dropna(subset=needed).copy()

    print("Model-ready rows:", df_model.shape[0], "| Firms:", df_model["Ticker"].nunique())
    print("Years in model-ready:", sorted(df_model["Year"].unique()))

    # -----------------------------
    # 4) TEMPORAL SPLIT 
    # Train years: <= 2020 (targets 2016-2020 typically)
    # Test years : 2021-2024
    # -----------------------------
    train = df_model[df_model["Year"] <= 2020].copy()
    test  = df_model[(df_model["Year"] >= 2021) & (df_model["Year"] <= 2024)].copy()

    print("Train rows:", train.shape[0], "| Test rows:", test.shape[0])
    print("Train years:", sorted(train["Year"].unique()))
    print("Test years:", sorted(test["Year"].unique()))

    # Keep only firms present in both (important for FE prediction)
    common_firms = sorted(set(train["Ticker"]).intersection(set(test["Ticker"])))
    train = train[train["Ticker"].isin(common_firms)].copy()
    test  = test[test["Ticker"].isin(common_firms)].copy()
    print("Common firms:", len(common_firms), "| Train rows:", train.shape[0], "| Test rows:", test.shape[0])

    y_train = train["dlog_dps"].to_numpy()
    y_test  = test["dlog_dps"].to_numpy()

    # -----------------------------
    # 5) RANDOM FOREST (A) Controls only vs (B) + Working Capital
    # Using lagged features to avoid leakage
    # -----------------------------
    Xcols_A = ["ROA_lag1", "Leverage_lag1", "Size_lag1"]
    Xcols_B = ["dwc_ta_lag1", "ROA_lag1", "Leverage_lag1", "Size_lag1"]

    X_train_A = train[Xcols_A].to_numpy()
    X_test_A  = test[Xcols_A].to_numpy()

    X_train_B = train[Xcols_B].to_numpy()
    X_test_B  = test[Xcols_B].to_numpy()

    rf_params = dict(
        n_estimators=500,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    rf_A = RandomForestRegressor(**rf_params).fit(X_train_A, y_train)
    rf_B = RandomForestRegressor(**rf_params).fit(X_train_B, y_train)

    pred_A = rf_A.predict(X_test_A)
    pred_B = rf_B.predict(X_test_B)

    def metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return rmse, mae, r2

    rmse_A, mae_A, r2_A = metrics(y_test, pred_A)
    rmse_B, mae_B, r2_B = metrics(y_test, pred_B)

    ml_summary = pd.DataFrame({
        "Model": ["RF Controls only (lagged)", "RF Controls + Δ(WC/TA)_lag1"],
        "RMSE": [rmse_A, rmse_B],
        "MAE": [mae_A, mae_B],
        "R2_out_of_sample": [r2_A, r2_B],
        "Train_rows": [len(train), len(train)],
        "Test_rows": [len(test), len(test)]
    })

    print("\n=== ML PERFORMANCE (Temporal split) ===")
    print(ml_summary)

    imp_A = pd.DataFrame({"feature": Xcols_A, "importance": rf_A.feature_importances_}).sort_values("importance", ascending=False)
    imp_B = pd.DataFrame({"feature": Xcols_B, "importance": rf_B.feature_importances_}).sort_values("importance", ascending=False)

    print("\n=== Feature importance A ===")
    print(imp_A)
    print("\n=== Feature importance B ===")
    print(imp_B)

    # -----------------------------
    # 6) ECONOMETRIC PREDICTION 
    # (i) Pooled OLS on train -> predict test
    # (ii) Firm FE on train -> predict test (for same firms)
    # -----------------------------

    # --- (i) Pooled OLS (controls + WC lag1) ---
    Y_tr = train["dlog_dps"].to_numpy()
    X_tr = sm.add_constant(train[["dwc_ta_lag1","ROA_lag1","Leverage_lag1","Size_lag1"]]).to_numpy(dtype=float)

    ols_tr = sm.OLS(Y_tr, X_tr).fit()
    X_te = sm.add_constant(test[["dwc_ta_lag1","ROA_lag1","Leverage_lag1","Size_lag1"]]).to_numpy(dtype=float)
    pred_ols = ols_tr.predict(X_te)

    rmse_ols, mae_ols, r2_ols = metrics(y_test, pred_ols)

    # --- (ii) Firm Fixed Effects on train (dummies), then predict on test ---
    # Build design matrices with identical columns
    def build_fe_matrix(data, firm_list, include_wc=True):
        base = ["ROA_lag1","Leverage_lag1","Size_lag1"]
        if include_wc:
            base = ["dwc_ta_lag1"] + base

        X0 = data[base].copy()
        dums = pd.get_dummies(data["Ticker"], prefix="FE")
        # ensure all firms columns exist
        for f in firm_list:
            col = f"FE_{f}"
            if col not in dums.columns:
                dums[col] = 0
        dums = dums[[f"FE_{f}" for f in firm_list]]
        # drop one dummy to avoid collinearity
        dums = dums.drop(columns=[f"FE_{firm_list[0]}"])
        X = pd.concat([X0, dums], axis=1)
        X = sm.add_constant(X)
        return X

    firm_list = common_firms[:]  # same firms in train & test

    X_fe_tr = build_fe_matrix(train, firm_list, include_wc=True).to_numpy(dtype=float)
    Y_fe_tr = train["dlog_dps"].to_numpy(dtype=float)

    fe_tr = sm.OLS(Y_fe_tr, X_fe_tr).fit()

    X_fe_te = build_fe_matrix(test, firm_list, include_wc=True).to_numpy(dtype=float)
    pred_fe = fe_tr.predict(X_fe_te)

    rmse_fe, mae_fe, r2_fe = metrics(y_test, pred_fe)

    econ_pred = pd.DataFrame({
        "Model": ["Pooled OLS (lagged, train->test)", "Firm FE (lagged, train->test)"], 
        "RMSE": [rmse_ols, rmse_fe],
        "MAE": [mae_ols, mae_fe],
        "R2_out_of_sample": [r2_ols, r2_fe],
        "Train_years": ["<=2020", "<=2020"],
        "Test_years": ["2021-2024", "2021-2024"]
    })

    print("\n=== ECONOMETRIC PREDICTION (Temporal split) ===")
    print(econ_pred)
    import sys
    print(type(ml_summary))
    with open(FIGURES_DIR / "table_pred.tex", "w") as f:
        f.write(econ_pred.to_latex(index=False, escape=False))

    print("✅ Exporté :", FIGURES_DIR / "table_pred.tex")
    import matplotlib.pyplot as plt
    import numpy as np

    # Data
    models = ["RF Controls only", "RF Controls + Δ(WC/TA)"]
    rmse_values = [0.507165, 0.544684]

    # Positions for bars
    x = np.arange(len(models))

    plt.figure(figsize=(4, 5))  # taille plus compacte

    # Thin bars
    plt.bar(x, rmse_values, width=0.85)

    # Labels
    plt.xticks(x, models, rotation=0)
    plt.ylabel("RMSE (out-of-sample)")
    plt.title("Out-of-sample Prediction Error (Random Forest)")

    # Clean layout
    plt.tight_layout()

    # Save for Overleaf
    plt.savefig(FIGURES_DIR / "fig2_rf_rmse_comparison.png",
                dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()
    pass