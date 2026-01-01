# app.py
# JAMS Capital – Investor Portal (Streamlit Cloud + localhost compatible)
# Fixes requested:
# 1) Logo too big -> hard cap height + sidebar placement (no giant whitespace block)
# 2) Sector chart: if sector lookup missing/Unknown -> bucket into "Other" (not "Unknown")
# 3) Cloud-safe padding (no top clipping) + white readable theme
# 4) Compact pie + compact P&L bars (top-N + Other; winners/losers only)

import re
import io
import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
except Exception:
    yf = None


# ============================================================
# BRAND / THEME
# ============================================================
JAMS_RED = "#921515"
DARK_TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#E5E7EB"
BG = "#FFFFFF"
CARD_BG = "#FFFFFF"
PLOT_BG = "#FFFFFF"
GRID = "#E5E7EB"

# Put one of these files in repo root (same folder as app.py)
LOGO_FILES = [
    "capital logo-02.png",
    "jc_logo.png",
    "jams_logo.png",
    "logo.png",
]

MASTER_URL = "https://docs.google.com/spreadsheets/d/15C7vLJLkyJGumLwD4Ld76awOTi26JJcrDZUxwLKsDO0/edit?gid=0#gid=0"
MASTER_TAB_NAME = None
MASTER_TAB_CANDIDATES = ["Investor Data Room", "Data Room", "Master", "Data", "Sheet1"]


# ============================================================
# PAGE CONFIG + CSS
# ============================================================
st.set_page_config(
    page_title="JAMS Capital | Investor Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      html, body, [class*="stApp"] {{
        overflow: visible !important;
      }}

      section.main > div.block-container {{
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
      }}

      .block-container > div:first-child {{
        margin-top: 0rem !important;
      }}

      .stApp {{ background-color: {BG}; color: {DARK_TEXT}; }}
      [data-testid="stSidebar"] {{ background-color: #F9FAFB; }}

      .topbar {{
        display:flex; align-items:center; gap:16px;
        padding: 12px 14px; background:{CARD_BG};
        border: 1px solid {BORDER}; border-radius: 16px;
      }}
      .brand-title {{ font-size: 20px; font-weight: 900; color: {DARK_TEXT}; line-height: 1.1; }}
      .brand-title span {{ color: {JAMS_RED}; }}
      .brand-sub {{ color:{MUTED}; font-size:12px; margin-top:2px; }}

      .card {{
        background:{CARD_BG}; border:1px solid {BORDER}; border-radius: 16px;
        padding: 14px 16px;
      }}
      .kpi-label {{ color:{MUTED}; font-size:12px; }}
      .kpi-value {{ color:{DARK_TEXT}; font-size:24px; font-weight:900; }}
      .kpi-sub {{ color:{MUTED}; font-size:12px; margin-top:2px; }}

      .stButton>button {{
        background:{JAMS_RED}; color:white; border-radius: 10px; border:0;
        padding: 0.55rem 1rem; font-weight:900;
      }}
      .stButton>button:hover {{ opacity: 0.92; color:white; }}

      /* Keep images from blowing up */
      img {{
        max-height: 1000px !important;
        width: auto !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPERS
# ============================================================
def extract_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("Could not parse spreadsheet ID from URL.")
    return m.group(1)


def gviz_csv_url(spreadsheet_id: str, sheet: str, cell_range: str | None = None) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={sheet}"
    if cell_range:
        base += f"&range={cell_range}"
    return base


@st.cache_data(show_spinner=False, ttl=300)
def read_public_sheet_range(spreadsheet_id: str, sheet: str, cell_range: str) -> pd.DataFrame:
    url = gviz_csv_url(spreadsheet_id, sheet=sheet, cell_range=cell_range)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        raise ValueError(f"Sheet fetch failed (HTTP {r.status_code}). URL: {url}")
    if "<html" in r.text.lower():
        raise ValueError("Google returned HTML instead of CSV. Sheet is not public/published.")
    return pd.read_csv(io.StringIO(r.text), header=None)


def clean_str(x) -> str:
    if isinstance(x, pd.Series):
        x = x.iloc[0] if len(x) else ""
    elif isinstance(x, (np.ndarray, list, tuple)):
        x = x[0] if len(x) else ""
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_header(x) -> str:
    s = clean_str(x).lower()
    s = s.replace(":", "")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_num(x):
    if isinstance(x, pd.Series):
        x = x.iloc[0] if len(x) else np.nan
    elif isinstance(x, (np.ndarray, list, tuple)):
        x = x[0] if len(x) else np.nan
    if pd.isna(x):
        return np.nan
    s = clean_str(x)
    if s == "" or s.lower() == "nan":
        return np.nan
    s = s.replace(",", "").replace("$", "").replace("NT$", "").replace("USD", "").strip()
    s = s.replace("(", "-").replace(")", "")
    if s.endswith("%"):
        v = pd.to_numeric(s[:-1], errors="coerce")
        return (v / 100.0) if pd.notna(v) else np.nan
    return pd.to_numeric(s, errors="coerce")


def safe_dt(x):
    return pd.to_datetime(clean_str(x), errors="coerce").tz_localize(None)


def get_col_as_series(df: pd.DataFrame, colname: str) -> pd.Series:
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):  # duplicate col names
        return obj.iloc[:, 0]
    return obj


def show_logo_small_in_header():
    """
    Avoid large logo blocks. We place the logo in the sidebar (small) by default.
    Return whether logo exists.
    """
    for f in LOGO_FILES:
        try:
            st.sidebar.image(f, width=1000)  # hard cap
            return True
        except Exception:
            continue
    return False


# ============================================================
# MASTER PARSING
# ============================================================
def _scan_master_for_table(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.copy().applymap(clean_str)
    gl = g.applymap(lambda v: v.lower())

    target = ["name", "password", "portfolio link"]
    header_row = None
    start_col = None

    for r in range(gl.shape[0]):
        row_vals = list(gl.iloc[r, :])
        for c in range(gl.shape[1] - 2):
            if row_vals[c:c + 3] == target:
                header_row = r
                start_col = c
                break
        if header_row is not None:
            break

    if header_row is None:
        for r in range(gl.shape[0]):
            row_list = list(gl.iloc[r, :])
            row_set = set([v for v in row_list if v])
            if all(t in row_set for t in target):
                header_row = r
                start_col = min(row_list.index("name"), row_list.index("password"), row_list.index("portfolio link"))
                break

    if header_row is None:
        preview = g.iloc[:12, :12]
        raise ValueError(
            "Could not locate master headers in A1:Z200. Preview top-left 12x12:\n"
            f"{preview}"
        )

    df = g.iloc[header_row + 1:, start_col:start_col + 3].copy()
    df.columns = ["Name", "Password", "Portfolio Link"]
    df = df.replace("", np.nan).dropna(how="all")

    df["Password"] = df["Password"].astype(str).str.strip()
    df["Portfolio Link"] = df["Portfolio Link"].astype(str).str.strip()
    df = df[df["Portfolio Link"].str.contains("docs.google.com/spreadsheets", na=False)]

    if len(df) == 0:
        raise ValueError("Master table found but contains no valid Portfolio Link rows.")
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_master_credentials() -> pd.DataFrame:
    mid = extract_sheet_id(MASTER_URL)
    sheets = [MASTER_TAB_NAME] if MASTER_TAB_NAME else MASTER_TAB_CANDIDATES

    last_err = None
    for s in sheets:
        if not s:
            continue
        try:
            grid = read_public_sheet_range(mid, sheet=s, cell_range="A1:Z200")
            return _scan_master_for_table(grid)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Failed to load master sheet. Tabs tried: {sheets}. Last error: {last_err}")


def find_investor_by_password(password: str, master_df: pd.DataFrame) -> pd.Series | None:
    p = str(password).strip()
    hit = master_df[master_df["Password"].astype(str).str.strip() == p]
    if len(hit) == 0:
        return None
    return hit.iloc[0]


# ============================================================
# PORTFOLIO PARSING
# ============================================================
@dataclass
class PortfolioData:
    twr: pd.DataFrame
    weights: pd.DataFrame
    pnl_current: pd.DataFrame
    pnl_all: pd.DataFrame
    invested_capital: float
    current_balance: float


def extract_list_from_grid(grid: pd.DataFrame, row_start: int, col_label: int, col_value: int, stop_if_blank=True):
    out = []
    for r in range(row_start, grid.shape[0]):
        a = clean_str(grid.iat[r, col_label] if col_label < grid.shape[1] else "")
        b = grid.iat[r, col_value] if col_value < grid.shape[1] else ""
        if stop_if_blank and (a == "" or a.lower() == "nan"):
            break
        out.append((a, b))
    return pd.DataFrame(out, columns=["A", "B"])


def detect_twr_header_row(grid: pd.DataFrame, max_rows=60) -> int:
    g = grid.copy().iloc[:max_rows, :].applymap(norm_header)
    for r in range(g.shape[0]):
        row = list(g.iloc[r, :])
        row_set = set([x for x in row if x])
        has_date = ("date" in row_set) or any(x.startswith("date") for x in row_set)
        has_spy = any("spy" in x for x in row_set)
        has_begin = any("begin" in x for x in row_set)
        has_end = any("ending" in x or (x.startswith("end") and "value" in x) for x in row_set)
        has_cf = any(x == "cf" or "cash flow" in x or "cashflow" in x for x in row_set)
        if has_date and has_spy and (has_begin or has_end or has_cf):
            return r
    return 0


def map_twr_columns(headers: list[str]) -> dict[str, str | None]:
    hnorm = [norm_header(h) for h in headers]

    def find_col(predicate):
        for i, h in enumerate(hnorm):
            if predicate(h):
                return headers[i]
        return None

    col_date = find_col(lambda h: h == "date" or h.startswith("date "))
    col_begin = find_col(lambda h: "begin" in h and ("value" in h or "val" in h)) or find_col(lambda h: "begin" in h)
    col_cf = find_col(lambda h: h == "cf" or "cash flow" in h or "cashflow" in h)
    col_end = find_col(lambda h: ("ending" in h and ("value" in h or "val" in h)) or ("ending" in h) or ("end" in h and "value" in h))
    col_spy = find_col(lambda h: "spy" in h and ("balance" in h or "value" in h)) or find_col(lambda h: "spy" in h)

    return {"Date": col_date, "Beginning Value": col_begin, "CF": col_cf, "Ending Value": col_end, "SPY Balance": col_spy}


def load_twr(pid: str) -> pd.DataFrame:
    grid = read_public_sheet_range(pid, sheet="TWR", cell_range="A1:Q6000")

    header_row = detect_twr_header_row(grid)
    header = [clean_str(x) for x in grid.iloc[header_row].tolist()]

    df = grid.iloc[header_row + 1:].copy()
    df.columns = header
    df = df.dropna(axis=1, how="all")

    mapped = map_twr_columns(list(df.columns))

    critical = ["Date", "SPY Balance", "Beginning Value", "CF", "Ending Value"]
    if any(mapped[k] is None for k in critical):
        cols = list(df.columns)
        if len(cols) >= 7:
            mapped = {"Date": cols[0], "SPY Balance": cols[3], "Beginning Value": cols[4], "CF": cols[5], "Ending Value": cols[6]}
        else:
            raise ValueError("TWR parsing failed (cannot map required columns).")

    out = pd.DataFrame()
    out["Date"] = get_col_as_series(df, mapped["Date"]).map(safe_dt)
    out["Beginning Value"] = get_col_as_series(df, mapped["Beginning Value"]).map(to_num)
    out["CF"] = get_col_as_series(df, mapped["CF"]).map(to_num).fillna(0.0)
    out["Ending Value"] = get_col_as_series(df, mapped["Ending Value"]).map(to_num)
    out["SPY Balance"] = get_col_as_series(df, mapped["SPY Balance"]).map(to_num)

    out = out.dropna(subset=["Date", "Beginning Value", "Ending Value", "SPY Balance"]).copy()
    out = out.sort_values("Date").reset_index(drop=True)

    today = pd.Timestamp(dt.date.today())
    out = out[out["Date"] <= today].copy()

    out["Port_DailyRet"] = (out["Ending Value"] - out["CF"]) / out["Beginning Value"] - 1.0
    out["SPY_DailyRet"] = out["SPY Balance"].pct_change()

    out["Port_Cum"] = (1.0 + out["Port_DailyRet"]).cumprod() - 1.0
    out["SPY_Cum"] = (1.0 + out["SPY_DailyRet"].fillna(0.0)).cumprod() - 1.0
    return out


def find_overview_totals_gviz_heuristic(ov: pd.DataFrame) -> tuple[float, float]:
    g = ov.copy().applymap(clean_str)
    best = None  # (balance, invested, row_index)

    for r in range(g.shape[0]):
        row_raw = [clean_str(x) for x in g.iloc[r, :].tolist()]
        row_l = " ".join([x.lower() for x in row_raw if x and x.lower() != "none"])
        if "total" not in row_l:
            continue

        nums = []
        for x in row_raw:
            v = to_num(x)
            if pd.notna(v) and abs(v) > 1.5:
                nums.append(float(v))

        if len(nums) < 2:
            continue

        nums_sorted = sorted(nums, key=lambda z: abs(z), reverse=True)
        top2 = nums_sorted[:2]
        invested = min(top2)
        balance = max(top2)

        if best is None or balance > best[0]:
            best = (balance, invested, r)

    if best is None:
        return np.nan, np.nan
    return float(best[1]), float(best[0])  # invested, balance


def load_portfolio(portfolio_url: str) -> PortfolioData:
    pid = extract_sheet_id(portfolio_url)

    twr = load_twr(pid)

    g = read_public_sheet_range(pid, sheet="Graphing", cell_range="A1:P300")
    r0 = 7  # row 8

    # H/I weights
    w = extract_list_from_grid(g, row_start=r0, col_label=7, col_value=8)
    w.columns = ["Security", "Weight"]
    w["Security"] = w["Security"].astype(str).str.strip()
    w["Weight"] = w["Weight"].apply(to_num)
    w = w.dropna(subset=["Weight"])
    w = w[w["Security"].str.lower().ne("nan")]
    w = w[w["Security"].str.len() > 0]

    wsum = w["Weight"].sum()
    if wsum > 1.5:
        w["Weight"] = w["Weight"] / 100.0
        wsum = w["Weight"].sum()
    if wsum > 0 and (wsum < 0.98 or wsum > 1.02):
        w["Weight"] = w["Weight"] / wsum

    # K/L current pnl
    pnl_cur = extract_list_from_grid(g, row_start=r0, col_label=10, col_value=11)
    pnl_cur.columns = ["Ticker", "PnL"]
    pnl_cur["Ticker"] = pnl_cur["Ticker"].astype(str).str.strip().str.upper()
    pnl_cur["PnL"] = pnl_cur["PnL"].apply(to_num)
    pnl_cur = pnl_cur.dropna(subset=["PnL"])
    pnl_cur = pnl_cur[pnl_cur["Ticker"].str.len() > 0]
    pnl_cur = pnl_cur[pnl_cur["Ticker"].str.lower().ne("nan")]

    # O/P all pnl
    pnl_all = extract_list_from_grid(g, row_start=r0, col_label=14, col_value=15)
    pnl_all.columns = ["Ticker", "PnL"]
    pnl_all["Ticker"] = pnl_all["Ticker"].astype(str).str.strip().str.upper()
    pnl_all["PnL"] = pnl_all["PnL"].apply(to_num)
    pnl_all = pnl_all.dropna(subset=["PnL"])
    pnl_all = pnl_all[pnl_all["Ticker"].str.len() > 0]
    pnl_all = pnl_all[pnl_all["Ticker"].str.lower().ne("nan")]

    # Overview heuristic
    ov = read_public_sheet_range(pid, sheet="Overview", cell_range="A1:K300")
    invested_capital, current_balance = find_overview_totals_gviz_heuristic(ov)

    return PortfolioData(
        twr=twr,
        weights=w,
        pnl_current=pnl_cur,
        pnl_all=pnl_all,
        invested_capital=invested_capital,
        current_balance=current_balance,
    )


# ============================================================
# ANALYTICS
# ============================================================
def annualized_return(cum_ret: float, n_days: int) -> float:
    if n_days <= 1 or pd.isna(cum_ret):
        return np.nan
    years = n_days / 252.0
    if years <= 0:
        return np.nan
    return (1.0 + cum_ret) ** (1.0 / years) - 1.0


def annualized_vol(daily_r: pd.Series) -> float:
    r = daily_r.dropna()
    if len(r) < 10:
        return np.nan
    return float(np.std(r, ddof=1) * math.sqrt(252))


def beta_vs_spy(port: pd.Series, spy: pd.Series) -> float:
    port = port.dropna()
    spy = spy.dropna()
    idx = port.index.intersection(spy.index)
    if len(idx) < 20:
        return np.nan
    x = spy.loc[idx].values
    y = port.loc[idx].values
    var = np.var(x)
    return float(np.cov(x, y)[0, 1] / var) if var != 0 else np.nan


def max_drawdown(cum_curve: pd.Series) -> float:
    if len(cum_curve) == 0:
        return np.nan
    eq = 1.0 + cum_curve
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def hist_var_cvar(daily_r: pd.Series, alpha=0.05):
    r = daily_r.dropna()
    if len(r) < 60:
        return np.nan, np.nan
    q = np.quantile(r, alpha)
    var = -q
    tail = r[r <= q]
    cvar = -(tail.mean()) if len(tail) else np.nan
    return float(var), float(cvar)


@st.cache_data(show_spinner=False, ttl=86400)
def lookup_sectors(tickers: list[str]) -> dict[str, str]:
    """
    If sector cannot be resolved -> return "Other" per your request.
    """
    out = {}
    tickers = [t.upper().strip() for t in tickers if t and str(t).strip()]

    for t in tickers:
        if t in {"CASH", "USD"}:
            out[t] = "Cash"

    if yf is None:
        for t in tickers:
            out.setdefault(t, "Other")
        return out

    for t in tickers:
        if t in out:
            continue
        try:
            tk = yf.Ticker(t)
            sector = None
            quote_type = None

            try:
                fi = getattr(tk, "fast_info", None)
                if fi and isinstance(fi, dict):
                    sector = fi.get("sector", None)
            except Exception:
                pass

            if not sector:
                try:
                    info = tk.info
                    if isinstance(info, dict):
                        sector = info.get("sector", None)
                        quote_type = info.get("quoteType", None)
                except Exception:
                    sector = None
                    quote_type = None

            # If it is an ETF and no sector, treat as Other
            if not sector and quote_type == "ETF":
                sector = "Other"

            out[t] = sector if sector else "Other"
        except Exception:
            out[t] = "Other"

    return out


# ============================================================
# CHARTING HELPERS
# ============================================================
def make_topn_other(df: pd.DataFrame, label_col: str, value_col: str, top_n=8):
    d = df[[label_col, value_col]].dropna().copy()
    d[value_col] = d[value_col].astype(float)
    d = d.sort_values(value_col, ascending=False)
    if len(d) <= top_n:
        return d
    top = d.head(top_n).copy()
    other_val = float(d.iloc[top_n:][value_col].sum())
    other = pd.DataFrame([{label_col: "Other", value_col: other_val}])
    return pd.concat([top, other], ignore_index=True)


def plot_returns(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Port_Cum_Window"],
        mode="lines", name="Portfolio",
        line=dict(color=JAMS_RED, width=3)
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SPY_Cum_Window"],
        mode="lines", name="S&P 500 (SPY)",
        line=dict(color="#111827", width=2)
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=DARK_TEXT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickformat=".0%", gridcolor=GRID, zerolinecolor=GRID),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        title="Portfolio Return vs S&P 500 (SPY)"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_donut(df: pd.DataFrame, label_col: str, value_col: str, title: str, top_n=8):
    d2 = make_topn_other(df, label_col, value_col, top_n=top_n)
    fig = go.Figure(data=[go.Pie(
        labels=d2[label_col],
        values=d2[value_col],
        hole=0.60,
        textinfo="percent",
        textposition="inside",
        marker=dict(line=dict(color=BG, width=1))
    )])
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=DARK_TEXT, size=13),
        legend=dict(orientation="v", x=1.02, y=0.5),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show full breakdown"):
        d_full = df[[label_col, value_col]].dropna().copy()
        d_full[value_col] = d_full[value_col].astype(float)
        d_full = d_full.sort_values(value_col, ascending=False)
        st.dataframe(d_full, use_container_width=True, hide_index=True)


def plot_bar_compact(df: pd.DataFrame, title: str, top_each_side=12, height_cap=420):
    d = df.dropna().copy()
    if len(d) == 0:
        st.warning("No data.")
        return

    d["PnL"] = d["PnL"].astype(float)

    winners = d.sort_values("PnL", ascending=False).head(top_each_side)
    losers = d.sort_values("PnL", ascending=True).head(top_each_side)
    dd = pd.concat([losers, winners]).drop_duplicates(subset=["Ticker"], keep="first")
    dd = dd.sort_values("PnL")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dd["PnL"], y=dd["Ticker"], orientation="h",
        marker=dict(color=JAMS_RED)
    ))
    fig.update_layout(
        title=title,
        height=min(height_cap, max(320, 18 * len(dd) + 140)),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=DARK_TEXT, size=13),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show full P&L table"):
        full = d.sort_values("PnL", ascending=False)
        st.dataframe(full, use_container_width=True, hide_index=True)


# ============================================================
# HEADER
# ============================================================
logo_ok = show_logo_small_in_header()

st.markdown(
    f"""
    <div class="topbar">
      <div>
        <div class="brand-title">JAMS Capital <span>Investor Portal</span></div>
        <div class="brand-sub">Password-based access to performance and risk analytics.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

if not logo_ok:
    st.sidebar.warning(
        "Logo not found. Commit your logo image into the GitHub repo (same folder as app.py). "
        "Accepted filenames: " + ", ".join([f'"{x}"' for x in LOGO_FILES])
    )


# ============================================================
# SIDEBAR LOGIN
# ============================================================
st.sidebar.markdown("## Investor Login")
password = st.sidebar.text_input("Password", type="password", placeholder="Enter password")
login = st.sidebar.button("Access Portfolio")

master = load_master_credentials()

if login:
    row = find_investor_by_password(password, master)
    if row is None:
        st.sidebar.error("Invalid password.")
        st.session_state["authed"] = False
    else:
        st.session_state["authed"] = True
        st.session_state["investor_name"] = str(row.get("Name", "Investor"))
        st.session_state["portfolio_link"] = str(row.get("Portfolio Link", "")).strip()

if not st.session_state.get("authed", False):
    st.info("Enter your password in the sidebar to view your portfolio dashboard.")
    st.stop()

portfolio_link = st.session_state.get("portfolio_link", "")
if "docs.google.com/spreadsheets" not in portfolio_link:
    st.error("Portfolio Link is missing/invalid in the master sheet.")
    st.stop()


# ============================================================
# LOAD PORTFOLIO
# ============================================================
with st.spinner("Loading portfolio..."):
    pdata = load_portfolio(portfolio_link)

twr = pdata.twr.copy()


# ============================================================
# DATE CONTROLS
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("## Date Range")
preset = st.sidebar.selectbox("Preset", ["1M", "3M", "6M", "YTD", "All", "Custom"], index=2)

min_d = twr["Date"].min().date()
max_d = min(twr["Date"].max().date(), dt.date.today())

if preset == "Custom":
    d1, d2 = st.sidebar.date_input(
        "Select range",
        value=(max(min_d, max_d - dt.timedelta(days=90)), max_d),
        min_value=min_d,
        max_value=max_d
    )
    if isinstance(d1, (tuple, list)):
        d1, d2 = d1
else:
    d2 = max_d
    if preset == "1M":
        d1 = max(min_d, d2 - dt.timedelta(days=30))
    elif preset == "3M":
        d1 = max(min_d, d2 - dt.timedelta(days=90))
    elif preset == "6M":
        d1 = max(min_d, d2 - dt.timedelta(days=180))
    elif preset == "YTD":
        d1 = max(min_d, dt.date(d2.year, 1, 1))
    else:
        d1 = min_d

dfw = twr[(twr["Date"].dt.date >= d1) & (twr["Date"].dt.date <= d2)].copy()
dfw = dfw.dropna(subset=["Port_DailyRet", "SPY_DailyRet"], how="any")

df_plot = dfw[["Date", "Port_DailyRet", "SPY_DailyRet"]].copy()
df_plot["Port_Cum_Window"] = (1.0 + df_plot["Port_DailyRet"]).cumprod() - 1.0
df_plot["SPY_Cum_Window"] = (1.0 + df_plot["SPY_DailyRet"]).cumprod() - 1.0

port_window_cum = float(df_plot["Port_Cum_Window"].iloc[-1]) if len(df_plot) else np.nan
spy_window_cum = float(df_plot["SPY_Cum_Window"].iloc[-1]) if len(df_plot) else np.nan

n_days = len(df_plot)
ann_ret = annualized_return(port_window_cum, n_days)
vol = annualized_vol(df_plot["Port_DailyRet"])
beta = beta_vs_spy(df_plot["Port_DailyRet"], df_plot["SPY_DailyRet"])
mdd = max_drawdown(df_plot["Port_Cum_Window"]) if len(df_plot) else np.nan
var95, cvar95 = hist_var_cvar(df_plot["Port_DailyRet"], 0.05)

all_outperf = float(twr["Port_Cum"].iloc[-1] - twr["SPY_Cum"].iloc[-1]) if len(twr) else np.nan


# ============================================================
# MAIN DASHBOARD
# ============================================================
st.markdown(f"### Welcome, **{st.session_state.get('investor_name','Investor')}**")

k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(
        f"""
        <div class="card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

kpi(k1, "Total Invested Amount", f"${pdata.invested_capital:,.2f}" if pd.notna(pdata.invested_capital) else "—")
kpi(k2, "Current Balance", f"${pdata.current_balance:,.2f}" if pd.notna(pdata.current_balance) else "—")
kpi(k3, "Total Return (Selected)", f"{port_window_cum:.2%}" if pd.notna(port_window_cum) else "—", f"{d1} to {d2}")
kpi(k4, "Annualized Return (Selected)", f"{ann_ret:.2%}" if pd.notna(ann_ret) else "—")
kpi(k5, "All-Time vs S&P (SPY)", f"{all_outperf:.2%}" if pd.notna(all_outperf) else "—")

st.markdown("")

left, right = st.columns([2, 1], gap="large")
with left:
    if len(df_plot) < 5:
        st.warning("Not enough valid return rows in the selected window to plot.")
    else:
        plot_returns(df_plot)

with right:
    st.markdown("#### Risk Snapshot (Selected Range)")
    risk = pd.DataFrame([
        ["Beta vs SPY", beta, "beta"],
        ["Volatility (Ann.)", vol, "pct"],
        ["Max Drawdown", mdd, "pct"],
        ["VaR 95% (1D)", var95, "pct"],
        ["CVaR 95% (1D)", cvar95, "pct"],
        ["S&P Return (Selected)", spy_window_cum, "pct"],
    ], columns=["Metric", "Value", "Fmt"])

    def fmt(row):
        v = row["Value"]
        if pd.isna(v):
            return "—"
        if row["Fmt"] == "beta":
            return f"{v:.2f}"
        return f"{v:.2%}"

    risk["Value"] = risk.apply(fmt, axis=1)
    st.dataframe(risk[["Metric", "Value"]], use_container_width=True, hide_index=True)

st.markdown("---")

cA, cB = st.columns(2, gap="large")
with cA:
    st.markdown("#### Current Allocation (Weights)")
    if len(pdata.weights) == 0:
        st.error("Weights not found at Graphing!H8:I.")
    else:
        plot_donut(pdata.weights, "Security", "Weight", "Portfolio Weights (Top 8 + Other)", top_n=8)

with cB:
    st.markdown("#### Sector Allocation (Current Holdings)")
    tickers = []
    if len(pdata.pnl_current):
        tickers = pdata.pnl_current["Ticker"].dropna().astype(str).tolist()
    else:
        tickers = pdata.weights["Security"].dropna().astype(str).tolist()

    tickers = [t.upper().strip() for t in tickers if t and t.upper().strip() not in {"CASH", "OTHER"}]
    tickers = [t for t in tickers if re.fullmatch(r"[A-Z\.\-]{1,10}", t)]

    if len(tickers) == 0:
        st.warning("No tickers available to map sectors.")
    else:
        sectors = lookup_sectors(tickers)  # returns Other when missing
        wmap = {str(s).upper(): float(w) for s, w in zip(pdata.weights["Security"], pdata.weights["Weight"])}

        rows = []
        for t in tickers:
            sec = sectors.get(t, "Other")
            if sec in (None, "", "Unknown"):
                sec = "Other"
            rows.append((sec, wmap.get(t, 0.0)))

        sdf = pd.DataFrame(rows, columns=["Sector", "Weight"]).groupby("Sector", as_index=False)["Weight"].sum()
        # If yfinance gave nothing useful, everything will end in Other; that's OK per request.
        sdf = sdf[sdf["Weight"] > 0].sort_values("Weight", ascending=False)
        if len(sdf) == 0:
            sdf = pd.DataFrame([["Other", 1.0]], columns=["Sector", "Weight"])

        plot_donut(sdf, "Sector", "Weight", "Sector Weights (Top 8 + Other)", top_n=8)

st.markdown("")

cC, cD = st.columns(2, gap="large")
with cC:
    st.markdown("#### Profit Contribution (Current Holdings)")
    if len(pdata.pnl_current) == 0:
        st.error("Current holdings P&L not found at Graphing!K8:L.")
    else:
        plot_bar_compact(pdata.pnl_current, "P&L Contribution — Current Holdings (Top Winners/Losers)", top_each_side=12)

with cD:
    st.markdown("#### Profit Contribution (All Holdings: Current + Liquidated)")
    if len(pdata.pnl_all) == 0:
        st.error("All holdings P&L not found at Graphing!O8:P.")
    else:
        plot_bar_compact(pdata.pnl_all, "P&L Contribution — All Holdings (Top Winners/Losers)", top_each_side=12)

with st.expander("Debug (Overview Export)", expanded=False):
    st.write("Invested Capital (detected):", pdata.invested_capital)
    st.write("Current Balance (detected):", pdata.current_balance)
    pid_dbg = extract_sheet_id(portfolio_link)
    ov_dbg = read_public_sheet_range(pid_dbg, sheet="Overview", cell_range="A1:K120")
    st.dataframe(ov_dbg, use_container_width=True)
