from __future__ import annotations

import os
import warnings
import functools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from pandas.tseries.holiday import USFederalHolidayCalendar

warnings.filterwarnings("ignore")

COMMISSION_PER_CONTRACT: float = 0.65
COMMISSION_PER_SHARE: float    = 0.005
MINIMUM_COMMISSION: float      = 1.00
OPTION_MULTIPLIER: int         = 100

USE_EXECUTION_SPREAD: bool     = True
USE_REAL_VIX: bool = True

VIX_MAP: dict[str, str] = {
    "SPY":   "^VIX",
    "QQQ":   "^VXN",
    "IWM":   "^RVX",
    "DIA":   "^VXD",
    "^GSPC": "^VIX",
    "^SPX":  "^VIX",
    "^NDX":  "^VXN",
    "^RUT":  "^RVX",
    "^DJI":  "^VXD",
}

STRIKE_TICK_LIQUID_ETF: set[str] = {
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","XLF",
    "EEM","FXI","VXX","UVXY","HYG","LQD","XLE","XLK",
    "ARKK","SOXX","SMH","XBI","IBB","JETS","GDX","SLB",
}
STRIKE_TICK_INDEX: set[str] = {"SPX","^GSPC","NDX","^NDX","RUT","^RUT","VIX","^VIX"}


_BROAD_INDEX_ETF: set[str] = {
    "SPY","QQQ","IWM","DIA","^GSPC","^SPX","^NDX","^RUT","^DJI",
}
_SECTOR_ETF: set[str] = {
    "GLD","SLV","TLT","XLF","EEM","FXI","VXX","UVXY","HYG","LQD",
    "XLE","XLK","ARKK","SOXX","SMH","XBI","IBB","JETS","GDX","SLB",
}

VVIX_MAP: dict[str, str] = {
    "SPY":   "^VVIX",
    "^GSPC": "^VVIX",
    "^SPX":  "^VVIX",
}

VVIX_NORM:          float = 94.0
SKEW_VVIX_BASE:     float = 0.25
SKEW_VVIX_EXPONENT: float = 1.20
SKEW_VVIX_MIN:      float = 0.05
SKEW_VVIX_MAX:      float = 0.80

SKEW_CALL_OTM_FACTOR: float = 0.30

SPREAD_IV_ALPHA: float = 0.02
SPREAD_IV_BETA:  float = 0.12
SPREAD_IV_MAX_BASE: float = 0.18

SPREAD_OI_VERY_OTM_SHORT: float = 2.5
SPREAD_OI_VERY_OTM_LONG:  float = 1.8
SPREAD_OI_DEEP_ITM:       float = 1.4
SPREAD_OI_OTM_THRESHOLD:  float = 0.25
SPREAD_OI_ITM_THRESHOLD:  float = 0.20

USE_TERM_STRUCTURE: bool = True
VIX9D_MAP: dict[str, str] = {
    "SPY":   "^VIX9D",
    "QQQ":   "^VIX9D",
    "IWM":   "^VIX9D",
    "^GSPC": "^VIX9D",
    "^SPX":  "^VIX9D",
}
VIX3M_MAP: dict[str, str] = {
    "SPY":   "^VIX3M",
    "QQQ":   "^VIX3M",
    "IWM":   "^VIX3M",
    "^GSPC": "^VIX3M",
    "^SPX":  "^VIX3M",
}
TERM_T_SHORT: float = 9.0  / 365.0
TERM_T_MID:   float = 30.0 / 365.0
TERM_T_LONG:  float = 91.0 / 365.0
TERM_EXTRAP_CAP: float = 0.15

USE_EARNINGS_EFFECT: bool = True
EARNINGS_PRE_DAYS:   int   = 14
EARNINGS_POST_HALFLIFE: float = 1.0
EARNINGS_POST_DAYS:  int   = 5

def _earnings_day_mult(iv_base: float) -> float:
    iv = float(np.clip(iv_base, 0.05, 2.0))
    if iv < 0.25:
        return 1.25
    elif iv < 0.45:
        return 1.40
    else:
        return 1.60

def _earnings_pre_mult(iv_base: float) -> float:
    return 1.0 + (_earnings_day_mult(iv_base) - 1.0) * 0.7

EARNINGS_ETF_SET: set[str] = {
    "SPY","QQQ","IWM","DIA","GLD","SLV","TLT","XLF","EEM","FXI",
    "VXX","UVXY","HYG","LQD","XLE","XLK","ARKK","SOXX","SMH",
    "XBI","IBB","JETS","GDX","^GSPC","^NDX","^RUT","^DJI","SYNTHETIC",
}

DEFAULT_BID_ASK_SPREAD_PCT: float = 0.05
SPREAD_PCT_MAX: float             = 0.35
MIN_OPTION_TICK: float            = 0.01
PENNY_TICK_CUTOFF: float          = 3.00
TICK_ABOVE_CUTOFF: float          = 0.05

SPREAD_SATURATION_LOG: list = []
SPREAD_TOTAL_CALLS: list = [0]

USE_AMERICAN_PRICING: bool         = True
AMERICAN_ONLY_IF_EXERCISE_RELEVANT = True
BINOMIAL_STEPS_PER_DAY: int        = 5
BINOMIAL_STEPS_MIN: int            = 50
BINOMIAL_STEPS_MAX: int            = 250

IV_SURFACE_ENABLE: bool   = True
IV_SKEW_STRENGTH: float   = 0.25
IV_SMILE_STRENGTH: float  = 0.08
IV_TERM_STRENGTH: float   = 0.10
IV_MIN: float             = 0.03
IV_MAX: float             = 2.50

IV_PROXY_SMOOTH_SPAN: int          = 5
IV_PROXY_MIN: float                = 0.05
IV_PROXY_MAX: float                = 1.50
GJR_ALPHA: float                   = 0.05
GJR_BETA: float                    = 0.92
GJR_GAMMA: float                   = 0.05
GJR_EWMA_LAMBDA: float             = 0.97

TRADING_DAYS_PER_YEAR: int    = 252
CALENDAR_DAYS_PER_YEAR: float = 365.0

PRICING_NOISE_ENABLE: bool    = True
NOISE_BASE_SIGMA: float       = 0.002
NOISE_OTM_MULT: float         = 2.5
NOISE_NEAREXP_MULT: float     = 2.0
NOISE_ILLIQUID_MULT: float    = 3.5
NOISE_SEED_SALT: int          = 0

ROLL_SLIPPAGE_MULTIPLIER: float = 1.05

def roll_slippage_multiplier(dte_close: float) -> float:
    dte = max(float(dte_close), 0.0)
    mult = 1.05 + 0.02 * max(0.0, 7.0 - dte)
    return float(np.clip(mult, 1.05, 1.20))

IV_PROXY_VRP_AR1_SPEED: float    = 0.05
IV_PROXY_VRP_LONG_RUN: float     = 1.18
IV_PROXY_VRP_FALLBACK_NEUTRAL: float = 1.10

WARMUP_DAYS: int = 126

REG_T_MARGIN_PCT: float     = 0.20

MARGIN_CALL_BUFFER: float = 1.05
MARGIN_CALL_THRESHOLD: float  = 1.0 / MARGIN_CALL_BUFFER

STRIKE_DELTA_TOLERANCE: float = 0.30
EQUITY_SIZING_FLOOR_PCT: float = 0.20

MAX_MARGIN_PCT_EQUITY: float    = 0.65
MAX_NOTIONAL_PCT_EQUITY: float  = 0.40
MAX_WING_RISK_PCT_EQUITY: float = 0.45

IV_PROXY_VRP_VOL: float          = 0.015
IV_SKEW_AR1_SPEED: float   = 0.08
IV_SKEW_LONG_RUN: float    = 0.25
IV_SKEW_VOL: float         = 0.03
IV_SMILE_AR1_SPEED: float  = 0.10
IV_SMILE_LONG_RUN: float   = 0.07
IV_SMILE_VOL: float        = 0.015
IV_PROXY_VRP_DYNAMIC_WINDOW: int = 63
IV_PROXY_SPIKE_RATIO: float      = 2.0
IV_PROXY_SPIKE_HALFLIFE: int     = 10

IV_SKEW_STRENGTH_DEFAULT: float  = 0.25
IV_SMILE_STRENGTH_DEFAULT: float = 0.08


def vrp_from_hv(hv_annual: float) -> float:
    hv = float(np.clip(hv_annual, 0.05, 2.0))
    if hv < 0.15:
        vrp = 1.28 - (hv - 0.05) / 0.10 * 0.02
    elif hv < 0.25:
        vrp = 1.26 - (hv - 0.15) / 0.10 * 0.08
    elif hv < 0.40:
        vrp = 1.18 - (hv - 0.25) / 0.15 * 0.06
    elif hv < 0.60:
        vrp = 1.12 - (hv - 0.40) / 0.20 * 0.05
    else:
        vrp = max(1.04, 1.07 - (hv - 0.60) * 0.05)
    return float(np.clip(vrp, 1.04, 1.30))


def _otm_spread_mults(ticker: str) -> tuple[float, float]:
    t = (ticker or "").upper()
    if   t in _BROAD_INDEX_ETF: return 1.6, 1.3
    elif t in _SECTOR_ETF:      return 1.9, 1.5
    else:                        return SPREAD_OI_VERY_OTM_SHORT, SPREAD_OI_VERY_OTM_LONG


@dataclass
class MarketContext:
    S:             float
    r:             float
    q:             float
    sigma:         float
    date:          pd.Timestamp
    S_ref:         float = 0.0
    surface_skew:  float = IV_SKEW_STRENGTH_DEFAULT
    surface_smile: float = IV_SMILE_STRENGTH_DEFAULT
    day_idx:       int   = 0
    strike_tick:   float = 1.0
    iv_base:       float = 0.20
    ts_tenors:     object = None
    ts_ivs:        object = None
    ticker:        str   = ""

    def __post_init__(self):
        self.S     = max(float(self.S), 1e-8)
        self.r     = float(np.clip(self.r, -0.01, 0.15))
        self.q     = float(np.clip(self.q, 0.0, 0.20))
        self.sigma = float(np.clip(self.sigma, IV_MIN, IV_MAX))
        self.iv_base = float(np.clip(self.iv_base, IV_MIN, IV_MAX))
        if self.S_ref <= 0:
            self.S_ref = self.S


def parse_number(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    s = str(value).strip().replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        last  = parts[-1]
        if len(last) == 3 and last.isdigit():
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%Y/%m/%d"):
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    return datetime(2020,1,1)


def year_fraction(start, end) -> float:
    try:
        delta = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        return max(delta / (CALENDAR_DAYS_PER_YEAR * 86400), 0.0)
    except Exception:
        return 0.0


_US_HOLIDAYS_CACHE = None

def _us_holidays(start_year: int, end_year: int) -> set:
    global _US_HOLIDAYS_CACHE
    if _US_HOLIDAYS_CACHE is None:
        cal = USFederalHolidayCalendar()
        hols = cal.holidays(start=f"{start_year-1}-01-01",
                             end=f"{end_year+1}-12-31")
        _US_HOLIDAYS_CACHE = set(pd.Timestamp(h).normalize() for h in hols)
    return _US_HOLIDAYS_CACHE


def _is_us_holiday(date: pd.Timestamp) -> bool:
    d = pd.Timestamp(date).normalize()
    return d in _us_holidays(d.year, d.year)


def _third_friday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d + timedelta(weeks=2)


def expiration_date_from_days(ref_date: pd.Timestamp, days: int) -> pd.Timestamp:
    days = max(int(days), 1)
    ref = pd.Timestamp(ref_date)
    raw = ref + timedelta(days=days)

    if days < 14:
        while raw.weekday() != 4:
            raw += timedelta(days=1)
        target = pd.Timestamp(raw)
    else:
        target = _third_friday(raw.year, raw.month)
        if raw > target:
            next_month = raw + pd.DateOffset(months=1)
            target = _third_friday(next_month.year, next_month.month)

    if _is_us_holiday(target):
        target = target - timedelta(days=1)

    return target


def calculate_commissions(quantity: int, price: float = 0.0,
                           is_option: bool = True) -> float:
    if quantity <= 0:
        return 0.0
    comm = quantity * (COMMISSION_PER_CONTRACT if is_option else COMMISSION_PER_SHARE)
    return max(comm, MINIMUM_COMMISSION)


def _option_tick(price: float) -> float:
    return MIN_OPTION_TICK if price < PENNY_TICK_CUTOFF else TICK_ABOVE_CUTOFF


def _round_to_tick(price: float, side: str = "BUY") -> float:
    if price <= 0:
        return 0.0
    tick = _option_tick(price)
    if side.upper() in ("SELL","SHORT"):
        return float(np.floor(price / tick) * tick)
    return float(np.ceil(price / tick) * tick)


def estimate_bid_ask_spread_pct(S: float, K: float, mid: float,
                                 T: float, option_type: str = "call",
                                 iv_sigma: float = 0.20,
                                 ticker: str = "") -> float:

    otm_short_mult, otm_long_mult = _otm_spread_mults(ticker)

    iv_sigma = float(np.clip(iv_sigma, 0.05, 2.0))

    spread_base = SPREAD_IV_ALPHA + SPREAD_IV_BETA * iv_sigma
    spread_base = float(np.clip(spread_base, 0.02, SPREAD_IV_MAX_BASE))

    spread = spread_base

    if S > 0 and K > 0:
        log_m = abs(float(np.log(K / S)))
        spread *= 1.0 + min(log_m / 0.15, 1.5)

    if mid > 0:
        if   mid < 0.05: spread = max(spread, 0.50)
        elif mid < 0.15: spread = max(spread, 0.30)
        elif mid < 0.50: spread = max(spread, 0.18)
        elif mid < 1.00: spread = max(spread, 0.12)
        elif mid < 3.00: spread = max(spread, 0.08)

    if T > 0:
        dte = T * CALENDAR_DAYS_PER_YEAR
        if   dte <  5: spread *= 1.8
        elif dte < 14: spread *= 1.35
        elif dte < 21: spread *= 1.15

    log_m_abs = 0.0
    dte_val   = T * CALENDAR_DAYS_PER_YEAR if T > 0 else 30.0
    if S > 0 and K > 0:
        log_m_abs = abs(float(np.log(K / S)))
        if log_m_abs > SPREAD_OI_OTM_THRESHOLD:

            if dte_val < 14:
                spread *= otm_short_mult
            else:
                spread *= otm_long_mult
        elif log_m_abs > SPREAD_OI_ITM_THRESHOLD:
            spread *= SPREAD_OI_DEEP_ITM

    SPREAD_TOTAL_CALLS[0] += 1
    if spread > SPREAD_PCT_MAX:
        SPREAD_SATURATION_LOG.append({
            "dte":      dte_val,
            "log_m":    log_m_abs,
            "iv":       iv_sigma,
            "uncapped": spread,
        })

    return float(np.clip(spread, 0.0, SPREAD_PCT_MAX))


def reset_spread_audit() -> None:
    SPREAD_SATURATION_LOG.clear()
    SPREAD_TOTAL_CALLS[0] = 0


def report_spread_audit() -> None:
    total = SPREAD_TOTAL_CALLS[0]
    if total == 0:
        return
    n_sat = len(SPREAD_SATURATION_LOG)
    pct = 100.0 * n_sat / total
    print(f"\n  📐 Auditoría spreads: {n_sat:,}/{total:,} llamadas saturadas "
          f"({pct:.2f}%)")
    if n_sat == 0:
        return
    sat_df = pd.DataFrame(SPREAD_SATURATION_LOG)
    print(f"     Spread teórico medio (sin cap): {sat_df['uncapped'].mean():.1%}  "
          f"max: {sat_df['uncapped'].max():.1%}")
    print(f"     DTE medio en saturaciones: {sat_df['dte'].mean():.1f}d  "
          f"|log_m| medio: {sat_df['log_m'].mean():.3f}  "
          f"IV media: {sat_df['iv'].mean():.1%}")
    if pct > 15.0:
        print(f"     ⚠️  Saturación > 15% — considera subir SPREAD_PCT_MAX a 0.40-0.45")


def apply_execution_spread(mid: float, side: str, option_type: str,
                            ctx: MarketContext, K: float, T: float) -> float:
    if mid <= 0:
        return 0.0
    if not USE_EXECUTION_SPREAD:
        return _round_to_tick(mid, side)

    spread_pct = estimate_bid_ask_spread_pct(ctx.S, K, mid, T, option_type,
                                              iv_sigma=ctx.iv_base,
                                              ticker=getattr(ctx, "ticker", ""))
    half = max(mid * spread_pct * 0.5, _option_tick(mid) * 0.5)
    if side.upper() in ("SELL","SHORT"):
        return max(_round_to_tick(mid - half, "SELL"), 0.0)
    return _round_to_tick(mid + half, "BUY")


def estimate_reg_t_margin_per_short(S: float, K: float,
                                     option_type: str, premium: float,
                                     sigma: float = 0.20) -> float:
    S = max(float(S), 0.0)
    K = max(float(K), 0.0)
    premium = max(float(premium), 0.0)

    if option_type == "call":
        otm_amount = max(K - S, 0.0)
    else:
        otm_amount = max(S - K, 0.0)

    base    = REG_T_MARGIN_PCT * S * OPTION_MULTIPLIER
    otm_ded = otm_amount * OPTION_MULTIPLIER
    prem_cr = premium * OPTION_MULTIPLIER

    method_a = base - otm_ded + prem_cr
    min_base  = 0.10 * S * OPTION_MULTIPLIER
    method_b  = min_base + prem_cr

    margin = max(method_a, method_b, prem_cr + 1.0)

    if S > 0 and K > 0:
        log_m = abs(float(np.log(K / S)))
        if log_m > 0.15:
            discount = max(0.70, 1.0 - (log_m - 0.15) * 1.0)
            margin *= discount

    margin *= iv_margin_multiplier(sigma)
    return float(max(margin, prem_cr + 1.0))


def estimate_span_margin_condor(S: float,
                                 K_put_long: float, K_put_short: float,
                                 K_call_short: float, K_call_long: float,
                                 total_credit: float,
                                 sigma: float = 0.20) -> float:
    put_width  = max(K_put_short  - K_put_long,   0.0)
    call_width = max(K_call_long  - K_call_short,  0.0)
    margin = max(put_width, call_width) * OPTION_MULTIPLIER
    margin = max(margin, total_credit * OPTION_MULTIPLIER + 1.0, 1.0)
    margin *= iv_margin_multiplier(sigma)
    return float(margin)


def estimate_span_margin_strangle(S: float, K_call: float, K_put: float,
                                   prem_call: float, prem_put: float,
                                   sigma: float = 0.20) -> float:
    m_call = estimate_reg_t_margin_per_short(S, K_call, 'call', prem_call, sigma)
    m_put  = estimate_reg_t_margin_per_short(S, K_put,  'put',  prem_put,  sigma)
    if m_call >= m_put:
        return float(m_call + prem_put  * OPTION_MULTIPLIER)
    else:
        return float(m_put  + prem_call * OPTION_MULTIPLIER)


def iv_margin_multiplier(sigma: float) -> float:
    sigma = float(sigma)
    if sigma < 0.30:
        return 1.00
    elif sigma < 0.50:
        return 1.10
    elif sigma < 0.80:
        return 1.25
    else:
        return 1.50


class IVFactorModel:
    __slots__ = (
        "var","vbar","alpha","beta","gamma","kappa","lam",
        "vrp","vrp_long_run_asset","_ret_window",
        "skew_factor","_slope_window",
        "smile_factor","_kurt_window",
        "n_obs",
        "_warmup_rets",
    )

    def __init__(self):
        alpha, beta, gamma = GJR_ALPHA, GJR_BETA, GJR_GAMMA
        pers = alpha + beta + 0.5 * gamma
        if pers >= 0.999:
            s = 0.999 / pers
            alpha *= s; beta *= s; gamma *= s
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.kappa  = max(1.0 - (alpha + beta + 0.5 * gamma), 1e-6)
        self.lam    = float(GJR_EWMA_LAMBDA)
        self.var    = (0.20 ** 2) / TRADING_DAYS_PER_YEAR
        self.vbar   = self.var
        self._warmup_rets: list[float] = []
        self.vrp          = float(IV_PROXY_VRP_LONG_RUN)
        self.vrp_long_run_asset: float = float(IV_PROXY_VRP_LONG_RUN)
        self._ret_window: list[float] = []
        self.skew_factor   = float(IV_SKEW_LONG_RUN)
        self._slope_window: list[float] = []
        self.smile_factor  = float(IV_SMILE_LONG_RUN)
        self._kurt_window:  list[float] = []
        self.n_obs = 0

    def warm_start(self, initial_rets: np.ndarray,
                   vrp_long_run: float = IV_PROXY_VRP_LONG_RUN) -> None:
        if len(initial_rets) == 0:
            return
        hv_warmup = float(np.std(initial_rets)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        hv_warmup = float(np.clip(hv_warmup, 0.05, 2.0))
        self.var  = (hv_warmup ** 2) / TRADING_DAYS_PER_YEAR
        self.vbar = self.var
        for r in initial_rets:
            if np.isfinite(r):
                self._garch_update(float(r))
                self._vrp_update(float(r))
                self._skew_update(float(r))
                self._smile_update(float(r))
        low_hv, high_hv = 0.10, 0.50
        low_vrp  = vrp_long_run + 0.06
        high_vrp = vrp_long_run - 0.08
        vrp_init = low_vrp + (high_vrp - low_vrp) * (hv_warmup - low_hv) / (high_hv - low_hv)
        self.vrp_long_run_asset = float(vrp_long_run)
        self.vrp = float(np.clip(vrp_init,
                                  vrp_long_run - 0.12,
                                  vrp_long_run + 0.10))

    def _garch_update(self, eps: float) -> None:
        eps2  = eps * eps
        ind   = 1.0 if eps < 0 else 0.0
        self.vbar = self.lam * self.vbar + (1.0 - self.lam) * eps2
        omega = self.kappa * self.vbar
        new_v = (omega + self.alpha * eps2 + self.gamma * ind * eps2
                 + self.beta * self.var)
        self.var = new_v if (np.isfinite(new_v) and new_v > 0) else self.var

    def _hv_annual(self) -> float:
        return float(np.sqrt(max(self.var, 0.0) * TRADING_DAYS_PER_YEAR))

    def _vrp_update(self, eps: float) -> None:
        hv_now  = self._hv_annual()
        shock_mag = abs(eps) * np.sqrt(TRADING_DAYS_PER_YEAR) - hv_now
        asymmetry = 1.5 if eps < 0 else 1.0
        innov = asymmetry * float(np.clip(shock_mag, -0.30, 0.30))

        self._ret_window.append(float(eps))
        win = int(IV_PROXY_VRP_DYNAMIC_WINDOW)
        if len(self._ret_window) > win * 2:
            self._ret_window = self._ret_window[-win:]

        vrp_lr_asset = float(getattr(self, 'vrp_long_run_asset', IV_PROXY_VRP_LONG_RUN))
        if len(self._ret_window) >= 20:
            hv_long = float(np.std(self._ret_window[-win:])) * np.sqrt(TRADING_DAYS_PER_YEAR)
            hv_long = float(np.clip(hv_long, 0.05, 2.0))
            low_hv, high_hv   = 0.10, 0.50
            low_vrp  = vrp_lr_asset + 0.08
            high_vrp = vrp_lr_asset - 0.10
            lr_dynamic = low_vrp + (high_vrp - low_vrp) * (hv_long - low_hv) / (high_hv - low_hv)
            lr = float(np.clip(lr_dynamic,
                               vrp_lr_asset - 0.12,
                               vrp_lr_asset + 0.10))
        else:
            hv_long = hv_now
            lr      = vrp_lr_asset

        spike_ratio = float(IV_PROXY_SPIKE_RATIO)
        in_spike    = (hv_now > spike_ratio * max(hv_long, 0.05))
        if in_spike:
            speed = float(IV_PROXY_VRP_AR1_SPEED)
        else:
            halflife = float(IV_PROXY_SPIKE_HALFLIFE)
            speed = float(np.log(2.0) / halflife) if self.vrp > lr * 1.05 else float(IV_PROXY_VRP_AR1_SPEED)

        noise = float(IV_PROXY_VRP_VOL)
        self.vrp = (self.vrp
                    + speed * (lr - self.vrp)
                    + noise * innov)
        self.vrp = float(np.clip(self.vrp, 1.02, 1.38))

    def _skew_update(self, eps: float) -> None:
        window = 60
        self._slope_window.append(float(eps))
        if len(self._slope_window) > window * 2:
            self._slope_window = self._slope_window[-window:]
        if len(self._slope_window) >= 10:
            r = np.array(self._slope_window[-window:])
            std = r.std()
            if std > 0:
                skewness = float(np.mean(((r - r.mean()) / std) ** 3))
                innov = -skewness
            else:
                innov = 0.0
        else:
            innov = 0.0
        speed = float(IV_SKEW_AR1_SPEED)
        lr    = float(IV_SKEW_LONG_RUN)
        noise = float(IV_SKEW_VOL)
        self.skew_factor = (self.skew_factor
                            + speed * (lr - self.skew_factor)
                            + noise * innov)
        self.skew_factor = float(np.clip(self.skew_factor, 0.04, 0.60))

    def _smile_update(self, eps: float) -> None:
        window = 60
        self._kurt_window.append(float(eps))
        if len(self._kurt_window) > window * 2:
            self._kurt_window = self._kurt_window[-window:]
        if len(self._kurt_window) >= 10:
            r   = np.array(self._kurt_window[-window:])
            std = r.std()
            if std > 0:
                kurt_excess = float(np.mean(((r - r.mean()) / std) ** 4)) - 3.0
                innov = float(np.clip(kurt_excess / 3.0, -1.0, 1.0))
            else:
                innov = 0.0
        else:
            innov = 0.0
        speed = float(IV_SMILE_AR1_SPEED)
        lr    = float(IV_SMILE_LONG_RUN)
        noise = float(IV_SMILE_VOL)
        self.smile_factor = (self.smile_factor
                             + speed * (lr - self.smile_factor)
                             + noise * innov)
        self.smile_factor = float(np.clip(self.smile_factor, 0.01, 0.25))

    def update(self, log_ret: float) -> None:
        eps = float(log_ret) if np.isfinite(log_ret) else 0.0
        self._garch_update(eps)
        self._vrp_update(eps)
        self._skew_update(eps)
        self._smile_update(eps)
        self.n_obs += 1

    def iv_atm(self) -> float:
        return float(np.clip(self._hv_annual() * self.vrp,
                             IV_PROXY_MIN, IV_PROXY_MAX))

    def surface_params(self) -> tuple[float, float]:
        return self.skew_factor, self.smile_factor

    def iv_proxy(self) -> float:
        return self.iv_atm()

    def vrp_multiplier(self) -> float:
        return self.vrp


GJRGARCHState = IVFactorModel


def build_iv_proxy_series(
    prices: np.ndarray,
    warmup_rets: Optional[np.ndarray] = None,
    vrp_long_run: float = IV_PROXY_VRP_LONG_RUN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, "IVFactorModel"]:
    n   = len(prices)
    raw_iv    = np.empty(n, dtype=float)
    raw_skew  = np.empty(n, dtype=float)
    raw_smile = np.empty(n, dtype=float)
    state = IVFactorModel()

    if warmup_rets is not None and len(warmup_rets) >= 5:
        hv_warmup_est = float(np.std(warmup_rets)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        vrp_long_run  = vrp_from_hv(hv_warmup_est)
    elif len(prices) >= 20:
        early_rets   = np.log(prices[1:21] / prices[:20])
        early_rets   = early_rets[np.isfinite(early_rets)]
        if len(early_rets) >= 5:
            hv_early_est = float(np.std(early_rets)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            vrp_long_run = vrp_from_hv(hv_early_est)
        else:
            vrp_long_run = IV_PROXY_VRP_FALLBACK_NEUTRAL
    else:
        vrp_long_run = IV_PROXY_VRP_FALLBACK_NEUTRAL

    state.vrp                = float(vrp_long_run)
    state.vrp_long_run_asset = float(vrp_long_run)

    if warmup_rets is not None and len(warmup_rets) > 5:
        state.warm_start(warmup_rets, vrp_long_run=vrp_long_run)

    for i in range(n):
        raw_iv[i]    = state.iv_atm()
        sk, sm       = state.surface_params()
        raw_skew[i]  = sk
        raw_smile[i] = sm
        if i > 0:
            log_ret = float(np.log(max(prices[i], 1e-8) / max(prices[i-1], 1e-8)))
            state.update(log_ret)

    smooth_iv = (pd.Series(raw_iv)
                 .ewm(span=IV_PROXY_SMOOTH_SPAN, adjust=False)
                 .mean()
                 .values)
    return (np.clip(smooth_iv, IV_PROXY_MIN, IV_PROXY_MAX),
            raw_skew, raw_smile, state)


def effective_sigma(ctx: MarketContext, K: float, T: float,
                    iv_adjust: bool = True) -> float:
    if not iv_adjust or not IV_SURFACE_ENABLE:
        return float(np.clip(ctx.sigma, IV_MIN, IV_MAX))

    S, sigma = ctx.S, ctx.sigma
    S_ref = ctx.S_ref if ctx.S_ref > 0 else S
    if K <= 0 or S <= 0 or sigma <= 0 or T <= 0:
        return float(np.clip(sigma, IV_MIN, IV_MAX))

    T_eff  = max(T, 1.0 / CALENDAR_DAYS_PER_YEAR)
    sqrt_T = np.sqrt(T_eff)


    log_m_sk  = float(np.log(K / S_ref))
    norm_m_sk = log_m_sk / max(sigma * sqrt_T, 0.01)
    blend = float(0.5 * (1.0 + np.tanh(2.0 * norm_m_sk)))
    skew_factor_l = 1.0 + blend * (SKEW_CALL_OTM_FACTOR - 1.0)
    skew = -ctx.surface_skew * skew_factor_l * norm_m_sk / (1.0 + abs(norm_m_sk))

    log_m_sm  = float(np.log(K / S))
    norm_m_sm = log_m_sm / max(sigma * sqrt_T, 0.01)
    smile     = ctx.surface_smile * norm_m_sm**2 / (1.0 + norm_m_sm**2)

    if (USE_TERM_STRUCTURE
            and ctx.ts_tenors is not None
            and ctx.ts_ivs is not None):
        t1, t2, t3 = ctx.ts_tenors
        v1, v2, v3 = ctx.ts_ivs
        T_q = float(np.clip(T_eff, t1 * 0.5, t3 * 2.0))

        if T_q <= t2:
            w = (T_q - t1) / max(t2 - t1, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            iv_ts = v1 + w * (v2 - v1)
        else:
            w = (T_q - t2) / max(t3 - t2, 1e-6)

            w = float(np.clip(w, 0.0, 1.0))
            iv_ts = v2 + w * (v3 - v2)

        iv_ts = float(np.clip(iv_ts, IV_PROXY_MIN, IV_PROXY_MAX))
        ts_adj = float(np.clip(iv_ts - sigma, -TERM_EXTRAP_CAP, TERM_EXTRAP_CAP))
    else:
        if T_eff >= 2.0 / CALENDAR_DAYS_PER_YEAR:
            ts_adj = IV_TERM_STRENGTH * 0.3 * max(
                np.exp(-T_eff / 0.03) - np.exp(-T_eff / 0.008), 0.0
            )
        else:
            ts_adj = 0.0

    sigma_eff = sigma * (1.0 + skew + smile) + ts_adj
    return float(np.clip(sigma_eff, IV_MIN, IV_MAX))


def add_pricing_noise(mid: float, ctx: MarketContext, K: float, T: float,
                       option_type: str) -> float:
    if not PRICING_NOISE_ENABLE or mid <= 0:
        return mid

    dte  = max(T * CALENDAR_DAYS_PER_YEAR, 0.0)
    log_m = abs(np.log(K / ctx.S)) if ctx.S > 0 else 0.0

    noise_sigma = NOISE_BASE_SIGMA
    if log_m > 0.30:
        noise_sigma *= NOISE_OTM_MULT
    elif log_m > 0.15:
        noise_sigma *= 1.5
    if dte < 7:
        noise_sigma *= NOISE_NEAREXP_MULT
    elif dte < 14:
        noise_sigma *= 1.4
    if mid < 0.15:
        noise_sigma *= NOISE_ILLIQUID_MULT
    elif mid < 0.50:
        noise_sigma *= 1.8

    seed = (ctx.day_idx * 10007
            + int(round(K * 100)) * 31
            + NOISE_SEED_SALT)
    rng  = np.random.default_rng(seed)
    noise_factor = float(np.exp(rng.normal(-noise_sigma**2 / 2.0, noise_sigma)))
    noisy_mid    = mid * noise_factor
    intrinsic = max(ctx.S - K, 0.0) if option_type == "call" else max(K - ctx.S, 0.0)
    return float(max(noisy_mid, intrinsic, _option_tick(mid) / 2.0))


def _bsm(S, K, r, q, sigma, T, option_type):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    disc_r, disc_q = np.exp(-r*T), np.exp(-q*T)
    N, n = norm.cdf, norm.pdf
    if option_type == "call":
        price = S*disc_q*N(d1) - K*disc_r*N(d2)
        delta = disc_q*N(d1)
        lo, hi = max(S*disc_q - K*disc_r, 0.0), S*disc_q
    else:
        price = K*disc_r*N(-d2) - S*disc_q*N(-d1)
        delta = disc_q*(N(d1) - 1.0)
        lo, hi = max(K*disc_r - S*disc_q, 0.0), K*disc_r
    price = float(np.clip(price, lo, hi))
    return max(price, 0.0), float(delta)


@functools.lru_cache(maxsize=8192)
def _binomial_cached(S_r, K_r, r_r, q_r, sigma_r, T_r, option_type, steps):
    S,K,r,q,sigma,T = float(S_r),float(K_r),float(r_r),float(q_r),float(sigma_r),float(T_r)
    dt = T/steps
    if dt <= 0 or sigma <= 0:
        i = max(S-K,0.) if option_type=="call" else max(K-S,0.)
        return float(i), 0.0
    u = np.exp(sigma*np.sqrt(dt)); d = 1.0/u
    disc = np.exp(-r*dt)
    a    = np.exp((r-q)*dt)
    p    = float(np.clip((a-d)/(u-d), 0.0, 1.0))
    j  = np.arange(steps+1)
    ST = S*(u**j)*(d**(steps-j))
    V  = np.maximum(ST-K,0.) if option_type=="call" else np.maximum(K-ST,0.)
    Vu=Vd=Su=Sd=None
    for i in range(steps-1,-1,-1):
        V  = disc*(p*V[1:]+(1.-p)*V[:-1])
        ST = ST[:-1]/d
        intr = np.maximum(ST-K,0.) if option_type=="call" else np.maximum(K-ST,0.)
        V = np.maximum(V, intr)
        if i==1: Vu,Vd,Su,Sd = float(V[1]),float(V[0]),float(ST[1]),float(ST[0])
    price = float(V[0])
    delta = float((Vu-Vd)/(Su-Sd)) if (Vu is not None and (Su-Sd)!=0) else 0.0
    return max(price,0.0), delta


def _round_params(S,K,r,q,sigma,T):
    return round(S,2), round(K,2), round(r,4), round(q,4), round(sigma,4), round(T,4)


def price_option(ctx: MarketContext, K: float, T: float,
                 option_type: str = "call",
                 iv_adjust: bool = True,
                 add_noise: bool = True,
                 force_european: bool = False) -> tuple[float, float]:
    S, r, q = ctx.S, ctx.r, ctx.q
    T = max(float(T), 1.0/CALENDAR_DAYS_PER_YEAR)
    K = max(float(K), 0.01)

    sigma_eff = effective_sigma(ctx, K, T, iv_adjust)

    if T <= 1.0/CALENDAR_DAYS_PER_YEAR*0.5:
        intr  = max(S-K,0.) if option_type=="call" else max(K-S,0.)
        delta = (1. if S>K else 0.) if option_type=="call" else (-1. if S<K else 0.)
        return float(intr), float(delta)

    use_american = USE_AMERICAN_PRICING and not force_european
    if AMERICAN_ONLY_IF_EXERCISE_RELEVANT and use_american:
        if option_type=="call" and q <= 0.001:
            use_american = False
        if option_type=="put" and K <= S:
            use_american = False

    if use_american:
        steps = int(np.clip(T*TRADING_DAYS_PER_YEAR*BINOMIAL_STEPS_PER_DAY,
                            BINOMIAL_STEPS_MIN, BINOMIAL_STEPS_MAX))
        price, delta = _binomial_cached(*_round_params(S,K,r,q,sigma_eff,T),
                                         option_type, steps)
    else:
        price, delta = _bsm(S,K,r,q,sigma_eff,T,option_type)

    price = max(float(price), _option_tick(price)/2.0 if price>0 else 0.0)

    if add_noise:
        price = add_pricing_noise(price, ctx, K, T, option_type)

    return float(price), float(delta)


def get_strike_tick(ticker: str, S: float) -> float:
    t = ticker.upper() if ticker else ""
    if t in STRIKE_TICK_INDEX:
        return 5.0
    if t in STRIKE_TICK_LIQUID_ETF:
        return 1.0
    S = float(S)
    if S < 50:    return 0.50
    elif S < 200: return 1.00
    elif S < 500: return 2.50
    else:         return 5.00


def round_to_strike_tick(K: float, tick: float) -> float:
    if tick <= 0:
        return K
    return float(round(round(K / tick) * tick, 4))


def find_strike_by_delta(ctx: MarketContext, target_delta: float,
                          T: float, option_type: str = "call",
                          tolerance: float = STRIKE_DELTA_TOLERANCE,
                          strike_tick: float = 0.0) -> Optional[float]:
    S = ctx.S
    if option_type == "call":
        target_delta = float(np.clip(target_delta, 0.01, 0.99))
        lo, hi = S*0.20, S*4.0
        def f(K): return price_option(ctx, K, T, "call", add_noise=False)[1] - target_delta
    else:
        target_delta = float(np.clip(target_delta, -0.99, -0.01))
        lo, hi = S*0.20, S*4.0
        def f(K): return price_option(ctx, K, T, "put", add_noise=False)[1] - target_delta

    K_result = None
    if f(lo) * f(hi) <= 0:
        try:
            K_result = float(brentq(f, lo, hi, xtol=0.01, rtol=1e-4, maxiter=60))
        except Exception:
            K_result = None

    if K_result is None:
        if option_type == "call":
            K_result = S * (1.0 + (0.5 - target_delta) * 0.4)
        else:
            K_result = S * (1.0 - (0.5 + target_delta) * 0.4)

    if strike_tick > 0:
        K_result = round_to_strike_tick(K_result, strike_tick)

    _, actual_delta = price_option(ctx, K_result, T, option_type, add_noise=False)
    if target_delta != 0:
        delta_error = abs(actual_delta - target_delta) / abs(target_delta)
        if delta_error > tolerance:
            return None

    return K_result


def all_greeks(ctx: MarketContext, K: float, T: float,
               option_type: str = "call", iv_adjust: bool = True) -> dict:
    S, r, q = ctx.S, ctx.r, ctx.q
    sigma = effective_sigma(ctx, K, T, iv_adjust)
    zero = {k: 0.0 for k in ("delta","gamma","theta","vega","rho","vanna","charm","vomma")}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return zero

    T       = max(T, 1.0 / CALENDAR_DAYS_PER_YEAR)
    sqrt_T  = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    N, n_pdf = norm.cdf, norm.pdf

    gamma = disc_q * n_pdf(d1) / (S * sigma * sqrt_T)
    vega = S * disc_q * n_pdf(d1) * sqrt_T / 100.0

    if option_type == "call":
        delta = disc_q * N(d1)
        theta_annual = (
            - S * disc_q * n_pdf(d1) * sigma / (2.0 * sqrt_T)
            - r * K * disc_r * N(d2)
            + q * S * disc_q * N(d1)
        )
        theta = theta_annual / TRADING_DAYS_PER_YEAR
        rho   = K * T * disc_r * N(d2)  / 100.0
    else:
        delta = disc_q * (N(d1) - 1.0)
        theta_annual = (
            - S * disc_q * n_pdf(d1) * sigma / (2.0 * sqrt_T)
            + r * K * disc_r * N(-d2)
            - q * S * disc_q * N(-d1)
        )
        theta = theta_annual / TRADING_DAYS_PER_YEAR
        rho   = -K * T * disc_r * N(-d2) / 100.0

    vanna = -disc_q * n_pdf(d1) * d2 / sigma / 100.0
    denom_charm = 2.0 * T * sigma * sqrt_T
    if denom_charm > 0:
        charm_annual = -disc_q * n_pdf(d1) * (
            2.0 * (r - q) * T - d2 * sigma * sqrt_T
        ) / denom_charm
        charm = charm_annual / TRADING_DAYS_PER_YEAR
    else:
        charm = 0.0

    vomma = vega * (d1 * d2) / sigma / 100.0

    return {
        "delta": delta, "gamma": gamma, "theta": theta, "vega": vega,
        "rho":   rho,   "vanna": vanna, "charm": charm, "vomma": vomma,
    }


def implied_vol(price, ctx, K, T, option_type="call"):
    S,r,q = ctx.S,ctx.r,ctx.q
    intr = max(S-K,0.) if option_type=="call" else max(K-S,0.)
    if price<=intr*0.5 or T<=0: return 0.20
    def obj(sigma): p,_=_bsm(S,K,r,q,sigma,T,option_type); return p-price
    try:    return float(brentq(obj,1e-4,5.,xtol=1e-5,maxiter=100))
    except: return 0.20


class DataManager:
    TICKERS: dict = {
        "TECH":      {"Apple":"AAPL","Microsoft":"MSFT","Amazon":"AMZN",
                      "Alphabet":"GOOGL","Meta":"META","NVIDIA":"NVDA","Tesla":"TSLA"},
        "SEMIS":     {"AMD":"AMD","Intel":"INTC","Broadcom":"AVGO",
                      "Qualcomm":"QCOM","TSMC":"TSM","Micron":"MU"},
        "SOFTWARE":  {"Adobe":"ADBE","Salesforce":"CRM","Oracle":"ORCL",
                      "ServiceNow":"NOW","Snowflake":"SNOW"},
        "INTERNET":  {"Netflix":"NFLX","PayPal":"PYPL","Shopify":"SHOP",
                      "Booking":"BKNG","Uber":"UBER"},
        "FINANCIALS":{"JPMorgan":"JPM","Goldman":"GS","Visa":"V","Mastercard":"MA"},
        "HEALTH":    {"J&J":"JNJ","Pfizer":"PFE","Merck":"MRK","Eli Lilly":"LLY"},
        "CONSUMER":  {"P&G":"PG","Coca-Cola":"KO","McDonald's":"MCD","Walmart":"WMT"},
        "ENERGY":    {"Exxon":"XOM","Chevron":"CVX","NextEra":"NEE"},
        "ETFs":      {"S&P500":"SPY","NASDAQ100":"QQQ","Russell2000":"IWM","Dow":"DIA"},
    }

    @classmethod
    def download(cls, ticker, start, end):
        print(f"\n📊 Descargando {ticker} ({start} → {end})…")
        start_ts = pd.Timestamp(start)
        warmup_start = (start_ts - pd.Timedelta(days=WARMUP_DAYS + 30)).strftime("%Y-%m-%d")
        try:
            raw = yf.download(ticker, start=warmup_start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError(f"Sin datos para {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            warmup_data = raw[raw.index < start_ts]
            backtest_data = raw[raw.index >= start_ts]
            if len(warmup_data) < 10:
                print(f"  ⚠️  Solo {len(warmup_data)} días de warmup disponibles")
            else:
                print(f"  ✓ Warmup: {len(warmup_data)} días previos al período")
            return cls._build(ticker, backtest_data, warmup_data)
        except Exception as e:
            print(f"  ⚠️  {e} → datos sintéticos")
            return cls._synthetic(500, 150., 0.25)

    @classmethod
    def _build(cls, ticker, raw, warmup_raw=None):
        def col(cs, df=raw):
            for c in cs:
                if c in df.columns:
                    return df[c].values.flatten().astype(float)
            return df.iloc[:,0].values.flatten().astype(float)

        close  = col(["Close"])
        open_  = col(["Open"])
        high   = col(["High"])
        low    = col(["Low"])
        volume = col(["Volume"]) if "Volume" in raw.columns else np.zeros(len(close))
        dates  = pd.to_datetime(raw.index)
        n      = len(close)

        warmup_rets = None
        if warmup_raw is not None and len(warmup_raw) > 5:
            warmup_close = col(["Close"], df=warmup_raw)
            warmup_rets = np.log(warmup_close[1:] / warmup_close[:-1])
            warmup_rets = warmup_rets[np.isfinite(warmup_rets)]
            if len(warmup_rets) > WARMUP_DAYS:
                warmup_rets = warmup_rets[-WARMUP_DAYS:]

        data = pd.DataFrame({"Date":dates,"Open":open_,"High":high,
                              "Low":low,"S":close,"Volume":volume})

        log_ret = np.log(pd.Series(close).clip(lower=1e-8)).diff().fillna(0.).values
        data["log_returns"] = log_ret
        data["returns"]     = pd.Series(close).pct_change().fillna(0.).values

        for w in (5,10,20,30,60):
            data[f"hv_{w}d"] = (pd.Series(log_ret).rolling(w,min_periods=1).std()
                                 * np.sqrt(TRADING_DAYS_PER_YEAR)).values
        data["volatility_realized"] = data["hv_30d"]

        print("  📈 Calibrando modelo IV 3 factores (GJR-GARCH + AR(1) VRP/Skew/Smile)…")
        if warmup_rets is not None and len(warmup_rets) >= 5:
            hv_est  = float(np.std(warmup_rets)) * np.sqrt(TRADING_DAYS_PER_YEAR)
            vrp_est = vrp_from_hv(hv_est)
            print(f"  ✓ HV warmup: {hv_est:.1%}  →  VRP long-run: {vrp_est:.4f}")
        elif n >= 20:
            early_rets = log_ret[1:21]
            early_rets = early_rets[np.isfinite(early_rets)]
            if len(early_rets) >= 5:
                hv_est = float(np.std(early_rets)) * np.sqrt(TRADING_DAYS_PER_YEAR)
                vrp_est = vrp_from_hv(hv_est)
                print(f"  ✓ HV early-sample: {hv_est:.1%}  →  VRP long-run: {vrp_est:.4f} (warmup escaso)")
            else:
                vrp_est = IV_PROXY_VRP_FALLBACK_NEUTRAL
                print(f"  ✓ VRP long-run: {vrp_est:.4f} (fallback neutro)")
        else:
            vrp_est = IV_PROXY_VRP_FALLBACK_NEUTRAL
            print(f"  ✓ VRP long-run: {vrp_est:.4f} (fallback neutro — sample muy corto)")
        iv_arr, skews, smiles, _ = build_iv_proxy_series(close, warmup_rets)
        data["surface_skew"]  = skews
        data["surface_smile"] = smiles

        vix_ticker = VIX_MAP.get(ticker.upper()) if USE_REAL_VIX else None
        vix_arr    = None
        if vix_ticker:
            vix_arr = cls._download_vix(vix_ticker, dates, n, iv_arr)
            if vix_arr is not None:
                data["volatility_implied"] = vix_arr
                data["iv_source"] = "VIX_real"
                corr = float(np.corrcoef(vix_arr, iv_arr)[0, 1])
                print(f"  ✓ IV fuente:   VIX real ({vix_ticker})")
                print(f"  ✓ VIX media:   {vix_arr.mean():.1%}  "
                      f"Modelo media: {iv_arr.mean():.1%}  "
                      f"Correlación: {corr:.3f}")
            else:
                print(f"  ⚠️  VIX {vix_ticker} no disponible → GJR-GARCH activo")

        if vix_arr is None:
            data["volatility_implied"] = iv_arr
            data["iv_source"] = "GJR_GARCH"
            print(f"  ✓ IV fuente:   GJR-GARCH + VRP dinámico")
            print(f"  ✓ IV media:    {iv_arr.mean():.1%}  HV media: {data['volatility_realized'].mean():.1%}")

        vvix_ticker  = VVIX_MAP.get(ticker.upper()) if USE_REAL_VIX else None
        vvix_skews   = None
        if vvix_ticker:
            vvix_skews = cls._download_vvix(vvix_ticker, dates, n)
            if vvix_skews is not None:
                data["surface_skew"] = vvix_skews
                data["skew_source"]  = "VVIX"
                print(f"  ✓ Skew fuente: VVIX real ({vvix_ticker})")
                print(f"  ✓ Skew medio:  {vvix_skews.mean():.3f}  "
                      f"rango=[{vvix_skews.min():.3f},{vvix_skews.max():.3f}]")
            else:
                print(f"  ⚠️  VVIX {vvix_ticker} no disponible → AR(1) GJR-GARCH activo para skew")

        if vvix_skews is None:
            data["skew_source"] = "GJR_GARCH"
            print(f"  ✓ Skew medio:  {skews.mean():.3f}  rango=[{skews.min():.3f},{skews.max():.3f}]")

        print(f"  ✓ Smile medio: {smiles.mean():.3f}  rango=[{smiles.min():.3f},{smiles.max():.3f}]")

        data["risk_free_rate"] = cls._download_rate(raw.index[0], raw.index[-1], n)
        data["dividend_yield"] = cls._div_yield(ticker, dates, close)

        print("  📅 Procesando calendar de earnings…")
        iv_base_avg = float(data["volatility_implied"].mean())
        earn_mult = cls._earnings_iv_multiplier(ticker, dates, n, iv_base_avg)
        data["earnings_iv_mult"] = earn_mult
        data["volatility_implied"] = (data["volatility_implied"] * earn_mult).clip(
            IV_PROXY_MIN, IV_PROXY_MAX)
        if earn_mult.max() > 1.005:
            print(f"  ✓ IV ajustada por earnings. Max={data['volatility_implied'].max():.1%}")

        ts_result = None
        vix9d_arr = None; vix3m_arr = None
        if USE_TERM_STRUCTURE:
            vt = VIX_MAP.get(ticker.upper(), "")
            ts_result = cls._download_term_structure(ticker, vt, dates, n)
        if ts_result is not None:
            vix9d_arr, vix3m_arr = ts_result
            data["vix9d"] = vix9d_arr
            data["vix3m"] = vix3m_arr
            data["ts_available"] = True
            print(f"  ✓ Term structure dinámica activa para {ticker}")
        else:
            data["vix9d"] = np.nan
            data["vix3m"] = np.nan
            data["ts_available"] = False
            if USE_TERM_STRUCTURE and ticker.upper() not in EARNINGS_ETF_SET:
                print(f"  ℹ️  Term structure plana (VIX9D/VIX3M no disponibles para {ticker})")

        S = pd.Series(close)
        for w in (20,50,200):
            data[f"MA_{w}"] = S.rolling(w,min_periods=1).mean().values
        bbm = S.rolling(20,min_periods=1).mean()
        bbs = S.rolling(20,min_periods=1).std().fillna(0)
        data["BB_mid"]   = bbm.values
        data["BB_upper"] = (bbm+2*bbs).values
        data["BB_lower"] = (bbm-2*bbs).values
        d = S.diff()
        g = d.where(d>0,0.).rolling(14,min_periods=1).mean()
        l = (-d.where(d<0,0.)).rolling(14,min_periods=1).mean()
        data["RSI"] = (100-100/(1+g/l.replace(0,np.nan))).fillna(50).values
        e12 = S.ewm(span=12,adjust=False).mean()
        e26 = S.ewm(span=26,adjust=False).mean()
        data["MACD"]        = (e12-e26).values
        data["MACD_signal"] = ((e12-e26).ewm(span=9,adjust=False).mean()).values
        hl  = pd.Series(high)-pd.Series(low)
        hc  = (pd.Series(high)-S.shift()).abs()
        lc  = (pd.Series(low) -S.shift()).abs()
        tr  = pd.concat([hl,hc,lc],axis=1).max(axis=1)
        data["ATR"] = tr.rolling(14,min_periods=1).mean().values

        data[data.select_dtypes(include=[np.number]).columns] = \
            data.select_dtypes(include=[np.number]).ffill()
        data = data.reset_index(drop=True)
        data["day"]    = range(n)
        data["ticker"] = ticker

        cls._save_monthly(data, ticker)
        cls._print_summary(data, ticker)
        return data

    @classmethod
    def _download_term_structure(cls, ticker: str, vix_ticker: str,
                                  dates: pd.DatetimeIndex,
                                  n: int) -> Optional[tuple]:
        if not USE_TERM_STRUCTURE:
            return None
        vix9d_tk = VIX9D_MAP.get(ticker.upper())
        vix3m_tk = VIX3M_MAP.get(ticker.upper())
        if not vix9d_tk or not vix3m_tk:
            return None
        try:
            start_dl = (dates[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            end_dl   = (dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

            def _fetch(tk):
                raw = yf.download(tk, start=start_dl, end=end_dl,
                                   progress=False, auto_adjust=False)
                if raw.empty:
                    return None
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = [c[0] for c in raw.columns]
                raw.index = pd.to_datetime(raw.index)
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                ser = raw[col].astype(float) / 100.0
                aligned = ser.reindex(dates, method="ffill").values.flatten()
                if len(aligned) != n:
                    return None
                nan_mask = ~np.isfinite(aligned) | (aligned <= 0)
                if nan_mask.all():
                    return None
                aligned[nan_mask] = np.nan
                aligned = pd.Series(aligned).ffill().bfill().fillna(0.20).values
                return np.clip(aligned.astype(float), IV_PROXY_MIN, IV_PROXY_MAX)

            arr9d = _fetch(vix9d_tk)
            arr3m = _fetch(vix3m_tk)
            if arr9d is None or arr3m is None:
                return None

            n_ok = int(np.sum(np.isfinite(arr9d) & np.isfinite(arr3m)))
            print(f"  ✓ Term structure: {vix9d_tk} + {vix3m_tk}  "
                  f"({n_ok}/{n} días con ambos datos)")
            print(f"    VIX9D medio: {arr9d.mean():.1%}  "
                  f"VIX3M medio: {arr3m.mean():.1%}  "
                  f"Pendiente media: {(arr3m.mean()-arr9d.mean())*100:+.2f}pp")
            return arr9d, arr3m
        except Exception as e:
            print(f"  ⚠️  Term structure no disponible ({e})")
            return None

    @classmethod
    def _earnings_iv_multiplier(cls, ticker: str,
                                 dates: pd.DatetimeIndex, n: int,
                                 iv_base: float = 0.30) -> np.ndarray:
        result = np.ones(n, dtype=float)
        if not USE_EARNINGS_EFFECT or ticker.upper() in EARNINGS_ETF_SET:
            return result
        try:
            tkr = yf.Ticker(ticker)
            ed  = getattr(tkr, "earnings_dates", None)
            if ed is None or len(ed) == 0:
                return result
            earn_dates  = pd.DatetimeIndex(ed.index).normalize().unique()
            dates_norm  = pd.DatetimeIndex(dates).normalize()

            day_mult = _earnings_day_mult(iv_base)
            pre_mult = _earnings_pre_mult(iv_base)

            for ed_ts in earn_dates:
                delta_days = (dates_norm - ed_ts).days
                for i, dd in enumerate(delta_days):
                    if dd < -EARNINGS_PRE_DAYS or dd > EARNINGS_POST_DAYS:
                        continue
                    if dd < 0:
                        t = (dd + EARNINGS_PRE_DAYS) / EARNINGS_PRE_DAYS
                        mult = 1.0 + (pre_mult - 1.0) * t
                    elif dd == 0:
                        mult = day_mult
                    else:
                        mult = 1.0 + (day_mult - 1.0) * np.exp(
                            -dd / EARNINGS_POST_HALFLIFE)
                    result[i] = max(result[i], float(mult))
            n_affected = int(np.sum(result > 1.005))
            if n_affected > 0:
                print(f"  ✓ Earnings effect: {len(earn_dates)} fechas  "
                      f"{n_affected} días afectados  "
                      f"day_mult={day_mult:.2f}× (IV base={iv_base:.1%})  "
                      f"max={result.max():.2f}×")
            else:
                print(f"  ℹ️  Earnings: {len(earn_dates)} fechas descargadas, "
                      f"ninguna dentro del período")
        except Exception as e:
            print(f"  ⚠️  Earnings no disponibles para {ticker} ({e})")
        return result

    @classmethod
    def _download_rate(cls, start, end, n):
        try:
            raw = yf.download("^IRX", start=start, end=end,
                              progress=False, auto_adjust=False)
            if raw.empty:
                return pd.Series(np.full(n, 0.05))
            c = "Close" if "Close" in raw.columns else raw.columns[0]
            rate = raw[c].values.flatten().astype(float)
            rate = rate / 100.0
            rate = np.clip(rate, -0.01, 0.15)
            if len(rate) < n:
                rate = np.concatenate([rate, np.full(n-len(rate), rate[-1] if len(rate)>0 else 0.05)])
            return pd.Series(rate[:n]).ffill().fillna(0.05)
        except Exception:
            return pd.Series(np.full(n, 0.05))

    @classmethod
    def _download_vix(cls, vix_ticker, dates, n, fallback):
        try:
            start_dl = (dates[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            end_dl   = (dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            raw = yf.download(vix_ticker, start=start_dl, end=end_dl,
                              progress=False, auto_adjust=False)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            vix_series = raw[col].astype(float) / 100.0
            vix_aligned = vix_series.reindex(dates, method="ffill").values.flatten()
            if len(vix_aligned) != n:
                return None
            nan_mask = ~np.isfinite(vix_aligned) | (vix_aligned <= 0)
            if nan_mask.any():
                n_gaps = nan_mask.sum()
                vix_aligned[nan_mask] = fallback[nan_mask]
                if n_gaps > 5:
                    print(f"  ⚠️  {n_gaps} días sin VIX → relleno con GJR-GARCH")
            return np.clip(vix_aligned.astype(float), IV_PROXY_MIN, IV_PROXY_MAX)
        except Exception:
            return None

    @classmethod
    def _download_vvix(cls, vvix_ticker, dates, n):
        try:
            start_dl = (dates[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            end_dl   = (dates[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            raw = yf.download(vvix_ticker, start=start_dl, end=end_dl,
                              progress=False, auto_adjust=False)
            if raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            col = "Close" if "Close" in raw.columns else raw.columns[0]
            vvix_series = raw[col].astype(float)
            vvix_aligned = vvix_series.reindex(dates, method="ffill").values.flatten()
            if len(vvix_aligned) != n:
                return None
            nan_mask = ~np.isfinite(vvix_aligned) | (vvix_aligned <= 0)
            if nan_mask.any():
                vvix_aligned[nan_mask] = VVIX_NORM
            ratio       = vvix_aligned / VVIX_NORM
            skew_factor = SKEW_VVIX_BASE * (ratio ** SKEW_VVIX_EXPONENT)
            skew_factor = np.clip(skew_factor, SKEW_VVIX_MIN, SKEW_VVIX_MAX)
            return skew_factor.astype(float)
        except Exception:
            return None

    @classmethod
    def _div_yield(cls, ticker, dates, close):
        n = len(dates); dyd = np.zeros(n)
        try:
            tkr  = yf.Ticker(ticker)
            divs = getattr(tkr,"dividends",None)
            if divs is None or len(divs)==0: return dyd
            div_df = divs.reset_index()
            div_df.columns = ["Date","Div"]
            div_df["Date"] = pd.to_datetime(div_df["Date"]).dt.normalize()
            date_s = pd.Series(dates).dt.normalize()
            div_map = pd.Series(0.0, index=date_s)
            for _,row in div_df.iterrows():
                mask = date_s == row["Date"]
                div_map[mask] = float(row["Div"])
            div_arr = div_map.values
            for i in range(n):
                si = max(0,i-252)
                ttm = div_arr[si:i].sum()
                if close[i]>0: dyd[i] = float(np.clip(ttm/close[i],0.,0.20))
        except Exception: pass
        return dyd

    @classmethod
    def _save_monthly(cls, data, ticker):
        try:
            df = data.set_index("Date")
            agg = {"S":["first","last","max","min","mean"],"Volume":"sum",
                   "volatility_implied":"mean","volatility_realized":"mean",
                   "risk_free_rate":"mean","dividend_yield":"mean",
                   "surface_skew":"mean","surface_smile":"mean"}
            if "RSI" in df.columns: agg["RSI"] = "mean"
            m = df.resample("ME").agg(agg)
            m.columns = ["_".join(c) if isinstance(c,tuple) else c for c in m.columns]
            m["monthly_return"] = m["S_last"]/m["S_first"] - 1
            m.reset_index().to_csv(f"monthly_{ticker}.csv",index=False,float_format="%.6f")
            print(f"  💾 monthly_{ticker}.csv guardado")
        except Exception as e:
            print(f"  ⚠️  monthly: {e}")

    @classmethod
    def _print_summary(cls, data, ticker):
        s0,s1 = data["S"].iloc[0],data["S"].iloc[-1]
        print(f"\n{'─'*60}")
        print(f"  {ticker} | {data['Date'].iloc[0].date()} → {data['Date'].iloc[-1].date()}")
        print(f"  Días: {len(data)} | ${s0:.2f}→${s1:.2f} ({(s1/s0-1)*100:+.1f}%)")
        iv_src = data.get("iv_source", pd.Series(["GJR_GARCH"]*len(data)))
        src_str = iv_src.iloc[0] if hasattr(iv_src, 'iloc') else str(iv_src)
        print(f"  IV media: {data['volatility_implied'].mean():.1%}"
              f"  HV media: {data['volatility_realized'].mean():.1%}"
              f"  VRP: {data['volatility_implied'].mean()/max(data['volatility_realized'].mean(),0.001):.2f}"
              f"  [fuente: {src_str}]")
        print(f"  Skew medio: {data['surface_skew'].mean():.3f}"
              f"  Smile medio: {data['surface_smile'].mean():.3f}")
        print(f"  Tasa rf: {data['risk_free_rate'].mean():.2%}")
        print(f"{'─'*60}")

    @classmethod
    def _synthetic(cls, days=252, S0=100., vol=0.20, r=0.05, seed=42):
        np.random.seed(seed)
        dt   = 1/TRADING_DAYS_PER_YEAR
        path = [S0]
        for _ in range(1, days):
            path.append(path[-1]*np.exp((r-.5*vol**2)*dt+vol*np.sqrt(dt)*np.random.normal()))
        prices = np.array(path)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
        dates = dates[:len(prices)]
        if len(dates) < len(prices):
            dates = pd.bdate_range(start="2018-01-01", periods=days)[:len(prices)]
        iv_arr, skews, smiles, _ = build_iv_proxy_series(prices)
        log_ret = np.concatenate([[0], np.log(prices[1:]/prices[:-1])])
        df = pd.DataFrame({
            "Date":dates,"Open":prices,"High":prices*1.01,"Low":prices*0.99,
            "S":prices,"Volume":1_000_000,"log_returns":log_ret,
            "returns":np.concatenate([[0],prices[1:]/prices[:-1]-1]),
            "volatility_realized":np.full(len(prices),vol),
            "volatility_implied":iv_arr,"surface_skew":skews,"surface_smile":smiles,
            "risk_free_rate":np.full(len(prices),r),"dividend_yield":np.zeros(len(prices)),
            "RSI":np.full(len(prices),50.),"MACD":np.zeros(len(prices)),
            "MACD_signal":np.zeros(len(prices)),"ATR":prices*0.01,
        })
        for w in (5,10,20,30,60): df[f"hv_{w}d"] = vol
        for w in (20,50,200):     df[f"MA_{w}"]  = prices
        df["BB_mid"]=prices; df["BB_upper"]=prices*1.02; df["BB_lower"]=prices*0.98
        df["day"]=range(len(prices)); df["iv_source"]="GJR_GARCH"
        df["skew_source"]="GJR_GARCH"; df["ticker"]="SYNTHETIC"
        df["earnings_iv_mult"] = 1.0
        df["vix9d"]            = np.nan
        df["vix3m"]            = np.nan
        df["ts_available"]     = False
        print(f"  ✓ Datos sintéticos: {len(df)} días S0={S0} vol={vol:.0%}")
        return df


@dataclass
class OptionLeg:
    option_type:      str
    K:                float
    T_exp:            pd.Timestamp
    contracts:        int
    sign:             int
    entry_mid:        float
    entry_exec:       float
    entry_commission: float
    open_date:        pd.Timestamp
    is_open:          bool = True

    def intrinsic(self, S: float) -> float:
        return max(S - self.K, 0.0) if self.option_type == "call" else max(self.K - S, 0.0)

    def expiration_cashflow(self, S: float) -> float:
        return self.sign * self.intrinsic(S) * self.contracts * OPTION_MULTIPLIER

    def mtm_value(self, ctx: "MarketContext", iv_adjust: bool = True) -> float:
        T_remaining = year_fraction(ctx.date, self.T_exp)
        if T_remaining <= 0:
            return self.sign * self.intrinsic(ctx.S) * self.contracts * OPTION_MULTIPLIER
        mid, _ = price_option(ctx, self.K, T_remaining, self.option_type,
                               iv_adjust, add_noise=False)
        liq_side = "SELL" if self.sign > 0 else "BUY"
        liq_px   = apply_execution_spread(mid, liq_side, self.option_type,
                                           ctx, self.K, T_remaining)
        intrinsic = self.intrinsic(ctx.S)
        tick_floor = _option_tick(mid) / 2.0 if mid > 0 else 0.0
        liq_px = float(max(liq_px, intrinsic, tick_floor))
        return self.sign * liq_px * self.contracts * OPTION_MULTIPLIER

    def margin_required(self, ctx: "MarketContext") -> float:
        if self.sign > 0:
            return 0.0
        T_remaining = year_fraction(ctx.date, self.T_exp)
        if T_remaining <= 0:
            return 0.0
        mid, _ = price_option(ctx, self.K, T_remaining, self.option_type,
                               add_noise=False)
        return (estimate_reg_t_margin_per_short(
                    ctx.S, self.K, self.option_type, mid, sigma=ctx.sigma)
                * self.contracts)

    def close_cashflow(self, ctx: "MarketContext",
                        iv_adjust: bool = True) -> tuple:
        T_remaining = year_fraction(ctx.date, self.T_exp)
        T_eff = max(T_remaining, 0.001)
        mid, _  = price_option(ctx, self.K, T_eff,
                                self.option_type, iv_adjust, add_noise=False)
        side    = "SELL" if self.sign > 0 else "BUY"
        exec_px = apply_execution_spread(mid, side, self.option_type,
                                          ctx, self.K, T_eff)
        intrinsic   = self.intrinsic(ctx.S)
        tick_floor  = _option_tick(mid) / 2.0
        exec_px     = float(max(exec_px, intrinsic, tick_floor if mid > 0 else 0.0))
        cashflow = self.sign * exec_px * self.contracts * OPTION_MULTIPLIER
        return cashflow, mid, exec_px


@dataclass
class OptionPosition:
    legs:              list = field(default_factory=list)
    open_date:         object = None
    is_open:           bool = True
    fixed_margin:      float = 0.0
    cycle_id:          str = field(default_factory=lambda: uuid4().hex[:12])

    def mtm_value(self, ctx: MarketContext) -> float:
        return sum(leg.mtm_value(ctx) for leg in self.legs if leg.is_open)

    def margin_required(self, ctx: MarketContext) -> float:
        if self.fixed_margin > 0:
            return self.fixed_margin

        short_legs = [l for l in self.legs if l.sign < 0 and l.is_open]
        long_legs  = [l for l in self.legs if l.sign > 0 and l.is_open]
        if not short_legs:
            return 0.0

        def _is_ratio(otype: str) -> bool:
            n_short = sum(l.contracts for l in short_legs if l.option_type == otype)
            n_long  = sum(l.contracts for l in long_legs  if l.option_type == otype)
            return n_short > n_long

        is_ratio_structure = _is_ratio("call") or _is_ratio("put")

        net_cost = sum(
            l.sign * (-l.entry_exec) * l.contracts * OPTION_MULTIPLIER
            for l in self.legs if l.is_open
        )
        if net_cost < 0 and not is_ratio_structure:
            return 0.0

        long_capacity = {id(l): l.contracts for l in long_legs}

        call_spread_margins: list[float] = []
        put_spread_margins:  list[float] = []
        naked_margins:       list[float] = []

        for sl in sorted(short_legs, key=lambda l: l.K):
            T_r = max(year_fraction(ctx.date, sl.T_exp), 0.001)
            remaining_short = sl.contracts

            candidates = [
                l for l in long_legs
                if long_capacity.get(id(l), 0) > 0
                and l.option_type == sl.option_type
                and abs((l.T_exp - sl.T_exp).days) <= 1
            ]

            if sl.option_type == "call":
                candidates = [l for l in candidates if l.K >= sl.K]
                candidates.sort(key=lambda l: l.K)
            else:
                candidates = [l for l in candidates if l.K <= sl.K]
                candidates.sort(key=lambda l: l.K, reverse=True)

            for cover in candidates:
                if remaining_short <= 0:
                    break
                avail = long_capacity[id(cover)]
                if avail <= 0:
                    continue
                used = min(remaining_short, avail)
                width = abs(cover.K - sl.K)
                sm = max(width * used * OPTION_MULTIPLIER, 1.0)
                if sl.option_type == "call":
                    call_spread_margins.append(sm)
                else:
                    put_spread_margins.append(sm)
                long_capacity[id(cover)] = avail - used
                remaining_short -= used

            if remaining_short > 0:
                mid_nx, _ = price_option(ctx, sl.K, T_r, sl.option_type,
                                          add_noise=False)
                naked_margins.append(
                    estimate_reg_t_margin_per_short(
                        ctx.S, sl.K, sl.option_type, mid_nx,
                        sigma=ctx.sigma) * remaining_short
                )

        if call_spread_margins and put_spread_margins:
            margin = max(sum(call_spread_margins), sum(put_spread_margins))
        else:
            margin = sum(call_spread_margins) + sum(put_spread_margins)

        margin += sum(naked_margins)

        return float(max(margin, 0.0))

    def expiration_cashflow(self, S: float) -> float:
        return sum(leg.expiration_cashflow(S) for leg in self.legs if leg.is_open)

    def entry_net_cost(self) -> float:
        total = 0.0
        for leg in self.legs:
            total += leg.sign * (-1) * leg.entry_exec * leg.contracts * OPTION_MULTIPLIER
        return total

    def entry_commission(self) -> float:
        return sum(leg.entry_commission for leg in self.legs)

    def days_to_exp(self, current_date: pd.Timestamp) -> int:
        if not self.legs:
            return 0
        return max(int((self.legs[0].T_exp - current_date).days), 0)

    def capital_at_risk(self) -> float:
        longs  = [l for l in self.legs if l.is_open and l.sign > 0]
        shorts = [l for l in self.legs if l.is_open and l.sign < 0]


        if longs and not shorts:
            return sum(l.entry_exec * l.contracts * OPTION_MULTIPLIER
                       for l in longs)


        if shorts and not longs:

            return sum(l.entry_exec * l.contracts * OPTION_MULTIPLIER
                       for l in shorts)


        net_credit = 0.0
        for l in self.legs:
            if not l.is_open:
                continue

            net_credit += l.sign * (-l.entry_exec) * l.contracts * OPTION_MULTIPLIER

        if net_credit >= 0:

            return float(net_credit * -1) if net_credit < 0 else sum(
                l.entry_exec * l.contracts * OPTION_MULTIPLIER
                for l in longs
            )


        call_shorts = [l for l in shorts if l.option_type == "call"]
        call_longs  = [l for l in longs  if l.option_type == "call"]
        put_shorts  = [l for l in shorts if l.option_type == "put"]
        put_longs   = [l for l in longs  if l.option_type == "put"]

        call_wing = 0.0
        if call_shorts and call_longs:
            call_wing = max(
                abs(cs.K - cl.K)
                for cs in call_shorts for cl in call_longs
            )
        put_wing = 0.0
        if put_shorts and put_longs:
            put_wing = max(
                abs(ps.K - pl.K)
                for ps in put_shorts for pl in put_longs
            )

        max_wing = max(call_wing, put_wing)
        n_contracts = max((l.contracts for l in shorts), default=1)

        if max_wing > 0:

            max_loss = max_wing * n_contracts * OPTION_MULTIPLIER + net_credit
            return float(max(max_loss, 0.0))
        else:

            return sum(l.entry_exec * l.contracts * OPTION_MULTIPLIER
                       for l in longs)


class BaseStrategy:
    name: str = "Base"
    description: str = ""

    def __init__(self, name: str = "", description: str = ""):
        if name:
            self.name = name
        if description:
            self.description = description
        self._reset()

    def _reset(self):
        self.initial_capital   = 0.0
        self.cash              = 0.0
        self.total_commissions = 0.0
        self.bankrupted        = False
        self.positions: list = []
        self.trades:    list = []
        self.portfolio_history: list = []
        self._warnings:         list = []
        self._commission_log:   list = []

    def initialize(self, capital: float, **params):
        self._reset()
        self.initial_capital = float(capital)
        self.cash            = float(capital)
        for k, v in params.items():
            setattr(self, k, v)

    def _accrue_daily_interest(self, ctx: MarketContext) -> None:
        return

    @staticmethod
    def _commission_preview(n_contracts: int, price: float = 0.0,
                            is_option: bool = True) -> float:
        return calculate_commissions(n_contracts, price, is_option)

    def _commission(self, n_contracts: int, price: float = 0.0,
                     is_option: bool = True, note: str = "") -> float:
        comm = calculate_commissions(n_contracts, price, is_option)
        self.total_commissions += comm
        if note:
            self._commission_log.append({
                "contracts": n_contracts,
                "price":     price,
                "comm":      comm,
                "note":      note,
            })
        return comm

    def _open_leg(self, ctx: MarketContext, K: float, T_years: float,
                   option_type: str, contracts: int, sign: int,
                   iv_adjust: bool = True) -> OptionLeg:
        mid, _ = price_option(ctx, K, T_years, option_type, iv_adjust)
        side = "BUY" if sign > 0 else "SELL"
        exec_px = apply_execution_spread(mid, side, option_type, ctx, K, T_years)
        comm = self._commission(contracts, exec_px, note=f"{side} {option_type} K={K:.2f}")

        exp_date = expiration_date_from_days(ctx.date,
                                              int(T_years * CALENDAR_DAYS_PER_YEAR))

        self.cash += sign * (-exec_px) * contracts * OPTION_MULTIPLIER - comm

        return OptionLeg(
            option_type=option_type, K=K, T_exp=exp_date,
            contracts=contracts, sign=sign,
            entry_mid=mid, entry_exec=exec_px,
            entry_commission=comm,
            open_date=ctx.date,
        )

    def _register_open_trade(self, leg: OptionLeg, ctx: MarketContext,
                              cycle_id: str, action: str = "open") -> None:
        self.trades.append({
            "date":        ctx.date,
            "action":      action,
            "cycle_id":    cycle_id,
            "option_type": leg.option_type,
            "K":           leg.K,
            "sign":        leg.sign,
            "contracts":   leg.contracts,
            "entry_exec":  leg.entry_exec,
            "commission":  leg.entry_commission,
            "pnl":         None,
        })

    def _close_leg(self, leg: OptionLeg, ctx: MarketContext,
                    reason: str = "close", cycle_id: str = "") -> float:
        cashflow, mid, exec_px = leg.close_cashflow(ctx)

        if reason == "roll":
            T_remaining = year_fraction(ctx.date, leg.T_exp)
            dte_close = T_remaining * CALENDAR_DAYS_PER_YEAR
            roll_mult = roll_slippage_multiplier(dte_close)
            if roll_mult > 1.0:

                extra_slip_pct = estimate_bid_ask_spread_pct(
                    ctx.S, leg.K, mid,
                    max(T_remaining, 0.001),
                    leg.option_type, iv_sigma=ctx.iv_base,
                    ticker=getattr(ctx, "ticker", "")
                ) * (roll_mult - 1.0)
                extra_slip = mid * extra_slip_pct
                if leg.sign > 0:
                    exec_px = max(exec_px - extra_slip, 0.0)
                    cashflow = leg.sign * exec_px * leg.contracts * OPTION_MULTIPLIER
                else:
                    exec_px = exec_px + extra_slip
                    cashflow = leg.sign * exec_px * leg.contracts * OPTION_MULTIPLIER

        comm = self._commission(leg.contracts, exec_px,
                                 note=f"close {leg.option_type} K={leg.K:.2f}")
        self.cash += cashflow - comm
        leg.is_open = False

        pnl = (self.sign_adjusted_pnl(leg, exec_px) - leg.entry_commission - comm)
        self.trades.append({
            "date": ctx.date, "action": reason,
            "cycle_id": cycle_id,
            "option_type": leg.option_type, "K": leg.K,
            "sign": leg.sign, "contracts": leg.contracts,
            "entry_exec": leg.entry_exec, "exit_exec": exec_px,
            "pnl": pnl, "commission": comm,
            "roll_slippage": reason == "roll",
        })
        return pnl

    def sign_adjusted_pnl(self, leg: OptionLeg, exit_px: float) -> float:
        return leg.sign * (exit_px - leg.entry_exec) * leg.contracts * OPTION_MULTIPLIER

    def _expire_leg(self, leg: OptionLeg, ctx: MarketContext,
                     cycle_id: str = "") -> float:
        payoff   = leg.intrinsic(ctx.S)
        cashflow = leg.sign * payoff * leg.contracts * OPTION_MULTIPLIER

        if payoff > 0:
            if leg.sign < 0:

                ASSIGNMENT_COST_PER_CONTRACT = 2.00
                assignment_cost = leg.contracts * ASSIGNMENT_COST_PER_CONTRACT
                comm = self._commission(leg.contracts, payoff,
                                         note=f"expire ITM short {leg.option_type} K={leg.K:.2f}")
            else:

                shares_delivered = leg.contracts * OPTION_MULTIPLIER
                assignment_cost  = 0.0
                comm = max(shares_delivered * COMMISSION_PER_SHARE, MINIMUM_COMMISSION)
                self.total_commissions += comm
                self._commission_log.append({
                    "contracts": leg.contracts,
                    "price":     payoff,
                    "comm":      comm,
                    "note":      (f"expire ITM long {leg.option_type} K={leg.K:.2f} "
                                  f"({shares_delivered} acciones)"),
                })
        else:
            comm = 0.0
            assignment_cost = 0.0

        self.cash += cashflow - comm - assignment_cost
        leg.is_open = False

        pnl = (leg.sign * (payoff - leg.entry_exec)
               * leg.contracts * OPTION_MULTIPLIER
               - leg.entry_commission - comm - assignment_cost)
        self.trades.append({
            "date": ctx.date, "action": "expire",
            "cycle_id": cycle_id,
            "option_type": leg.option_type, "K": leg.K,
            "sign": leg.sign, "contracts": leg.contracts,
            "entry_exec": leg.entry_exec, "payoff": payoff,
            "pnl": pnl, "commission": comm,
            "assignment_cost": assignment_cost,
        })
        return pnl

    def _close_all_open_end_of_period(self, ctx: MarketContext) -> None:
        for pos in [p for p in self.positions if p.is_open]:
            for leg in pos.legs:
                if leg.is_open:
                    cashflow, mid, exec_px = leg.close_cashflow(ctx)
                    comm = self._commission(
                        leg.contracts, exec_px,
                        note=f"end_of_period {leg.option_type} K={leg.K:.2f}"
                    )
                    self.cash += cashflow - comm
                    leg.is_open = False
                    pnl = (self.sign_adjusted_pnl(leg, exec_px)
                           - leg.entry_commission - comm)
                    self.trades.append({
                        "date": ctx.date, "action": "end_of_period",
                        "cycle_id": pos.cycle_id,
                        "option_type": leg.option_type, "K": leg.K,
                        "sign": leg.sign, "contracts": leg.contracts,
                        "entry_exec": leg.entry_exec, "exit_exec": exec_px,
                        "pnl": pnl, "commission": comm,
                        "roll_slippage": False,
                    })
            pos.is_open = False

    def _total_margin_required(self, ctx: MarketContext) -> float:
        return sum(p.margin_required(ctx) for p in self.positions if p.is_open)

    def _total_capital_at_risk(self, ctx: MarketContext) -> float:
        cap = 0.0
        for p in self.positions:
            if p.is_open:
                cap += p.capital_at_risk() + p.margin_required(ctx)
        return cap

    def _positions_mtm(self, ctx: MarketContext) -> float:
        return sum(p.mtm_value(ctx) for p in self.positions if p.is_open)

    def _equity(self, ctx: MarketContext) -> float:
        return self.cash + self._positions_mtm(ctx)

    def _sizing_base(self, ctx: MarketContext) -> float:
        equity = self._equity(ctx)
        floor  = EQUITY_SIZING_FLOOR_PCT * self.initial_capital
        return max(equity, floor)

    def _apply_margin_cap(self, n: int, margin_per: float,
                           ctx: MarketContext, wing_width: float = 0.0) -> int:
        if n <= 0:
            return 0
        eq = self._sizing_base(ctx)

        if margin_per > 0:
            margin_in_use = self._total_margin_required(ctx)
            avail_margin  = max(MAX_MARGIN_PCT_EQUITY * eq - margin_in_use, 0.0)
            n = min(n, int(avail_margin / margin_per))

        if wing_width > 0:
            max_by_notional = int(eq * MAX_WING_RISK_PCT_EQUITY
                                  / max(wing_width * OPTION_MULTIPLIER, 1.0))
        else:
            iv_mult = iv_margin_multiplier(ctx.sigma)
            margin_proxy = 0.12 * ctx.S * OPTION_MULTIPLIER
            max_by_notional = int(eq * MAX_NOTIONAL_PCT_EQUITY
                                  / max(margin_proxy, 1.0))

        n = min(n, max_by_notional)
        return max(n, 0)

    def _check_margin_call(self, ctx: MarketContext) -> bool:
        if self.bankrupted:
            return False

        equity = self.cash + self._positions_mtm(ctx)
        margin_req = self._total_margin_required(ctx)

        if equity <= 0:
            self._liquidate_all(ctx, reason="bankruptcy")
            self.bankrupted = True
            return True

        if margin_req > 0 and equity < margin_req * MARGIN_CALL_BUFFER:
            self._liquidate_all(ctx, reason="margin_call")
            return True

        return False

    def _liquidate_all(self, ctx: MarketContext, reason: str = "margin_call"):
        open_positions = [p for p in self.positions if p.is_open]
        open_positions.sort(key=lambda p: p.mtm_value(ctx))
        for pos in open_positions:
            for leg in pos.legs:
                if leg.is_open:
                    self._close_leg(leg, ctx, reason=reason, cycle_id=pos.cycle_id)
            pos.is_open = False

    def _record_portfolio(self, ctx: MarketContext, mtm: float):
        total = self.cash + mtm
        if self.bankrupted:
            total = max(total, 0.0)


        n_open = sum(
            sum(l.contracts for l in p.legs if l.is_open)
            for p in self.positions if p.is_open
        )

        net_g = {"delta": 0., "gamma": 0., "theta": 0., "vega": 0.}
        for pos in self.positions:
            if not pos.is_open:
                continue
            for leg in pos.legs:
                if not leg.is_open:
                    continue
                T_r = year_fraction(ctx.date, leg.T_exp)
                if T_r <= 0:
                    continue
                g = all_greeks(ctx, leg.K, T_r, leg.option_type)
                scale = leg.sign * leg.contracts * OPTION_MULTIPLIER
                net_g["delta"] += g["delta"] * scale
                net_g["gamma"] += g["gamma"] * scale
                net_g["theta"] += g["theta"] * scale
                net_g["vega"]  += g["vega"]  * scale

        margin_req  = self._total_margin_required(ctx)
        cap_at_risk = self._total_capital_at_risk(ctx)

        self.portfolio_history.append({
            "date":        ctx.date,
            "cash":        self.cash,
            "mtm":         mtm,
            "total":       total,
            "margin":      margin_req,
            "margin_pct":  margin_req / max(self.initial_capital, 1.0),
            "capital_at_risk":      cap_at_risk,
            "capital_at_risk_pct":  cap_at_risk / max(self.initial_capital, 1.0),
            "commissions": self.total_commissions,
            "n_open":      n_open,
            "rf":          ctx.r,
            "S":           ctx.S,
            "iv":          ctx.sigma,
            "net_delta":   net_g["delta"],
            "net_gamma":   net_g["gamma"],
            "net_theta":   net_g["theta"],
            "net_vega":    net_g["vega"],
            "bankrupted":  self.bankrupted,
        })
        return total

    def execute(self, data: pd.DataFrame) -> dict:
        raise NotImplementedError

    def metrics(self) -> dict:
        if not self.portfolio_history:
            return self._empty_metrics()

        values = pd.Series([h["total"] for h in self.portfolio_history], dtype=float)
        if len(values) < 2:
            return self._empty_metrics()

        pct_rets = values.pct_change().dropna()

        base_value = float(self.initial_capital) if self.initial_capital > 0 else float(values.iloc[0])
        total_ret  = values.iloc[-1] / base_value - 1
        n          = len(pct_rets)

        annual_ret = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / max(n, 1)) - 1
        vol = float(pct_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        rf_daily = 0.02 / TRADING_DAYS_PER_YEAR
        if self.portfolio_history and 'rf' in self.portfolio_history[0]:
            rf_mean = np.mean([h.get('rf', 0.02) for h in self.portfolio_history])
            rf_daily = rf_mean / TRADING_DAYS_PER_YEAR

        excess = pct_rets - rf_daily
        sharpe = float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / pct_rets.std()
                        if pct_rets.std() > 0 else 0.0)

        neg = pct_rets[pct_rets < 0]
        down_vol = float(neg.std()) if len(neg) > 1 else 1e-9
        sortino  = float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / down_vol)

        cum   = values / values.iloc[0]
        peak  = cum.cummax()
        dd    = (cum - peak) / peak
        max_dd = float(dd.min())
        calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        avg_equity = values.mean()
        var95_pct = float(np.percentile(pct_rets, 5))
        es95_pct  = float(pct_rets[pct_rets <= np.percentile(pct_rets, 5)].mean())
        var95_dol = var95_pct * avg_equity
        es95_dol  = es95_pct  * avg_equity

        cycle_pnl: dict = {}
        for t in self.trades:
            pnl = t.get("pnl")
            if pnl is None:
                continue
            action = t.get("action", "")
            if action in ("roll", "expire", "close", "margin_call",
                          "bankruptcy", "end_of_period"):
                cid = t.get("cycle_id", "")
                if not cid:
                    cid = f"fallback_{t.get('date', '?')}"
                cycle_pnl.setdefault(cid, 0.0)
                cycle_pnl[cid] += float(pnl)

        cycle_profits = list(cycle_pnl.values())
        if not cycle_profits:
            cycle_profits = [t.get("pnl", 0.0) for t in self.trades
                             if "pnl" in t and t["pnl"] is not None]

        wins    = [p for p in cycle_profits if p > 0]
        losses  = [p for p in cycle_profits if p < 0]
        win_rate  = len(wins) / max(len(cycle_profits), 1)
        pf        = sum(wins) / abs(sum(losses)) if losses else float("inf")
        avg_win   = float(np.mean(wins)) if wins else 0.0
        avg_loss  = float(np.mean(losses)) if losses else 0.0

        comm_impact = self.total_commissions / self.initial_capital

        return {
            "Retorno Total":            total_ret,
            "Retorno Anual":            annual_ret,
            "Volatilidad Anual":        vol,
            "Sharpe Ratio":             sharpe,
            "Sortino Ratio":            sortino,
            "Calmar Ratio":             calmar,
            "Max Drawdown":             max_dd,
            "VaR 95% (1d, %)":          var95_pct,
            "VaR 95% (1d, $)":          var95_dol,
            "ES 95% (1d, %)":           es95_pct,
            "ES 95% (1d, $)":           es95_dol,
            "Win Rate":                 win_rate,
            "Profit Factor":            pf,
            "Avg Win ($)":              avg_win,
            "Avg Loss ($)":             avg_loss,
            "Nº Ciclos":                len(cycle_profits),
            "Nº Wins":                  len(wins),
            "Nº Losses":                len(losses),
            "Días Positivos":           int((pct_rets > 0).sum()),
            "Días Negativos":           int((pct_rets < 0).sum()),
            "Asimetría":                float(pct_rets.skew()),
            "Curtosis":                 float(pct_rets.kurtosis()),
            "Comisiones Totales ($)":   self.total_commissions,
            "Impacto Comisiones":       comm_impact,
            "Delta Neto Medio":         self._mean_greek_open("net_delta"),
            "Gamma Neto Medio":         self._mean_greek_open("net_gamma"),
            "Theta Neto $/día":         self._mean_greek_open("net_theta"),
            "Vega Neto $/1pp":          self._mean_greek_open("net_vega"),
            "Margen Medio (%)":         self._mean_margin_pct(),
            "Capital en Riesgo Medio (%)": self._mean_capital_at_risk_pct(),
            "Contratos Abiertos Medio": self._mean_greek_open("n_open"),
            "Roll Trades":              self._count_roll_trades(),
            "Bancarrota":               bool(self.bankrupted),
        }

    def _mean_greek_open(self, key: str) -> float:
        if not self.portfolio_history: return 0.0
        vals = [h.get(key, 0.0) for h in self.portfolio_history
                if isinstance(h.get(key), (int, float)) and h.get("n_open", 0) > 0]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_margin_pct(self) -> float:
        if not self.portfolio_history or self.initial_capital <= 0: return 0.0
        vals = [h.get("margin", 0.0) / self.initial_capital
                for h in self.portfolio_history if isinstance(h.get("margin"), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    def _mean_capital_at_risk_pct(self) -> float:
        if not self.portfolio_history or self.initial_capital <= 0: return 0.0
        vals = [h.get("capital_at_risk", 0.0) / self.initial_capital
                for h in self.portfolio_history if isinstance(h.get("capital_at_risk"), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    def _count_roll_trades(self) -> int:
        return sum(1 for t in self.trades
                   if t.get("action") == "roll" or t.get("roll_slippage"))

    def _empty_metrics(self) -> dict:
        keys = ["Retorno Total","Retorno Anual","Volatilidad Anual","Sharpe Ratio",
                "Sortino Ratio","Calmar Ratio","Max Drawdown",
                "VaR 95% (1d, %)","VaR 95% (1d, $)","ES 95% (1d, %)","ES 95% (1d, $)",
                "Win Rate","Profit Factor","Avg Win ($)","Avg Loss ($)","Nº Ciclos",
                "Nº Wins","Nº Losses","Días Positivos","Días Negativos",
                "Asimetría","Curtosis","Comisiones Totales ($)","Impacto Comisiones"]
        return {k: 0.0 for k in keys}

    def portfolio_series(self) -> pd.Series:
        if not self.portfolio_history:
            return pd.Series(dtype=float)
        return pd.Series(
            [h["total"] for h in self.portfolio_history],
            index=[h["date"] for h in self.portfolio_history],
        )


class BuyAndHoldStrategy(BaseStrategy):
    name = "Buy & Hold"
    description = "Compra y mantiene el activo subyacente (total return)"

    def execute(self, data: pd.DataFrame) -> dict:
        S0   = float(data["S"].iloc[0])
        approx_shares = int(self.initial_capital / S0)
        comm   = self._commission(approx_shares, S0, is_option=False,
                                   note="Compra inicial B&H")
        capital_efectivo = self.initial_capital - comm
        shares = capital_efectivo / S0
        self.cash -= comm
        self.trades = [{"date": data["Date"].iloc[0], "action": "BUY",
                         "pnl": 0.0, "commission": comm, "cycle_id": "bah_initial"}]
        for _, row in data.iterrows():
            S     = float(row["S"])
            total = shares * S
            self.portfolio_history.append({
                "date": pd.Timestamp(row["Date"]),
                "cash": 0.0, "mtm": total, "total": total,
                "margin": 0.0, "capital_at_risk": self.initial_capital,
                "commissions": self.total_commissions,
                "S": S, "iv": float(row.get("volatility_implied", 0.20)),
                "rf": float(row.get("risk_free_rate", 0.05)),
                "n_open": 0,
                "net_delta": shares, "net_gamma": 0, "net_theta": 0, "net_vega": 0,
            })
        return {"portfolio_history": self.portfolio_history}


def ctx_from_row(row, S_ref: float, day_idx: int) -> MarketContext:
    S      = float(row["S"])
    ticker = str(row.get("ticker", ""))
    iv_val = float(row.get("volatility_implied", 0.20))

    ts_tenors = None; ts_ivs = None
    if row.get("ts_available", False):
        v9d = row.get("vix9d", np.nan)
        v3m = row.get("vix3m", np.nan)
        if np.isfinite(v9d) and np.isfinite(v3m) and v9d > 0 and v3m > 0:
            ts_tenors = (TERM_T_SHORT, TERM_T_MID, TERM_T_LONG)
            ts_ivs    = (float(v9d), float(iv_val), float(v3m))

    return MarketContext(
        S            = S,
        r            = float(row.get("risk_free_rate", 0.05)),
        q            = float(row.get("dividend_yield", 0.0)),
        sigma        = iv_val,
        date         = pd.Timestamp(row["Date"]),
        S_ref        = S_ref,
        surface_skew = float(row.get("surface_skew",  IV_SKEW_STRENGTH_DEFAULT)),
        surface_smile= float(row.get("surface_smile", IV_SMILE_STRENGTH_DEFAULT)),
        day_idx      = day_idx,
        strike_tick  = get_strike_tick(ticker, S),
        iv_base      = iv_val,
        ts_tenors    = ts_tenors,
        ts_ivs       = ts_ivs,
        ticker       = ticker,
    )


def ctx_from_row_reactive(row, S_ref: float, day_idx: int,
                           recent_log_rets: list) -> MarketContext:
    sigma_model = float(row.get("volatility_implied", 0.20))
    if len(recent_log_rets) >= 3:
        hv_short = float(np.std(recent_log_rets[-5:])) * np.sqrt(TRADING_DAYS_PER_YEAR)
        hv_short = float(np.clip(hv_short, IV_PROXY_MIN, IV_PROXY_MAX))
        sigma_reactive = 0.70 * sigma_model + 0.30 * hv_short
    else:
        sigma_reactive = sigma_model
    sigma_reactive = float(np.clip(sigma_reactive, IV_MIN, IV_MAX))
    S_val  = float(row["S"])
    ticker = str(row.get("ticker", ""))

    ts_tenors = None; ts_ivs = None
    if row.get("ts_available", False):
        v9d = row.get("vix9d", np.nan)
        v3m = row.get("vix3m", np.nan)
        if np.isfinite(v9d) and np.isfinite(v3m) and v9d > 0 and v3m > 0:
            ts_tenors = (TERM_T_SHORT, TERM_T_MID, TERM_T_LONG)
            ts_ivs    = (float(v9d), float(sigma_reactive), float(v3m))

    return MarketContext(
        S            = S_val,
        r            = float(row.get("risk_free_rate", 0.05)),
        q            = float(row.get("dividend_yield", 0.0)),
        sigma        = sigma_reactive,
        date         = pd.Timestamp(row["Date"]),
        S_ref        = S_ref,
        surface_skew = float(row.get("surface_skew",  IV_SKEW_STRENGTH_DEFAULT)),
        surface_smile= float(row.get("surface_smile", IV_SMILE_STRENGTH_DEFAULT)),
        day_idx      = day_idx,
        strike_tick  = get_strike_tick(ticker, S_val),
        iv_base      = sigma_model,
        ts_tenors    = ts_tenors,
        ts_ivs       = ts_ivs,
        ticker       = ticker,
    )


class _SingleLegStrategy(BaseStrategy):
    option_type: str    = "call"
    sign: int           = 1
    target_delta: float = 0.50
    expiration_days: int = 30
    roll_enabled: bool   = True
    roll_days_before: int = 5
    max_positions: int   = 1
    max_pct_cash: float  = 0.10

    def execute(self, data: pd.DataFrame) -> dict:
        S_ref        = float(data["S"].iloc[0])
        just_closed  = False
        log_rets_buf: list = []
        rows = list(data.iterrows())

        for i, (_, row) in enumerate(rows):
            ctx = ctx_from_row(row, S_ref, i)
            if i > 0:
                self._accrue_daily_interest(ctx)
                S_prev = float(rows[i-1][1]["S"])
                S_curr = float(row["S"])
                if S_prev > 0 and S_curr > 0:
                    log_rets_buf.append(float(np.log(S_curr / S_prev)))
                if len(log_rets_buf) > 20:
                    log_rets_buf = log_rets_buf[-20:]

            just_closed = False
            if self.bankrupted:
                self._record_portfolio(ctx, 0.0)
                continue

            for pos in [p for p in self.positions if p.is_open]:
                leg = pos.legs[0]
                days_left = (leg.T_exp - ctx.date).days
                if days_left <= 0:
                    self._expire_leg(leg, ctx, cycle_id=pos.cycle_id)
                    pos.is_open = False
                    just_closed = True
                elif self.roll_enabled and days_left <= self.roll_days_before:
                    self._close_leg(leg, ctx, reason="roll", cycle_id=pos.cycle_id)
                    pos.is_open = False
                    just_closed = True

            self.positions = [p for p in self.positions if p.is_open]

            need_open = (len(self.positions) < self.max_positions and
                         (just_closed or len(self.positions) == 0))

            if need_open:
                margin_in_use = self._total_margin_required(ctx)
                available     = self.cash - margin_in_use

                if available > 50:
                    T = self.expiration_days / CALENDAR_DAYS_PER_YEAR
                    ctx_for_strike = ctx_from_row_reactive(row, S_ref, i, log_rets_buf)
                    K = find_strike_by_delta(ctx_for_strike, self.target_delta, T,
                                             self.option_type,
                                             strike_tick=ctx_for_strike.strike_tick)
                    if K is None:
                        self._warnings.append(
                            f"[{ctx.date.date()}] Strike delta={self.target_delta} "
                            f"{self.option_type} no convergió; trade omitido")
                    else:
                        mid, _ = price_option(ctx, K, T, self.option_type)
                        side   = "BUY" if self.sign > 0 else "SELL"
                        exec_px = apply_execution_spread(mid, side, self.option_type, ctx, K, T)

                        if exec_px > 0:
                            if self.sign > 0:
                                eq_base   = self._sizing_base(ctx)
                                max_cost  = eq_base * self.max_pct_cash
                                contracts = int(max_cost / max(exec_px * OPTION_MULTIPLIER, 0.01))
                                contracts = min(contracts, 50)
                                if contracts == 0 and available > exec_px * OPTION_MULTIPLIER:
                                    contracts = 1
                            else:
                                margin_per = estimate_reg_t_margin_per_short(
                                    ctx.S, K, self.option_type, exec_px, sigma=ctx.sigma)
                                sizing_cap = max(available,
                                                 EQUITY_SIZING_FLOOR_PCT * self.initial_capital)
                                contracts  = int((sizing_cap * 0.5) / max(margin_per, 1.0))
                                contracts  = min(contracts, 50)
                                contracts  = self._apply_margin_cap(contracts, margin_per, ctx)

                            if contracts > 0:
                                leg = self._open_leg(ctx, K, T, self.option_type,
                                                      contracts, self.sign)
                                pos = OptionPosition(legs=[leg], open_date=ctx.date)
                                self._register_open_trade(leg, ctx, pos.cycle_id)
                                self.positions.append(pos)

            self._check_margin_call(ctx)

            is_last_day = (i == len(rows) - 1)
            if is_last_day:
                self._close_all_open_end_of_period(ctx)

            mtm = self._positions_mtm(ctx)
            self._record_portfolio(ctx, mtm)

        return {"portfolio_history": self.portfolio_history}


class LongCallStrategy(_SingleLegStrategy):
    def __init__(self, delta=0.50, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Call Comprada {delta}Δ",
                         description=f"Long call delta={delta} DTE={expiration_days}")
        self.option_type=     "call"; self.sign=1
        self.target_delta=    delta;  self.expiration_days=expiration_days
        self.roll_enabled=    roll_enabled; self.roll_days_before=roll_days_before
        self.max_positions=   1;      self.max_pct_cash=0.05


class ShortCallStrategy(_SingleLegStrategy):
    def __init__(self, delta=0.30, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Call Vendida {delta}Δ",
                         description=f"Short call delta={delta} DTE={expiration_days}")
        self.option_type=     "call"; self.sign=-1
        self.target_delta=    delta;  self.expiration_days=expiration_days
        self.roll_enabled=    roll_enabled; self.roll_days_before=roll_days_before
        self.max_positions=   1


class LongPutStrategy(_SingleLegStrategy):
    def __init__(self, delta=-0.50, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Put Comprada {delta}Δ",
                         description=f"Long put delta={delta} DTE={expiration_days}")
        self.option_type=     "put"; self.sign=1
        self.target_delta=    delta; self.expiration_days=expiration_days
        self.roll_enabled=    roll_enabled; self.roll_days_before=roll_days_before
        self.max_positions=   1;    self.max_pct_cash=0.05


class ShortPutStrategy(_SingleLegStrategy):
    def __init__(self, delta=-0.30, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Put Vendida {delta}Δ",
                         description=f"Short put delta={delta} DTE={expiration_days}")
        self.option_type=     "put"; self.sign=-1
        self.target_delta=    delta; self.expiration_days=expiration_days
        self.roll_enabled=    roll_enabled; self.roll_days_before=roll_days_before
        self.max_positions=   1


class _TwoLegSpreadStrategy(BaseStrategy):
    expiration_days: int  = 30
    roll_days_before: int = 5
    roll_enabled: bool    = True

    @property
    def reopen_every_days(self):
        return 1 if self.roll_enabled else self.expiration_days

    def execute(self, data: pd.DataFrame) -> dict:
        S_ref         = float(data["S"].iloc[0])
        last_open_day = -9999
        log_rets_buf: list = []
        rows = list(data.iterrows())

        for i, (_, row) in enumerate(rows):
            ctx = ctx_from_row(row, S_ref, i)
            if i > 0:
                self._accrue_daily_interest(ctx)
                S_prev = float(rows[i-1][1]["S"])
                S_curr = float(row["S"])
                if S_prev > 0 and S_curr > 0:
                    log_rets_buf.append(float(np.log(S_curr / S_prev)))
                if len(log_rets_buf) > 20:
                    log_rets_buf = log_rets_buf[-20:]

            just_closed = False
            if self.bankrupted:
                self._record_portfolio(ctx, 0.0)
                continue

            for pos in [p for p in self.positions if p.is_open]:
                days_left = pos.days_to_exp(ctx.date)
                if days_left <= 0:
                    for leg in pos.legs:
                        if leg.is_open:
                            self._expire_leg(leg, ctx, cycle_id=pos.cycle_id)
                    pos.is_open = False; just_closed = True
                elif self.roll_enabled and days_left <= self.roll_days_before:
                    for leg in pos.legs:
                        if leg.is_open:
                            self._close_leg(leg, ctx, reason="roll", cycle_id=pos.cycle_id)
                    pos.is_open = False; just_closed = True

            self.positions = [p for p in self.positions if p.is_open]

            days_since_last = i - last_open_day
            should_open = (
                len(self.positions) == 0
                and (just_closed or days_since_last >= self.reopen_every_days)
            )
            if should_open:
                current_total = self.cash + self._positions_mtm(ctx)
                if current_total < 0.10 * self.initial_capital:
                    should_open = False

            if should_open:
                T = self.expiration_days / CALENDAR_DAYS_PER_YEAR
                ctx_build = ctx_from_row_reactive(row, S_ref, i, log_rets_buf)
                new_pos = self._build_position(ctx_build, T)
                if new_pos is not None:
                    self.positions.append(new_pos)
                    last_open_day = i

            self._check_margin_call(ctx)

            is_last_day = (i == len(rows) - 1)
            if is_last_day:
                self._close_all_open_end_of_period(ctx)

            mtm = self._positions_mtm(ctx)
            self._record_portfolio(ctx, mtm)

        return {"portfolio_history": self.portfolio_history}

    def _build_position(self, ctx, T):
        raise NotImplementedError


class _MultiLegStrategy(_TwoLegSpreadStrategy):
    width: float = 0.10

    def _n_debit(self, debit_per, pct=0.05, ctx=None):
        if debit_per <= 0:
            return 0
        base = self._sizing_base(ctx) if ctx is not None else self.initial_capital
        return min(int((base * pct) / max(debit_per, 0.01)), 50)


class BullSpreadStrategy(_TwoLegSpreadStrategy):
    def __init__(self, long_delta=0.50, short_delta=0.25,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Bull Spread {long_delta}/{short_delta}Δ",
                         description=f"Long call {long_delta}Δ + Short call {short_delta}Δ")
        self.long_delta=long_delta; self.short_delta=short_delta
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        K_long  = find_strike_by_delta(ctx, self.long_delta,  T, "call", strike_tick=ctx.strike_tick)
        K_short = find_strike_by_delta(ctx, self.short_delta, T, "call", strike_tick=ctx.strike_tick)
        if K_long is None or K_short is None: return None
        if K_short <= K_long:
            K_short = round_to_strike_tick(K_long * 1.025, ctx.strike_tick)

        m_l,_ = price_option(ctx,K_long, T,"call"); m_s,_ = price_option(ctx,K_short,T,"call")
        e_l = apply_execution_spread(m_l,"BUY", "call",ctx,K_long, T)
        e_s = apply_execution_spread(m_s,"SELL","call",ctx,K_short,T)
        debit_per = (e_l - e_s) * OPTION_MULTIPLIER
        if debit_per <= 0: return None

        n = min(int((self._sizing_base(ctx)*0.05)/max(debit_per,0.01)),40)
        if n <= 0: return None

        c_l_prev = self._commission_preview(n, e_l)
        c_s_prev = self._commission_preview(n, e_s)
        total_cost = debit_per*n + c_l_prev + c_s_prev
        if total_cost > self.cash:
            n = max(1, int(self.cash*0.90/max(debit_per,0.01)))
            if n <= 0: return None
            c_l_prev = self._commission_preview(n, e_l)
            c_s_prev = self._commission_preview(n, e_s)
            if debit_per*n + c_l_prev + c_s_prev > self.cash:
                return None

        c_l = self._commission(n, e_l, note=f"buy call K={K_long:.2f}")
        c_s = self._commission(n, e_s, note=f"sell call K={K_short:.2f}")
        self.cash -= debit_per*n + c_l + c_s

        exp = expiration_date_from_days(ctx.date, int(T*CALENDAR_DAYS_PER_YEAR))
        legs = [OptionLeg("call",K_long, exp,n,+1,m_l,e_l,c_l,ctx.date),
                OptionLeg("call",K_short,exp,n,-1,m_s,e_s,c_s,ctx.date)]
        pos = OptionPosition(legs=legs, open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg, ctx, pos.cycle_id, "open_bull_spread")
        return pos


class BearSpreadStrategy(_TwoLegSpreadStrategy):
    def __init__(self, long_delta=-0.50, short_delta=-0.25,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Bear Spread {long_delta}/{short_delta}Δ",
                         description=f"Long put {long_delta}Δ + Short put {short_delta}Δ")
        self.long_delta=long_delta; self.short_delta=short_delta
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        K_long  = find_strike_by_delta(ctx, self.long_delta,  T,"put",strike_tick=ctx.strike_tick)
        K_short = find_strike_by_delta(ctx, self.short_delta, T,"put",strike_tick=ctx.strike_tick)
        if K_long is None or K_short is None: return None
        if K_long <= K_short:
            K_long = round_to_strike_tick(K_short*1.025, ctx.strike_tick)

        m_l,_ = price_option(ctx,K_long, T,"put"); m_s,_ = price_option(ctx,K_short,T,"put")
        e_l = apply_execution_spread(m_l,"BUY", "put",ctx,K_long, T)
        e_s = apply_execution_spread(m_s,"SELL","put",ctx,K_short,T)
        debit_per = (e_l - e_s) * OPTION_MULTIPLIER
        if debit_per <= 0: return None

        n = min(int((self._sizing_base(ctx)*0.05)/max(debit_per,0.01)),40)
        if n <= 0: return None

        c_l_prev = self._commission_preview(n, e_l)
        c_s_prev = self._commission_preview(n, e_s)
        total_cost = debit_per*n + c_l_prev + c_s_prev
        if total_cost > self.cash:
            n = max(1, int(self.cash*0.90/max(debit_per,0.01)))
            if n <= 0: return None
            c_l_prev = self._commission_preview(n, e_l)
            c_s_prev = self._commission_preview(n, e_s)
            if debit_per*n + c_l_prev + c_s_prev > self.cash:
                return None

        c_l = self._commission(n, e_l, note=f"buy put K={K_long:.2f}")
        c_s = self._commission(n, e_s, note=f"sell put K={K_short:.2f}")
        self.cash -= debit_per*n + c_l + c_s

        exp = expiration_date_from_days(ctx.date, int(T*CALENDAR_DAYS_PER_YEAR))
        legs = [OptionLeg("put",K_long, exp,n,+1,m_l,e_l,c_l,ctx.date),
                OptionLeg("put",K_short,exp,n,-1,m_s,e_s,c_s,ctx.date)]
        pos = OptionPosition(legs=legs, open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg, ctx, pos.cycle_id, "open_bear_spread")
        return pos


class LongStraddleStrategy(_TwoLegSpreadStrategy):
    def __init__(self, delta=0.50, use_calls=True, use_puts=True,
                 expiration_days=30, roll_enabled=True,
                 roll_days_before=2, max_pct_cap=0.05):
        legs_desc = []
        if use_calls: legs_desc.append(f"call {delta}Δ")
        if use_puts:  legs_desc.append(f"put {delta}Δ")
        super().__init__(name=f"Straddle Comprado {'&'.join(legs_desc)}",
                         description=f"Long {' + '.join(legs_desc)} ATM/NTM")
        self.delta=float(delta); self.use_calls=use_calls; self.use_puts=use_puts
        self.expiration_days=expiration_days; self.roll_enabled=roll_enabled
        self.roll_days_before=roll_days_before; self.max_pct_cap=float(max_pct_cap)

    def _build_position(self, ctx, T):
        if not self.use_calls and not self.use_puts: return None
        if abs(self.delta-0.50) < 0.01:
            K = ctx.S
        else:
            K_c = find_strike_by_delta(ctx,self.delta, T,"call",strike_tick=ctx.strike_tick) if self.use_calls else ctx.S
            K_p = find_strike_by_delta(ctx,-self.delta,T,"put", strike_tick=ctx.strike_tick) if self.use_puts  else ctx.S
            Ks  = [x for x in [K_c,K_p] if x is not None]
            K   = round_to_strike_tick(float(np.mean(Ks)),ctx.strike_tick) if Ks else ctx.S

        legs_data = []; debit_per = 0.0
        if self.use_calls:
            mc,_=price_option(ctx,K,T,"call"); ec=apply_execution_spread(mc,"BUY","call",ctx,K,T)
            debit_per+=ec*OPTION_MULTIPLIER; legs_data.append(("call",K,mc,ec))
        if self.use_puts:
            mp,_=price_option(ctx,K,T,"put");  ep=apply_execution_spread(mp,"BUY","put", ctx,K,T)
            debit_per+=ep*OPTION_MULTIPLIER; legs_data.append(("put",K,mp,ep))
        if debit_per <= 0: return None


        n_base = min(int((self.initial_capital*self.max_pct_cap)/max(debit_per,0.01)),60)
        max_n_by_price = int(ctx.S*0.04*OPTION_MULTIPLIER/max(debit_per,0.01))
        n = min(n_base, max(max_n_by_price,1), 50)
        if n <= 0: return None

        comm_total_prev = sum(self._commission_preview(n, ex) for _,_,_,ex in legs_data)
        if debit_per*n + comm_total_prev > self.cash:
            return None

        built_legs = []; total_comm_real = 0.0
        for otype,K_leg,mid,exec_px in legs_data:
            comm = self._commission(n, exec_px, note=f"buy {otype} straddle K={K_leg:.2f}")
            total_comm_real += comm
            built_legs.append(OptionLeg(otype,K_leg,None,n,+1,mid,exec_px,comm,ctx.date))
        self.cash -= debit_per*n + total_comm_real

        exp = expiration_date_from_days(ctx.date, int(T*CALENDAR_DAYS_PER_YEAR))
        for leg in built_legs: leg.T_exp = exp
        pos = OptionPosition(legs=built_legs, open_date=ctx.date)
        for leg in built_legs:
            self._register_open_trade(leg, ctx, pos.cycle_id, "open_straddle")
        return pos


class ShortStraddleStrategy(_TwoLegSpreadStrategy):
    def __init__(self, delta=0.50, use_calls=True, use_puts=True,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        legs_desc = []
        if use_calls: legs_desc.append(f"call {delta}Δ")
        if use_puts:  legs_desc.append(f"put {delta}Δ")
        super().__init__(name=f"Straddle Vendido {'&'.join(legs_desc)}",
                         description=f"Short {' + '.join(legs_desc)} ATM/NTM")
        self.delta=float(delta); self.use_calls=use_calls; self.use_puts=use_puts
        self.expiration_days=expiration_days; self.roll_enabled=roll_enabled
        self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        if not self.use_calls and not self.use_puts: return None
        if abs(self.delta-0.50)<0.01:
            K = ctx.S
        else:
            K_c = find_strike_by_delta(ctx,self.delta, T,"call",strike_tick=ctx.strike_tick) if self.use_calls else ctx.S
            K_p = find_strike_by_delta(ctx,-self.delta,T,"put", strike_tick=ctx.strike_tick) if self.use_puts  else ctx.S
            Ks  = [x for x in [K_c,K_p] if x is not None]
            K   = round_to_strike_tick(float(np.mean(Ks)),ctx.strike_tick) if Ks else ctx.S

        exec_prices={}; mid_prices={}
        if self.use_calls:
            mc,_=price_option(ctx,K,T,"call"); ec=apply_execution_spread(mc,"SELL","call",ctx,K,T)
            exec_prices["call"]=ec; mid_prices["call"]=mc
        if self.use_puts:
            mp,_=price_option(ctx,K,T,"put");  ep=apply_execution_spread(mp,"SELL","put", ctx,K,T)
            exec_prices["put"]=ep; mid_prices["put"]=mp

        ec_=exec_prices.get("call",0.0); ep_=exec_prices.get("put",0.0)
        credit_per=(ec_+ep_)*OPTION_MULTIPLIER
        margin_per=estimate_span_margin_strangle(ctx.S,K,K,ec_,ep_,sigma=ctx.sigma)
        margin_per=max(margin_per,credit_per*1.2,1.0)

        n=min(int((self._sizing_base(ctx)*0.35)/max(margin_per,1.0)),60)
        n=self._apply_margin_cap(n,margin_per,ctx)
        if n<=0: return None

        types_used = (["call"] if self.use_calls else []) + (["put"] if self.use_puts else [])
        comm_prev = sum(self._commission_preview(n, exec_prices[ot]) for ot in types_used)
        net_prev = credit_per*n - comm_prev
        if net_prev <= 0:
            return None

        built_legs=[]; total_comm=0.0
        for otype in types_used:
            ex=exec_prices[otype]; mi=mid_prices[otype]
            comm=self._commission(n,ex,note=f"sell {otype} straddle K={K:.2f}")
            total_comm+=comm
            built_legs.append(OptionLeg(otype,K,None,n,-1,mi,ex,comm,ctx.date))

        net=credit_per*n - total_comm
        self.cash += net

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        for leg in built_legs: leg.T_exp=exp
        pos=OptionPosition(legs=built_legs,open_date=ctx.date)
        for leg in built_legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_short_straddle")
        return pos


class LongStrangleStrategy(_TwoLegSpreadStrategy):
    def __init__(self, call_delta=0.25, put_delta=-0.25,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Strangle Comprado {call_delta}/{put_delta}Δ",
                         description=f"Long call {call_delta}Δ + Long put {put_delta}Δ")
        self.call_delta=call_delta; self.put_delta=put_delta
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        Kc=find_strike_by_delta(ctx,self.call_delta,T,"call",strike_tick=ctx.strike_tick)
        Kp=find_strike_by_delta(ctx,self.put_delta, T,"put", strike_tick=ctx.strike_tick)
        if Kc is None or Kp is None: return None
        if Kc<=Kp:
            Kc=round_to_strike_tick(max(Kc,ctx.S*1.01),ctx.strike_tick)
            Kp=round_to_strike_tick(min(Kp,ctx.S*0.99),ctx.strike_tick)

        mc,_=price_option(ctx,Kc,T,"call"); mp,_=price_option(ctx,Kp,T,"put")
        ec=apply_execution_spread(mc,"BUY","call",ctx,Kc,T)
        ep=apply_execution_spread(mp,"BUY","put", ctx,Kp,T)
        debit_per=(ec+ep)*OPTION_MULTIPLIER
        if debit_per<=0: return None


        n_base=min(int((self._sizing_base(ctx)*0.05)/max(debit_per,0.01)),60)
        max_n=int(ctx.S*0.04*OPTION_MULTIPLIER/max(debit_per,0.01))
        n=min(n_base,max(max_n,1),50)
        if n<=0: return None

        cc_prev = self._commission_preview(n, ec)
        cp_prev = self._commission_preview(n, ep)
        if debit_per*n + cc_prev + cp_prev > self.cash:
            return None

        cc=self._commission(n,ec,note=f"buy call strangle K={Kc:.2f}")
        cp=self._commission(n,ep,note=f"buy put strangle K={Kp:.2f}")
        self.cash -= debit_per*n + cc + cp

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("call",Kc,exp,n,+1,mc,ec,cc,ctx.date),
              OptionLeg("put", Kp,exp,n,+1,mp,ep,cp,ctx.date)]
        pos=OptionPosition(legs=legs,open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_strangle")
        return pos


class ShortStrangleStrategy(_TwoLegSpreadStrategy):
    def __init__(self, call_delta=0.25, put_delta=-0.25,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Strangle Vendido {call_delta}/{put_delta}Δ",
                         description=f"Short call {call_delta}Δ + Short put {put_delta}Δ")
        self.call_delta=call_delta; self.put_delta=put_delta
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        Kc=find_strike_by_delta(ctx,self.call_delta,T,"call",strike_tick=ctx.strike_tick)
        Kp=find_strike_by_delta(ctx,self.put_delta, T,"put", strike_tick=ctx.strike_tick)
        if Kc is None or Kp is None: return None
        if Kc<=Kp:
            Kc=round_to_strike_tick(max(Kc,ctx.S*1.01),ctx.strike_tick)
            Kp=round_to_strike_tick(min(Kp,ctx.S*0.99),ctx.strike_tick)

        mc,_=price_option(ctx,Kc,T,"call"); mp,_=price_option(ctx,Kp,T,"put")
        ec=apply_execution_spread(mc,"SELL","call",ctx,Kc,T)
        ep=apply_execution_spread(mp,"SELL","put", ctx,Kp,T)
        credit_per=(ec+ep)*OPTION_MULTIPLIER
        margin_per=estimate_span_margin_strangle(ctx.S,Kc,Kp,ec,ep,sigma=ctx.sigma)
        margin_per=max(margin_per,credit_per*1.2,1.0)

        n=min(int((self._sizing_base(ctx)*0.35)/max(margin_per,1.0)),60)
        n=self._apply_margin_cap(n,margin_per,ctx)
        if n<=0: return None

        cc_prev = self._commission_preview(n, ec)
        cp_prev = self._commission_preview(n, ep)
        net_prev = credit_per*n - cc_prev - cp_prev
        if net_prev <= 0: return None

        cc=self._commission(n,ec,note=f"sell call strangle K={Kc:.2f}")
        cp=self._commission(n,ep,note=f"sell put strangle K={Kp:.2f}")
        net=credit_per*n - cc - cp
        self.cash += net

        fixed_m = float(estimate_span_margin_strangle(
            ctx.S, Kc, Kp, ec, ep, sigma=ctx.sigma)) * n

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("call",Kc,exp,n,-1,mc,ec,cc,ctx.date),
              OptionLeg("put", Kp,exp,n,-1,mp,ep,cp,ctx.date)]
        pos=OptionPosition(legs=legs,open_date=ctx.date,fixed_margin=fixed_m)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_short_strangle")
        return pos


class LongButterflyStrategy(_MultiLegStrategy):
    def __init__(self, width=0.10, option_type="call",
                 width_lower=None, width_upper=None,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        w_lo=width_lower if width_lower is not None else width
        w_hi=width_upper if width_upper is not None else width
        asym="" if abs(w_lo-w_hi)<0.001 else f" asim {w_lo*100:.0f}/{w_hi*100:.0f}%"
        name=f"Mariposa Comprada {option_type} {width*100:.0f}%{asym}"
        super().__init__(name=name,
                         description=f"Long butterfly {option_type}: Long ala + Short 2 ATM + Long ala")
        self.option_type=option_type.lower(); self.width=width
        self.width_lower=w_lo; self.width_upper=w_hi
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        S=ctx.S; tick=ctx.strike_tick
        K2=round_to_strike_tick(S,tick)
        K1=round_to_strike_tick(S*(1-self.width_lower),tick)
        K3=round_to_strike_tick(S*(1+self.width_upper),tick)
        if K2-K1<tick: K1=K2-tick
        if K3-K2<tick: K3=K2+tick

        if self.option_type=="iron":
            mp2,_=price_option(ctx,K2,T,"put");  mc2,_=price_option(ctx,K2,T,"call")
            mp1,_=price_option(ctx,K1,T,"put");  mc3,_=price_option(ctx,K3,T,"call")
            ep2=apply_execution_spread(mp2,"BUY", "put", ctx,K2,T)
            ec2=apply_execution_spread(mc2,"BUY", "call",ctx,K2,T)
            ep1=apply_execution_spread(mp1,"SELL","put", ctx,K1,T)
            ec3=apply_execution_spread(mc3,"SELL","call",ctx,K3,T)
            debit_per=(ep2+ec2-ep1-ec3)*OPTION_MULTIPLIER
            if debit_per<=0: return None


            n=self._n_debit(debit_per,0.05,ctx=ctx)
            max_n=int(ctx.S*0.04*OPTION_MULTIPLIER/max(debit_per,0.01))
            n=min(n,max(max_n,1))
            if n<=0: return None

            preview = (self._commission_preview(n,ep2) + self._commission_preview(n,ec2)
                       + self._commission_preview(n,ep1) + self._commission_preview(n,ec3))
            if debit_per*n + preview > self.cash:
                n=max(1,int(self.cash*0.90/max(debit_per,0.01)))
                if n<=0: return None
                preview = (self._commission_preview(n,ep2) + self._commission_preview(n,ec2)
                           + self._commission_preview(n,ep1) + self._commission_preview(n,ec3))
                if debit_per*n + preview > self.cash:
                    return None

            c2p=self._commission(n,ep2); c2c=self._commission(n,ec2)
            c1 =self._commission(n,ep1); c3 =self._commission(n,ec3)
            self.cash -= debit_per*n + c2p + c2c + c1 + c3

            exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
            legs=[OptionLeg("put", K2,exp,n,+1,mp2,ep2,c2p,ctx.date),
                  OptionLeg("call",K2,exp,n,+1,mc2,ec2,c2c,ctx.date),
                  OptionLeg("put", K1,exp,n,-1,mp1,ep1,c1, ctx.date),
                  OptionLeg("call",K3,exp,n,-1,mc3,ec3,c3, ctx.date)]
            pos=OptionPosition(legs=legs,open_date=ctx.date)
            for leg in legs:
                self._register_open_trade(leg,ctx,pos.cycle_id,"open_long_iron_butterfly")
            return pos
        else:
            ot=self.option_type if self.option_type in ("call","put") else "call"
            m1,_=price_option(ctx,K1,T,ot); m2,_=price_option(ctx,K2,T,ot)
            m3,_=price_option(ctx,K3,T,ot)
            e1=apply_execution_spread(m1,"BUY", ot,ctx,K1,T)
            e2=apply_execution_spread(m2,"SELL",ot,ctx,K2,T)
            e3=apply_execution_spread(m3,"BUY", ot,ctx,K3,T)
            debit_per=(e1-2*e2+e3)*OPTION_MULTIPLIER
            if debit_per<=0: return None


            n=self._n_debit(debit_per,0.05,ctx=ctx)
            max_n=int(S*0.04*OPTION_MULTIPLIER/max(debit_per,0.01))
            n=min(n,max(max_n,1))
            if n<=0: return None

            preview = (self._commission_preview(n,e1) + self._commission_preview(2*n,e2)
                       + self._commission_preview(n,e3))
            if debit_per*n + preview > self.cash:
                n=max(1,int(self.cash*0.90/max(debit_per,0.01)))
                if n<=0: return None
                preview = (self._commission_preview(n,e1) + self._commission_preview(2*n,e2)
                           + self._commission_preview(n,e3))
                if debit_per*n + preview > self.cash:
                    return None

            c1=self._commission(n,e1); c2=self._commission(2*n,e2); c3=self._commission(n,e3)
            self.cash -= debit_per*n + c1 + c2 + c3

            exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
            legs=[OptionLeg(ot,K1,exp, n,+1,m1,e1,c1,ctx.date),
                  OptionLeg(ot,K2,exp,2*n,-1,m2,e2,c2,ctx.date),
                  OptionLeg(ot,K3,exp, n,+1,m3,e3,c3,ctx.date)]
            pos=OptionPosition(legs=legs,open_date=ctx.date)
            for leg in legs:
                self._register_open_trade(leg,ctx,pos.cycle_id,f"open_long_butterfly_{ot}")
            return pos


class ShortIronButterflyStrategy(_MultiLegStrategy):
    def __init__(self, width=0.10, width_lower=None, width_upper=None,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        w_lo=width_lower if width_lower is not None else width
        w_hi=width_upper if width_upper is not None else width
        asym="" if abs(w_lo-w_hi)<0.001 else f" asim {w_lo*100:.0f}/{w_hi*100:.0f}%"
        super().__init__(name=f"Iron Butterfly Corta {width*100:.0f}%{asym}",
                         description="Short iron butterfly: Short straddle ATM + Long alas (crédito)")
        self.width=width; self.width_lower=w_lo; self.width_upper=w_hi
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        S=ctx.S; tick=ctx.strike_tick
        K2=round_to_strike_tick(S,tick)
        K1=round_to_strike_tick(S*(1-self.width_lower),tick)
        K3=round_to_strike_tick(S*(1+self.width_upper),tick)
        if K2-K1<tick: K1=K2-tick
        if K3-K2<tick: K3=K2+tick

        mp1,_=price_option(ctx,K1,T,"put");  mp2,_=price_option(ctx,K2,T,"put")
        mc2,_=price_option(ctx,K2,T,"call"); mc3,_=price_option(ctx,K3,T,"call")
        ep1=apply_execution_spread(mp1,"BUY", "put", ctx,K1,T)
        ep2=apply_execution_spread(mp2,"SELL","put", ctx,K2,T)
        ec2=apply_execution_spread(mc2,"SELL","call",ctx,K2,T)
        ec3=apply_execution_spread(mc3,"BUY", "call",ctx,K3,T)
        credit_per=(ep2+ec2-ep1-ec3)*OPTION_MULTIPLIER
        if credit_per<=0: return None

        wing_w=max(K2-K1,K3-K2)
        margin_per=wing_w*OPTION_MULTIPLIER*iv_margin_multiplier(ctx.sigma)
        margin_per=max(margin_per,credit_per*1.2,1.0)

        n=min(int((self._sizing_base(ctx)*0.28)/max(margin_per,1.0)),40)
        n=self._apply_margin_cap(n,margin_per,ctx,wing_width=wing_w)
        if n<=0: return None

        preview = (self._commission_preview(n,ep1) + self._commission_preview(n,ep2)
                   + self._commission_preview(n,ec2) + self._commission_preview(n,ec3))
        net_prev = credit_per*n - preview
        if net_prev <= 0: return None

        c1=self._commission(n,ep1); c2p=self._commission(n,ep2)
        c2c=self._commission(n,ec2); c3=self._commission(n,ec3)
        net_cash = credit_per*n - c1 - c2p - c2c - c3
        self.cash += net_cash

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("put", K1,exp,n,+1,mp1,ep1,c1, ctx.date),
              OptionLeg("put", K2,exp,n,-1,mp2,ep2,c2p,ctx.date),
              OptionLeg("call",K2,exp,n,-1,mc2,ec2,c2c,ctx.date),
              OptionLeg("call",K3,exp,n,+1,mc3,ec3,c3, ctx.date)]
        fixed_m=max(wing_w*OPTION_MULTIPLIER*n,1.0)
        pos=OptionPosition(legs=legs,open_date=ctx.date,fixed_margin=fixed_m)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_short_iron_butterfly")
        return pos


class ShortButterflyStrategy(_MultiLegStrategy):
    def __init__(self, width=0.10, option_type="call",
                 width_lower=None, width_upper=None,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        w_lo=width_lower if width_lower is not None else width
        w_hi=width_upper if width_upper is not None else width
        asym="" if abs(w_lo-w_hi)<0.001 else f" asim {w_lo*100:.0f}/{w_hi*100:.0f}%"
        super().__init__(name=f"Mariposa Vendida {option_type} {width*100:.0f}%{asym}",
                         description=f"Short butterfly {option_type}")
        self.option_type=option_type.lower(); self.width=width
        self.width_lower=w_lo; self.width_upper=w_hi
        self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        S=ctx.S; tick=ctx.strike_tick
        K2=round_to_strike_tick(S,tick)
        K1=round_to_strike_tick(S*(1-self.width_lower),tick)
        K3=round_to_strike_tick(S*(1+self.width_upper),tick)
        if K2-K1<tick: K1=K2-tick
        if K3-K2<tick: K3=K2+tick
        ot=self.option_type if self.option_type in ("call","put") else "call"

        m1,_=price_option(ctx,K1,T,ot); m2,_=price_option(ctx,K2,T,ot)
        m3,_=price_option(ctx,K3,T,ot)
        e1=apply_execution_spread(m1,"SELL",ot,ctx,K1,T)
        e2=apply_execution_spread(m2,"BUY", ot,ctx,K2,T)
        e3=apply_execution_spread(m3,"SELL",ot,ctx,K3,T)
        credit_per=(e1-2*e2+e3)*OPTION_MULTIPLIER
        if credit_per<=0: return None

        wing_w=max(K2-K1,K3-K2,0.0)
        max_loss=max(wing_w*OPTION_MULTIPLIER-credit_per,1.0)
        margin_floor=wing_w*OPTION_MULTIPLIER
        margin=max(max_loss,credit_per*2.0,margin_floor,1.0)
        margin*=iv_margin_multiplier(ctx.sigma)

        n=min(int((self._sizing_base(ctx)*0.28)/max(margin,1.0)),40)
        n=self._apply_margin_cap(n,margin,ctx,wing_width=wing_w)
        if n<=0: return None

        preview = (self._commission_preview(n,e1) + self._commission_preview(2*n,e2)
                   + self._commission_preview(n,e3))
        if credit_per*n - preview <= 0: return None

        c1=self._commission(n,e1); c2=self._commission(2*n,e2); c3=self._commission(n,e3)
        net=credit_per*n - c1 - c2 - c3
        self.cash += net

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg(ot,K1,exp, n,-1,m1,e1,c1,ctx.date),
              OptionLeg(ot,K2,exp,2*n,+1,m2,e2,c2,ctx.date),
              OptionLeg(ot,K3,exp, n,-1,m3,e3,c3,ctx.date)]
        fixed_m=max(max_loss*n,margin_floor*n,1.0)
        pos=OptionPosition(legs=legs,open_date=ctx.date,fixed_margin=fixed_m)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,f"open_short_butterfly_{ot}")
        return pos


class LongCondorStrategy(_MultiLegStrategy):
    def __init__(self, width=0.10, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Cóndor Comprado {width*100:.0f}%",
                         description="Debit iron condor (4 strikes)")
        self.width=width; self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        w=self.width; tick=ctx.strike_tick
        pi=round_to_strike_tick(ctx.S*(1-0.5*w),tick)
        po=round_to_strike_tick(ctx.S*(1-1.5*w),tick)
        ci=round_to_strike_tick(ctx.S*(1+0.5*w),tick)
        co=round_to_strike_tick(ctx.S*(1+1.5*w),tick)
        if pi-po<tick: po=pi-tick
        if ci-pi<tick: ci=pi+tick
        if co-ci<tick: co=ci+tick

        mpi,_=price_option(ctx,pi,T,"put");  mpo,_=price_option(ctx,po,T,"put")
        mci,_=price_option(ctx,ci,T,"call"); mco,_=price_option(ctx,co,T,"call")
        epi=apply_execution_spread(mpi,"BUY", "put", ctx,pi,T)
        epo=apply_execution_spread(mpo,"SELL","put", ctx,po,T)
        eci=apply_execution_spread(mci,"BUY", "call",ctx,ci,T)
        eco=apply_execution_spread(mco,"SELL","call",ctx,co,T)
        debit_per=(epi-epo+eci-eco)*OPTION_MULTIPLIER
        if debit_per<=0: return None


        n=self._n_debit(debit_per,0.05,ctx=ctx)
        max_n=int(ctx.S*0.04*OPTION_MULTIPLIER/max(debit_per,0.01))
        n=min(n,max(max_n,1))
        if n<=0: return None

        preview = (self._commission_preview(n,epi) + self._commission_preview(n,epo)
                   + self._commission_preview(n,eci) + self._commission_preview(n,eco))
        if debit_per*n + preview > self.cash:
            n=max(1,int(self.cash*0.90/max(debit_per,0.01)))
            if n<=0: return None
            preview = (self._commission_preview(n,epi) + self._commission_preview(n,epo)
                       + self._commission_preview(n,eci) + self._commission_preview(n,eco))
            if debit_per*n + preview > self.cash:
                return None

        cpi=self._commission(n,epi); cpo=self._commission(n,epo)
        cci=self._commission(n,eci); cco=self._commission(n,eco)
        self.cash -= debit_per*n + cpi + cpo + cci + cco

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("put", pi,exp,n,+1,mpi,epi,cpi,ctx.date),
              OptionLeg("put", po,exp,n,-1,mpo,epo,cpo,ctx.date),
              OptionLeg("call",ci,exp,n,+1,mci,eci,cci,ctx.date),
              OptionLeg("call",co,exp,n,-1,mco,eco,cco,ctx.date)]
        pos=OptionPosition(legs=legs,open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_long_condor")
        return pos


class ShortCondorStrategy(_MultiLegStrategy):
    def __init__(self, width=0.10, expiration_days=30,
                 roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Cóndor Vendido {width*100:.0f}%",
                         description="Credit iron condor (4 strikes)")
        self.width=width; self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        w=self.width; tick=ctx.strike_tick
        pi=round_to_strike_tick(ctx.S*(1-0.5*w),tick)
        po=round_to_strike_tick(ctx.S*(1-1.5*w),tick)
        ci=round_to_strike_tick(ctx.S*(1+0.5*w),tick)
        co=round_to_strike_tick(ctx.S*(1+1.5*w),tick)
        if pi-po<tick: po=pi-tick
        if ci-pi<tick: ci=pi+tick
        if co-ci<tick: co=ci+tick

        mpi,_=price_option(ctx,pi,T,"put");  mpo,_=price_option(ctx,po,T,"put")
        mci,_=price_option(ctx,ci,T,"call"); mco,_=price_option(ctx,co,T,"call")
        epi=apply_execution_spread(mpi,"SELL","put", ctx,pi,T)
        epo=apply_execution_spread(mpo,"BUY", "put", ctx,po,T)
        eci=apply_execution_spread(mci,"SELL","call",ctx,ci,T)
        eco=apply_execution_spread(mco,"BUY", "call",ctx,co,T)
        credit_per=(epi-epo+eci-eco)*OPTION_MULTIPLIER
        if credit_per<=0: return None

        wing_put=pi-po; wing_call=co-ci; max_wing=max(wing_put,wing_call)
        margin=estimate_span_margin_condor(ctx.S,po,pi,ci,co,
                                           credit_per/OPTION_MULTIPLIER,sigma=ctx.sigma)
        n=min(int((self._sizing_base(ctx)*0.28)/max(margin,1.0)),40)
        n=self._apply_margin_cap(n,margin,ctx,wing_width=max_wing)
        if n<=0: return None

        preview = (self._commission_preview(n,epi) + self._commission_preview(n,epo)
                   + self._commission_preview(n,eci) + self._commission_preview(n,eco))
        if credit_per*n - preview <= 0: return None

        cpi=self._commission(n,epi); cpo=self._commission(n,epo)
        cci=self._commission(n,eci); cco=self._commission(n,eco)
        net=credit_per*n - cpi - cpo - cci - cco
        self.cash += net

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("put", pi,exp,n,-1,mpi,epi,cpi,ctx.date),
              OptionLeg("put", po,exp,n,+1,mpo,epo,cpo,ctx.date),
              OptionLeg("call",ci,exp,n,-1,mci,eci,cci,ctx.date),
              OptionLeg("call",co,exp,n,+1,mco,eco,cco,ctx.date)]
        fixed_m = float(estimate_span_margin_condor(
            ctx.S, po, pi, ci, co, credit_per/OPTION_MULTIPLIER, sigma=ctx.sigma)) * n
        pos=OptionPosition(legs=legs,open_date=ctx.date,fixed_margin=fixed_m)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_short_condor")
        return pos


class RatioBullStrategy(_TwoLegSpreadStrategy):
    def __init__(self, long_delta=0.60, short_delta=0.40, ratio=2,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Ratio Alcista {long_delta}Δ/{ratio}×{short_delta}Δ",
                         description=f"Short 1 call {long_delta}Δ + Long {ratio} calls {short_delta}Δ")
        self.long_delta=long_delta; self.short_delta=short_delta
        self.ratio=int(ratio); self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        Kl=find_strike_by_delta(ctx,self.long_delta, T,"call",strike_tick=ctx.strike_tick)
        Ks=find_strike_by_delta(ctx,self.short_delta,T,"call",strike_tick=ctx.strike_tick)
        if Kl is None or Ks is None: return None
        if Ks<=Kl: Ks=round_to_strike_tick(Kl*1.02,ctx.strike_tick)

        ml,_=price_option(ctx,Kl,T,"call"); ms,_=price_option(ctx,Ks,T,"call")

        el=apply_execution_spread(ml,"SELL","call",ctx,Kl,T)
        es=apply_execution_spread(ms,"BUY", "call",ctx,Ks,T)

        net_per=(el-self.ratio*es)*OPTION_MULTIPLIER
        naked=max(self.ratio-1,0)
        margin=max(estimate_reg_t_margin_per_short(ctx.S,Ks,"call",es,sigma=ctx.sigma)*naked,
                   abs(net_per)*2.0,1.0)

        n=min(int((self._sizing_base(ctx)*0.28)/max(margin,1.0)),40)
        n=self._apply_margin_cap(n,margin,ctx)
        if n<=0: return None

        cl_prev = self._commission_preview(n, el)
        cs_prev = self._commission_preview(n*self.ratio, es)
        cash_change_prev = net_per*n - cl_prev - cs_prev
        if self.cash + cash_change_prev < -0.05*self.initial_capital: return None

        cl=self._commission(n,el,note=f"sell call K={Kl:.2f}")
        cs=self._commission(n*self.ratio,es,note=f"buy call K={Ks:.2f}")
        self.cash += net_per*n - cl - cs

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("call",Kl,exp,n,             -1,ml,el,cl,ctx.date),
              OptionLeg("call",Ks,exp,n*self.ratio,  +1,ms,es,cs,ctx.date)]
        pos=OptionPosition(legs=legs,open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_ratio_bull")
        return pos


class RatioBearStrategy(_TwoLegSpreadStrategy):
    def __init__(self, long_delta=-0.60, short_delta=-0.40, ratio=2,
                 expiration_days=30, roll_enabled=True, roll_days_before=2):
        super().__init__(name=f"Ratio Bajista {long_delta}Δ/{ratio}×{short_delta}Δ",
                         description=f"Short 1 put {long_delta}Δ + Long {ratio} puts {short_delta}Δ")
        self.long_delta=long_delta; self.short_delta=short_delta
        self.ratio=int(ratio); self.expiration_days=expiration_days
        self.roll_enabled=roll_enabled; self.roll_days_before=roll_days_before

    def _build_position(self, ctx, T):
        Kl=find_strike_by_delta(ctx,self.long_delta, T,"put",strike_tick=ctx.strike_tick)
        Ks=find_strike_by_delta(ctx,self.short_delta,T,"put",strike_tick=ctx.strike_tick)
        if Kl is None or Ks is None: return None
        if Kl<=Ks: Kl=round_to_strike_tick(Ks*1.02,ctx.strike_tick)

        ml,_=price_option(ctx,Kl,T,"put"); ms,_=price_option(ctx,Ks,T,"put")

        el=apply_execution_spread(ml,"SELL","put",ctx,Kl,T)
        es=apply_execution_spread(ms,"BUY", "put",ctx,Ks,T)
        net_per=(el-self.ratio*es)*OPTION_MULTIPLIER
        naked=max(self.ratio-1,0)
        margin=max(estimate_reg_t_margin_per_short(ctx.S,Ks,"put",es,sigma=ctx.sigma)*naked,
                   abs(net_per)*2.0,1.0)

        n=min(int((self._sizing_base(ctx)*0.28)/max(margin,1.0)),40)
        n=self._apply_margin_cap(n,margin,ctx)
        if n<=0: return None

        cl_prev = self._commission_preview(n, el)
        cs_prev = self._commission_preview(n*self.ratio, es)
        cash_change_prev = net_per*n - cl_prev - cs_prev
        if self.cash + cash_change_prev < -0.05*self.initial_capital: return None

        cl=self._commission(n,el,note=f"sell put K={Kl:.2f}")
        cs=self._commission(n*self.ratio,es,note=f"buy put K={Ks:.2f}")
        self.cash += net_per*n - cl - cs

        exp=expiration_date_from_days(ctx.date,int(T*CALENDAR_DAYS_PER_YEAR))
        legs=[OptionLeg("put",Kl,exp,n,            -1,ml,el,cl,ctx.date),
              OptionLeg("put",Ks,exp,n*self.ratio, +1,ms,es,cs,ctx.date)]
        pos=OptionPosition(legs=legs,open_date=ctx.date)
        for leg in legs:
            self._register_open_trade(leg,ctx,pos.cycle_id,"open_ratio_bear")
        return pos


class Backtester:
    def __init__(self):
        self.strategies: dict = {}
        self.data = None
        self.all_metrics: dict = {}

    def add(self, s: BaseStrategy) -> "Backtester":
        self.strategies[s.name] = s
        return self

    def load(self, ticker: str, start: str, end: str) -> "Backtester":
        self.data = DataManager.download(ticker, start, end)
        return self

    def run(self, capital: float = 100_000) -> "Backtester":
        if self.data is None:
            raise RuntimeError("Primero carga datos con .load()")

        _binomial_cached.cache_clear()
        reset_spread_audit()

        if not any(type(s).__name__ == "BuyAndHoldStrategy"
                   for s in self.strategies.values()):
            bh = BuyAndHoldStrategy()
            self.strategies = {bh.name: bh, **self.strategies}

        iv_src   = self.data["iv_source"].iloc[0]   if "iv_source"   in self.data.columns else "GJR_GARCH"
        sk_src   = self.data["skew_source"].iloc[0] if "skew_source" in self.data.columns else "GJR_GARCH"
        tk_src   = self.data["ticker"].iloc[0]      if "ticker"      in self.data.columns else "?"
        tick_s0  = self.data["S"].iloc[0]           if self.data is not None else 0
        stk_tick = get_strike_tick(tk_src, tick_s0)

        print(f"\n{'='*70}")
        print(f"🚀 BACKTEST v34.7  Capital=${capital:,.0f}  Estrategias={len(self.strategies)}")
        print(f"   Ticker: {tk_src} | Strike tick: ${stk_tick} | IV: {iv_src} | Skew: {sk_src}")
        print(f"   v28-29: IronButterflyFix|RegTFix|CashInterest|MarginFix|EndOfPeriod")
        print(f"   v30:    VIX real para índices | VVIX para skew")
        print(f"   v31:    Tercer viernes real | Strikes discretos")
        print(f"   v32:    SortinoFix | SizingFix | TradesAudit")
        print(f"   v33:    CycleIdFix | Warmup126 | Binomial5 | SkewCallOTM")
        print(f"   v34.0:  SpreadIVDyn | ProxyOI | TermStructure | EarningsEffect")
        print(f"   v34.5:  VRPFallback | SpreadAudit | EarnVolByIV | IBMargin")
        print(f"           | RollSlipByDTE | MultiShortCover | NoPhantomCommish")
        print(f"           | MtMIntrinsicFloor | MarginCapNoForce")
        print(f"   v34.7:  NoCashInterest | LongITMAssignCost | TermClipFix")
        print(f"           | SpreadOTMByLiquidity | SkewTanhSmooth")
        print(f"{'='*70}")

        for name, strat in self.strategies.items():
            print(f"\n  ▶ {name}")
            try:
                strat.initialize(capital)
                strat.execute(self.data)
                m = strat.metrics()
                self.all_metrics[name] = m
                bankrupt_str = " [BANCARROTA]" if m.get("Bancarrota", False) else ""
                print(f"    Retorno:{m['Retorno Total']:+.1%}  "
                      f"Sharpe:{m['Sharpe Ratio']:.2f}  "
                      f"MaxDD:{m['Max Drawdown']:.1%}  "
                      f"Ciclos:{int(m['Nº Ciclos'])}  "
                      f"Ctos:{m['Contratos Abiertos Medio']:.1f}  "
                      f"Comis:${m['Comisiones Totales ($)']:,.0f}"
                      f"{bankrupt_str}")
                if getattr(strat, '_warnings', []):
                    print(f"    ⚠️  {len(strat._warnings)} warnings (strike fallbacks)")
            except Exception as e:
                print(f"    ❌ {e}")
                import traceback; traceback.print_exc()

        report_spread_audit()
        return self

    def report(self) -> pd.DataFrame:
        if not self.all_metrics:
            print("⚠️  Sin métricas.")
            return pd.DataFrame()
        df = pd.DataFrame(self.all_metrics).T
        self._print_table(df)
        self._save_csvs()
        self._delta_equivalence_analysis(df)
        self._plot_dashboard(df)
        self._plot_returns_tab()
        self._plot_risk_tab()
        return df

    def _delta_equivalence_analysis(self, df: pd.DataFrame):
        print(f"\n{'='*90}")
        print("📐 ANÁLISIS DE DELTA EQUIVALENTE (ex-post)")
        print(f"{'='*90}")
        bh_name  = next((n for n,s in self.strategies.items()
                         if type(s).__name__=="BuyAndHoldStrategy"), None)
        bh_delta = None
        if bh_name and self.strategies[bh_name].portfolio_history:
            ph  = self.strategies[bh_name].portfolio_history
            S0  = ph[0].get("S", None)
            cap = self.strategies[bh_name].initial_capital
            if S0 and S0>0:
                bh_delta = cap/S0
                if not (10<=bh_delta<=10_000): bh_delta=None

        if bh_delta is None:
            print("  ⚠️  No se pudo calcular delta de referencia del Buy & Hold.")
            return

        print(f"  Delta de referencia (B&H): {bh_delta:.1f} acciones equivalentes\n")
        rows=[]; num_df=df.apply(pd.to_numeric,errors="coerce")
        print(f"  {'Estrategia':<38} {'Delta media':>12} {'Ret.anual':>10} {'Ret.Δ-adj':>11} {'Ratio':>8} {'Nota'}")
        print(f"  {'-'*95}")

        for name,strat in self.strategies.items():
            if not strat.portfolio_history or type(strat).__name__=="BuyAndHoldStrategy": continue
            deltas=[h.get("net_delta",0) for h in strat.portfolio_history if h.get("n_open",0)>0]
            delta_mean=float(np.mean(deltas)) if deltas else 0.0
            ret_ann=float(num_df.loc[name,"Retorno Anual"]) if name in num_df.index and "Retorno Anual" in num_df.columns else 0.0
            abs_delta=abs(delta_mean)
            if abs_delta<5:
                ret_adj=ret_ann; ratio=float("nan"); nota="delta-neutral"
            else:
                scale=abs(bh_delta)/abs_delta; ret_adj=ret_ann*scale
                ratio=ret_adj/(ret_ann if abs(ret_ann)>0.001 else 0.001); nota=f"×{scale:.2f} escala"
            rows.append({"Estrategia":name,"Delta media":round(delta_mean,2),
                         "Retorno Anual (%)":round(ret_ann*100,2),
                         "Ret. Delta-ajustado (%)":round(ret_adj*100,2) if not np.isnan(ret_adj) else float("nan"),
                         "Factor escala":round(abs(bh_delta)/max(abs_delta,0.01),3),"Nota":nota})
            ret_adj_str=f"{ret_adj:+.2%}" if not np.isnan(ret_adj) else "N/A"
            ratio_str=f"{ratio:.2f}×" if not np.isnan(ratio) else "—"
            print(f"  {name:<38} {delta_mean:>+11.1f} {ret_ann:>+9.2%} {ret_adj_str:>11} {ratio_str:>8}  {nota}")

        if rows:
            pd.DataFrame(rows).to_csv("delta_equivalente.csv",index=False,float_format="%.4f")
            print(f"\n  💾 delta_equivalente.csv guardado")
        print(f"{'='*90}")

    def _print_table(self, df):
        print(f"\n{'='*100}\n📊 MÉTRICAS COMPARATIVAS v34.7\n{'='*100}")
        fmt={
            "Retorno Total":"{:+.2%}","Retorno Anual":"{:+.2%}",
            "Volatilidad Anual":"{:.2%}","Sharpe Ratio":"{:.3f}",
            "Sortino Ratio":"{:.3f}","Calmar Ratio":"{:.3f}",
            "Max Drawdown":"{:.2%}","VaR 95% (1d, $)":"${:,.0f}",
            "ES 95% (1d, $)":"${:,.0f}","Win Rate":"{:.1%}",
            "Profit Factor":"{:.2f}","Nº Ciclos":"{:.0f}",
            "Roll Trades":"{:.0f}","Comisiones Totales ($)":"${:,.0f}",
            "Impacto Comisiones":"{:.2%}","Theta Neto $/día":"{:.2f}",
            "Vega Neto $/1pp":"{:.2f}","Delta Neto Medio":"{:.3f}",
            "Margen Medio (%)":"{:.1%}","Capital en Riesgo Medio (%)":"{:.1%}",
            "Contratos Abiertos Medio":"{:.1f}",
        }
        disp=df.copy()
        for col,f in fmt.items():
            if col in disp.columns:
                disp[col]=disp[col].apply(
                    lambda x: f.format(float(x)) if pd.notna(x) and isinstance(x,(int,float)) else "—")
        key_cols=[c for c in fmt if c in disp.columns]
        pd.set_option("display.max_columns",None); pd.set_option("display.width",140)
        print(disp[key_cols])
        pd.reset_option("display.max_columns"); pd.reset_option("display.width")

        num=df.copy().apply(pd.to_numeric,errors="coerce")
        print(f"\n{'-'*70}\n🏆 RANKINGS")
        for label,col,op in [
            ("Mejor Retorno Anual","Retorno Anual","max"),
            ("Mejor Sharpe","Sharpe Ratio","max"),
            ("Menor Max Drawdown","Max Drawdown","max"),
            ("Mayor Win Rate","Win Rate","max"),
            ("Menores Comisiones","Comisiones Totales ($)","min"),
        ]:
            if col in num.columns:
                idx=num[col].idxmax() if op=="max" else num[col].idxmin()
                if pd.notna(num.loc[idx,col]):
                    print(f"  {label:25s}: {idx}  ({num.loc[idx,col]:.3f})")

    def _save_csvs(self):
        print("\n💾 Guardando CSVs…")
        pd.DataFrame(self.all_metrics).T.to_csv("metricas_completas_v34_7.csv", float_format="%.6f")
        print("  ✓ metricas_completas_v34_7.csv")

        bh_strat = None; bh_name = None; opt_strats = {}
        for name, strat in self.strategies.items():
            if type(strat).__name__ == "BuyAndHoldStrategy":
                bh_strat = strat; bh_name = name
            else:
                opt_strats[name] = strat

        for name, strat in opt_strats.items():
            safe = (name.replace(" ","_").replace("/","_")
                    .replace("Δ","d").replace("%","pct").replace("×","x"))
            if strat.trades:
                pd.DataFrame(strat.trades).to_csv(f"trades_{safe}.csv", index=False)
            if strat.portfolio_history:
                pd.DataFrame(strat.portfolio_history).to_csv(
                    f"portfolio_{safe}.csv", index=False)
                ph = strat.portfolio_history; rows_g = []
                for h in ph:
                    n = max(h.get("n_open", 0), 1)
                    has_pos = h.get("n_open", 0) > 0
                    rows_g.append({
                        "date": h["date"].strftime("%Y-%m-%d") if hasattr(h["date"], "strftime") else str(h["date"])[:10],
                        "S":            round(h.get("S",          0.0), 4),
                        "iv_atm":       round(h.get("iv",         0.0), 6),
                        "n_contratos":  h.get("n_open", 0),
                        "net_delta":    round(h.get("net_delta",  0.0), 4),
                        "net_gamma":    round(h.get("net_gamma",  0.0), 6),
                        "net_theta":    round(h.get("net_theta",  0.0), 4),
                        "net_vega":     round(h.get("net_vega",   0.0), 4),
                        "delta_por_cto":round(h.get("net_delta",  0.0) / n, 4) if has_pos else 0.0,
                        "theta_por_cto":round(h.get("net_theta",  0.0) / n, 4) if has_pos else 0.0,
                        "vega_por_cto": round(h.get("net_vega",   0.0) / n, 4) if has_pos else 0.0,
                        "total_capital":round(h.get("total",      0.0), 2),
                        "cash":         round(h.get("cash",       0.0), 2),
                        "mtm":          round(h.get("mtm",        0.0), 2),
                        "margin_pct":   round(h.get("margin_pct", 0.0), 6),
                        "rf":           round(h.get("rf",         0.0), 6),
                    })
                pd.DataFrame(rows_g).to_csv(f"greeks_{safe}.csv", index=False)

        if bh_strat is not None:
            lines_out = []
            lines_out.append("# SECCIÓN 1: Portfolio diario Buy & Hold")
            if bh_strat.portfolio_history:
                pf_df = pd.DataFrame(bh_strat.portfolio_history)
                if "date" in pf_df.columns:
                    pf_df["date"] = pf_df["date"].apply(
                        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10])
                lines_out.append(pf_df.to_csv(index=False).strip())
            else:
                lines_out.append("(sin datos de portfolio)")

            lines_out.append("\n\n# SECCIÓN 2: Trades Buy & Hold")
            if bh_strat.trades:
                tr_df = pd.DataFrame(bh_strat.trades)
                if "date" in tr_df.columns:
                    tr_df["date"] = tr_df["date"].apply(
                        lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10])
                lines_out.append(tr_df.to_csv(index=False).strip())
            else:
                lines_out.append("(sin trades registrados)")

            lines_out.append("\n\n# SECCIÓN 3: Métricas comparativas")
            lines_out.append("# Formato: porcentajes en %, ratios con 3 decimales, $ en dólares.")
            lines_out.append("# N/A en Greeks del Buy & Hold (no tiene posiciones de opciones).")

            all_met = {}
            if bh_name and bh_name in self.all_metrics:
                all_met[bh_name] = self.all_metrics[bh_name]
            for n in opt_strats:
                if n in self.all_metrics:
                    all_met[n] = self.all_metrics[n]

            if all_met:
                met_df = pd.DataFrame(all_met).T.T
                greek_rows = [
                    "Delta Neto Medio", "Gamma Neto Medio",
                    "Theta Neto $/día", "Vega Neto $/1pp",
                    "Contratos Abiertos Medio", "Roll Trades",
                    "Margen Medio (%)",
                ]
                bh_col = bh_name if bh_name in met_df.columns else None
                if bh_col:
                    for row in greek_rows:
                        if row in met_df.index:
                            met_df.loc[row, bh_col] = float("nan")

                pct_rows   = {"Retorno Total","Retorno Anual","Volatilidad Anual",
                               "Max Drawdown","VaR 95% (1d, %)","ES 95% (1d, %)",
                               "Win Rate","Impacto Comisiones","Margen Medio (%)",
                               "Capital en Riesgo Medio (%)"}
                money_rows = {"Avg Win ($)","Avg Loss ($)","Comisiones Totales ($)",
                               "Theta Neto $/día","Vega Neto $/1pp",
                               "VaR 95% (1d, $)","ES 95% (1d, $)"}
                int_rows   = {"Nº Ciclos","Nº Wins","Nº Losses","Días Positivos",
                               "Días Negativos","Roll Trades","Contratos Abiertos Medio"}
                dec3_rows  = {"Sharpe Ratio","Sortino Ratio","Calmar Ratio",
                               "Profit Factor","Delta Neto Medio","Gamma Neto Medio",
                               "Asimetría","Curtosis"}

                formatted_rows = []
                for metric_name, row in met_df.iterrows():
                    fmt_row = {"Métrica": metric_name}
                    for col in met_df.columns:
                        val = row[col]
                        if pd.isna(val):
                            fmt_row[col] = "N/A"
                        elif metric_name in pct_rows:
                            try:    fmt_row[col] = f"{float(val)*100:.2f}%"
                            except: fmt_row[col] = str(val)
                        elif metric_name in money_rows:
                            try:    fmt_row[col] = f"{float(val):,.2f}"
                            except: fmt_row[col] = str(val)
                        elif metric_name in int_rows:
                            try:    fmt_row[col] = f"{int(round(float(val)))}"
                            except: fmt_row[col] = str(val)
                        elif metric_name in dec3_rows:
                            try:    fmt_row[col] = f"{float(val):.3f}"
                            except: fmt_row[col] = str(val)
                        else:
                            try:    fmt_row[col] = f"{float(val):.4f}"
                            except: fmt_row[col] = str(val)
                    formatted_rows.append(fmt_row)

                fmt_df = pd.DataFrame(formatted_rows).set_index("Métrica")
                lines_out.append(fmt_df.to_csv())
            else:
                lines_out.append("(sin métricas disponibles)")

            with open("buy_and_hold_completo.csv", "w", encoding="utf-8") as f:
                f.write("\n".join(lines_out))
            print("  ✓ buy_and_hold_completo.csv  (portfolio + trades + métricas comparativas)")

        if SPREAD_SATURATION_LOG:
            pd.DataFrame(SPREAD_SATURATION_LOG).to_csv(
                "spread_saturation_log.csv", index=False, float_format="%.4f")
            print(f"  ✓ spread_saturation_log.csv ({len(SPREAD_SATURATION_LOG)} registros)")

        print(f"  ✓ {len(self.strategies)} estrategias exportadas")
        print(f"  ✓ greeks_<estrategia>.csv — Greeks diarios con contexto")

    @staticmethod
    def _fmt_dates(ax, dates):
        try:
            loc=mdates.AutoDateLocator(minticks=4,maxticks=10)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            plt.setp(ax.xaxis.get_majorticklabels(),rotation=30,ha="right")
        except Exception: pass

    def _color_cycle(self,n):
        cmap=plt.cm.get_cmap("tab20",max(n,1))
        return [cmap(i) for i in range(n)]

    def _portfolio_series_all(self):
        return {name:strat.portfolio_series()
                for name,strat in self.strategies.items()
                if len(strat.portfolio_series())>1}

    def _cum_returns(self,ser): return ser/ser.iloc[0]
    def _drawdown(self,ser):
        cum=self._cum_returns(ser); peak=cum.cummax()
        return (cum-peak)/peak

    def _plot_dashboard(self,df):
        try: plt.style.use("seaborn-v0_8-darkgrid")
        except Exception: pass
        n_strats=len(self.strategies); colors=self._color_cycle(n_strats+1)
        portfolios=self._portfolio_series_all()
        fig=plt.figure(figsize=(26,18))
        gs=fig.add_gridspec(3,3,hspace=0.50,wspace=0.32)

        ax1=fig.add_subplot(gs[0,:2])
        if self.data is not None:
            d=pd.to_datetime(self.data["Date"]); S=self.data["S"]
            ax1.plot(d,S,color="#1f77b4",lw=2,label="Precio",zorder=4)
            for ma,col,ls in [("MA_50","#ff7f0e","--"),("MA_200","#d62728","--")]:
                if ma in self.data.columns: ax1.plot(d,self.data[ma],ls,color=col,lw=1.2,alpha=0.75,label=ma)
            if "BB_upper" in self.data.columns:
                ax1.fill_between(d,self.data["BB_lower"],self.data["BB_upper"],alpha=0.10,color="steelblue",label="Bollinger ±2σ")
            ax1.set_title("SUBYACENTE — Precio & Medias Móviles",fontweight="bold",fontsize=12)
            ax1.set_ylabel("Precio ($)"); ax1.legend(fontsize=8,ncol=4,loc="upper left")
            ax1.grid(alpha=0.3); self._fmt_dates(ax1,d)

        ax2=fig.add_subplot(gs[0,2])
        if self.data is not None:
            d=pd.to_datetime(self.data["Date"])
            iv_src=self.data.get("iv_source",pd.Series(["GJR_GARCH"]*len(self.data)))
            iv_label="VIX real" if str(iv_src.iloc[0])=="VIX_real" else "IV proxy (GJR-GARCH)"
            ax2.plot(d,self.data["volatility_implied"]*100,color="crimson",lw=1.8,label=iv_label)
            ax2.plot(d,self.data["volatility_realized"]*100,color="steelblue",lw=1.5,alpha=0.7,label="HV 30d")
            ax2.fill_between(d,self.data["volatility_realized"]*100,self.data["volatility_implied"]*100,
                              alpha=0.15,color="crimson",label="VRP")
            ax2.set_title("IV proxy vs HV realizada (%)",fontweight="bold",fontsize=11)
            ax2.set_ylabel("Volatilidad (%)"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
            self._fmt_dates(ax2,d)

        ax4=fig.add_subplot(gs[1,:])
        if self.data is not None:
            bh_rets=pd.to_numeric(self.data["returns"],errors="coerce").fillna(0)
            bh_cum=(1+bh_rets).cumprod()
            ax4.plot(pd.to_datetime(self.data["Date"]),bh_cum,"--",color="black",lw=2.5,
                      alpha=0.55,label="Buy & Hold (subyacente)",zorder=6)
        for ci,(name,ser) in enumerate(portfolios.items()):
            cum=self._cum_returns(ser)
            ax4.plot(cum.index,cum.values,lw=2.2,color=colors[ci],label=name[:30],alpha=0.88,zorder=4)
            ax4.annotate(f"{cum.iloc[-1]:.2f}×",xy=(cum.index[-1],cum.iloc[-1]),
                          xytext=(5,0),textcoords="offset points",fontsize=7,color=colors[ci],va="center")
        ax4.axhline(1,color="gray",ls=":",lw=1,alpha=0.5)
        ax4.set_title("RETORNOS ACUMULADOS — Todas las estrategias vs Buy & Hold",fontweight="bold",fontsize=12)
        ax4.set_ylabel("Crecimiento del capital (×1 = capital inicial)")
        ax4.legend(fontsize=7.5,ncol=4,loc="upper left",framealpha=0.85); ax4.grid(alpha=0.3)
        if portfolios: self._fmt_dates(ax4,list(portfolios.values())[0].index)

        num_df=df.apply(pd.to_numeric,errors="coerce")
        ax5=fig.add_subplot(gs[2,0])
        if "Sharpe Ratio" in num_df.columns:
            vals=num_df["Sharpe Ratio"].dropna().sort_values()
            bc=["#2ca02c" if v>=0 else "#d62728" for v in vals]
            bars=ax5.barh(range(len(vals)),vals,color=bc,alpha=0.80,edgecolor="white",lw=0.5)
            ax5.set_yticks(range(len(vals))); ax5.set_yticklabels([n[:20] for n in vals.index],fontsize=8)
            ax5.axvline(0,color="black",lw=0.8); ax5.set_title("SHARPE RATIO",fontweight="bold",fontsize=11)
            ax5.set_xlabel("Sharpe"); ax5.grid(alpha=0.3,axis="x")
            for bar,v in zip(bars,vals):
                ax5.text(v+(0.03 if v>=0 else -0.08),bar.get_y()+bar.get_height()/2,f"{v:.2f}",va="center",fontsize=7.5)

        ax6=fig.add_subplot(gs[2,1])
        if "Max Drawdown" in num_df.columns:
            vals=num_df["Max Drawdown"].dropna().sort_values()
            cmap_dd=plt.cm.RdYlGn; norm_dd=plt.Normalize(vals.min(),0)
            bc_dd=[cmap_dd(norm_dd(v)) for v in vals]
            bars_dd=ax6.barh(range(len(vals)),vals*100,color=bc_dd,alpha=0.85,edgecolor="white",lw=0.5)
            ax6.set_yticks(range(len(vals))); ax6.set_yticklabels([n[:20] for n in vals.index],fontsize=8)
            ax6.set_title("MÁXIMO DRAWDOWN (%)",fontweight="bold",fontsize=11)
            ax6.set_xlabel("Drawdown (%)"); ax6.axvline(0,color="black",lw=0.8); ax6.grid(alpha=0.3,axis="x")
            for bar,v in zip(bars_dd,vals):
                ax6.text(v*100-0.5,bar.get_y()+bar.get_height()/2,f"{v:.1%}",va="center",ha="right",fontsize=7.5)

        ax7=fig.add_subplot(gs[2,2])
        if "Retorno Anual" in num_df.columns and "Volatilidad Anual" in num_df.columns:
            ret_s=num_df["Retorno Anual"].dropna(); vol_s=num_df["Volatilidad Anual"].dropna()
            idx=ret_s.index.intersection(vol_s.index)
            sharpe_s=num_df.get("Sharpe Ratio",pd.Series(dtype=float)).reindex(idx).fillna(0)
            sr=max(sharpe_s.max()-sharpe_s.min(),0.01)
            sizes=100+300*(sharpe_s-sharpe_s.min())/sr
            sc=ax7.scatter(vol_s[idx]*100,ret_s[idx]*100,c=sharpe_s.values,cmap="RdYlGn",
                            s=sizes,alpha=0.85,edgecolors="black",lw=0.6,zorder=5)
            plt.colorbar(sc,ax=ax7,label="Sharpe",shrink=0.8)
            for name in idx:
                ax7.annotate(name[:14],(vol_s[name]*100,ret_s[name]*100),
                              xytext=(4,3),textcoords="offset points",fontsize=7.0,alpha=0.85)
            ax7.axhline(0,color="gray",ls="--",lw=0.8)
            ax7.set_xlabel("Volatilidad Anual (%)"); ax7.set_ylabel("Retorno Anual (%)")
            ax7.set_title("RETORNO vs VOLATILIDAD\n(tamaño = Sharpe Ratio)",fontweight="bold",fontsize=11)
            ax7.grid(alpha=0.3)

        fig.suptitle("DASHBOARD BACKTESTING DE OPCIONES — v34.7  |  "
                     "Pricing: BSM-Merton + Binomial CRR  |  "
                     "v34.7: NoCashInterest + LongITMAssign + TermClipFix + SpreadOTMByLiquidity + SkewTanh",
                     fontsize=12,fontweight="bold",y=1.005)
        plt.savefig("dashboard_principal.png",dpi=200,bbox_inches="tight",facecolor="white")
        print("\n  ✓ dashboard_principal.png"); plt.close()

    def _plot_returns_tab(self):
        try: plt.style.use("seaborn-v0_8-darkgrid")
        except Exception: pass
        portfolios=self._portfolio_series_all()
        if not portfolios: return
        colors=self._color_cycle(len(portfolios)+1)
        num_df=pd.DataFrame(self.all_metrics).T.apply(pd.to_numeric,errors="coerce")
        fig=plt.figure(figsize=(26,20))
        gs=fig.add_gridspec(3,2,hspace=0.48,wspace=0.28,height_ratios=[2.2,1.2,1.2])

        ax1=fig.add_subplot(gs[0,:])
        if self.data is not None:
            bh_rets=pd.to_numeric(self.data["returns"],errors="coerce").fillna(0)
            bh_cum=(1+bh_rets).cumprod()
            ax1.plot(pd.to_datetime(self.data["Date"]),(bh_cum-1)*100,"--",color="black",
                      lw=2.5,alpha=0.50,label="Subyacente (Buy & Hold)",zorder=6)
        for ci,(name,ser) in enumerate(portfolios.items()):
            cum=self._cum_returns(ser); pct=(cum-1)*100
            ax1.plot(cum.index,pct.values,lw=2.5,color=colors[ci],label=name[:32],alpha=0.90,zorder=5)
            ax1.annotate(f"  {name[:18]}: {pct.iloc[-1]:+.1f}%",
                          xy=(cum.index[-1],pct.iloc[-1]),xytext=(8,0),textcoords="offset points",
                          fontsize=7.5,color=colors[ci],va="center",
                          bbox=dict(boxstyle="round,pad=0.15",fc="white",alpha=0.7,lw=0))
        ax1.axhline(0,color="black",lw=1,ls="--",alpha=0.4)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:+.0f}%"))
        ax1.set_title("RENDIMIENTO ACUMULADO DE TODAS LAS ESTRATEGIAS (%)",fontweight="bold",fontsize=14,pad=10)
        ax1.set_ylabel("Rentabilidad acumulada (%)"); ax1.legend(fontsize=8,ncol=3,loc="upper left")
        ax1.grid(alpha=0.3)
        if portfolios: self._fmt_dates(ax1,list(portfolios.values())[0].index)

        ax2=fig.add_subplot(gs[1,0])
        if "Retorno Anual" in num_df.columns:
            vals=num_df["Retorno Anual"].dropna().sort_values(ascending=False)
            bc=["#2ca02c" if v>=0 else "#d62728" for v in vals]
            bars=ax2.bar(range(len(vals)),vals*100,color=bc,alpha=0.82,edgecolor="white",lw=0.8)
            ax2.set_xticks(range(len(vals)))
            ax2.set_xticklabels([n[:16] for n in vals.index],rotation=35,ha="right",fontsize=8)
            ax2.axhline(0,color="black",lw=0.8); ax2.set_ylabel("Retorno Anual (%)")
            ax2.set_title("RETORNO ANUAL POR ESTRATEGIA",fontweight="bold",fontsize=11)
            ax2.grid(alpha=0.3,axis="y")
            for bar,v in zip(bars,vals):
                ax2.text(bar.get_x()+bar.get_width()/2,v*100+(0.3 if v>=0 else -0.8),
                          f"{v:+.1%}",ha="center",va="bottom" if v>=0 else "top",fontsize=7.5,fontweight="bold")

        ax3=fig.add_subplot(gs[1,1])
        for ci,(name,ser) in enumerate(portfolios.items()):
            dd=self._drawdown(ser)*100
            ax3.fill_between(dd.index,dd.values,0,color=colors[ci],alpha=0.22)
            ax3.plot(dd.index,dd.values,lw=1.5,color=colors[ci],alpha=0.80,label=name[:22])
        ax3.set_title("DRAWDOWN POR ESTRATEGIA (%)",fontweight="bold",fontsize=11)
        ax3.set_ylabel("Drawdown (%)")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0f}%"))
        ax3.legend(fontsize=7,ncol=2,loc="lower left"); ax3.grid(alpha=0.3)
        if portfolios: self._fmt_dates(ax3,list(portfolios.values())[0].index)

        ax4=fig.add_subplot(gs[2,0])
        for ci,(name,ser) in enumerate(portfolios.items()):
            log_r=np.log(ser/ser.shift(1)).dropna()*100
            if len(log_r)<5: continue
            ax4.hist(log_r,bins=50,alpha=0.30,color=colors[ci],density=True,label=name[:20])
            try:
                from scipy.stats import gaussian_kde
                kde=gaussian_kde(log_r); xs=np.linspace(log_r.min(),log_r.max(),300)
                ax4.plot(xs,kde(xs),color=colors[ci],lw=1.8,alpha=0.85)
            except Exception: pass
        ax4.axvline(0,color="black",lw=1,ls="--",alpha=0.5)
        ax4.set_xlabel("Retorno diario log (%)"); ax4.set_ylabel("Densidad")
        ax4.set_title("DISTRIBUCIÓN DE RETORNOS DIARIOS",fontweight="bold",fontsize=11)
        ax4.legend(fontsize=7,ncol=2); ax4.grid(alpha=0.3)

        ax5=fig.add_subplot(gs[2,1])
        monthly_matrix={}
        for name,ser in portfolios.items():
            mret=ser.resample("ME").last().pct_change().dropna()*100
            for ts,v in mret.items():
                label=ts.strftime("%Y-%m")
                if label not in monthly_matrix: monthly_matrix[label]={}
                monthly_matrix[label][name]=v
        if monthly_matrix:
            hmap_df=pd.DataFrame(monthly_matrix).T.sort_index()
            hmap_df.columns=[c[:16] for c in hmap_df.columns]
            vals_arr=hmap_df.values[np.isfinite(hmap_df.values)]
            vmax=max(abs(vals_arr).max(),5) if len(vals_arr)>0 else 5
            im=ax5.imshow(hmap_df.T.values,cmap="RdYlGn",aspect="auto",vmin=-vmax,vmax=vmax)
            plt.colorbar(im,ax=ax5,label="Ret. mensual (%)",shrink=0.85)
            ax5.set_xticks(range(len(hmap_df.index)))
            ax5.set_xticklabels(hmap_df.index,rotation=60,ha="right",fontsize=6.5)
            ax5.set_yticks(range(len(hmap_df.columns)))
            ax5.set_yticklabels(hmap_df.columns,fontsize=8)
            ax5.set_title("RETORNOS MENSUALES POR ESTRATEGIA (%)",fontweight="bold",fontsize=11)

        fig.suptitle("PESTAÑA DE RENDIMIENTOS — v34.7",fontsize=14,fontweight="bold",y=1.005)
        plt.savefig("rendimientos_estrategias.png",dpi=200,bbox_inches="tight",facecolor="white")
        print("  ✓ rendimientos_estrategias.png"); plt.close()

    def _plot_risk_tab(self):
        try: plt.style.use("seaborn-v0_8-darkgrid")
        except Exception: pass
        portfolios=self._portfolio_series_all()
        if not portfolios: return
        n=len(portfolios); colors=self._color_cycle(n)
        fig,axes=plt.subplots(2,2,figsize=(22,14))
        fig.suptitle("ANÁLISIS DE RIESGO — Margen · Greeks · Contratos Abiertos",
                     fontsize=14,fontweight="bold",y=1.01)

        ax1=axes[0,0]
        for ci,(name,strat) in enumerate(self.strategies.items()):
            if not strat.portfolio_history: continue
            ph=strat.portfolio_history
            dts=[h["date"] for h in ph]
            mgs=[h.get("margin",0.0)/max(strat.initial_capital,1.0)*100 for h in ph]
            ax1.plot(dts,mgs,lw=1.8,color=colors[ci%len(colors)],label=name[:22],alpha=0.85)
        ax1.axhline(MAX_MARGIN_PCT_EQUITY*100,color="red",ls="--",lw=1.5,label=f"Cap {MAX_MARGIN_PCT_EQUITY*100:.0f}%")
        ax1.set_title("MARGEN UTILIZADO (% del capital)",fontweight="bold")
        ax1.set_ylabel("Margen (%)"); ax1.legend(fontsize=7,ncol=2); ax1.grid(alpha=0.3)
        if portfolios: self._fmt_dates(ax1,list(portfolios.values())[0].index)

        ax2=axes[0,1]; has_any=False; max_contracts=0
        for ci,(name,strat) in enumerate(self.strategies.items()):
            if not strat.portfolio_history or name=="Buy & Hold": continue
            ph=strat.portfolio_history
            dts=[h["date"] for h in ph]; ns=[h.get("n_open",0) for h in ph]
            if max(ns,default=0)==0: continue
            max_contracts=max(max_contracts,max(ns))
            ax2.fill_between(dts,ns,alpha=0.25,color=colors[ci%len(colors)])
            ax2.plot(dts,ns,lw=1.5,color=colors[ci%len(colors)],label=name[:22],alpha=0.90)
            has_any=True
        ax2.set_title("CONTRATOS ABIERTOS POR DÍA",fontweight="bold"); ax2.set_ylabel("Nº contratos")
        if has_any and max_contracts>0: ax2.set_ylim(0,max_contracts*1.15)
        ax2.legend(fontsize=7,ncol=2); ax2.grid(alpha=0.3)
        if portfolios: self._fmt_dates(ax2,list(portfolios.values())[0].index)

        ax3=axes[1,0]
        num_df=pd.DataFrame(self.all_metrics).T.apply(pd.to_numeric,errors="coerce")
        if "Theta Neto $/día" in num_df.columns:
            vals=num_df["Theta Neto $/día"].dropna().sort_values()
            bc=["#2ca02c" if v>=0 else "#d62728" for v in vals]
            bars=ax3.barh(range(len(vals)),vals,color=bc,alpha=0.80,edgecolor="white",lw=0.5)
            ax3.set_yticks(range(len(vals))); ax3.set_yticklabels([n[:22] for n in vals.index],fontsize=8)
            ax3.axvline(0,color="black",lw=0.8)
            ax3.set_title("THETA NETO MEDIO ($/día por cartera)",fontweight="bold")
            ax3.set_xlabel("Theta $/día  (>0 = vendedor de tiempo)"); ax3.grid(alpha=0.3,axis="x")
            max_abs=max([abs(v) for v in vals],default=1.0)
            for bar,v in zip(bars,vals):
                offset=max_abs*0.02
                ax3.text(v+(offset if v>=0 else -offset),bar.get_y()+bar.get_height()/2,
                          f"{v:.2f}",va="center",ha="left" if v>=0 else "right",fontsize=7.5)

        ax4=axes[1,1]
        if "Delta Neto Medio" in num_df.columns:
            strat_names_d=[n for n in num_df.index if pd.notna(num_df.loc[n,"Delta Neto Medio"])]
            strat_names_d=sorted(strat_names_d,key=lambda n: float(num_df.loc[n,"Delta Neto Medio"]))
            y_pos=np.arange(len(strat_names_d))
            delta_vals=[float(num_df.loc[n,"Delta Neto Medio"]) for n in strat_names_d]
            bc_d=["#2ca02c" if v>20 else("#d62728" if v<-20 else "#aec7e8") for v in delta_vals]
            ax4.barh(y_pos,delta_vals,height=0.6,color=bc_d,alpha=0.82,edgecolor="white",lw=0.5)
            ax4.set_yticks(y_pos); ax4.set_yticklabels([n[:22] for n in strat_names_d],fontsize=8)
            ax4.axvline(0,color="black",lw=0.8)
            ax4.set_title("DELTA NETO MEDIO (exposición direccional)",fontweight="bold")
            ax4.set_xlabel("Delta neto total (>0 alcista, <0 bajista)")
            max_abs=max([abs(v) for v in delta_vals],default=1.0)
            for i,v in enumerate(delta_vals):
                offset=max_abs*0.01
                ax4.text(v+(offset if v>=0 else -offset),y_pos[i],f"{v:.0f}",
                          va="center",ha="left" if v>=0 else "right",fontsize=7.5)
            ax4.grid(alpha=0.3,axis="x")

        plt.tight_layout()
        plt.savefig("riesgo_greeks.png",dpi=200,bbox_inches="tight",facecolor="white")
        print("  ✓ riesgo_greeks.png"); plt.close()


def _yn(prompt: str, default: bool = True) -> bool:
    d = "sí" if default else "no"
    ans = input(f"  {prompt} [{d}]: ").strip().lower()
    if ans in ("s","si","sí","y","yes"): return True
    if ans in ("n","no"): return False
    return default

def _ask_delta(prompt: str, default: float) -> float:
    raw = input(f"    {prompt} [{default}]: ").strip()
    return parse_number(raw) if raw else default

def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"  {prompt} [{default}]: ").strip()
    try: return int(raw) if raw else default
    except ValueError: return default

def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    return parse_number(raw) if raw else default


def interactive_config():
    print("=" * 72)
    print("🎓  BACKTESTER DE OPCIONES v34.7")
    print("    v34.5 + cherry-pick: NoCashInterest | LongITMAssign | TermClipFix")
    print("                          SpreadOTMByLiquidity | SkewTanhSmooth")
    print("=" * 72)

    print("\n⚙️  PARÁMETROS DE TRADING")
    global USE_EXECUTION_SPREAD, COMMISSION_PER_CONTRACT
    USE_EXECUTION_SPREAD    = _yn("Usar bid-ask spread realista (recomendado)", True)
    COMMISSION_PER_CONTRACT = _ask_float("Comisión por contrato ($)", 0.65)

    print("\n📈 SELECCIÓN DE ACTIVO")
    all_t = []
    for cat, assets in DataManager.TICKERS.items():
        print(f"\n  ── {cat} ──")
        for name, tk in assets.items():
            idx = len(all_t) + 1
            print(f"    [{idx:2d}] {name:<28} ({tk})")
            all_t.append((name, tk))

    while True:
        try:
            sel = int(input(f"\n  Selecciona activo (1-{len(all_t)}): ").strip())
            if 1 <= sel <= len(all_t):
                ticker_name, ticker = all_t[sel - 1]
                break
        except (ValueError, IndexError): pass

    print("\n📅 PERIODO DE BACKTEST")
    print("  [1] 2020-2023   [2] 2021-2024   [3] 2018-2023")
    print("  [4] 2015-2023   [5] 2010-2023   [6] Personalizado")
    p = input("  Opción [2]: ").strip() or "2"
    periods = {"1":("2020-01-01","2023-12-31"),"2":("2021-01-01","2024-12-31"),
               "3":("2018-01-01","2023-12-31"),"4":("2015-01-01","2023-12-31"),
               "5":("2010-01-01","2023-12-31")}
    if p in periods: start, end = periods[p]
    else:
        start = input("  Fecha inicio (YYYY-MM-DD) [2021-01-01]: ").strip() or "2021-01-01"
        end   = input("  Fecha fin    (YYYY-MM-DD) [2024-12-31]: ").strip() or "2024-12-31"

    cap_inp = input("\n💰 Capital inicial [$100,000]: ").strip()
    capital = parse_number(cap_inp) or 100_000.0

    print("\n⏱  VENCIMIENTO Y ROLLING")
    dte          = _ask_int("Días hasta vencimiento (DTE)", 30)
    roll_enabled = _yn("¿Hacer rolling automático antes del vencimiento?", True)
    roll_days    = _ask_int("Días antes del vencimiento para iniciar el roll", 5) if roll_enabled else 0
    if roll_enabled and dte < roll_days + 3:
        print(f"  ⚠️  Advertencia: DTE={dte} con roll_days={roll_days} → posición abierta")
        print(f"     solo {dte - roll_days} días por ciclo. Considera DTE ≥ {roll_days + 7}.")

    print("\n🎯 SELECCIÓN DE ESTRATEGIAS")
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Buy & Hold se ejecuta siempre como benchmark               │")
    print("  │  [A] Simples (Call/Put comprada/vendida)                    │")
    print("  │  [B] Spreads (Bull, Bear, Ratio alcista/bajista)            │")
    print("  │  [C] Conos y Cuñas (Straddle/Strangle ±)                    │")
    print("  │  [D] Mariposas y Cóndores                                   │")
    print("  │  [E] TODAS las estrategias (18 total + benchmark)           │")
    print("  │  [F] Selección individual con deltas personalizados         │")
    print("  └─────────────────────────────────────────────────────────────┘")
    cat = input("  Categoría [E]: ").strip().upper() or "E"

    bt = Backtester()
    common = dict(expiration_days=dte, roll_enabled=roll_enabled, roll_days_before=roll_days)

    def add_single(StratClass, default_delta, opt_type):
        d = _ask_delta(f"Delta para {StratClass.__name__}", default_delta)
        bt.add(StratClass(delta=d, **common))

    def add_spread(StratClass, def_long, def_short, label_long, label_short):
        ld = _ask_delta(f"Delta LARGA para {StratClass.__name__} ({label_long})", def_long)
        sd = _ask_delta(f"Delta CORTA para {StratClass.__name__} ({label_short})", def_short)
        bt.add(StratClass(long_delta=ld, short_delta=sd, **common))

    def add_ratio(StratClass, def_long, def_short, def_ratio, lbl):
        ld = _ask_delta(f"Delta LARGA para {lbl}", def_long)
        sd = _ask_delta(f"Delta CORTA para {lbl}", def_short)
        r  = _ask_int(f"Ratio de ventas para {lbl}", def_ratio)
        bt.add(StratClass(long_delta=ld, short_delta=sd, ratio=r, **common))

    def add_strangle(StratClass, def_call, def_put, lbl):
        cd  = _ask_delta(f"Delta CALL para {lbl}", def_call)
        pd_ = _ask_delta(f"Delta PUT para {lbl}", def_put)
        bt.add(StratClass(call_delta=cd, put_delta=pd_, **common))

    def add_wing(StratClass, def_width, lbl):
        w = _ask_float(f"Ancho (width) para {lbl} [0.10 = 10%]", def_width)
        bt.add(StratClass(width=w, **common))

    def add_straddle_long():
        print("    Configuración Straddle Comprado:")
        delta = _ask_delta("Delta de cada pata (0.50=ATM exacto)", 0.50)
        calls = _yn("¿Incluir pata call?", True)
        puts  = _yn("¿Incluir pata put?",  True)
        pct   = _ask_float("% máximo del capital por ciclo (ej: 0.025 = 2.5%)", 0.025)
        bt.add(LongStraddleStrategy(delta=delta, use_calls=calls, use_puts=puts,
                                     max_pct_cap=pct, **common))

    def add_straddle_short():
        print("    Configuración Straddle Vendido:")
        delta = _ask_delta("Delta de cada pata (0.50=ATM exacto)", 0.50)
        calls = _yn("¿Incluir pata call?", True)
        puts  = _yn("¿Incluir pata put?",  True)
        bt.add(ShortStraddleStrategy(delta=delta, use_calls=calls, use_puts=puts, **common))

    def add_butterfly_long():
        print("    Configuración Mariposa Comprada:")
        otype = input("    Tipo: call / put / iron [call]: ").strip().lower() or "call"
        w     = _ask_float("Ancho simétrico de alas (ej: 0.10 = 10%)", 0.10)
        asym  = _yn("¿Alas asimétricas?", False)
        w_lo  = _ask_float("Ancho ala inferior", w) if asym else w
        w_hi  = _ask_float("Ancho ala superior", w) if asym else w
        bt.add(LongButterflyStrategy(option_type=otype, width=w,
                                      width_lower=w_lo, width_upper=w_hi, **common))

    def add_butterfly_short():
        print("    Configuración Mariposa Vendida:")
        otype = input("    Tipo: call / put [call]: ").strip().lower() or "call"
        w     = _ask_float("Ancho simétrico de alas (ej: 0.10 = 10%)", 0.10)
        asym  = _yn("¿Alas asimétricas?", False)
        w_lo  = _ask_float("Ancho ala inferior", w) if asym else w
        w_hi  = _ask_float("Ancho ala superior", w) if asym else w
        bt.add(ShortButterflyStrategy(option_type=otype, width=w,
                                       width_lower=w_lo, width_upper=w_hi, **common))

    def add_short_iron_butterfly():
        print("    Configuración Short Iron Butterfly (crédito: vende straddle ATM + compra alas):")
        w    = _ask_float("Ancho simétrico de alas (ej: 0.10 = 10%)", 0.10)
        asym = _yn("¿Alas asimétricas?", False)
        w_lo = _ask_float("Ancho ala inferior", w) if asym else w
        w_hi = _ask_float("Ancho ala superior", w) if asym else w
        bt.add(ShortIronButterflyStrategy(width=w, width_lower=w_lo, width_upper=w_hi, **common))

    def add_long_iron_butterfly():
        print("    Configuración Long Iron Butterfly (debit: compra straddle ATM + vende alas):")
        w    = _ask_float("Ancho simétrico de alas (ej: 0.10 = 10%)", 0.10)
        asym = _yn("¿Alas asimétricas?", False)
        w_lo = _ask_float("Ancho ala inferior", w) if asym else w
        w_hi = _ask_float("Ancho ala superior", w) if asym else w
        bt.add(LongButterflyStrategy(option_type="iron", width=w,
                                      width_lower=w_lo, width_upper=w_hi, **common))

    if cat in ("A", "E"):
        print("\n  📋 Configurando estrategias SIMPLES…")
        if cat == "E" or _yn("  ¿Incluir Call Comprada?", True):
            if cat == "E": bt.add(LongCallStrategy(**common))
            else: add_single(LongCallStrategy, 0.50, "call")
        if cat == "E" or _yn("  ¿Incluir Call Vendida?", True):
            if cat == "E": bt.add(ShortCallStrategy(**common))
            else: add_single(ShortCallStrategy, 0.30, "call")
        if cat == "E" or _yn("  ¿Incluir Put Comprada?", True):
            if cat == "E": bt.add(LongPutStrategy(**common))
            else: add_single(LongPutStrategy, -0.50, "put")
        if cat == "E" or _yn("  ¿Incluir Put Vendida?", True):
            if cat == "E": bt.add(ShortPutStrategy(**common))
            else: add_single(ShortPutStrategy, -0.30, "put")

    if cat in ("B", "E"):
        print("\n  📋 Configurando SPREADS…")
        if cat == "E" or _yn("  ¿Incluir Bull Spread?", True):
            if cat == "E": bt.add(BullSpreadStrategy(**common))
            else: add_spread(BullSpreadStrategy, 0.60, 0.30, "call ITM", "call OTM")
        if cat == "E" or _yn("  ¿Incluir Bear Spread?", True):
            if cat == "E": bt.add(BearSpreadStrategy(**common))
            else: add_spread(BearSpreadStrategy, -0.60, -0.30, "put ITM", "put OTM")
        if cat == "E" or _yn("  ¿Incluir Ratio Alcista?", True):
            if cat == "E": bt.add(RatioBullStrategy(**common))
            else: add_ratio(RatioBullStrategy, 0.60, 0.40, 2, "Ratio Alcista")
        if cat == "E" or _yn("  ¿Incluir Ratio Bajista?", True):
            if cat == "E": bt.add(RatioBearStrategy(**common))
            else: add_ratio(RatioBearStrategy, -0.60, -0.40, 2, "Ratio Bajista")

    if cat in ("C", "E"):
        print("\n  📋 Configurando CONOS Y CUÑAS…")
        if cat == "E" or _yn("  ¿Incluir Straddle Comprado?", True):
            if cat == "E": bt.add(LongStraddleStrategy(**common))
            else: add_straddle_long()
        if cat == "E" or _yn("  ¿Incluir Straddle Vendido?", True):
            if cat == "E": bt.add(ShortStraddleStrategy(**common))
            else: add_straddle_short()
        if cat == "E" or _yn("  ¿Incluir Strangle Comprado?", True):
            if cat == "E": bt.add(LongStrangleStrategy(**common))
            else: add_strangle(LongStrangleStrategy, 0.25, -0.25, "Strangle Comprado")
        if cat == "E" or _yn("  ¿Incluir Strangle Vendido?", True):
            if cat == "E": bt.add(ShortStrangleStrategy(**common))
            else: add_strangle(ShortStrangleStrategy, 0.25, -0.25, "Strangle Vendido")

    if cat in ("D", "E"):
        print("\n  📋 Configurando MARIPOSAS Y CÓNDORES…")
        if cat == "E" or _yn("  ¿Incluir Mariposa Comprada?", True):
            if cat == "E": bt.add(LongButterflyStrategy(**common))
            else: add_butterfly_long()
        if cat == "E" or _yn("  ¿Incluir Mariposa Vendida?", True):
            if cat == "E": bt.add(ShortButterflyStrategy(**common))
            else: add_butterfly_short()
        if cat == "E" or _yn("  ¿Incluir Long Iron Butterfly?", True):
            if cat == "E": bt.add(LongButterflyStrategy(option_type="iron", **common))
            else: add_long_iron_butterfly()
        if cat == "E" or _yn("  ¿Incluir Short Iron Butterfly?", True):
            if cat == "E": bt.add(ShortIronButterflyStrategy(**common))
            else: add_short_iron_butterfly()
        if cat == "E" or _yn("  ¿Incluir Cóndor Comprado?", True):
            if cat == "E": bt.add(LongCondorStrategy(**common))
            else: add_wing(LongCondorStrategy, 0.10, "Cóndor Comprado")
        if cat == "E" or _yn("  ¿Incluir Cóndor Vendido?", True):
            if cat == "E": bt.add(ShortCondorStrategy(**common))
            else: add_wing(ShortCondorStrategy, 0.10, "Cóndor Vendido")

    if cat == "F":
        print("\n  📋 SELECCIÓN INDIVIDUAL — configura cada estrategia")
        print("  ℹ️  Buy & Hold se ejecuta siempre como benchmark")
        all_options = [
            ("Call Comprada",        lambda: add_single(LongCallStrategy,   0.50, "call")),
            ("Call Vendida",         lambda: add_single(ShortCallStrategy,  0.30, "call")),
            ("Put Comprada",         lambda: add_single(LongPutStrategy,   -0.50, "put")),
            ("Put Vendida",          lambda: add_single(ShortPutStrategy,  -0.30, "put")),
            ("Bull Spread",          lambda: add_spread(BullSpreadStrategy,  0.60,  0.30, "call ITM", "call OTM")),
            ("Bear Spread",          lambda: add_spread(BearSpreadStrategy, -0.60, -0.30, "put ITM",  "put OTM")),
            ("Ratio Alcista",        lambda: add_ratio(RatioBullStrategy,    0.60,  0.40, 2, "Ratio Alcista")),
            ("Ratio Bajista",        lambda: add_ratio(RatioBearStrategy,   -0.60, -0.40, 2, "Ratio Bajista")),
            ("Straddle Comprado",    lambda: add_straddle_long()),
            ("Straddle Vendido",     lambda: add_straddle_short()),
            ("Strangle Comprado",    lambda: add_strangle(LongStrangleStrategy,  0.25, -0.25, "Strangle Comprado")),
            ("Strangle Vendido",     lambda: add_strangle(ShortStrangleStrategy, 0.25, -0.25, "Strangle Vendido")),
            ("Mariposa Comprada",    lambda: add_butterfly_long()),
            ("Mariposa Vendida",     lambda: add_butterfly_short()),
            ("Long Iron Butterfly",  lambda: add_long_iron_butterfly()),
            ("Short Iron Butterfly", lambda: add_short_iron_butterfly()),
            ("Cóndor Comprado",      lambda: add_wing(LongCondorStrategy,  0.10, "Cóndor Comprado")),
            ("Cóndor Vendido",       lambda: add_wing(ShortCondorStrategy, 0.10, "Cóndor Vendido")),
        ]
        print()
        for i, (name, _) in enumerate(all_options, 1):
            print(f"    [{i:2d}] {name}")
        sel_str = input("\n  Escribe los números separados por comas (ej: 1,3,5-8): ").strip()
        selected = set()
        for part in sel_str.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    a, b = part.split("-")
                    selected.update(range(int(a), int(b) + 1))
                except ValueError: pass
            elif part.isdigit():
                selected.add(int(part))
        for idx in sorted(selected):
            if 1 <= idx <= len(all_options):
                name, fn = all_options[idx - 1]
                print(f"\n  ▸ Configurando: {name}")
                fn()

    print(f"\n✅ Configuración completada:")
    print(f"   Activo:      {ticker_name} ({ticker})")
    print(f"   Periodo:     {start} → {end}")
    print(f"   Capital:     ${capital:,.0f}")
    print(f"   DTE:         {dte} días  |  Rolling: {'Sí (' + str(roll_days) + 'd antes)' if roll_enabled else 'No'}")
    print(f"   Estrategias: {len(bt.strategies)} + Buy & Hold (benchmark automático)")
    return bt, ticker, start, end, capital


def run_from_config(config: dict) -> "Backtester":
    global USE_EXECUTION_SPREAD, COMMISSION_PER_CONTRACT, USE_REAL_VIX
    USE_EXECUTION_SPREAD    = config.get("execution_spread", True)
    COMMISSION_PER_CONTRACT = config.get("commission", 0.65)
    USE_REAL_VIX            = config.get("use_real_vix", True)

    dte   = config.get("dte",  30)
    roll  = config.get("roll", True)
    rdays = config.get("roll_days", 5)
    common = dict(expiration_days=dte, roll_enabled=roll, roll_days_before=rdays)

    bt  = Backtester()
    MAP = {
        "long_call":            lambda: bt.add(LongCallStrategy(**common)),
        "short_call":           lambda: bt.add(ShortCallStrategy(**common)),
        "long_put":             lambda: bt.add(LongPutStrategy(**common)),
        "short_put":            lambda: bt.add(ShortPutStrategy(**common)),
        "bull_spread":          lambda: bt.add(BullSpreadStrategy(**common)),
        "bear_spread":          lambda: bt.add(BearSpreadStrategy(**common)),
        "ratio_bull":           lambda: bt.add(RatioBullStrategy(**common)),
        "ratio_bear":           lambda: bt.add(RatioBearStrategy(**common)),
        "long_straddle":        lambda: bt.add(LongStraddleStrategy(**common)),
        "short_straddle":       lambda: bt.add(ShortStraddleStrategy(**common)),
        "long_strangle":        lambda: bt.add(LongStrangleStrategy(**common)),
        "short_strangle":       lambda: bt.add(ShortStrangleStrategy(**common)),
        "long_butterfly":       lambda: bt.add(LongButterflyStrategy(**common)),
        "short_butterfly":      lambda: bt.add(ShortButterflyStrategy(**common)),
        "long_iron_butterfly":  lambda: bt.add(LongButterflyStrategy(option_type="iron",**common)),
        "short_iron_butterfly": lambda: bt.add(ShortIronButterflyStrategy(**common)),
        "long_condor":          lambda: bt.add(LongCondorStrategy(**common)),
        "short_condor":         lambda: bt.add(ShortCondorStrategy(**common)),
    }
    for key in config.get("strategies", ["all"]):
        if key == "all":
            for fn in MAP.values(): fn()
            break
        if key in MAP: MAP[key]()

    (bt.load(config["ticker"], config["start"], config["end"])
       .run(capital=config.get("capital", 100_000)))
    bt.report()
    return bt


def main():
    print("=" * 72)
    print("🎓  BACKTESTER PROFESIONAL DE OPCIONES v34.7")
    print("=" * 72)
    try:
        bt, ticker, start, end, capital = interactive_config()
        bt.load(ticker, start, end).run(capital=capital)
        bt.report()
    except KeyboardInterrupt:
        print("\n\nInterrumpido.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    try:
        import yfinance
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                                "yfinance","pandas","numpy","matplotlib","seaborn","scipy"])
    main()
