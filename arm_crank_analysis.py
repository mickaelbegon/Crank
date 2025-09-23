#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm-Crank (Hand-Cycling) Analysis Toolkit
=========================================

This script provides a complete pipeline to load, process, and visualize
arm-cranking data recorded from pedals/crank sensors (e.g., LabVIEW .lvm export).
It is tailored for datasets that include the following *intended* column headers:

    Time
    Left pedal_Fx
    Left pedal_Fy
    Left pedal_Fz
    Left pedal_Mx
    Left pedal_My
    Left pedal_Mz
    Left pedal_CoPx
    Left pedal_CoPy
    Right pedal_Fx
    Right pedal_Fy
    Right pedal_Fz
    Right pedal_Mx
    Right pedal_My
    Right pedal_Mz
    Right pedal_CoPx
    Right pedal_CoPy
    Left pedal Angular position
    Right pedal Angular position
    Crank Angular position
    Time
    Left pedal in crank_Fx
    Left pedal in crank_Fy
    Left pedal in crank_Fz
    Left pedal in crank_Mx
    Left pedal in crank_My
    Left pedal in crank_Mz
    Right pedal in crank_Fx
    Right pedal in crank_Fy
    Right pedal in crank_Fz
    Right pedal in crank_Mx
    Right pedal in crank_My
    Right pedal in crank_Mz
    Left pedal Torque
    Right pedal Torque
    Total Torque
    Angular velocity
    Left pedal Power
    Right pedal Power
    Total Power
    Left pedal Work
    Right pedal Work
    Total Work
    Left pedal Effectivness Index
    Right pedal Effectivness Index
    Total Effectivness Index

Some files contain duplicate header names (e.g., "Time" appears twice).
This script renames duplicates as "Time" and "Time_2" automatically.

Main features
-------------
- Load .lvm data (robust header skipping) and assign the provided headers.
- Compute sampling frequency (mean 1/dt over the file).
- Trim the trial to the steady section: keep the longest contiguous segment
  where "Angular velocity" is within ±tol% of its mean (default tol = 5%).
- Build a continuous crank angle by unwrapping "Crank Angular position"
  (auto-detect degrees vs radians).
- Recompute a precise angular velocity from the unwrapped angle.
- Compute total force norms for left/right pedals in both lab and crank frames.
- Detect cycle boundaries from the unwrapped crank angle (2π multiples).
- Time-normalize each cycle to 0–100% and compute mean & std across cycles
  for *all* variables.
- Per-cycle metrics:
    * Work (L/R/Total) = cumulative work difference between end and start
    * Max power (L/R/Total) within the cycle
    * Propulsive force (Fz in crank): max and mean
    * Total force norm (lab & crank frames): max and mean
    * Torques (L/R/Total): max and mean
- Simple Tkinter GUI to:
    * choose the .lvm file
    * pick figure type (XY or polar)
    * choose x-axis (time, normalized cycle %, crank angle, cycle #)
    * choose one or multiple y variables
    * pick whether to plot all cycles (gradient) or mean±SD

How to run
----------
- With Conda environment (see environment.yml):
    conda env create -f environment.yml
    conda activate armcrank
    python arm_crank_analysis.py

Notes
-----
- The GUI uses Tkinter (cross-platform). If launching from certain IDEs or
  environments (e.g., headless servers), GUI windows may not appear.
- The plotting code aims to be robust but keep it simple and transparent.
- Everything is heavily commented to make adaptation easy for your lab.
"""

from __future__ import annotations

import os
import sys
import math
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


# ------------------------------
# Configuration & Column Names
# ------------------------------

# Intended header list (order matters). Duplicate "Time" is expected.
INTENDED_HEADERS = [
    "Time",
    "Left pedal_Fx", "Left pedal_Fy", "Left pedal_Fz",
    "Left pedal_Mx", "Left pedal_My", "Left pedal_Mz",
    "Left pedal_CoPx", "Left pedal_CoPy",
    "Right pedal_Fx", "Right pedal_Fy", "Right pedal_Fz",
    "Right pedal_Mx", "Right pedal_My", "Right pedal_Mz",
    "Right pedal_CoPx", "Right pedal_CoPy",
    "Left pedal Angular position",
    "Right pedal Angular position",
    "Crank Angular position",
    "Time",  # duplicate in the provided spec
    "Left pedal in crank_Fx", "Left pedal in crank_Fy", "Left pedal in crank_Fz",
    "Left pedal in crank_Mx", "Left pedal in crank_My", "Left pedal in crank_Mz",
    "Right pedal in crank_Fx", "Right pedal in crank_Fy", "Right pedal in crank_Fz",
    "Right pedal in crank_Mx", "Right pedal in crank_My", "Right pedal in crank_Mz",
    "Left pedal Torque", "Right pedal Torque", "Total Torque",
    "Angular velocity",
    "Left pedal Power", "Right pedal Power", "Total Power",
    "Left pedal Work", "Right pedal Work", "Total Work",
    "Left pedal Effectivness Index", "Right pedal Effectivness Index", "Total Effectivness Index",
]


def uniqueify_headers(headers: list[str]) -> list[str]:
    """
    Make header names unique by appending suffixes: col, col_2, col_3, ...
    """
    seen = {}
    result = []
    for h in headers:
        if h not in seen:
            seen[h] = 1
            result.append(h)
        else:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
    return result


def first_data_row_index(lines: list[str], expected_min_cols: int = 8) -> int:
    """
    Heuristically find the first line that looks like *numeric data*.
    - Skip LabVIEW .lvm headers (non-numeric / key:value lines).
    - expected_min_cols: minimal number of numeric columns to trust as data.
    Returns the zero-based line index.
    """
    for i, line in enumerate(lines):
        # Try splitting by tab first, else whitespace.
        parts = line.strip().split('\t')
        if len(parts) < expected_min_cols:
            parts = line.strip().split()
        # Try to parse as floats.
        good = 0
        for p in parts:
            try:
                float(p.replace(',', '.'))  # be tolerant to decimal commas
                good += 1
            except ValueError:
                pass
        # If line is "mostly" numeric and has enough columns, call it data.
        if good >= expected_min_cols and good >= len(parts) * 0.8:
            return i
    # Fallback: return 0 (let pandas throw an error later rather than crash here)
    return 0


def read_lvm(filepath: str, intended_headers: list[str] = INTENDED_HEADERS, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Robustly read a LabVIEW .lvm file and set the provided headers.

    Strategy:
    - Read file as text, find the first data row (numeric line).
    - Use pandas.read_csv from that line onward.
    - Force the number of columns to match the count of numeric tokens in the first data line.
    - Assign unique header names derived from intended_headers, truncated/padded accordingly.
    """
    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()

    # Locate first data row; try to infer delimiter by that row dynamically.
    start_idx = first_data_row_index(lines, expected_min_cols=8)
    data_preview = lines[start_idx].strip()

    # Infer delimiter: prefer tab, else whitespace
    if '\t' in data_preview:
        sep = '\t'
    else:
        sep = r'\s+'  # regex for any whitespace

    # How many columns do we have?
    if sep == '\t':
        ncols = len(data_preview.split('\t'))
    else:
        # split on any whitespace; avoid regex at this step
        ncols = len(data_preview.split())

    # Build a header list exactly of length ncols
    unique_headers = uniqueify_headers(intended_headers)
    if len(unique_headers) >= ncols:
        use_headers = unique_headers[:ncols]
    else:
        # pad with generic names if file has more columns than expected
        extra = [f"Extra_{i+1}" for i in range(ncols - len(unique_headers))]
        use_headers = unique_headers + extra

    # Read with pandas from the first data line
    df = pd.read_csv(
        filepath,
        sep=sep,
        engine='python',
        header=None,
        skiprows=start_idx,
        names=use_headers,
        na_values=['NaN', 'nan', ''],
    )

    # Convert all columns to numeric when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


# ------------------------------
# Core Analysis Class
# ------------------------------

class ArmCrankAnalyzer:
    """
    Encapsulates all computations required for arm-cranking analysis.

    Typical usage:
        df = read_lvm(path_to_lvm)
        ana = ArmCrankAnalyzer(df)
        ana.run_full_pipeline()
        # Access results:
        ana.sampling_hz
        ana.df_trim        # trimmed dataframe (steady section only)
        ana.cycles         # list of (start_idx, end_idx) for cycles in trimmed data
        ana.cycle_summary  # DataFrame with per-cycle metrics
        ana.stats_mean     # DataFrame (100x variables) mean over normalized cycles
        ana.stats_std      # DataFrame (100x variables) std over normalized cycles
    """

    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.df = df  # working copy (will become trimmed)
        self.sampling_hz: float | None = None

        # Column names we rely on (try to find best match if duplicates were renamed)
        # We pick the *first* matching column name found.
        self.col_time = self._find_first_col(["Time", "Time_2", "time", "time_2"])
        self.col_crank_ang = self._find_first_col(["Crank Angular position", "Crank Angular position_2", "Crank angle"])
        self.col_ang_vel = self._find_first_col(["Angular velocity", "Angular velocity_2", "Crank angular velocity"])

        # Force (lab frame)
        self.cols_left_F = [self._find_first_col(["Left pedal_Fx"]),
                            self._find_first_col(["Left pedal_Fy"]),
                            self._find_first_col(["Left pedal_Fz"])]
        self.cols_right_F = [self._find_first_col(["Right pedal_Fx"]),
                             self._find_first_col(["Right pedal_Fy"]),
                             self._find_first_col(["Right pedal_Fz"])]
        # Force (crank frame)
        self.cols_left_F_crank = [self._find_first_col(["Left pedal in crank_Fx"]),
                                  self._find_first_col(["Left pedal in crank_Fy"]),
                                  self._find_first_col(["Left pedal in crank_Fz"])]
        self.cols_right_F_crank = [self._find_first_col(["Right pedal in crank_Fx"]),
                                   self._find_first_col(["Right pedal in crank_Fy"]),
                                   self._find_first_col(["Right pedal in crank_Fz"])]

        # Torques
        self.col_torque_L = self._find_first_col(["Left pedal Torque"])
        self.col_torque_R = self._find_first_col(["Right pedal Torque"])
        self.col_torque_T = self._find_first_col(["Total Torque"])

        # Power
        self.col_power_L = self._find_first_col(["Left pedal Power"])
        self.col_power_R = self._find_first_col(["Right pedal Power"])
        self.col_power_T = self._find_first_col(["Total Power"])

        # Work (cumulative)
        self.col_work_L = self._find_first_col(["Left pedal Work"])
        self.col_work_R = self._find_first_col(["Right pedal Work"])
        self.col_work_T = self._find_first_col(["Total Work"])

        # Derived columns to be created
        self.col_crank_rad = "Crank angle [rad]"
        self.col_crank_rad_unwrapped = "Crank angle unwrapped [rad]"
        self.col_ang_vel_from_unwrap = "Angular velocity from unwrapped [rad/s]"

        self.col_left_Fnorm = "Left pedal |F| (lab)"
        self.col_right_Fnorm = "Right pedal |F| (lab)"
        self.col_left_Fnorm_crank = "Left pedal |F| (crank)"
        self.col_right_Fnorm_crank = "Right pedal |F| (crank)"

        # Results containers
        self.trim_start_idx: int | None = None
        self.trim_end_idx: int | None = None
        self.df_trim: pd.DataFrame | None = None
        self.cycles: list[tuple[int, int]] = []
        self.cycle_summary: pd.DataFrame | None = None
        self.stats_mean: pd.DataFrame | None = None
        self.stats_std: pd.DataFrame | None = None

    # ---------- Utility: flexible column resolution ----------
    def _find_first_col(self, candidates: list[str]) -> str | None:
        """
        Return the first column name that exists in self.df_raw among candidates.
        """
        for c in candidates:
            if c in self.df_raw.columns:
                return c
        return None

    # -------------------- Public pipeline --------------------
    def run_full_pipeline(self,
                          steady_tol_ratio: float = 0.05,
                          min_steady_seconds: float = 1.0,
                          sg_window: int | None = None,
                          sg_poly: int = 3,
                          verbose: bool = True):
        """
        Execute all major steps in order.
        - steady_tol_ratio: angular velocity within ±tol% of mean for steady segment
        - min_steady_seconds: minimal continuous duration to accept a steady segment
        - sg_window/sg_poly: optional Savitzky–Golay smoothing window for angle before differentiating
        """
        if verbose: print("[1/9] Compute sampling frequency...")
        self.compute_sampling_hz()

        if verbose: print("[2/9] Trim to steady-speed segment...")
        self.trim_to_steady_speed(tol_ratio=steady_tol_ratio, min_steady_seconds=min_steady_seconds)

        if verbose: print("[3/9] Detect angle units (deg/rad) and convert...")
        self.add_crank_angle_radians()

        if verbose: print("[4/9] Unwrap crank angle and recompute angular velocity...")
        self.unwrap_and_compute_ang_vel(sg_window=sg_window, sg_poly=sg_poly)

        if verbose: print("[5/9] Compute force norms (lab & crank frames)...")
        self.compute_force_norms()

        if verbose: print("[6/9] Detect cycles from unwrapped crank angle...")
        self.detect_cycles_from_unwrap()

        if verbose: print("[7/9] Time-normalize cycles and compute mean/std...")
        self.compute_cycle_stats(n_points=100)

        if verbose: print("[8/9] Compute per-cycle metrics...")
        self.compute_per_cycle_metrics()

        if verbose: print("[9/9] Done.")

    # -------------------- Step 1: Sampling -------------------
    def compute_sampling_hz(self):
        """
        Compute sampling frequency as the inverse of the mean time step.
        Uses the first available time column.
        """
        if self.col_time is None:
            raise ValueError("No 'Time' column found. Check your header mapping.")
        t = self.df[self.col_time].to_numpy(dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            raise ValueError("Time column has non-increasing or invalid values.")
        self.sampling_hz = 1.0 / float(np.mean(dt))

    # -------------------- Step 2: Trim to steady --------------
    def trim_to_steady_speed(self, tol_ratio: float = 0.05, min_steady_seconds: float = 1.0):
        """
        Keep the *longest contiguous* segment where Angular velocity is within ±tol_ratio
        of its own mean over the file. This is a robust way to isolate the "steady" trial.

        If Angular velocity column is missing, fallback to recomputed angular velocity later
        (but trimming will be skipped).
        """
        if self.col_ang_vel is None or self.col_ang_vel not in self.df.columns:
            # fallback: no trimming
            self.trim_start_idx = 0
            self.trim_end_idx = len(self.df) - 1
            self.df_trim = self.df.copy()
            return

        w = self.df[self.col_ang_vel].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        if w.size == 0:
            self.trim_start_idx, self.trim_end_idx = 0, len(self.df) - 1
            self.df_trim = self.df.copy()
            return

        w_full = self.df[self.col_ang_vel].to_numpy(dtype=float)
        mu = np.nanmean(w_full)
        if not np.isfinite(mu) or mu == 0:
            self.trim_start_idx, self.trim_end_idx = 0, len(self.df) - 1
            self.df_trim = self.df.copy()
            return

        tol = abs(mu) * tol_ratio
        mask = np.isfinite(w_full) & (np.abs(w_full - mu) <= tol)

        # Find the longest contiguous True segment in mask
        best_len = 0
        best_start = 0
        cur_len = 0
        cur_start = 0
        for i, ok in enumerate(mask):
            if ok:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_len = 0
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

        # Enforce minimal steady duration
        if self.sampling_hz is None:
            self.compute_sampling_hz()
        min_len = int(max(1, min_steady_seconds * self.sampling_hz))

        if best_len >= min_len:
            self.trim_start_idx = best_start
            self.trim_end_idx = best_start + best_len - 1
        else:
            # If no decent steady segment is found, keep the whole file.
            self.trim_start_idx = 0
            self.trim_end_idx = len(self.df) - 1

        self.df_trim = self.df.iloc[self.trim_start_idx:self.trim_end_idx + 1].reset_index(drop=True)

    # -------------------- Step 3: Angle in radians ------------
    def add_crank_angle_radians(self):
        """
        Create a column (self.col_crank_rad) with crank angle in radians.
        Auto-detect degrees vs radians:
          - If max(abs(angle)) > 2π + small margin OR max(angle) > 6.5, assume degrees.
        """
        if self.col_crank_ang is None:
            raise ValueError("No 'Crank Angular position' column found.")

        ang = self.df_trim[self.col_crank_ang].to_numpy(dtype=float)
        amax = np.nanmax(np.abs(ang[np.isfinite(ang)])) if np.isfinite(ang).any() else 0.0

        # Heuristic: if values exceed ~6.5, likely degrees
        if amax > 6.5:
            ang_rad = np.deg2rad(ang)
        else:
            ang_rad = ang

        self.df_trim[self.col_crank_rad] = ang_rad

    # -------------------- Step 4: Unwrap & angular velocity ---
    def unwrap_and_compute_ang_vel(self, sg_window: int | None = None, sg_poly: int = 3):
        """
        Unwrap crank angle and compute angular velocity by differentiation.
        Optionally smooth the angle with a Savitzky–Golay filter before differentiating.
        """
        t = self.df_trim[self.col_time].to_numpy(dtype=float)
        ang = self.df_trim[self.col_crank_rad].to_numpy(dtype=float)

        # Optional smoothing for a cleaner derivative
        if sg_window is not None and sg_window > 3 and sg_window % 2 == 1:
            ang_smooth = savgol_filter(ang, window_length=sg_window, polyorder=sg_poly, mode='interp')
        else:
            ang_smooth = ang

        # Unwrap (makes angle continuous, removing 2π jumps)
        ang_unwrap = np.unwrap(ang_smooth)
        self.df_trim[self.col_crank_rad_unwrapped] = ang_unwrap

        # Compute angular velocity via gradient
        dt = np.gradient(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.gradient(ang_unwrap) / dt
        self.df_trim[self.col_ang_vel_from_unwrap] = w

    # -------------------- Step 5: Force norms -----------------
    def compute_force_norms(self):
        """
        Compute total force magnitudes (Euclidean norms) from the 3 components,
        in both lab and crank frames.
        """
        def norm3(cols):
            if None in cols:
                return None
            arr = self.df_trim[cols].to_numpy(dtype=float)
            return np.linalg.norm(arr, axis=1)

        # Lab frame
        if None not in self.cols_left_F:
            self.df_trim[self.col_left_Fnorm] = norm3(self.cols_left_F)
        if None not in self.cols_right_F:
            self.df_trim[self.col_right_Fnorm] = norm3(self.cols_right_F)

        # Crank frame
        if None not in self.cols_left_F_crank:
            self.df_trim[self.col_left_Fnorm_crank] = norm3(self.cols_left_F_crank)
        if None not in self.cols_right_F_crank:
            self.df_trim[self.col_right_Fnorm_crank] = norm3(self.cols_right_F_crank)

    # -------------------- Step 6: Cycle detection -------------
    def detect_cycles_from_unwrap(self):
        """
        Detect cycle boundaries using the unwrapped crank angle.
        A new cycle starts each time the unwrapped angle crosses a multiple of 2π.
        """
        ang_u = self.df_trim[self.col_crank_rad_unwrapped].to_numpy(dtype=float)
        k = np.floor(ang_u / (2 * np.pi)).astype(int)
        dk = np.diff(k, prepend=k[0])

        # cycle start indices are where dk increases (i.e., a new 2π block begins)
        starts = np.where(dk > 0)[0]
        # define cycle ends one sample before the next start (and the last cycle ends at the last sample)
        ends = np.concatenate([starts[1:] - 1, [len(ang_u) - 1]])

        # Filter out too-short cycles (e.g., noise)
        min_len = int(max(3, (self.sampling_hz or 100) * 0.2))  # at least 0.2 s by default
        self.cycles = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]

    # -------------------- Step 7: Normalized cycle stats ------
    def compute_cycle_stats(self, n_points: int = 100):
        """
        For *every* column in df_trim, build a [n_points]-long time-normalized
        representation per cycle, then compute mean & std across cycles.

        Returns:
            self.stats_mean (DataFrame shape: n_points x n_vars)
            self.stats_std  (same shape)
        """
        if not self.cycles:
            self.stats_mean = pd.DataFrame()
            self.stats_std = pd.DataFrame()
            return

        # Normalized x from 0..1 with n_points
        xn = np.linspace(0.0, 1.0, n_points)

        cols = list(self.df_trim.columns)
        stacks = {c: [] for c in cols}

        for (s, e) in self.cycles:
            seg = self.df_trim.iloc[s:e+1]
            m = len(seg)
            if m < 2:
                continue

            # normalized position inside the cycle for interpolation
            xi = np.linspace(0.0, 1.0, m)

            for c in cols:
                y = seg[c].to_numpy(dtype=float)
                # interpolate onto the common normalized grid
                y_interp = np.interp(xn, xi, y)
                stacks[c].append(y_interp)

        # Build mean/std DataFrames
        mean_data = {}
        std_data = {}
        for c in cols:
            arr = np.vstack(stacks[c]) if len(stacks[c]) > 0 else np.empty((0, n_points))
            if arr.size == 0:
                mean_data[c] = np.full(n_points, np.nan)
                std_data[c] = np.full(n_points, np.nan)
            else:
                mean_data[c] = np.nanmean(arr, axis=0)
                std_data[c] = np.nanstd(arr, axis=0)

        self.stats_mean = pd.DataFrame(mean_data, index=(xn * 100.0))  # index = % cycle (0..100)
        self.stats_std = pd.DataFrame(std_data, index=(xn * 100.0))

    # -------------------- Step 8: Per-cycle metrics -----------
    def compute_per_cycle_metrics(self):
        """
        Compute summary metrics for each cycle:

            - Work per cycle (L/R/Total): end - start values of cumulative work.
            - Max power (L/R/Total).
            - Propulsive force in crank frame (Fz_in_crank): max and mean.
            - Total force (norm) in lab & crank frames: max and mean (per pedal).
            - Torques (L/R/Total): max and mean.

        Output:
            self.cycle_summary : DataFrame with one row per cycle and named metrics.
        """
        rows = []
        for idx, (s, e) in enumerate(self.cycles):
            seg = self.df_trim.iloc[s:e+1]

            # Work (end - start) using cumulative signals
            def delta(col):
                if col is None or col not in seg:
                    return np.nan
                a = seg[col].to_numpy(dtype=float)
                return np.nan if a.size == 0 else (a[-1] - a[0])

            work_L = delta(self.col_work_L)
            work_R = delta(self.col_work_R)
            work_T = delta(self.col_work_T)

            # Max power
            def vmax(col):
                if col is None or col not in seg:
                    return np.nan
                a = seg[col].to_numpy(dtype=float)
                return np.nanmax(a) if a.size else np.nan

            pL_max = vmax(self.col_power_L)
            pR_max = vmax(self.col_power_R)
            pT_max = vmax(self.col_power_T)

            # Propulsive force (Fz in crank): max & mean (both pedals separately)
            def max_mean(col):
                if col is None or col not in seg:
                    return (np.nan, np.nan)
                a = seg[col].to_numpy(dtype=float)
                return (np.nanmax(a), np.nanmean(a)) if a.size else (np.nan, np.nan)

            L_Fz_max, L_Fz_mean = max_mean(self._find_first_col(["Left pedal in crank_Fz"]))
            R_Fz_max, R_Fz_mean = max_mean(self._find_first_col(["Right pedal in crank_Fz"]))

            # Total force norms (lab & crank): max & mean
            def max_mean_col(col):
                if col not in seg:
                    return (np.nan, np.nan)
                a = seg[col].to_numpy(dtype=float)
                return (np.nanmax(a), np.nanmean(a)) if a.size else (np.nan, np.nan)

            L_Fnorm_lab_max, L_Fnorm_lab_mean = max_mean_col(self.col_left_Fnorm) if self.col_left_Fnorm in seg else (np.nan, np.nan)
            R_Fnorm_lab_max, R_Fnorm_lab_mean = max_mean_col(self.col_right_Fnorm) if self.col_right_Fnorm in seg else (np.nan, np.nan)
            L_Fnorm_cr_max,  L_Fnorm_cr_mean  = max_mean_col(self.col_left_Fnorm_crank) if self.col_left_Fnorm_crank in seg else (np.nan, np.nan)
            R_Fnorm_cr_max,  R_Fnorm_cr_mean  = max_mean_col(self.col_right_Fnorm_crank) if self.col_right_Fnorm_crank in seg else (np.nan, np.nan)

            # Torques: max & mean
            tauL_max, tauL_mean = max_mean(self.col_torque_L)
            tauR_max, tauR_mean = max_mean(self.col_torque_R)
            tauT_max, tauT_mean = max_mean(self.col_torque_T)

            rows.append({
                "cycle_index": idx,
                "start_idx": int(s),
                "end_idx": int(e),
                "duration_s": float(self.df_trim[self.col_time].iloc[e] - self.df_trim[self.col_time].iloc[s]),
                # Work
                "work_L": work_L,
                "work_R": work_R,
                "work_T": work_T,
                # Power maxima
                "Pmax_L": pL_max,
                "Pmax_R": pR_max,
                "Pmax_T": pT_max,
                # Propulsive Fz (in crank)
                "Fz_crank_L_max": L_Fz_max,
                "Fz_crank_L_mean": L_Fz_mean,
                "Fz_crank_R_max": R_Fz_max,
                "Fz_crank_R_mean": R_Fz_mean,
                # Force norms
                "Fnorm_lab_L_max": L_Fnorm_lab_max,
                "Fnorm_lab_L_mean": L_Fnorm_lab_mean,
                "Fnorm_lab_R_max": R_Fnorm_lab_max,
                "Fnorm_lab_R_mean": R_Fnorm_lab_mean,
                "Fnorm_crank_L_max": L_Fnorm_cr_max,
                "Fnorm_crank_L_mean": L_Fnorm_cr_mean,
                "Fnorm_crank_R_max": R_Fnorm_cr_max,
                "Fnorm_crank_R_mean": R_Fnorm_cr_mean,
                # Torques
                "Torque_L_max": tauL_max,
                "Torque_L_mean": tauL_mean,
                "Torque_R_max": tauR_max,
                "Torque_R_mean": tauR_mean,
                "Torque_T_max": tauT_max,
                "Torque_T_mean": tauT_mean,
            })

        self.cycle_summary = pd.DataFrame(rows)

    # -------------------- Plotting Utilities ------------------
    def list_numeric_columns(self) -> list[str]:
        """Return a list of numeric columns available for plotting."""
        cols = []
        for c in self.df_trim.columns:
            if pd.api.types.is_numeric_dtype(self.df_trim[c]):
                cols.append(c)
        return cols

    def plot(self,
             figure_type: str = "XY",  # "XY" or "Polar"
             x_axis: str = "Time",     # "Time", "NormalizedCycle%", "CrankAngle", "CycleIndex"
             y_vars: list[str] = None,
             show_all_cycles: bool = False,
             show_mean_std: bool = True):
        """
        Generic plotting function supporting XY and Polar charts.
        - If Polar: theta = Crank angle (rad). All y_vars are plotted against theta.
        - If x_axis == "NormalizedCycle%": we use stats_mean / stats_std (0..100%).
          If show_all_cycles is True, overlay individual cycles with a color gradient.
        - If x_axis == "CycleIndex": y_vars must be cycle-level metrics (from cycle_summary).
        """
        if y_vars is None or len(y_vars) == 0:
            raise ValueError("Please specify at least one y variable to plot.")

        if figure_type not in ("XY", "Polar"):
            raise ValueError("figure_type must be 'XY' or 'Polar'.")

        if x_axis not in ("Time", "NormalizedCycle%", "CrankAngle", "CycleIndex"):
            raise ValueError("x_axis must be one of: Time, NormalizedCycle%, CrankAngle, CycleIndex.")

        if figure_type == "Polar":
            # Polar plot: interpret x as crank angle (radians)
            theta = self.df_trim[self.col_crank_rad] if "Crank angle" in self.df_trim else self.df_trim[self.col_crank_rad]
            theta = theta.to_numpy(dtype=float)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            for y in y_vars:
                r = self.df_trim[y].to_numpy(dtype=float)
                ax.plot(theta, r, linewidth=1.2)
            ax.set_title("Polar plot vs crank angle (rad)")
            plt.show()
            return

        # XY PLOTS
        if x_axis == "Time":
            x = self.df_trim[self.col_time].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            for y in y_vars:
                ax.plot(x, self.df_trim[y].to_numpy(dtype=float), linewidth=1.2)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Time-series")
            ax.grid(True, which='both', alpha=0.3)
            plt.show()
            return

        if x_axis == "CrankAngle":
            x = self.df_trim[self.col_crank_rad].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            for y in y_vars:
                ax.plot(x, self.df_trim[y].to_numpy(dtype=float), linewidth=1.2)
            ax.set_xlabel("Crank angle [rad]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Signals vs crank angle")
            ax.grid(True, which='both', alpha=0.3)
            plt.show()
            return

        if x_axis == "CycleIndex":
            if self.cycle_summary is None or self.cycle_summary.empty:
                raise RuntimeError("No cycle_summary available. Run compute_per_cycle_metrics() first.")
            x = self.cycle_summary["cycle_index"].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            for y in y_vars:
                if y not in self.cycle_summary.columns:
                    raise ValueError(f"'{y}' is not a cycle-level metric.")
                ax.plot(x, self.cycle_summary[y].to_numpy(dtype=float), marker='o', linewidth=1.2)
            ax.set_xlabel("Cycle index")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Cycle-level metrics")
            ax.grid(True, which='both', alpha=0.3)
            plt.show()
            return

        if x_axis == "NormalizedCycle%":
            if self.stats_mean is None or self.stats_mean.empty:
                raise RuntimeError("No normalized stats available. Run compute_cycle_stats() first.")
            fig, ax = plt.subplots()

            # Overlay all cycles if requested (using a colormap for gradient)
            if show_all_cycles:
                xn = self.stats_mean.index.to_numpy(dtype=float)  # 0..100
                # Rebuild individual cycles with interpolation for requested y_vars
                for (s, e) in self.cycles:
                    seg = self.df_trim.iloc[s:e+1]
                    m = len(seg)
                    xi = np.linspace(0.0, 1.0, m)
                    for y in y_vars:
                        yv = seg[y].to_numpy(dtype=float)
                        y_interp = np.interp(xn / 100.0, xi, yv)
                        ax.plot(xn, y_interp, alpha=0.25, linewidth=0.8)

            # Mean ± std
            if show_mean_std:
                xn = self.stats_mean.index.to_numpy(dtype=float)
                for y in y_vars:
                    mu = self.stats_mean[y].to_numpy(dtype=float)
                    sd = self.stats_std[y].to_numpy(dtype=float)
                    ax.plot(xn, mu, linewidth=2.0)
                    ax.fill_between(xn, mu - sd, mu + sd, alpha=0.2)

            ax.set_xlabel("Cycle [%]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Time-normalized cycles")
            ax.grid(True, which='both', alpha=0.3)
            plt.show()
            return


# ------------------------------
# Tkinter GUI
# ------------------------------

class App(tk.Tk):
    """
    Minimal GUI to:
      - Choose an .lvm file
      - Run the pipeline
      - Select plotting options and visualize
      - Export CSVs for cycle_summary and normalized stats
    """
    def __init__(self):
        super().__init__()
        self.title("Arm-Crank Analysis")
        self.geometry("820x600")

        # State
        self.filepath = tk.StringVar(value="")
        self.analyzer: ArmCrankAnalyzer | None = None

        # --- File frame ---
        frm_file = ttk.LabelFrame(self, text="1) Data file (.lvm)")
        frm_file.pack(fill="x", padx=10, pady=8)
        ttk.Entry(frm_file, textvariable=self.filepath).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(frm_file, text="Browse...", command=self.browse_file).pack(side="left", padx=6, pady=6)

        # --- Run frame ---
        frm_run = ttk.LabelFrame(self, text="2) Run analysis")
        frm_run.pack(fill="x", padx=10, pady=8)
        self.tol_var = tk.DoubleVar(value=0.05)
        self.minsec_var = tk.DoubleVar(value=1.0)
        ttk.Label(frm_run, text="Steady tol (±ratio):").pack(side="left", padx=6)
        ttk.Entry(frm_run, width=6, textvariable=self.tol_var).pack(side="left")
        ttk.Label(frm_run, text="Min steady [s]:").pack(side="left", padx=6)
        ttk.Entry(frm_run, width=6, textvariable=self.minsec_var).pack(side="left")
        ttk.Button(frm_run, text="Run Pipeline", command=self.run_pipeline).pack(side="right", padx=6, pady=6)

        # --- Plot frame ---
        frm_plot = ttk.LabelFrame(self, text="3) Plotting")
        frm_plot.pack(fill="both", expand=True, padx=10, pady=8)

        # Figure type
        self.fig_type = tk.StringVar(value="XY")
        ttk.Label(frm_plot, text="Figure type:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(frm_plot, values=["XY", "Polar"], textvariable=self.fig_type, width=12).grid(row=0, column=1, sticky="w")

        # X axis
        self.x_axis = tk.StringVar(value="Time")
        ttk.Label(frm_plot, text="X axis:").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Combobox(frm_plot, values=["Time", "NormalizedCycle%", "CrankAngle", "CycleIndex"],
                     textvariable=self.x_axis, width=18).grid(row=0, column=3, sticky="w")

        # Checkboxes
        self.chk_all_cycles = tk.BooleanVar(value=False)
        self.chk_mean_std = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_plot, text="Show all cycles", variable=self.chk_all_cycles).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Checkbutton(frm_plot, text="Show mean ± std", variable=self.chk_mean_std).grid(row=1, column=1, sticky="w", padx=6)

        # Y variables listbox (multi-select)
        ttk.Label(frm_plot, text="Y variables (multi-select):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.lst_y = tk.Listbox(frm_plot, selectmode="extended", width=50, height=16, exportselection=False)
        self.lst_y.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
        frm_plot.grid_columnconfigure(0, weight=1)
        frm_plot.grid_rowconfigure(3, weight=1)

        # Buttons
        ttk.Button(frm_plot, text="Plot", command=self.plot).grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Button(frm_plot, text="Export cycle_summary.csv", command=self.export_cycle_summary).grid(row=4, column=2, sticky="e", padx=6, pady=6)
        ttk.Button(frm_plot, text="Export normalized_stats.csv", command=self.export_normalized_stats).grid(row=4, column=3, sticky="e", padx=6, pady=6)

        # Status
        self.status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status, foreground="gray").pack(fill="x", padx=10, pady=4)

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select .lvm file",
            filetypes=[("LabVIEW Measurement", "*.lvm"), ("All files", "*.*")]
        )
        if path:
            self.filepath.set(path)

    def run_pipeline(self):
        path = self.filepath.get().strip()
        if not path:
            messagebox.showerror("Error", "Please choose a .lvm file first.")
            return
        try:
            df = read_lvm(path)
            ana = ArmCrankAnalyzer(df)
            ana.run_full_pipeline(
                steady_tol_ratio=float(self.tol_var.get()),
                min_steady_seconds=float(self.minsec_var.get()),
                verbose=True
            )
            self.analyzer = ana

            # Populate y-variable listbox with available numeric columns (trimmed df)
            self.lst_y.delete(0, tk.END)
            for c in ana.list_numeric_columns():
                self.lst_y.insert(tk.END, c)

            # Report sampling frequency and detected cycles
            msg = f"Sampling: {ana.sampling_hz:.2f} Hz | Steady segment: [{ana.trim_start_idx}, {ana.trim_end_idx}] | Cycles: {len(ana.cycles)}"
            self.status.set(msg)
            messagebox.showinfo("Pipeline complete", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def selected_y_vars(self) -> list[str]:
        idxs = self.lst_y.curselection()
        return [self.lst_y.get(i) for i in idxs]

    def plot(self):
        if self.analyzer is None:
            messagebox.showerror("Error", "Run the pipeline first.")
            return
        y = self.selected_y_vars()
        if not y:
            messagebox.showerror("Error", "Select at least one Y variable.")
            return
        try:
            self.analyzer.plot(
                figure_type=self.fig_type.get(),
                x_axis=self.x_axis.get(),
                y_vars=y,
                show_all_cycles=self.chk_all_cycles.get(),
                show_mean_std=self.chk_mean_std.get()
            )
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    def export_cycle_summary(self):
        if self.analyzer is None or self.analyzer.cycle_summary is None:
            messagebox.showerror("Error", "No cycle_summary available. Run the pipeline first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save cycle_summary.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        try:
            self.analyzer.cycle_summary.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def export_normalized_stats(self):
        if self.analyzer is None or self.analyzer.stats_mean is None:
            messagebox.showerror("Error", "No normalized stats available. Run the pipeline first.")
            return

        # Save two files: mean and std
        base = filedialog.asksaveasfilename(
            title="Save normalized_stats (base name)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not base:
            return
        base_no_ext, _ = os.path.splitext(base)
        path_mean = base_no_ext + "_mean.csv"
        path_std  = base_no_ext + "_std.csv"
        try:
            self.analyzer.stats_mean.to_csv(path_mean, index_label="Cycle %")
            self.analyzer.stats_std.to_csv(path_std, index_label="Cycle %")
            messagebox.showinfo("Saved", f"Saved:\n{path_mean}\n{path_std}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


# ------------------------------
# CLI Entrypoint
# ------------------------------

def main():
    """
    If called with a path argument, run the pipeline headlessly and export summary CSVs.
    Otherwise, launch the GUI.
    """
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading: {path}")
        df = read_lvm(path)
        ana = ArmCrankAnalyzer(df)
        ana.run_full_pipeline(verbose=True)

        # Export next to the data file
        base_no_ext, _ = os.path.splitext(path)
        out_cycles = base_no_ext + "_cycle_summary.csv"
        out_mean   = base_no_ext + "_normalized_mean.csv"
        out_std    = base_no_ext + "_normalized_std.csv"

        ana.cycle_summary.to_csv(out_cycles, index=False)
        ana.stats_mean.to_csv(out_mean, index_label="Cycle %")
        ana.stats_std.to_csv(out_std, index_label="Cycle %")

        print(f"Exported:\n  {out_cycles}\n  {out_mean}\n  {out_std}")
    else:
        # Launch GUI
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()