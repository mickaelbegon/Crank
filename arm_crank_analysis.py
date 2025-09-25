#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arm-Crank (Hand-Cycling) Analysis Toolkit — v1.4
=================================================

Changes in v1.4
---------------
- UI: when X axis = "Time", the checkboxes **Show all cycles** and **Show mean ± std**
  are disabled (they don't apply to raw time plots).
- Plotting: when **Show all cycles** is enabled (for "CrankAngle360°"), all cycles
  are drawn using a gradient of the **same base color** as the corresponding mean curve.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
import pandas as pd

# Ensure backend compatible with Tkinter BEFORE importing pyplot
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


# ------------------------------
# Configuration & Column Names
# ------------------------------

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
    "Time",
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


def uniqueify_headers(headers):
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


def first_data_row_index(lines, expected_min_cols=8):
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) < expected_min_cols:
            parts = line.strip().split()
        good = 0
        for p in parts:
            try:
                float(p.replace(',', '.'))
                good += 1
            except ValueError:
                pass
        if good >= expected_min_cols and good >= len(parts) * 0.8:
            return i
    return 0


def read_lvm(filepath, intended_headers=INTENDED_HEADERS, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
        lines = f.readlines()
    start_idx = first_data_row_index(lines, expected_min_cols=8)
    data_preview = lines[start_idx].strip()
    sep = '\t' if '\t' in data_preview else r'\s+'
    ncols = len(data_preview.split('\t')) if sep == '\t' else len(data_preview.split())
    headers = uniqueify_headers(intended_headers)
    if len(headers) >= ncols:
        use_headers = headers[:ncols]
    else:
        use_headers = headers + [f"Extra_{i+1}" for i in range(ncols - len(headers))]
    df = pd.read_csv(filepath, sep=sep, engine='python', header=None,
                     skiprows=start_idx, names=use_headers, na_values=['NaN','nan',''])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ------------------------------
# Core Analysis
# ------------------------------

class ArmCrankAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df_raw = df.copy()
        self.df = df
        self.sampling_hz = None

        # Locate columns
        self.col_time = self._find_first_col(["Time", "Time_2", "time", "time_2"])
        self.col_crank_ang = self._find_first_col(["Crank Angular position", "Crank Angular position_2", "Crank angle"])
        self.col_ang_vel = self._find_first_col(["Angular velocity", "Angular velocity_2", "Crank angular velocity"])

        self.cols_left_F = [self._find_first_col(["Left pedal_Fx"]),
                            self._find_first_col(["Left pedal_Fy"]),
                            self._find_first_col(["Left pedal_Fz"])]
        self.cols_right_F = [self._find_first_col(["Right pedal_Fx"]),
                             self._find_first_col(["Right pedal_Fy"]),
                             self._find_first_col(["Right pedal_Fz"])]
        self.cols_left_F_crank = [self._find_first_col(["Left pedal in crank_Fx"]),
                                  self._find_first_col(["Left pedal in crank_Fy"]),
                                  self._find_first_col(["Left pedal in crank_Fz"])]
        self.cols_right_F_crank = [self._find_first_col(["Right pedal in crank_Fx"]),
                                   self._find_first_col(["Right pedal in crank_Fy"]),
                                   self._find_first_col(["Right pedal in crank_Fz"])]

        self.col_torque_L = self._find_first_col(["Left pedal Torque"])
        self.col_torque_R = self._find_first_col(["Right pedal Torque"])
        self.col_torque_T = self._find_first_col(["Total Torque"])

        self.col_power_L = self._find_first_col(["Left pedal Power"])
        self.col_power_R = self._find_first_col(["Right pedal Power"])
        self.col_power_T = self._find_first_col(["Total Power"])

        self.col_work_L = self._find_first_col(["Left pedal Work"])
        self.col_work_R = self._find_first_col(["Right pedal Work"])
        self.col_work_T = self._find_first_col(["Total Work"])

        self.col_crank_rad = "Crank angle [rad]"
        self.col_crank_rad_unwrapped = "Crank angle unwrapped [rad]"
        self.col_ang_vel_from_unwrap = "Angular velocity from unwrapped [rad/s]"

        self.col_left_Fnorm = "Left pedal |F| (lab)"
        self.col_right_Fnorm = "Right pedal |F| (lab)"
        self.col_left_Fnorm_crank = "Left pedal |F| (crank)"
        self.col_right_Fnorm_crank = "Right pedal |F| (crank)"

        self.trim_start_idx = None
        self.trim_end_idx = None
        self.df_trim = None
        self.cycles = []
        self.cycle_summary = None
        self.stats_mean = None
        self.stats_std = None
        self.stats_angle_mean = None
        self.stats_angle_std = None

        self._roi_cache = None

    def _find_first_col(self, candidates):
        for c in candidates:
            if c in self.df_raw.columns:
                return c
        return None

    def run_full_pipeline(self, steady_tol_ratio=0.05, non_zero_thr_deg_s=2.0,
                          sg_window=None, sg_poly=3, min_steady_seconds=1.0, verbose=True):
        if verbose: print("[1/11] Sampling frequency")
        self.compute_sampling_hz()

        if verbose: print("[2/11] Auto-trim to steady-speed segment")
        self.trim_to_steady_speed(tol_ratio=steady_tol_ratio,
                                  non_zero_thr_deg_s=non_zero_thr_deg_s,
                                  min_steady_seconds=min_steady_seconds)

        if verbose: print("[3/11] Angle units detection and conversion")
        self.add_crank_angle_radians()

        if verbose: print("[4/11] Unwrap and precise angular velocity")
        self.unwrap_and_compute_ang_vel(sg_window=sg_window, sg_poly=sg_poly)

        if verbose: print("[5/11] Force norms")
        self.compute_force_norms()

        if verbose: print("[6/11] Cycle detection")
        self.detect_cycles_from_unwrap()

        if verbose: print("[7/11] Cycle time-normalized stats (0..100%)")
        self.compute_cycle_stats(n_points=100)

        if verbose: print("[8/11] Angle-normalized stats (0..360°)")
        self.compute_angle_stats(n_degrees=360)

        if verbose: print("[9/11] Per-cycle metrics")
        self.compute_per_cycle_metrics()

        if verbose: print("[10/11] Done.")

    # ----- Step 1: Sampling -----
    def compute_sampling_hz(self):
        if self.col_time is None:
            raise ValueError("No 'Time' column found.")
        t = self.df[self.col_time].to_numpy(dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            raise ValueError("Time column has non-increasing or invalid values.")
        self.sampling_hz = 1.0 / float(np.mean(dt))

    # ----- Step 2: Trim to steady (auto) -----
    def trim_to_steady_speed(self, tol_ratio=0.05, non_zero_thr_deg_s=2.0, min_steady_seconds=1.0):
        if self.col_ang_vel is None or self.col_ang_vel not in self.df.columns:
            self.trim_start_idx = 0
            self.trim_end_idx = len(self.df) - 1
            self.df_trim = self.df.copy()
            return

        w_full = self.df[self.col_ang_vel].to_numpy(dtype=float)
        if not np.isfinite(w_full).any():
            self.trim_start_idx, self.trim_end_idx = 0, len(self.df) - 1
            self.df_trim = self.df.copy()
            return

        # Active mask and active mean
        thr = np.deg2rad(non_zero_thr_deg_s)
        active_mask = np.isfinite(w_full) & (np.abs(w_full) > thr)
        mu_active = np.nanmean(w_full[active_mask]) if active_mask.any() else np.nanmean(w_full)
        tol = abs(mu_active) * tol_ratio if np.isfinite(mu_active) else 0.0

        # Forward from start, backward from end to reach active mean within tol
        i0 = 0
        while i0 < len(w_full) and not (np.isfinite(w_full[i0]) and abs(w_full[i0] - mu_active) <= tol):
            i0 += 1
        i1 = len(w_full) - 1
        while i1 >= 0 and not (np.isfinite(w_full[i1]) and abs(w_full[i1] - mu_active) <= tol):
            i1 -= 1

        if i0 >= i1:
            i0, i1 = 0, len(w_full) - 1

        if self.sampling_hz is None:
            self.compute_sampling_hz()
        min_len = int(max(1, min_steady_seconds * self.sampling_hz))
        if (i1 - i0 + 1) < min_len:
            i0, i1 = 0, len(w_full) - 1

        self.trim_start_idx = i0
        self.trim_end_idx = i1
        self.df_trim = self.df.iloc[i0:i1 + 1].reset_index(drop=True)

        # Cache for ROI viz
        t = self.df[self.col_time].to_numpy(dtype=float)
        self._roi_cache = dict(t=t, w=w_full, active_mask=active_mask, mu_active=mu_active,
                               idx0=i0, idx1=i1, thr_deg_s=non_zero_thr_deg_s, tol_ratio=tol_ratio)

    # ----- Manual ROI -----
    def set_manual_trim(self, t_start: float, t_end: float):
        if self.col_time is None:
            raise ValueError("No 'Time' column found.")
        t = self.df[self.col_time].to_numpy(dtype=float)
        if t_end <= t_start:
            raise ValueError("t_end must be > t_start.")
        i0 = int(np.searchsorted(t, t_start, side='left'))
        i1 = int(np.searchsorted(t, t_end, side='right')) - 1
        i0 = max(0, min(i0, len(self.df) - 1))
        i1 = max(0, min(i1, len(self.df) - 1))
        if i1 <= i0:
            raise ValueError("Selected ROI too short or invalid indices.")
        self.trim_start_idx = i0
        self.trim_end_idx = i1
        self.df_trim = self.df.iloc[i0:i1 + 1].reset_index(drop=True)
        # Invalidate downstream
        self.cycles = []
        self.cycle_summary = None
        self.stats_mean = None
        self.stats_std = None
        self.stats_angle_mean = None
        self.stats_angle_std = None

    # ----- Step 3: Angle to radians -----
    def add_crank_angle_radians(self):
        if self.col_crank_ang is None:
            raise ValueError("No 'Crank Angular position' column found.")
        ang = self.df_trim[self.col_crank_ang].to_numpy(dtype=float)
        amax = np.nanmax(np.abs(ang[np.isfinite(ang)])) if np.isfinite(ang).any() else 0.0
        ang_rad = np.deg2rad(ang) if amax > 6.5 else ang
        self.df_trim[self.col_crank_rad] = ang_rad

    # ----- Step 4: Unwrap & ang vel -----
    def unwrap_and_compute_ang_vel(self, sg_window=None, sg_poly=3):
        t = self.df_trim[self.col_time].to_numpy(dtype=float)
        ang = self.df_trim[self.col_crank_rad].to_numpy(dtype=float)
        ang_u = np.unwrap(ang)
        self.df_trim[self.col_crank_rad_unwrapped] = ang_u
        dt = np.gradient(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.gradient(ang_u) / dt
        self.df_trim[self.col_ang_vel_from_unwrap] = w

    # ----- Step 5: Force norms -----
    def compute_force_norms(self):
        def norm3(cols):
            if None in cols:
                return None
            arr = self.df_trim[cols].to_numpy(dtype=float)
            return np.linalg.norm(arr, axis=1)
        if None not in self.cols_left_F:
            self.df_trim[self.col_left_Fnorm] = norm3(self.cols_left_F)
        if None not in self.cols_right_F:
            self.df_trim[self.col_right_Fnorm] = norm3(self.cols_right_F)
        if None not in self.cols_left_F_crank:
            self.df_trim[self.col_left_Fnorm_crank] = norm3(self.cols_left_F_crank)
        if None not in self.cols_right_F_crank:
            self.df_trim[self.col_right_Fnorm_crank] = norm3(self.cols_right_F_crank)

    # ----- Step 6: Cycle detection -----
    def detect_cycles_from_unwrap(self):
        ang_u = self.df_trim[self.col_crank_rad_unwrapped].to_numpy(dtype=float)
        k = np.floor(ang_u / (2 * np.pi)).astype(int)
        dk = np.diff(k, prepend=k[0])
        starts = np.where(dk > 0)[0]
        ends = np.concatenate([starts[1:] - 1, [len(ang_u) - 1]])
        min_len = int(max(3, (self.sampling_hz or 100) * 0.2))
        self.cycles = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]

    # ----- Step 7: Time-normalized stats -----
    def compute_cycle_stats(self, n_points=100):
        if not self.cycles:
            self.stats_mean = pd.DataFrame()
            self.stats_std = pd.DataFrame()
            return
        xn = np.linspace(0.0, 1.0, n_points)
        cols = list(self.df_trim.columns)
        stacks = {c: [] for c in cols}
        for (s, e) in self.cycles:
            seg = self.df_trim.iloc[s:e+1]
            m = len(seg)
            if m < 2:
                continue
            xi = np.linspace(0.0, 1.0, m)
            for c in cols:
                y = seg[c].to_numpy(dtype=float)
                y_interp = np.interp(xn, xi, y)
                stacks[c].append(y_interp)
        mean_data, std_data = {}, {}
        for c in cols:
            arr = np.vstack(stacks[c]) if len(stacks[c]) > 0 else np.empty((0, n_points))
            if arr.size == 0:
                mean_data[c] = np.full(n_points, np.nan)
                std_data[c] = np.full(n_points, np.nan)
            else:
                mean_data[c] = np.nanmean(arr, axis=0)
                std_data[c] = np.nanstd(arr, axis=0)
        self.stats_mean = pd.DataFrame(mean_data, index=(xn * 100.0))
        self.stats_std = pd.DataFrame(std_data, index=(xn * 100.0))

    # ----- Step 8: Angle-normalized stats (0..360°) -----
    def compute_angle_stats(self, n_degrees=360):
        if not self.cycles:
            self.stats_angle_mean = pd.DataFrame()
            self.stats_angle_std = pd.DataFrame()
            return
        cols = list(self.df_trim.columns)
        angle_grid_rad = np.deg2rad(np.arange(n_degrees))
        stacks = {c: [] for c in cols}
        for (s, e) in self.cycles:
            seg = self.df_trim.iloc[s:e+1]
            ang_u = seg[self.col_crank_rad_unwrapped].to_numpy(dtype=float)
            ang_mod = (ang_u - ang_u[0]) % (2 * np.pi)
            for c in cols:
                y = seg[c].to_numpy(dtype=float)
                y_interp = np.interp(angle_grid_rad, ang_mod, y, left=np.nan, right=np.nan)
                stacks[c].append(y_interp)
        mean_data, std_data = {}, {}
        for c in cols:
            arr = np.vstack(stacks[c]) if len(stacks[c]) > 0 else np.empty((0, n_degrees))
            if arr.size == 0:
                mean_data[c] = np.full(n_degrees, np.nan)
                std_data[c] = np.full(n_degrees, np.nan)
            else:
                mean_data[c] = np.nanmean(arr, axis=0)
                std_data[c] = np.nanstd(arr, axis=0)
        self.stats_angle_mean = pd.DataFrame(mean_data, index=np.arange(n_degrees))
        self.stats_angle_std  = pd.DataFrame(std_data, index=np.arange(n_degrees))

    # ----- Step 9: Per-cycle metrics -----
    def compute_per_cycle_metrics(self):
        rows = []
        for idx, (s, e) in enumerate(self.cycles):
            seg = self.df_trim.iloc[s:e+1]

            def delta(col):
                if col is None or col not in seg:
                    return np.nan
                a = seg[col].to_numpy(dtype=float)
                return np.nan if a.size == 0 else (a[-1] - a[0])

            def vmax(col):
                if col is None or col not in seg:
                    return np.nan
                a = seg[col].to_numpy(dtype=float)
                return np.nanmax(a) if a.size else np.nan

            def max_mean(col):
                if col is None or col not in seg:
                    return (np.nan, np.nan)
                a = seg[col].to_numpy(dtype=float)
                return (np.nanmax(a), np.nanmean(a)) if a.size else (np.nan, np.nan)

            work_L = delta(self.col_work_L)
            work_R = delta(self.col_work_R)
            work_T = delta(self.col_work_T)

            pL_max = vmax(self.col_power_L)
            pR_max = vmax(self.col_power_R)
            pT_max = vmax(self.col_power_T)

            L_Fz_max, L_Fz_mean = max_mean(self._find_first_col(["Left pedal in crank_Fz"]))
            R_Fz_max, R_Fz_mean = max_mean(self._find_first_col(["Right pedal in crank_Fz"]))

            def max_mean_col(col):
                if col not in seg:
                    return (np.nan, np.nan)
                a = seg[col].to_numpy(dtype=float)
                return (np.nanmax(a), np.nanmean(a)) if a.size else (np.nan, np.nan)

            L_Fnorm_lab_max, L_Fnorm_lab_mean = max_mean_col("Left pedal |F| (lab)") if "Left pedal |F| (lab)" in seg else (np.nan, np.nan)
            R_Fnorm_lab_max, R_Fnorm_lab_mean = max_mean_col("Right pedal |F| (lab)") if "Right pedal |F| (lab)" in seg else (np.nan, np.nan)
            L_Fnorm_cr_max,  L_Fnorm_cr_mean  = max_mean_col("Left pedal |F| (crank)") if "Left pedal |F| (crank)" in seg else (np.nan, np.nan)
            R_Fnorm_cr_max,  R_Fnorm_cr_mean  = max_mean_col("Right pedal |F| (crank)") if "Right pedal |F| (crank)" in seg else (np.nan, np.nan)

            tauL_max, tauL_mean = max_mean(self.col_torque_L)
            tauR_max, tauR_mean = max_mean(self.col_torque_R)
            tauT_max, tauT_mean = max_mean(self.col_torque_T)

            rows.append({
                "cycle_index": idx,
                "start_idx": int(s),
                "end_idx": int(e),
                "duration_s": float(self.df_trim[self.col_time].iloc[e] - self.df_trim[self.col_time].iloc[s]),
                "work_L": work_L, "work_R": work_R, "work_T": work_T,
                "Pmax_L": pL_max, "Pmax_R": pR_max, "Pmax_T": pT_max,
                "Fz_crank_L_max": L_Fz_max, "Fz_crank_L_mean": L_Fz_mean,
                "Fz_crank_R_max": R_Fz_max, "Fz_crank_R_mean": R_Fz_mean,
                "Fnorm_lab_L_max": L_Fnorm_lab_max, "Fnorm_lab_L_mean": L_Fnorm_lab_mean,
                "Fnorm_lab_R_max": R_Fnorm_lab_max, "Fnorm_lab_R_mean": R_Fnorm_lab_mean,
                "Fnorm_crank_L_max": L_Fnorm_cr_max, "Fnorm_crank_L_mean": L_Fnorm_cr_mean,
                "Fnorm_crank_R_max": R_Fnorm_cr_max, "Fnorm_crank_R_mean": R_Fnorm_cr_mean,
                "Torque_L_max": tauL_max, "Torque_L_mean": tauL_mean,
                "Torque_R_max": tauR_max, "Torque_R_mean": tauR_mean,
                "Torque_T_max": tauT_max, "Torque_T_mean": tauT_mean,
            })
        self.cycle_summary = pd.DataFrame(rows)

    # ----- Helpers for GUI variable lists -----
    def list_numeric_columns(self):
        return [c for c in self.df_trim.columns if pd.api.types.is_numeric_dtype(self.df_trim[c])]

    def list_cycle_metrics(self):
        if self.cycle_summary is None or self.cycle_summary.empty:
            return []
        exclude = {"cycle_index", "start_idx", "end_idx"}
        return [c for c in self.cycle_summary.columns if c not in exclude]

    # ----- Plotting -----
    def plot(self, figure_type="XY", x_axis="Time", y_vars=None, show_all_cycles=False, show_mean_std=True):
        if not y_vars:
            raise ValueError("Please specify at least one y variable to plot.")
        if figure_type not in ("XY", "Polar"):
            raise ValueError("figure_type must be 'XY' or 'Polar'.")
        if x_axis not in ("Time", "NormalizedCycle%", "CrankAngle", "CrankAngle360°", "CycleIndex"):
            raise ValueError("x_axis must be one of: Time, NormalizedCycle%, CrankAngle, CrankAngle360°, CycleIndex.")

        # --- Polar plot (theta in radians) ---
        if figure_type == "Polar":
            theta = self.df_trim[self.col_crank_rad].to_numpy(dtype=float)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            for y in y_vars:
                r = self.df_trim[y].to_numpy(dtype=float)
                ax.plot(theta, r, linewidth=1.2, label=y)
            ax.set_title("Polar plot vs crank angle (rad)")
            ax.legend()
            plt.show()
            return

        # --- XY Time ---
        if x_axis == "Time":
            x = self.df_trim[self.col_time].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            for y in y_vars:
                ax.plot(x, self.df_trim[y].to_numpy(dtype=float), linewidth=1.2, label=y)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Time-series")
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            plt.show()
            return

        # --- XY CrankAngle (in DEGREES) ---
        if x_axis == "CrankAngle":
            x = np.rad2deg(self.df_trim[self.col_crank_rad].to_numpy(dtype=float))
            fig, ax = plt.subplots()
            for y in y_vars:
                ax.plot(x, self.df_trim[y].to_numpy(dtype=float), linewidth=1.2, label=y)
            ax.set_xlabel("Crank angle [deg]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Signals vs crank angle (degrees)")
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            plt.show()
            return

        # --- Angle-normalized (0..360°) ---
        if x_axis == "CrankAngle360°":
            if self.stats_angle_mean is None or self.stats_angle_mean.empty:
                raise RuntimeError("Angle-normalized stats unavailable. Run compute_angle_stats() first.")
            degs = self.stats_angle_mean.index.to_numpy(dtype=float)
            fig, ax = plt.subplots()

            # Get a deterministic color list
            base_colors = matplotlib.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])

            if show_all_cycles and self.cycles:
                nC = len(self.cycles)
                # Gradient alphas for cycles (same base color as mean curve)
                grad_alphas = np.linspace(0.15, 0.6, max(nC, 2))
                for yi, y in enumerate(y_vars):
                    base = base_colors[yi % len(base_colors)]
                    grid_rad = np.deg2rad(degs)
                    for ci, (s, e) in enumerate(self.cycles):
                        seg = self.df_trim.iloc[s:e+1]
                        ang_u = seg[self.col_crank_rad_unwrapped].to_numpy(dtype=float)
                        ang_mod = (ang_u - ang_u[0]) % (2 * np.pi)
                        yv = seg[y].to_numpy(dtype=float)
                        y_interp = np.interp(grid_rad, ang_mod, yv, left=np.nan, right=np.nan)
                        ax.plot(degs, y_interp, linewidth=0.8, color=base, alpha=float(grad_alphas[ci]), label=None)

            if show_mean_std:
                for yi, y in enumerate(y_vars):
                    base = base_colors[yi % len(base_colors)]
                    mu = self.stats_angle_mean[y].to_numpy(dtype=float)
                    sd = self.stats_angle_std[y].to_numpy(dtype=float)
                    ax.plot(degs, mu, linewidth=2.0, color=base, label=f"{y} (mean)")
                    ax.fill_between(degs, mu - sd, mu + sd, alpha=0.18, color=base, label=f"{y} (±1 SD)")

            ax.set_xlabel("Crank angle [deg]")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Angle-normalized (0–360°)")
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            plt.show()
            return

        # --- CycleIndex (cycle-level metrics) ---
        if x_axis == "CycleIndex":
            if self.cycle_summary is None or self.cycle_summary.empty:
                raise RuntimeError("No cycle_summary available. Run compute_per_cycle_metrics() first.")
            x = self.cycle_summary["cycle_index"].to_numpy(dtype=float)
            fig, ax = plt.subplots()
            markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h', 'X']
            for i, y in enumerate(y_vars):
                if y not in self.cycle_summary.columns:
                    raise ValueError(f"'{y}' is not a cycle-level metric.")
                m = markers[i % len(markers)]
                ax.plot(x, self.cycle_summary[y].to_numpy(dtype=float), marker=m, linewidth=1.2, label=y)
            ax.set_xlabel("Cycle index")
            ax.set_ylabel(", ".join(y_vars))
            ax.set_title("Cycle-level metrics")
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
            plt.show()
            return

    # ----- ROI Visualization -----
    def show_roi_figure(self):
        if not self._roi_cache:
            raise RuntimeError("Run trim_to_steady_speed() first.")
        t = self._roi_cache["t"]
        w = self._roi_cache["w"]
        active = self._roi_cache["active_mask"]
        mu = self._roi_cache["mu_active"]
        idx0 = self._roi_cache["idx0"]
        idx1 = self._roi_cache["idx1"]
        thr_deg_s = self._roi_cache["thr_deg_s"]
        tol_ratio = self._roi_cache["tol_ratio"]
        t0, t1 = t[idx0], t[idx1]
        fig, ax = plt.subplots()
        ax.plot(t, w, label="Angular velocity [rad/s]", linewidth=1.2)
        ax.plot(t[active], w[active], '.', alpha=0.3, label=f"Active (|w|>{thr_deg_s:.1f}°/s)")
        ax.axhline(mu, color='k', linestyle='--', alpha=0.7, label=f"mean active = {mu:.3f} rad/s")
        ax.axvspan(t0, t1, color='orange', alpha=0.2, label=f"Auto ROI (±{tol_ratio*100:.1f}%)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular velocity [rad/s]")
        ax.set_title("ROI: active mask, mean active speed, and auto-selected window — drag to select manual ROI")
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc="best")
        return fig, ax


# ------------------------------
# Tkinter GUI
# ------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arm-Crank Analysis")
        self.geometry("980x780")

        self.filepath = tk.StringVar(value="")
        self.analyzer = None

        # 1) File chooser
        frm_file = ttk.LabelFrame(self, text="1) Data file (.lvm)")
        frm_file.pack(fill="x", padx=10, pady=8)
        ttk.Entry(frm_file, textvariable=self.filepath).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(frm_file, text="Browse...", command=self.browse_file).pack(side="left", padx=6, pady=6)

        # 2) Parameters
        frm_params = ttk.LabelFrame(self, text="2) Parameters (ROI detection)")
        frm_params.pack(fill="x", padx=10, pady=8)

        self.tol_pct = tk.DoubleVar(value=5.0)         # ±%
        self.nonzero_thr_deg = tk.DoubleVar(value=2.0)  # deg/s

        ttk.Label(frm_params, text="Tolerance around mean (±%)").grid(row=0, column=0, sticky="w", padx=6)
        tol_slider = tk.Scale(frm_params, from_=0.5, to=20.0, resolution=0.5, orient="horizontal",
                              variable=self.tol_pct, length=260)
        tol_slider.grid(row=0, column=1, sticky="we", padx=6)

        ttk.Label(frm_params, text="Non-zero speed threshold (deg/s)").grid(row=1, column=0, sticky="w", padx=6)
        thr_slider = tk.Scale(frm_params, from_=0.0, to=50.0, resolution=0.1, orient="horizontal",
                              variable=self.nonzero_thr_deg, length=260)
        thr_slider.grid(row=1, column=1, sticky="we", padx=6)

        self.minsec_var = tk.DoubleVar(value=1.0)
        ttk.Label(frm_params, text="Min steady segment (s)").grid(row=2, column=0, sticky="w", padx=6)
        ttk.Entry(frm_params, width=8, textvariable=self.minsec_var).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Button(frm_params, text="Run Pipeline", command=self.run_pipeline).grid(row=0, column=2, rowspan=3, padx=10, pady=4, sticky="ns")

        # 3) Plotting
        frm_plot = ttk.LabelFrame(self, text="3) Plotting")
        frm_plot.pack(fill="both", expand=True, padx=10, pady=8)

        self.fig_type = tk.StringVar(value="XY")
        ttk.Label(frm_plot, text="Figure type:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(frm_plot, values=["XY", "Polar"], textvariable=self.fig_type, width=12).grid(row=0, column=1, sticky="w")

        self.x_axis = tk.StringVar(value="Time")
        ttk.Label(frm_plot, text="X axis:").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        self.cmb_xaxis = ttk.Combobox(frm_plot, values=["Time", "NormalizedCycle%", "CrankAngle", "CrankAngle360°", "CycleIndex"],
                                      textvariable=self.x_axis, width=18)
        self.cmb_xaxis.grid(row=0, column=3, sticky="w")
        self.cmb_xaxis.bind("<<ComboboxSelected>>", self._on_xaxis_changed)

        # Checkboxes to (de)activate depending on x-axis
        self.chk_all_cycles_var = tk.BooleanVar(value=False)
        self.chk_mean_std_var = tk.BooleanVar(value=True)
        self.chk_all_cycles_btn = ttk.Checkbutton(frm_plot, text="Show all cycles", variable=self.chk_all_cycles_var)
        self.chk_mean_std_btn = ttk.Checkbutton(frm_plot, text="Show mean ± std", variable=self.chk_mean_std_var)
        self.chk_all_cycles_btn.grid(row=1, column=0, sticky="w", padx=6)
        self.chk_mean_std_btn.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(frm_plot, text="Y variables (multi-select):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.lst_y = tk.Listbox(frm_plot, selectmode="extended", width=60, height=16, exportselection=False)
        self.lst_y.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=6, pady=6)
        frm_plot.grid_columnconfigure(0, weight=1)
        frm_plot.grid_rowconfigure(3, weight=1)

        ttk.Button(frm_plot, text="Plot", command=self.plot).grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Button(frm_plot, text="Export cycle_summary.csv", command=self.export_cycle_summary).grid(row=4, column=2, sticky="e", padx=6, pady=6)
        ttk.Button(frm_plot, text="Export normalized_stats.csv", command=self.export_normalized_stats).grid(row=4, column=3, sticky="e", padx=6, pady=6)

        # 4) ROI controls
        frm_roi = ttk.LabelFrame(self, text="4) ROI (Region Of Interest)")
        frm_roi.pack(fill="x", padx=10, pady=8)
        ttk.Button(frm_roi, text="Afficher ROI & Sélection manuelle", command=self.show_and_select_roi).pack(side="left", padx=6, pady=6)
        ttk.Button(frm_roi, text="Recalculer après ROI manuel", command=self.recompute_after_manual_roi).pack(side="left", padx=6, pady=6)

        self.status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status, foreground="gray").pack(fill="x", padx=10, pady=4)

        self._manual_roi = None
        # Initialize control states for default x-axis ("Time")
        self._apply_xaxis_dependent_states()

    # ----- GUI helpers -----
    def _apply_xaxis_dependent_states(self):
        xa = self.x_axis.get()
        if xa == "Time":
            # Disable checkboxes
            self.chk_all_cycles_btn.state(["disabled"])
            self.chk_mean_std_btn.state(["disabled"])
        else:
            # Enable checkboxes
            self.chk_all_cycles_btn.state(["!disabled"])
            self.chk_mean_std_btn.state(["!disabled"])
        # Update Y list
        self._refresh_y_list()

    def _on_xaxis_changed(self, event):
        self._apply_xaxis_dependent_states()

    def browse_file(self):
        path = filedialog.askopenfilename(title="Select .lvm file",
                                          filetypes=[("LabVIEW Measurement", "*.lvm"), ("All files", "*.*")])
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
                steady_tol_ratio=float(self.tol_pct.get())/100.0,
                non_zero_thr_deg_s=float(self.nonzero_thr_deg.get()),
                min_steady_seconds=float(self.minsec_var.get()),
                verbose=True
            )
            self.analyzer = ana
            self._apply_xaxis_dependent_states()
            msg = (f"Sampling: {ana.sampling_hz:.2f} Hz | Auto ROI: "
                   f"[{ana.trim_start_idx}, {ana.trim_end_idx}] | Cycles: {len(ana.cycles)} | "
                   f"thr={self.nonzero_thr_deg.get():.1f}°/s, tol=±{self.tol_pct.get():.1f}%")
            self.status.set(msg)
            messagebox.showinfo("Pipeline complete", msg)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _refresh_y_list(self):
        self.lst_y.delete(0, tk.END)
        if self.analyzer is None:
            return
        xa = self.x_axis.get()
        if xa == "CycleIndex":
            for c in self.analyzer.list_cycle_metrics():
                self.lst_y.insert(tk.END, c)
        else:
            for c in self.analyzer.list_numeric_columns():
                self.lst_y.insert(tk.END, c)

    def selected_y_vars(self):
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
                show_all_cycles=self.chk_all_cycles_var.get(),
                show_mean_std=self.chk_mean_std_var.get()
            )
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    def export_cycle_summary(self):
        if self.analyzer is None or self.analyzer.cycle_summary is None:
            messagebox.showerror("Error", "No cycle_summary available. Run the pipeline first.")
            return
        path = filedialog.asksaveasfilename(title="Save cycle_summary.csv",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
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
        base = filedialog.asksaveasfilename(title="Save normalized_stats (base name)",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
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

    # ----- ROI selection -----
    def show_and_select_roi(self):
        if self.analyzer is None:
            messagebox.showerror("Error", "Run the pipeline first.")
            return
        try:
            fig, ax = self.analyzer.show_roi_figure()
            ax.set_title(ax.get_title() + " — Drag to select manual ROI")
            _ = SpanSelector(ax, onselect=self._on_span_select, direction='horizontal',
                             useblit=True, interactive=True, props=dict(alpha=0.25))
            plt.show()
        except Exception as e:
            messagebox.showerror("ROI error", str(e))

    def _on_span_select(self, t_min, t_max):
        self._manual_roi = (min(t_min, t_max), max(t_min, t_max))
        self.status.set(f"Manual ROI selected: {self._manual_roi[0]:.3f}s → {self._manual_roi[1]:.3f}s")

    def recompute_after_manual_roi(self):
        if self.analyzer is None:
            messagebox.showerror("Error", "Run the pipeline first.")
            return
        if self._manual_roi is None:
            messagebox.showerror("Error", "Select a manual ROI first (button above).")
            return
        try:
            t0, t1 = self._manual_roi
            self.analyzer.set_manual_trim(t0, t1)
            # Recompute downstream
            self.analyzer.add_crank_angle_radians()
            self.analyzer.unwrap_and_compute_ang_vel()
            self.analyzer.compute_force_norms()
            self.analyzer.detect_cycles_from_unwrap()
            self.analyzer.compute_cycle_stats(n_points=100)
            self.analyzer.compute_angle_stats(n_degrees=360)
            self.analyzer.compute_per_cycle_metrics()
            self._refresh_y_list()
            msg = (f"Manual ROI applied | Segment: "
                   f"[{self.analyzer.trim_start_idx}, {self.analyzer.trim_end_idx}] | Cycles: {len(self.analyzer.cycles)}")
            self.status.set(msg)
            messagebox.showinfo("ROI applied", msg)
        except Exception as e:
            messagebox.showerror("ROI apply error", str(e))


# ------------------------------
# CLI Entrypoint
# ------------------------------

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Loading: {path}")
        df = read_lvm(path)
        ana = ArmCrankAnalyzer(df)
        # Defaults align with GUI defaults
        ana.run_full_pipeline(steady_tol_ratio=0.05, non_zero_thr_deg_s=2.0, verbose=True)

        base_no_ext, _ = os.path.splitext(path)
        out_cycles = base_no_ext + "_cycle_summary.csv"
        out_mean   = base_no_ext + "_normalized_mean.csv"
        out_std    = base_no_ext + "_normalized_std.csv"
        out_ang_mu = base_no_ext + "_angle_mean_360.csv"
        out_ang_sd = base_no_ext + "_angle_std_360.csv"

        ana.cycle_summary.to_csv(out_cycles, index=False)
        ana.stats_mean.to_csv(out_mean, index_label="Cycle %")
        ana.stats_std.to_csv(out_std, index_label="Cycle %")
        if ana.stats_angle_mean is not None:
            ana.stats_angle_mean.to_csv(out_ang_mu, index_label="Angle [deg]")
            ana.stats_angle_std.to_csv(out_ang_sd, index_label="Angle [deg]")

        print("Exported:")
        print(" ", out_cycles)
        print(" ", out_mean)
        print(" ", out_std)
        print(" ", out_ang_mu)
        print(" ", out_ang_sd)
    else:
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()