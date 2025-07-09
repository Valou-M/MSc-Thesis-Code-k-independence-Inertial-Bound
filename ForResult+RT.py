#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:38:51 2025

@author: v_m
"""

import csv, time, sys, pathlib
from sage.all import *           # full Sage namespace
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import random
from itertools import combinations
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
# MILP solver wrapped in a function
# ──────────────────────────────────────────────────────────────────────────
def solve_graph(G, k: int = 3, big_M: float = 1_000, eps: float = 1.0):
    """Return (objective_value, run_time_seconds) or (None, run_time) if not optimal."""
    A = np.array(G.adjacency_matrix(), dtype=float)
    t0 = time.time()

    # 1) eigen-analysis
    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0
    eigs = np.round(eigs, 10)
    uniq, mult = np.unique(eigs, return_counts=True)
    order = np.argsort(-uniq)          # descending
    theta = uniq[order]
    m      = mult[order]
    d = len(theta) - 1

    # 2) pre-compute powers
    A_pow  = [np.linalg.matrix_power(A, i) for i in range(k + 1)]
    th_pow = [[theta[j] ** i for i in range(k + 1)] for j in range(d + 1)]

    # 3) build MILP
    mdl = gp.Model(f"MILP3_{G.name()}")
    mdl.Params.OutputFlag        = 0       # silence solver
    mdl.Params.IntegralityFocus  = 1
    mdl.Params.IntFeasTol        = 0.1

    a = mdl.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = mdl.addVars(d + 1, vtype=GRB.BINARY, name="b")

    mdl.setObjective(gp.quicksum(int(m[j]) * b[j] for j in range(d + 1)),
                     GRB.MINIMIZE)

    # diagonal-entry constraints
    for v in range(len(A)):
        mdl.addConstr(gp.quicksum(a[i] * A_pow[i][v, v]
                                  for i in range(k + 1)) >= 0)

    # eigen-value constraints
    for j in range(d + 1):
        mdl.addConstr(
            gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))
            - big_M * b[j] + eps <= 0
        )

    mdl.optimize()
    run_time = time.time() - t0
    return (mdl.ObjVal, run_time) if mdl.status == GRB.OPTIMAL else (None, run_time)

# ──────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────
def main(list_file="graph_list.txt", out_csv="milp_results_new.csv"):
    list_path = pathlib.Path(list_file)
    if not list_path.exists():
        sys.exit(f"List file '{list_file}' not found.")

    results = []
    for raw_name in list_path.read_text().splitlines():
        name = raw_name.strip()
        if not name or name.startswith("#"):
            continue        # skip blanks / comments

        try:
            G = getattr(graphs, name)()
        except AttributeError:
            print(f"[skip] {name}: not in sage.graphs.*")
            continue

        print(f"[solve] {name:30s}", end=" … ", flush=True)
        obj, secs = solve_graph(G)
        if obj is None:
            print("no OPTIMAL solution")
        else:
            print(f"obj = {obj:.0f},  t = {secs:.2f}s")
        results.append((name, obj, secs))

    # write CSV
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph", "objective", "time_seconds"])
        w.writerows(results)

    print(f"\nFinished ✔  →  {out_csv} written with {len(results)} rows.")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()