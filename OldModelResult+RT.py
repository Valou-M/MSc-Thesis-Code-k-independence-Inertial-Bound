#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:44:29 2025

@author: v_m
"""

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
# Solver for one graph
# ──────────────────────────────────────────────────────────────────────────
def solve_graph(G, k: int = 3, big_M: float = 1_000, eps: float = 1.0):
    """
    Solve MILP 2 for *every* vertex u of `G` and return

        (best_objective_value, elapsed_time_seconds)

    If *no* MILP reaches OPTIMAL status, return (None, elapsed_seconds).
    """
    A = np.array(G.adjacency_matrix(), dtype=float)
    n = len(A)
    t0 = time.time()                          # stopwatch ──┐

    # ── eigen-data ────────────────────────────────────────────────────────
    eigs = np.linalg.eigh(A)[0]
    eigs[np.abs(eigs) < 1e-10] = 0            # squash tiny noise
    eigs = np.round(eigs, 10)
    uniq, mult = np.unique(eigs, return_counts=True)
    order      = np.argsort(-uniq)            # descending
    theta      = uniq [order]
    m          = mult [order]
    d          = len(theta) - 1

    # ── pre-compute powers used in constraints ────────────────────────────
    A_pow  = [np.linalg.matrix_power(A, i) for i in range(k + 1)]
    th_pow = [[theta[j] ** i for i in range(k + 1)]
              for j in range(d + 1)]

    best_obj = float("inf")                   # store best objective so far

    # ── loop over every vertex u ──────────────────────────────────────────
    for u in range(n):
        mdl = gp.Model(f"MILP2_{G.name()}_u{u}")
        mdl.Params.OutputFlag       = 0       # silence solver chatter
        mdl.Params.IntegralityFocus = 1
        mdl.Params.IntFeasTol       = 0.1

        a = mdl.addVars(k + 1, lb=-GRB.INFINITY, name="a")
        b = mdl.addVars(d + 1, vtype=GRB.BINARY, name="b")

        # Objective   min Σ m_j  ·  b_j
        mdl.setObjective(gp.quicksum(int(m[j]) * b[j] for j in range(d + 1)),
                         GRB.MINIMIZE)

        # 1)  Non-negative diagonal entries except vertex u
        for v in range(n):
            if v == u:
                continue
            mdl.addConstr(gp.quicksum(a[i] * A_pow[i][v, v]
                                      for i in range(k + 1)) >= 0)

        # 2)  Vertex u diagonal equality
        mdl.addConstr(gp.quicksum(a[i] * A_pow[i][u, u]
                                  for i in range(k + 1)) == 0)

        # 3)  Eigen-value family
        for j in range(d + 1):
            mdl.addConstr(
                gp.quicksum(a[i] * th_pow[j][i] for i in range(k + 1))
                - big_M * b[j] + eps <= 0
            )

        mdl.optimize()

        if mdl.status == GRB.OPTIMAL and mdl.ObjVal < best_obj:
            best_obj = mdl.ObjVal

    elapsed = time.time() - t0               # stopwatch ──┘
    return (best_obj if best_obj < float("inf") else None, elapsed)

# ──────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────
def main(list_file="graph_list.txt", out_csv="milp_results_old.csv"):
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