#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 10:19:52 2025

@author: v_m
"""

from sage.all import *           # full Sage namespace
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import random
from itertools import combinations
import matplotlib.pyplot as plt

try:
    import networkx as nx
    _has_nx = True
except ImportError:
    _has_nx = False

# Generate the matrix
#A = generate_adjacency_matrix()

def build_adjacency_matrix_O4():
    """
    Build and return the adjacency matrix for O_3 = K(7, 3).
    Also returns the list of vertices (3-subsets).
    """
    # 1) Enumerate all 3-subsets of {1,2,...,7}
    vertices = list(combinations(range(1, 14), 6))  # There are C(7,3) = 35 subsets
    n = len(vertices)  # should be 35

    # 2) Initialize an n x n matrix of zeros
    A = np.zeros((n, n), dtype=int)

    # 3) Fill the matrix with 1 if the subsets are disjoint, otherwise 0
    for i in range(n):
        for j in range(i + 1, n):
            if set(vertices[i]).isdisjoint(vertices[j]):
                A[i, j] = 1
                A[j, i] = 1

    return A, vertices

def build_Gk_star(k):
    n = 1 + 4*k + 2           # total vertices
    A = [[0]*n for _ in range(n)]

    def add(i, j):            # undirected edge
        A[i][j] = A[j][i] = 1

    # add the k disjoint 4-cycles
    for c in range(k):
        v = 1 + 4*c
        cycle = [v, v+1, v+2, v+3]
        for u, w in zip(cycle, cycle[1:]+cycle[:1]):
            add(u, w)

    # add the K2
    add(1+4*k, 1+4*k+1)

    # connect the universal vertex 0 to every other vertex
    for v in range(1, n):
        add(0, v)

    return A

A = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
], dtype=float)
k = 2 # Given value for k

total_start = time.time()

# Function to calculate A^i
def matrix_power(A, i):
    return np.linalg.matrix_power(A, i)

# ─── 2.  eigen-values (ascending order) ───────────────────────────────────
eigs = np.linalg.eigh(A)[0]            # eigh returns them already sorted

# ─── 3.  squash tiny round-off to zero, round everything else once ────────
eigs[np.abs(eigs) < 1e-10] = 0
eigs = np.round(eigs, 10)              # keep 10-digit precision

# ─── 4.  distinct values & multiplicities, then order as you wish ────────
uniq, mult = np.unique(eigs, return_counts=True)

# descending (largest → smallest) order if that’s what you want:
order = np.argsort(-uniq)              # minus sign for descending
unique_eigenvalues = uniq [order]
m  = mult [order]

print("distinct eigen-values (↓):", unique_eigenvalues)
print("multiplicities:           ", m)
print("Check   ∑ multiplicities =", m.sum(), " = order of graph =", A.shape[0])

d = len(unique_eigenvalues) - 1

print("d is equal to", d)

# Placeholder for big-M value
M = 1000
e = 1

# Pre-calculate values for constraints
A_powers = [matrix_power(A, i) for i in range(k + 1)]
eigenvalue_powers = [[unique_eigenvalues[j] ** i for i in range(k + 1)] for j in range(d + 1)]

# Lists to store solutions for all MILPs
milp2_solutions = []

bestSol = float("inf")       
for u in range(len(A)):
    
    # MILP 2
    model2 = gp.Model(f"MILP2_vertex_{u}")
    a = model2.addVars(k + 1, lb=-GRB.INFINITY, name="a")
    b = model2.addVars(d + 1, vtype=GRB.BINARY, name="b")
    model2.setParam(GRB.Param.IntegralityFocus, 1)
    model2.setParam(GRB.Param.IntFeasTol, 0.1)

    # Objective: minimize m^T b
    model2.setObjective(gp.quicksum(m[j] * b[j] for j in range(d + 1)), GRB.MINIMIZE)

    # Constraints
    for v in range(len(A)):
        if v != u:
            model2.addConstr(gp.quicksum(a[i] * A_powers[i][v, v] for i in range(k + 1)) >= 0)

    model2.addConstr(gp.quicksum(a[i] * A_powers[i][u, u] for i in range(k + 1)) == 0)

    for j in range(d + 1):
        model2.addConstr(gp.quicksum(a[i] * eigenvalue_powers[j][i] for i in range(k + 1)) - M * b[j] + e <= 0)
        
    # Optimize model
    model2.optimize()

    # Save solution
    if model2.status == GRB.OPTIMAL:
        if model2.objVal < bestSol:
            milp2_solutions.clear()
            milp2_solutions.append({
                'a': [a[i].x for i in range(k + 1)],
                'b': [b[j].x for j in range(d + 1)],
                'objective': model2.objVal
            })
            bestSol = model2.objVal


total_time2 = time.time() - total_start

print("\nMILP 2 Solutions:")
for solution in milp2_solutions:
    print(solution)

if milp2_solutions:
    min_obj_milp2 = min(sol['objective'] for sol in milp2_solutions)
    print(f"\nSmallest objective value for MILP 2: {min_obj_milp2}")
   
print(f"time MILP: {total_time2}")

""" print("\nUnique first set of equations with coefficients and variables:")
unique_equations = set()  # To store unique equation strings

for v in range(len(A)):
    eq_terms = []
    for i in range(k + 1):
        coeff = A_powers[i][v, v]
        # Optionally include only nonzero coefficients
        if coeff != 0:
            eq_terms.append(f"{coeff}*a{i}")
    # Build the equation string, using "0" if no nonzero terms exist.
    if eq_terms:
        equation = " + ".join(eq_terms) + " >= 0"
    else:
        equation = "0 >= 0"
   
    # Add to the set only if this equation wasn't printed before.
    if equation not in unique_equations:
        unique_equations.add(equation)
        print(equation)
"""
# ─────────────────────────────────────────────────────────────────────────
# Pretty utilities
# ─────────────────────────────────────────────────────────────────────────
def _fmt(val: float) -> str:
    """Return a string with exactly three decimals (rounded)."""
    return f"{val:.3f}"

# ─────────────────────────────────────────────────────────────────────────
# 1)  Eigen-values with multiplicities
# ─────────────────────────────────────────────────────────────────────────
theta_strings = [f"{_fmt(ev)}^({int(mult)})"
                 for ev, mult in zip(unique_eigenvalues, m)]

print("\nEigenvalues with multiplicities (θ^(m)):")
print(", ".join(theta_strings))

# ─────────────────────────────────────────────────────────────────────────
# 2)  Complete MILP in “min …  st …” form  (with deduplication)
# ─────────────────────────────────────────────────────────────────────────
print("\n========================  MILP 3  ========================")

# ─── Objective ───
print("\nmin")
obj_terms = [f"{int(m[j])} b{j}" if m[j].is_integer() else f"{m[j]:g} b{j}"
             for j in range(d + 1) if m[j] != 0]
print("    " + " + ".join(obj_terms))

# ─── Constraints ───
print("\nst")

# … first family (keep only unique) …
unique_equations = set()
for v in range(len(A)):
    lhs_terms = [f"{int(A_powers[i][v, v])} a{i}"
                 for i in range(k + 1) if A_powers[i][v, v] != 0]
    eq = " + ".join(lhs_terms) + " ≥ 0" if lhs_terms else "0 ≥ 0"
    if eq not in unique_equations:
        unique_equations.add(eq)
        print("    " + eq)

# … second family (one per eigen-value) …
for j in range(d + 1):
    lhs_terms = [f"{_fmt(eigenvalue_powers[j][i])} a{i}"
                 for i in range(k + 1) if eigenvalue_powers[j][i] != 0]
    lhs  = " + ".join(lhs_terms)
    rhs  = f"M b{j} + ϵ"
    print(f"    {lhs} - {rhs} ≤ 0")

# ─── Variable type ───
print("\nvariables")
print(f"    b_{{j}} ∈ {{0,1}}    for j = 0,…,{d}")

print("\nend\n")



def plot_graph(adj_matrix,
               layout: str = "spring",   # 'spring', 'circular', 'spectral', 'random'
               node_labels=None,
               node_size: int = 400,
               figsize: tuple = (6, 6),
               **draw_kwargs):
    """
    Plot an undirected graph from an adjacency matrix.

    Parameters
    ----------
    adj_matrix : array-like, shape (n, n)
        0/1 or weighted adjacency matrix (symmetry is assumed for undirected).
    layout : str, optional
        Layout to use when networkx is available: 'spring', 'circular',
        'spectral', or 'random'.  Ignored when networkx is absent.
    node_labels : list or None, optional
        Labels for the nodes; if None, nodes are labelled 0..n-1.
    node_size : int, optional
        Size of the nodes passed through to `nx.draw` / `plt.scatter`.
    figsize : tuple, optional
        Figure size in inches.
    **draw_kwargs :
        Extra keyword arguments forwarded to the drawing routine, e.g.
        `edge_color="gray", font_size=10`.
    """
    A = np.asarray(adj_matrix)
    n = A.shape[0]

    if node_labels is None:
        node_labels = list(range(n))

    plt.figure(figsize=figsize)

    if _has_nx:
        # ----- NetworkX branch (recommended) -----
        G = nx.from_numpy_array(A)
        layouts = {
            "spring":   nx.spring_layout,
            "circular": nx.circular_layout,
            "spectral": nx.spectral_layout,
            "random":   nx.random_layout,
        }
        pos = layouts.get(layout, nx.spring_layout)(G)

        nx.draw(G, pos,
                with_labels=True,
                labels={i: lbl for i, lbl in enumerate(node_labels)},
                node_size=node_size,
                **draw_kwargs)
    else:
        # ----- Fallback: simple circular layout -----
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.c_[np.cos(theta), np.sin(theta)]
        # draw edges
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] != 0:
                    plt.plot([pos[i, 0], pos[j, 0]],
                             [pos[i, 1], pos[j, 1]],
                             color=draw_kwargs.get("edge_color", "k"),
                             linewidth=draw_kwargs.get("linewidth", 1))
        # draw nodes
        plt.scatter(pos[:, 0], pos[:, 1],
                    s=node_size,
                    color=draw_kwargs.get("node_color", "#1f77b4"),
                    zorder=3)
        for i, lbl in enumerate(node_labels):
            plt.text(pos[i, 0], pos[i, 1], str(lbl),
                     ha="center", va="center",
                     fontsize=draw_kwargs.get("font_size", 10),
                     color=draw_kwargs.get("font_color", "w"))

        plt.axis("off")

    plt.show()


plot_graph(A, layout="spring", node_labels=[f"v{i+1}" for i in range(len(A))],
           node_size=600, edge_color="gray")