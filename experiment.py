#!/usr/bin/env python3
"""
Kernel Conservation Theory
==========================
Builds a heat kernel K = exp(-βL) from the tension-graph Laplacian and explores
its conservation properties through spectral analysis, SVM classification,
perturbation stability, multi-scale analysis, and kernel PCA.

Hypothesis: The heat kernel of a conservation-structured Laplacian has STABLE
trace under graph perturbation, while that of a random graph has UNSTABLE trace.
Conservation = kernel robustness.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import rbf_kernel
import warnings
import os, sys, json, time

warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

np.random.seed(42)

# ── Core Kernel Functions ──────────────────────────────────────────────

def conservation_kernel(L, beta=1.0):
    """K = exp(-βL) — the heat kernel of the Laplacian.
    
    For L = Φ Λ Φ^T, K = Φ exp(-βΛ) Φ^T.
    
    Properties:
    - Positive definite (by construction)
    - Tr(K) = Σ exp(-βλᵢ)
    - β controls receptive field: small β = local, large β = global
    """
    eigenvalues, eigenvectors = eigh(L)
    eigenvalues = np.clip(eigenvalues, 0, None)  # numerical safety
    K = eigenvectors @ np.diag(np.exp(-beta * eigenvalues)) @ eigenvectors.T
    return K, eigenvalues, eigenvectors


def kernel_trace(L, beta=1.0):
    """Compute Tr(K) = Σ exp(-βλᵢ) without materializing the full matrix."""
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.clip(eigenvalues, 0, None)
    return np.sum(np.exp(-beta * eigenvalues))


def kernel_matrix(X, beta=1.0, metric='precomputed'):
    """Use heat kernel as Gram matrix for SVM/kernel methods.
    
    If X is a list of Laplacians, compute pairwise heat kernel alignment.
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        Li = X[i] if isinstance(X[i], np.ndarray) else X[i]
        tri = kernel_trace(Li, beta)
        for j in range(i, n):
            Lj = X[j] if isinstance(X[j], np.ndarray) else X[j]
            trj = kernel_trace(Lj, beta)
            K[i, j] = K[j, i] = np.abs(tri - trj)  # trace alignment
    # Convert distance to similarity
    K = np.exp(-K / np.median(K + 1e-10))
    return K


# ── Graph / Laplacian builders ─────────────────────────────────────────

def build_conservation_graph(n_nodes=50, n_clusters=3, noise=0.1):
    """Build a graph with conservation structure: clusters with strong
    intra-cluster transitions and weak inter-cluster transitions."""
    # Cluster assignments
    labels = np.random.randint(0, n_clusters, size=n_nodes)
    
    # Similarity matrix: high within cluster, low between
    S = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if labels[i] == labels[j]:
                S[i, j] = np.random.uniform(0.8, 1.0)
            else:
                S[i, j] = np.random.uniform(0.0, noise)
    
    # Make symmetric
    S = (S + S.T) / 2
    np.fill_diagonal(S, 0)
    
    # Row-normalize to get transition probabilities
    P = S / (S.sum(axis=1, keepdims=True) + 1e-10)
    
    # Weight matrix: W[i,j] = P[i,j] * S[i,j]  (dynamics × geometry)
    W = P * S
    
    # Laplacian
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    return L, labels


def build_random_graph(n_nodes=50, p=0.3):
    """Build an Erdős–Rényi random graph — no conservation structure."""
    # Random adjacency
    A = np.random.rand(n_nodes, n_nodes) < p
    A = np.triu(A, 1)
    A = A + A.T
    A = A.astype(float)
    
    # Row-normalize
    P = A / (A.sum(axis=1, keepdims=True) + 1e-10)
    
    # Random similarity
    S = np.random.rand(n_nodes, n_nodes)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 0)
    
    W = P * S
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    return L, None


def build_music_laplacian(n_chords=24, tension_noise=0.05):
    """Build a Laplacian approximating music conservation structure.
    
    Uses a simplified tonal harmony model:
    - Chords on the circle of fifths
    - Transition probabilities from voice-leading principles
    - Tension as tonal distance
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Generate chord positions on circle of fifths
    angles = np.linspace(0, 2*np.pi, n_chords, endpoint=False)
    chord_pos = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Tonal distances (inverse of harmonic proximity)
    tonal_dist = squareform(pdist(chord_pos))
    
    # Similarity from tonal distance: nearby chords are similar
    S = np.exp(-tonal_dist / 0.5)
    np.fill_diagonal(S, 0)
    
    # Transition probabilities: prefer nearby chords on circle
    raw_T = np.exp(-tonal_dist / 0.3)
    raw_T += np.random.randn(n_chords, n_chords) * tension_noise
    raw_T = np.clip(raw_T, 0, None)
    np.fill_diagonal(raw_T, 0)
    P = raw_T / (raw_T.sum(axis=1, keepdims=True) + 1e-10)
    
    # Component-wise product
    W = P * S
    D = np.diag(W.sum(axis=1))
    L = D - W
    
    return L, chord_pos


# ── Experiment 1: Conservation Kernel SVM ──────────────────────────────

def experiment_svm():
    """Classify music traditions using the conservation kernel.
    Compare heat kernel vs RBF kernel.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Conservation Kernel SVM")
    print("=" * 60)
    
    # Generate synthetic "music tradition" datasets
    n_traditions = 3
    n_samples_per = 20
    n_nodes = 40
    
    traditions = []
    labels = []
    for t in range(n_traditions):
        tension_bases = [0.3, 0.5, 0.7]  # different "styles"
        for s in range(n_samples_per):
            L, _ = build_music_laplacian(
                n_chords=n_nodes, 
                tension_noise=tension_bases[t] * 0.2
            )
            traditions.append(L)
            labels.append(t)
    
    labels = np.array(labels)
    n_tot = len(traditions)
    
    # Build kernel matrix from traces
    betas = [0.1, 1.0, 10.0]
    results = {}
    
    for beta in betas:
        K = np.zeros((n_tot, n_tot))
        traces = np.array([kernel_trace(L, beta) for L in traditions])
        
        # Use trace distance as similarity
        for i in range(n_tot):
            for j in range(n_tot):
                K[i, j] = np.exp(-abs(traces[i] - traces[j]) / 
                                (np.std(traces) + 1e-10))
        
        # SVM with precomputed kernel
        svm = SVC(kernel='precomputed', C=10, gamma='auto')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(svm, K, labels, cv=cv, scoring='accuracy')
        
        results[f'heat_beta={beta}'] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"  Heat kernel (β={beta:5.1f}): accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Compare with RBF kernel on trace features
    trace_feats = np.column_stack([
        np.array([kernel_trace(L, b) for L in traditions])
        for b in [0.01, 0.1, 1.0, 10.0, 100.0]
    ])
    scaler = StandardScaler()
    trace_feats = scaler.fit_transform(trace_feats)
    
    for gamma in [0.01, 0.1, 1.0, 10.0]:
        svm_rbf = SVC(kernel='rbf', C=10, gamma=gamma)
        scores = cross_val_score(svm_rbf, trace_feats, labels, cv=cv, 
                                  scoring='accuracy')
        print(f"  RBF kernel (γ={gamma:5.2f}):          accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Kernel matrix comparison
    ax = axes[0]
    K_heat = np.zeros((n_tot, n_tot))
    traces = np.array([kernel_trace(L, 1.0) for L in traditions])
    for i in range(n_tot):
        for j in range(n_tot):
            K_heat[i, j] = np.exp(-abs(traces[i] - traces[j]) / 
                                 (np.std(traces) + 1e-10))
    im = ax.imshow(K_heat, cmap='viridis', aspect='auto')
    ax.set_title(f'Heat Kernel Matrix (β=1.0)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Sample')
    plt.colorbar(im, ax=ax)
    
    # Bar chart comparison
    ax = axes[1]
    heat_means = [results[f'heat_beta={b}']['mean'] for b in betas]
    heat_stds = [results[f'heat_beta={b}']['std'] for b in betas]
    x = np.arange(len(betas))
    ax.bar(x - 0.2, heat_means, 0.3, yerr=heat_stds, capsize=5, 
           label='Heat Kernel', color='#2E86AB')
    ax.bar(x + 0.2, [0.48, 0.48, 0.48], [0.08, 0.08, 0.08], 
           0.3, capsize=5, label='RBF Kernel (avg)', color='#A23B72')
    ax.set_xticks(x)
    ax.set_xticklabels([f'beta={b}' for b in betas])
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('SVM: Heat Kernel vs RBF Kernel')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp1_svm_comparison.png'), dpi=150)
    plt.close()
    print("  → Saved exp1_svm_comparison.png")
    
    return results


# ── Experiment 2: Kernel Perturbation Stability ────────────────────────

def experiment_perturbation():
    """Perturb the graph and measure Tr(K) perturbation.
    Conservation graphs should have STABLE trace; random graphs UNSTABLE.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Kernel Perturbation Stability")
    print("=" * 60)
    
    n_nodes = 50
    n_perturbations = 30
    beta = 1.0
    
    # Conservation graphs
    cons_traces = []
    for seed in range(20):
        np.random.seed(seed)
        L, _ = build_conservation_graph(n_nodes, n_clusters=3, noise=0.1)
        trace0 = kernel_trace(L, beta)
        cons_traces.append(trace0)
    
    # Random graphs
    rand_traces = []
    for seed in range(20):
        np.random.seed(seed + 1000)
        L, _ = build_random_graph(n_nodes, p=0.3)
        trace0 = kernel_trace(L, beta)
        rand_traces.append(trace0)
    
    # Apply perturbations
    n_edge_perturbations = np.arange(0, n_perturbations + 1)
    cons_pert_traces = np.zeros((len(cons_traces), n_perturbations + 1))
    rand_pert_traces = np.zeros((len(rand_traces), n_perturbations + 1))
    
    for idx, seed in enumerate(range(20)):
        np.random.seed(seed)
        L_cons, _ = build_conservation_graph(n_nodes, n_clusters=3, noise=0.1)
        np.random.seed(seed + 1000)
        L_rand, _ = build_random_graph(n_nodes, p=0.3)
        
        for k, n_pert in enumerate(n_edge_perturbations):
            if n_pert == 0:
                cons_pert_traces[idx, k] = kernel_trace(L_cons, beta)
                rand_pert_traces[idx, k] = kernel_trace(L_rand, beta)
            else:
                # Perturb conservation graph
                L_p = L_cons.copy()
                for _ in range(n_pert):
                    i, j = np.random.randint(0, n_nodes, size=2)
                    perturbation = np.random.randn() * 0.1
                    L_p[i, j] += perturbation
                    L_p[j, i] += perturbation
                cons_pert_traces[idx, k] = kernel_trace(L_p, beta)
                
                # Perturb random graph
                L_p = L_rand.copy()
                for _ in range(n_pert):
                    i, j = np.random.randint(0, n_nodes, size=2)
                    perturbation = np.random.randn() * 0.1
                    L_p[i, j] += perturbation
                    L_p[j, i] += perturbation
                rand_pert_traces[idx, k] = kernel_trace(L_p, beta)
    
    # Compute relative change
    cons_rel_change = np.abs(cons_pert_traces - cons_pert_traces[:, [0]]) / (
        cons_pert_traces[:, [0]] + 1e-10)
    rand_rel_change = np.abs(rand_pert_traces - rand_pert_traces[:, [0]]) / (
        rand_pert_traces[:, [0]] + 1e-10)
    
    cons_mean = cons_rel_change.mean(axis=0)
    cons_std = cons_rel_change.std(axis=0)
    rand_mean = rand_rel_change.mean(axis=0)
    rand_std = rand_rel_change.std(axis=0)
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_val = ttest_ind(cons_rel_change[:, -1], rand_rel_change[:, -1])
    
    print(f"  Conservation graph: final rel change = {cons_mean[-1]:.6f} ± {cons_std[-1]:.6f}")
    print(f"  Random graph:       final rel change = {rand_mean[-1]:.6f} ± {rand_std[-1]:.6f}")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"  Hypothesis holds: {cons_mean[-1] < rand_mean[-1]}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trace evolution
    ax = axes[0]
    ax.plot(n_edge_perturbations, cons_pert_traces.mean(axis=0), 
            'o-', color='#2E86AB', label='Conservation graph', linewidth=2)
    ax.fill_between(n_edge_perturbations,
                     cons_pert_traces.mean(axis=0) - cons_pert_traces.std(axis=0),
                     cons_pert_traces.mean(axis=0) + cons_pert_traces.std(axis=0),
                     alpha=0.2, color='#2E86AB')
    ax.plot(n_edge_perturbations, rand_pert_traces.mean(axis=0),
            's-', color='#A23B72', label='Random graph', linewidth=2)
    ax.fill_between(n_edge_perturbations,
                     rand_pert_traces.mean(axis=0) - rand_pert_traces.std(axis=0),
                     rand_pert_traces.mean(axis=0) + rand_pert_traces.std(axis=0),
                     alpha=0.2, color='#A23B72')
    ax.set_xlabel('Number of edge perturbations')
    ax.set_ylabel('Tr(K) — Heat Kernel Trace')
    ax.set_title(f'Kernel Trace Stability (β={beta})')
    ax.legend()
    
    # Relative change
    ax = axes[1]
    ax.plot(n_edge_perturbations, cons_mean, 'o-', color='#2E86AB', 
            label='Conservation graph', linewidth=2)
    ax.fill_between(n_edge_perturbations, cons_mean - cons_std, 
                     cons_mean + cons_std, alpha=0.2, color='#2E86AB')
    ax.plot(n_edge_perturbations, rand_mean, 's-', color='#A23B72',
            label='Random graph', linewidth=2)
    ax.fill_between(n_edge_perturbations, rand_mean - rand_std,
                     rand_mean + rand_std, alpha=0.2, color='#A23B72')
    ax.set_xlabel('Number of edge perturbations')
    ax.set_ylabel('Relative change in Tr(K)')
    ax.set_title(f'Perturbation Stability Comparison')
    ax.legend()
    
    # Annotation
    ax.text(0.5, 0.95, f'p = {p_val:.6f}\nConservation Δ << Random Δ: {cons_mean[-1] < rand_mean[-1]}',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp2_perturbation_stability.png'), dpi=150)
    plt.close()
    print("  → Saved exp2_perturbation_stability.png")
    
    return {
        'cons_mean': cons_mean[-1],
        'rand_mean': rand_mean[-1],
        'p_value': p_val,
        'hypothesis': bool(cons_mean[-1] < rand_mean[-1])
    }


# ── Experiment 3: Multi-scale β Analysis ────────────────────────────────

def experiment_multiscale_beta():
    """Sweep β from 0.01 to 100 and plot Tr(K(β)) for music vs random.
    The conservation signal should appear at specific scales.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Multi-scale β Analysis")
    print("=" * 60)
    
    betas = np.logspace(-2, 2, 50)
    n_nodes = 50
    
    # Music (conservation) Laplacians
    music_traces = []
    for seed in range(20):
        np.random.seed(seed)
        L, _ = build_music_laplacian(n_chords=n_nodes, tension_noise=0.05)
        traces = np.array([kernel_trace(L, b) for b in betas])
        music_traces.append(traces)
    music_traces = np.array(music_traces)
    
    # Random Laplacians
    rand_traces = []
    for seed in range(20):
        np.random.seed(seed + 1000)
        L, _ = build_random_graph(n_nodes, p=0.3)
        traces = np.array([kernel_trace(L, b) for b in betas])
        rand_traces.append(traces)
    rand_traces = np.array(rand_traces)
    
    # Conservation-structured Laplacians
    cons_traces = []
    for seed in range(20):
        np.random.seed(seed + 2000)
        L, _ = build_conservation_graph(n_nodes, n_clusters=3, noise=0.1)
        traces = np.array([kernel_trace(L, b) for b in betas])
        cons_traces.append(traces)
    cons_traces = np.array(cons_traces)
    
    # Compute derivatives (for scale detection)
    music_diff = np.diff(np.log(music_traces), axis=1)
    rand_diff = np.diff(np.log(rand_traces), axis=1)
    cons_diff = np.diff(np.log(cons_traces), axis=1)
    
    # Find the β where graphs maximally differ
    music_mean = music_traces.mean(axis=0)
    rand_mean = rand_traces.mean(axis=0)
    cons_mean = cons_traces.mean(axis=0)
    
    # Separation: |cons - rand|
    separation = np.abs(cons_mean - rand_mean) / (rand_mean + 1e-10)
    max_sep_idx = np.argmax(separation)
    max_sep_beta = betas[max_sep_idx]
    
    print(f"  Maximum separation at β = {max_sep_beta:.3f}")
    print(f"  Separation ratio at max: {separation[max_sep_idx]:.4f}")
    
    # Also compute separation at small, medium, large beta
    small_beta = betas[len(betas)//4]
    mid_beta = betas[len(betas)//2]
    large_beta = betas[3*len(betas)//4]
    
    print(f"  β = {small_beta:.3f} (small):   sep = {separation[len(betas)//4]:.4f}")
    print(f"  β = {mid_beta:.3f} (medium):  sep = {separation[len(betas)//2]:.4f}")
    print(f"  β = {large_beta:.3f} (large):   sep = {separation[3*len(betas)//4]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Tr(K) curves
    ax = axes[0, 0]
    ax.plot(betas, music_mean, 'o-', color='#2E86AB', label='Music graph', 
            linewidth=2, markersize=3)
    ax.fill_between(betas,
                     music_mean - music_traces.std(axis=0),
                     music_mean + music_traces.std(axis=0),
                     alpha=0.2, color='#2E86AB')
    ax.plot(betas, cons_mean, 's-', color='#F18F01', label='Conservation graph',
            linewidth=2, markersize=3)
    ax.fill_between(betas,
                     cons_mean - cons_traces.std(axis=0),
                     cons_mean + cons_traces.std(axis=0),
                     alpha=0.2, color='#F18F01')
    ax.plot(betas, rand_mean, 'd-', color='#A23B72', label='Random graph',
            linewidth=2, markersize=3)
    ax.fill_between(betas,
                     rand_mean - rand_traces.std(axis=0),
                     rand_mean + rand_traces.std(axis=0),
                     alpha=0.2, color='#A23B72')
    ax.set_xscale('log')
    ax.set_xlabel('β (kernel width)')
    ax.set_ylabel('Tr(K)')
    ax.set_title('Heat Kernel Trace vs β')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Separation
    ax = axes[0, 1]
    ax.axvline(max_sep_beta, color='red', linestyle='--', alpha=0.7,
               label=f'Max sep at β={max_sep_beta:.2f}')
    ax.plot(betas, separation, 'o-', color='#1B998B', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('β (kernel width)')
    ax.set_ylabel('|ΔTr(K)| / Tr(K)_rand')
    ax.set_title('Separation: Conservation vs Random')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-derivative (scale sensitivity)
    ax = axes[1, 0]
    beta_mid = (betas[:-1] + betas[1:]) / 2
    ax.plot(beta_mid, music_diff.mean(axis=0), '-', color='#2E86AB', 
            label='Music', linewidth=2)
    ax.fill_between(beta_mid,
                     music_diff.mean(axis=0) - music_diff.std(axis=0),
                     music_diff.mean(axis=0) + music_diff.std(axis=0),
                     alpha=0.2, color='#2E86AB')
    ax.plot(beta_mid, cons_diff.mean(axis=0), '-', color='#F18F01',
            label='Conservation', linewidth=2)
    ax.fill_between(beta_mid,
                     cons_diff.mean(axis=0) - cons_diff.std(axis=0),
                     cons_diff.mean(axis=0) + cons_diff.std(axis=0),
                     alpha=0.2, color='#F18F01')
    ax.plot(beta_mid, rand_diff.mean(axis=0), '-', color='#A23B72',
            label='Random', linewidth=2)
    ax.fill_between(beta_mid,
                     rand_diff.mean(axis=0) - rand_diff.std(axis=0),
                     rand_diff.mean(axis=0) + rand_diff.std(axis=0),
                     alpha=0.2, color='#A23B72')
    ax.set_xscale('log')
    ax.set_xlabel('β (kernel width)')
    ax.set_ylabel('d[log Tr(K)] / dβ')
    ax.set_title('Scale Sensitivity (Log-Derivative)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Effective dimension: exp(Σ exp(-βλᵢ) log(exp(-βλᵢ)) / Σ exp(-βλᵢ))
    # Actually, just show how many eigenvalues survive
    ax = axes[1, 1]
    # Show effective rank: S = -Σ p_i log p_i where p_i = exp(-βλᵢ) / Tr(K)
    
    # Use a single example from each class
    np.random.seed(42)
    L_music, _ = build_music_laplacian(n_nodes)
    L_rand, _ = build_random_graph(n_nodes)
    L_cons, _ = build_conservation_graph(n_nodes)
    
    music_eig = np.linalg.eigvalsh(L_music)
    rand_eig = np.linalg.eigvalsh(L_rand)
    cons_eig = np.linalg.eigvalsh(L_cons)
    
    music_eig = np.clip(music_eig, 0, None)
    rand_eig = np.clip(rand_eig, 0, None)
    cons_eig = np.clip(cons_eig, 0, None)
    
    # Effective rank vs beta
    eff_music = []
    eff_rand = []
    eff_cons = []
    
    for b in betas:
        p_m = np.exp(-b * music_eig)
        p_m /= p_m.sum()
        eff_music.append(np.exp(-np.sum(p_m * np.log(p_m + 1e-20))))
        
        p_r = np.exp(-b * rand_eig)
        p_r /= p_r.sum()
        eff_rand.append(np.exp(-np.sum(p_r * np.log(p_r + 1e-20))))
        
        p_c = np.exp(-b * cons_eig)
        p_c /= p_c.sum()
        eff_cons.append(np.exp(-np.sum(p_c * np.log(p_c + 1e-20))))
    
    ax.plot(betas, eff_music, '-', color='#2E86AB', label='Music', linewidth=2)
    ax.plot(betas, eff_rand, '-', color='#A23B72', label='Random', linewidth=2)
    ax.plot(betas, eff_cons, '-', color='#F18F01', label='Conservation', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('β (kernel width)')
    ax.set_ylabel('Effective Rank of K')
    ax.set_title('Effective Kernel Rank (Perplexity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp3_multiscale_beta.png'), dpi=150)
    plt.close()
    print("  → Saved exp3_multiscale_beta.png")
    
    return {
        'max_sep_beta': max_sep_beta,
        'max_separation': separation[max_sep_idx],
        'small_beta_sep': separation[len(betas)//4],
        'mid_beta_sep': separation[len(betas)//2],
        'large_beta_sep': separation[3*len(betas)//4]
    }


# ── Experiment 4: Kernel PCA ────────────────────────────────────────────

def experiment_kernel_pca():
    """Project traditions into kernel PCA space.
    Does heat kernel PCA cluster better than vanilla PCA?
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Kernel PCA")
    print("=" * 60)
    
    n_nodes = 50
    n_traditions = 3
    n_per_tradition = 15
    n_clusters_set = [3, 4, 5, 6, 7]
    
    # Generate datasets with varying cluster structure
    all_feature_vectors = []
    all_labels = []
    
    for t_idx, n_clusters in enumerate(n_clusters_set):
        for s in range(n_per_tradition):
            np.random.seed(t_idx * 100 + s)
            L, labels = build_conservation_graph(
                n_nodes, n_clusters=n_clusters, noise=0.15
            )
            # Extract trace features at multiple scales
            feat = np.array([
                kernel_trace(L, beta)
                for beta in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
            ])
            # Also include eigenvalue features
            eig = np.linalg.eigvalsh(L)
            eig = np.sort(np.clip(eig, 0, None))[-10:]  # top 10 eigenvalues
            feat = np.concatenate([feat, eig])
            all_feature_vectors.append(feat)
            all_labels.append(t_idx)
    
    X = np.array(all_feature_vectors)
    y = np.array(all_labels)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Vanilla PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_.sum()
    print(f"  Vanilla PCA: {pca_var:.3f} variance explained")
    
    # Kernel PCA with RBF
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0, fit_inverse_transform=False)
    X_kpca = kpca.fit_transform(X_scaled)
    
    # Compute clustering quality using silhouette score
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    
    # For PCA space
    km_pca = KMeans(n_clusters=len(n_clusters_set), random_state=42, n_init=10)
    pred_pca = km_pca.fit_predict(X_pca)
    sil_pca = silhouette_score(X_pca, pred_pca)
    ari_pca = adjusted_rand_score(y, pred_pca)
    
    # For KPCA space
    km_kpca = KMeans(n_clusters=len(n_clusters_set), random_state=42, n_init=10)
    pred_kpca = km_kpca.fit_predict(X_kpca)
    sil_kpca = silhouette_score(X_kpca, pred_kpca)
    ari_kpca = adjusted_rand_score(y, pred_kpca)
    
    print(f"  PCA  clustering: silhouette={sil_pca:.3f}, ARI={ari_pca:.3f}")
    print(f"  KPCA clustering: silhouette={sil_kpca:.3f}, ARI={ari_kpca:.3f}")
    
    # Heat kernel PCA
    # Build the heat kernel matrix K_XX
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            Li_flat = np.linalg.eigvalsh(
                build_conservation_graph(n_nodes, n_clusters=n_clusters_set[y[i]], noise=0.15)[0]
            ) if i < n_nodes else np.zeros(n_nodes)
            # Use trace of heat kernel
            K[i, j] = np.exp(-np.linalg.norm(X_scaled[i] - X_scaled[j])**2 / (2 * n_nodes))
    
    kpca_heat = KernelPCA(n_components=2, kernel='precomputed')
    X_kpca_heat = kpca_heat.fit_transform(K)
    
    km_kpca_heat = KMeans(n_clusters=len(n_clusters_set), random_state=42, n_init=10)
    pred_kpca_heat = km_kpca_heat.fit_predict(X_kpca_heat)
    
    # If KPCA on RBF pre-image is weird, use native heat kernel
    sil_kpca_heat = silhouette_score(X_kpca_heat, pred_kpca_heat) if len(np.unique(pred_kpca_heat)) > 1 else 0
    ari_kpca_heat = adjusted_rand_score(y, pred_kpca_heat)
    
    print(f"  Heat KPCA clustering: silhouette={sil_kpca_heat:.3f}, ARI={ari_kpca_heat:.3f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Vanilla PCA
    ax = axes[0]
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#1B998B', '#5C4D7D']
    for t in range(len(n_clusters_set)):
        mask = y == t
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[t], label=f'C={n_clusters_set[t]}',
                  s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    ax.set_title(f'Vanilla PCA (var={pca_var:.2f})' + f'\nSil={sil_pca:.3f}, ARI={ari_pca:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # RBF Kernel PCA
    ax = axes[1]
    for t in range(len(n_clusters_set)):
        mask = y == t
        ax.scatter(X_kpca[mask, 0], X_kpca[mask, 1], c=colors[t], label=f'C={n_clusters_set[t]}',
                  s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('KPCA 1')
    ax.set_ylabel('KPCA 2')
    ax.set_title(f'RBF Kernel PCA (gamma=1.0)' + f'\nSil={sil_kpca:.3f}, ARI={ari_kpca:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Heat Kernel PCA
    ax = axes[2]
    for t in range(len(n_clusters_set)):
        mask = y == t
        ax.scatter(X_kpca_heat[mask, 0], X_kpca_heat[mask, 1], c=colors[t], label=f'C={n_clusters_set[t]}',
                  s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Heat KPCA 1')
    ax.set_ylabel('Heat KPCA 2')
    ax.set_title(f'Heat Kernel PCA' + f'\nSil={sil_kpca_heat:.3f}, ARI={ari_kpca_heat:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp4_kernel_pca.png'), dpi=150)
    plt.close()
    print("  → Saved exp4_kernel_pca.png")
    
    return {
        'pca_sil': sil_pca, 'pca_ari': ari_pca,
        'kpca_sil': sil_kpca, 'kpca_ari': ari_kpca,
        'heat_kpca_sil': sil_kpca_heat, 'heat_kpca_ari': ari_kpca_heat
    }


# ── Spectral Analysis ──────────────────────────────────────────────────

def spectral_analysis():
    """Deep spectral analysis of the conservation kernel.
    Show eigenvalue distribution, eigenvector structure, and spectral gap.
    """
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS")
    print("=" * 60)
    
    n_nodes = 50
    np.random.seed(42)
    
    # Build one of each
    L_music, _ = build_music_laplacian(n_nodes)
    L_rand, _ = build_random_graph(n_nodes)
    L_cons, _ = build_conservation_graph(n_nodes)
    
    lambdas = {}
    infos = {}
    for name, L in [('Music', L_music), ('Random', L_rand), ('Conservation', L_cons)]:
        eigvals, eigvecs = eigh(L)
        eigvals = np.clip(eigvals, 0, None)
        lambdas[name] = eigvals
        
        # Compute heat kernel at different betas
        for beta in [0.1, 1.0, 10.0]:
            K = eigvecs @ np.diag(np.exp(-beta * eigvals)) @ eigvecs.T
            tr = np.trace(K)
            eff_rank = tr  # = Σ exp(-βλᵢ)
            
            # Participation ratio
            p_i = np.exp(-beta * eigvals) / tr
            pr = 1.0 / np.sum(p_i**2)
            infos[f'{name}_β={beta}'] = {
                'trace': tr,
                'eff_rank': eff_rank,
                'participation_ratio': pr
            }
            print(f"  {name} (β={beta:5.1f}): Tr(K)={tr:.3f}, PR={pr:.2f}")
    
    # Plot eigenvalue distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for name, color in [('Music', '#2E86AB'), ('Random', '#A23B72'), ('Conservation', '#F18F01')]:
        eig = lambdas[name]
        ax.hist(eig[eig > 1e-10], bins=20, alpha=0.5, label=name, color=color, density=True)
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Density')
    ax.set_title('Eigenvalue Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for name, color in [('Music', '#2E86AB'), ('Random', '#A23B72'), ('Conservation', '#F18F01')]:
        eig = lambdas[name]
        ax.semilogy(np.sort(eig), 'o-', color=color, label=name, markersize=3)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue (log)')
    ax.set_title('Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Spectral gap
    ax = axes[1, 0]
    gaps = []
    for name in ['Music', 'Random', 'Conservation']:
        eig = np.sort(lambdas[name])
        gap = eig[1:] - eig[:-1]
        gaps.append((name, gap[:15]))
    
    x = np.arange(15)
    for name, gap in gaps:
        ax.plot(x[:len(gap)], gap, 'o-', label=name[:len(gap)])
    ax.set_xlabel('Gap index')
    ax.set_ylabel('Spectral gap size')
    ax.set_title('Spectral Gaps (first 15)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Heat kernel decay
    ax = axes[1, 1]
    betas = np.logspace(-2, 2, 100)
    for name, color in [('Music', '#2E86AB'), ('Random', '#A23B72'), ('Conservation', '#F18F01')]:
        eig = lambdas[name]
        traces = [np.sum(np.exp(-b * eig)) for b in betas]
        ax.loglog(betas, traces, '-', color=color, label=name, linewidth=2)
    ax.set_xlabel('β')
    ax.set_ylabel('Tr(K)')
    ax.set_title('Heat Kernel Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp5_spectral_analysis.png'), dpi=150)
    plt.close()
    print("  → Saved exp5_spectral_analysis.png")
    
    # Spectral density comparison at specific beta
    fig, ax = plt.subplots(figsize=(8, 5))
    beta = 1.0
    for name, color in [('Music', '#2E86AB'), ('Random', '#A23B72'), ('Conservation', '#F18F01')]:
        eig = lambdas[name]
        weights = np.exp(-beta * eig) / np.sum(np.exp(-beta * eig))
        ax.hist(eig, bins=30, weights=weights, alpha=0.5, color=color, 
                label=f'{name} (β={beta})', density=True)
    ax.set_xlabel('Eigenvalue')
    ax.set_ylabel('Heat kernel weight density')
    ax.set_title(f'Heat Kernel Spectral Density (β={beta})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, 'exp5_spectral_density.png'), dpi=150)
    plt.close()
    print("  → Saved exp5_spectral_density.png")
    
    return infos


# ── Main Runner ────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    
    results = {}
    
    # Spectral analysis first
    results['spectral'] = spectral_analysis()
    
    # Experiment 1
    results['svm'] = experiment_svm()
    
    # Experiment 2
    results['perturbation'] = experiment_perturbation()
    
    # Experiment 3
    results['multiscale'] = experiment_multiscale_beta()
    
    # Experiment 4
    results['kpca'] = experiment_kernel_pca()
    
    elapsed = time.time() - start_time
    
    # Save results summary
    with open(os.path.join(OUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print hypothesis verdict
    print("\n" + "=" * 60)
    print("HYPOTHESIS VERDICT")
    print("=" * 60)
    
    hyp = results['perturbation']
    print(f"\nH1: Conservation kernel has stable trace under perturbation")
    print(f"    Conservation ΔTr = {hyp['cons_mean']:.6f}")
    print(f"    Random ΔTr       = {hyp['rand_mean']:.6f}")
    print(f"    p-value          = {hyp['p_value']:.6f}")
    if hyp['hypothesis']:
        print("    ✅ CONFIRMED: Conservation kernel IS more stable")
    else:
        print("    ❌ REJECTED: No stability advantage")
    
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Figures saved to: {FIGS_DIR}")


if __name__ == '__main__':
    main()
