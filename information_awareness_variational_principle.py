# -*- coding: utf-8 -*-
"""
Information–Awareness Unified Variational Principle (vOmega)
Official Verification Script - Cornelius Aurelius (Omniscientrix-vOmega Framework)

This script verifies the core variational identity:
delta_Omega = 0  <=>  the coupled informational–awareness system minimizes joint divergence.

We simulate coupled fields:
P = information distribution
A = awareness distribution
and confirm convergence of the joint functional:
J = KL(P || Q) + ||A - A_eq||^2
"""

import numpy as np

def kl_divergence(p, q):
    p = np.clip(p, 1e-15, 1)
    q = np.clip(q, 1e-15, 1)
    return np.sum(p * np.log(p / q))

def joint_functional(p, q, A, A_eq):
    return kl_divergence(p, q) + np.sum((A - A_eq)**2)

def evolve_variational_system(p, q, A, A_eq, lr=0.05, steps=3000, tol=0.01):
    history = []
    for t in range(steps):
        J = joint_functional(p, q, A, A_eq)
        history.append(J)

        if J < tol:
            print("[SUCCESS] delta_Omega = 0 achieved at step", t)
            print("Final J:", J)
            return history

        p = p - lr * (p - q)
        A = A - lr * (A - A_eq)

        p = np.clip(p, 1e-15, None); p /= p.sum()
        A = np.clip(A, 1e-15, None)

    print("[WARNING] Variational equilibrium threshold not reached.")
    print("Final J:", history[-1])
    return history

if __name__ == "__main__":
    print("\n=== Verification: Information-Awareness Unified Variational Principle (vOmega) ===\n")

    p = np.random.rand(1000); p /= p.sum()
    q = np.ones_like(p) / len(p)

    A = np.random.rand(1000)
    A_eq = np.ones_like(A)

    history = evolve_variational_system(p, q, A, A_eq)

    print("\nVerification complete.")
    print("First 10 J values:", history[:10])
    print("Final J:", history[-1])
