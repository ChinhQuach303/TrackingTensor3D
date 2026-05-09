# HO-RLSL Subspace Tracking & FCCA Implementation Guide

This document outlines the technical implementation and alignment of the High-Order Recursive Least Squares Subspace (HO-RLSL) tracking and Fiedler Consensus Clustering Algorithm (FCCA) within the `TrackingTensor3D` project, based on the framework by **Ozdemir et al. (2017)**.

---

## 1. HO-RLSL Subspace Tracking

### 1.1 Initialization (Baseline Training)
To ensure stability and eliminate artifactual change-points during the pre-stimulus period, the system implements a dedicated training phase:
- **Duration:** 1000ms (128 samples at 128Hz).
- **Method:** HOSVD (Higher-Order SVD) is performed on the baseline tensor to initialize the spatial basis $U$ (Channels) and subject basis $V$ (Subjects).
- **Adaptive Thresholding:** An initial `sigma_min` is estimated from the baseline singular values to anchor the tracking sensitivity.

### 1.2 Online Recursive Update
The core engine updates the subspace every $\alpha = 8$ samples (approx. 62.5ms), as recommended for EEG connectivity dynamics:
- **Windowing:** A sliding window of size $\alpha$ captures the most recent "clean" low-rank signal.
- **Orthogonal Projection:** Data is projected onto the complement of the current subspace: $D_{proj} = (I - UU^T) D_{window}$.
- **Basis Expansion:** If the energy in $D_{proj}$ exceeds a threshold $\sigma_{min}$, new directions are added to $U$.
- **Rank Constraint:** The updated basis is truncated back to rank $r=5$ using SVD to maintain a parsimonious representation.

### 1.3 Change-Point Detection
Change-points are identified directly from **Basis Reconfigurations**:
- A change-point is triggered if the HO-RLSL algorithm adds or removes directions from the subspace.
- **Tuning:** For this PLV dataset (34 subjects, 30 channels), the optimal sensitivity was found at $\sigma_{min} = 0.063$. This filters out noise jitter while capturing significant transitions during the ERN (50-250ms).

---

## 2. FCCA (Fiedler Consensus Clustering)

The FCCA module provides a group-level consensus on the brain's functional communities during the task period.

### 2.1 Mathematical Procedure
1. **Laplacian Construction:** For each subject and time point, the Laplacian $L = D - A$ is computed from the low-rank connectivity matrix.
2. **Fiedler Vector:** The eigenvector corresponding to the second smallest eigenvalue of $L$ is extracted.
3. **Bi-partitioning:** Nodes are assigned to two clusters (A or B) based on the sign of their Fiedler vector components.
4. **Consensus Matrix ($W$):** A co-occurrence matrix tracks how often pairs of nodes share the same cluster across all subjects and time points.
5. **Final Clustering:** A final Fiedler decomposition is performed on the averaged $W$ to determine the global consensus topology.

### 2.2 Functional Interpretation
In the current ERP task (ERN/CRN):
- **Cluster A (Frontal-Central):** Acts as the "Executive Hub" (including FCz, Cz, Fz), responsible for error monitoring.
- **Cluster B (Posterior-Parietal):** Acts as the "Sensory Hub" (including Oz, Pz, PO8), responsible for feedback processing.

---

## 3. Ozdemir (2017) Alignment Summary

- **Alpha Parameter:** Set to 8 samples (Matched).
- **Initialization:** Full 1s Baseline Training (Matched).
- **Subspace Velocity:** Derived from the Frobenius norm of projection changes (Matched).
- **Sparse Recovery:** $L_1$-norm regularization used to isolate transient connectivity artifacts (Matched).
- **Topology:** Hubs identified at Medial-Frontal regions during ERN peaks (Matched).

---
*Developed as part of the TrackingTensor3D Research Pipeline.*
