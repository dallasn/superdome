# SuperDOME Encoder — Orthogonal Sum + Projection Retrieval POC
#
# Motivation:
#
# Many models are now adapted by using low rank adapters (LoRAs) of the form:
# W_k = A_k @ B_k^T,   A_k ∈ R to improve storage and efficiency. It was found that 
# LoRAs can be extracted directly from weight deltas (W_T - W_S) using singular value
# decomposition (SVD) or alternating least squares (ALS).  Doing this iteratively produces
# a minimizing set of LoRAs {W_k} that approximate the full weight delta.  Further these LoRAs
# are nearly orthogonal and can be orthogonalized using a Gram–Schmidt process on the full
# weight space.  This allows all LoRAs to be superposed into a single "supertensor":
# W_super = Σ W_k which compresses the storage of K LoRAs (each N×N) into a single N×N matrix making
# the complete library of LoRAs efficient for storage in VRAM during inference.
#
# Rationale:
# If {W_k} are Frobenius-orthogonal, then W_super = Σ W_k and projection
#   W_hat_k = <W_super, Q_k> * Q_k, where Q_k = W_k / ||W_k||,
# exactly recovers W_k (up to numeric precision).
#
# This module extracts K LoRAs of rank R from a synthetic weight delta between teacher and student models, encodes them
# into a single supermatrix W_super, and then extracts them back via projection retrieval in order to reconstruct
# the teacher weights. It demonstrates one part of a potential framework for improved model compression 


import math, time
import torch
import torch.nn.functional as F

# -----------------------
# Configuration
# -----------------------
N = 4096         # hidden size typical for 7B models
R = 128          # LoRA rank 
K_TOTAL = 32     # target number of experts to extract & encode
USE_GPU = True
W_STD = 0.05 

FUNC_BATCH = 1024

# Device
DEVICE = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print(f"Device: {DEVICE}")

# -----------------------
# Utilities
# -----------------------
def check(t, name):
    print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

# -----------------------
# Randomized SVD implementation
# -----------------------
def randomized_svd(A, k, n_oversamples=10, n_iter=2):    
    orig_dtype = A.dtype
    A = A.to(torch.float32)
    m, n = A.shape
    do_transpose = False
    if m > n:
        A = A.T
        m, n = A.shape
        do_transpose = True
    l = min(k + n_oversamples, m)
    Qn = torch.randn(n, l, device=A.device, dtype=A.dtype)
    Y = A @ Qn
    Qm, _ = torch.linalg.qr(Y, mode='reduced')
    for _ in range(max(n_iter, 0)):
        Z = A.T @ Qm
        Qn, _ = torch.linalg.qr(Z, mode='reduced')
        Y = A @ Qn
        Qm, _ = torch.linalg.qr(Y, mode='reduced')
    B = Qm.T @ A
    Uh, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Qm @ Uh
    U = U[:, :k]; S = S[:k]; Vh = Vh[:k, :]
    if do_transpose:
        U, Vh = Vh.T, U.T
    return U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)

def calculate_aligned_residual(W_T_resized, W_S):
    print("-> Aligning teacher to student frame via SVD ...")
    M = (W_T_resized.T @ W_S).to(torch.float32)
    U, _, Vt = randomized_svd(M, min(M.shape))
    R = (U @ Vt).to(torch.float32)
    W_T_aligned = (W_T_resized.to(torch.float32) @ R)
    DeltaW = (W_T_aligned - W_S.to(torch.float32))
    print(f"   residual norm: {torch.linalg.norm(DeltaW).item():.4f}")
    return DeltaW, R, W_T_aligned

def iterative_factor_extraction(DeltaW, rank):
    svd_start = time.time()
    U, S, Vh = randomized_svd(DeltaW, rank, n_oversamples=10, n_iter=2)
    svd_time = time.time() - svd_start
    sqrtS = torch.diag(torch.sqrt(S.to(torch.float32)))
    U32 = U.to(torch.float32)
    V32 = Vh.to(torch.float32)
    A = U32 @ sqrtS
    B = (V32.T @ sqrtS)
    W_extracted = A @ B.T
    print(f"   SVD took {svd_time:.3f}s | A||={torch.linalg.norm(A):.4f} B||={torch.linalg.norm(B):.4f}")
    return A, B, W_extracted

# -----------------------
# Gram–Schmidt helpers (matrix Frobenius orthogonalization)
# -----------------------
def frob_inner(X, Y):
    return (X * Y).sum()

@torch.no_grad()
def gs_orthogonalize_matrix(W_new, Q_basis, eps=1e-12):
    if len(Q_basis) == 0:
        nrm = torch.linalg.norm(W_new).item()
        return W_new.clone(), [], nrm
    W = W_new.clone()
    coeffs = []
    for Q in Q_basis:
        alpha = frob_inner(W, Q)
        if torch.isfinite(alpha):
            W -= alpha * Q
            coeffs.append(alpha.item())
        else:
            coeffs.append(float('nan'))
    nrm = torch.linalg.norm(W).item()
    if nrm < eps:
        return W, coeffs, 0.0
    return W, coeffs, nrm

@torch.no_grad()
def refactor_rankR(W, R):
    r = min(R, min(W.shape))
    if r <= 0:
        raise ValueError("Non-positive target rank in refactor_rankR")
    U, S, Vh = randomized_svd(W, r, n_oversamples=10, n_iter=2)
    sqrtS = torch.diag(torch.sqrt(S.to(torch.float32)))
    U32 = U.to(torch.float32)
    V32 = Vh.to(torch.float32)
    A = U32 @ sqrtS
    B = (V32.T @ sqrtS)
    return A, B

# -----------------------
# Main pipeline
# -----------------------
def run():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # ----- Step 1: synthetic pair + alignment -----
    print("\n[Step 1] Build synthetic W_S and W_T_resized, then align")
    W_S = (torch.randn(N, N, device=DEVICE, dtype=torch.float16) * W_STD * 2).to(torch.float32)
    misalign = torch.linalg.svd(torch.randn(N, N, device=DEVICE, dtype=torch.float32))[0]
    W_T_resized = (W_S * 1.5) @ misalign
    DeltaW, R_align, W_T_aligned = calculate_aligned_residual(W_T_resized, W_S)

    print("-> Injecting additional structured residual energy ...")
    for _ in range(100):
        v1 = torch.randn(N, 1, device=DEVICE, dtype=torch.float32) * W_STD
        v2 = torch.randn(1, N, device=DEVICE, dtype=torch.float32) * W_STD
        DeltaW += v1 @ v2
    print(f"   residual norm after inject: {torch.linalg.norm(DeltaW).item():.4f}")

    # ----- Step 2: extract K experts of rank R with GS orthogonalization -----
    print(f"\n[Step 2] Factorization (GS): extract up to K={K_TOTAL} experts (rank R={R})")
    all_A, all_B, all_W, Q_basis = [], [], [], []
    Rem = DeltaW.clone()

    for k in range(K_TOTAL):
        print(f" - Expert {k+1}/{K_TOTAL} | ‖Rem‖={torch.linalg.norm(Rem).item():.6f}")
        A_k0, B_k0, Wk0 = iterative_factor_extraction(Rem, R)

        # Orthogonalize full matrix
        Wk_ortho, coeffs, nrm = gs_orthogonalize_matrix(Wk0, Q_basis, eps=1e-9)
        print(f"   GS proj: removed {len(coeffs)} components | ‖Wk_ortho‖={nrm:.6f}")
        if nrm < 1e-8:
            print("   -> Orthogonal component vanished (near-zero). Stopping early.")
            break

        # Re-factor to rank R (keeps LoRA interface), rebuild W_k
        A_k, B_k = refactor_rankR(Wk_ortho, R)
        Wk = A_k @ B_k.T

        # Normalize and append to orthonormal basis
        Wk_norm = torch.linalg.norm(Wk).clamp_min(1e-12)
        Qk = (Wk / Wk_norm).to(torch.float32)
        Q_basis.append(Qk)

        all_A.append(A_k); all_B.append(B_k); all_W.append(Wk)

        # Deflate the orthogonalized piece
        Rem -= Wk

    K_eff = len(all_W)
    print(f"Done. Extracted {K_eff} experts. Final residual norm: {torch.linalg.norm(Rem).item():.6f}")

    # ----- Step 3: Encode (simple sum) & Retrieve (projection) -----
    print("\n[Step 3] Superposition by simple sum and projection retrieval")
    W_super = torch.zeros(N, N, device=DEVICE, dtype=torch.float32)
    for Wk in all_W:
        W_super += Wk

    # Retrieval via Frobenius projection onto each Q_k = W_k / ||W_k||
    print("\n[Step 3a] Retrieval and errors vs. ground-truth (projection)")
    abs_list, rel_list, cos_list = [], [], []
    for k in range(K_eff):
        Wk = all_W[k]
        Qk = Q_basis[k]
        coeff = (W_super * Qk).sum()  # <W_super, Qk>_F
        W_rec = coeff * Qk
        diff = W_rec - Wk
        abs_err = torch.linalg.norm(diff).item()
        rel_err = (torch.linalg.norm(diff) / torch.linalg.norm(Wk).clamp_min(1e-12)).item()
        cos = F.cosine_similarity(W_rec.reshape(-1), Wk.reshape(-1), dim=0).item()
        abs_list.append(abs_err); rel_list.append(rel_err); cos_list.append(cos)
        print(f"  LoRA {k+1:>3}: abs={abs_err:.6e}, rel={rel_err:.3e}, cos={cos:.6f}")

    if len(rel_list) > 0:
        print(f"Retrieval rel-error (mean±std): {float(torch.tensor(rel_list).mean()):.3e} ± {float(torch.tensor(rel_list).std()):.3e}")
    else:
        print("Retrieval: no experts extracted.")

    # ----- Step 3b: Approximation vs teacher WITHOUT residual -----
    print("\n[Step 3b] Approximation error (no residual)")
    W_approx_aligned = W_S + W_super
    approx_abs = torch.linalg.norm(W_approx_aligned - W_T_aligned).item()
    approx_rel = (torch.linalg.norm(W_approx_aligned - W_T_aligned) /
                  (torch.linalg.norm(W_T_aligned) + 1e-12)).item()
    print(f"Approximation (no residual): abs={approx_abs:.6f}, rel={approx_rel:.6e}")

    # ----- Step 3c: Full reconstruction WITH consistent residual -----
    print("\n[Step 3c] Full reconstruction with consistent residual and rotation back")
    Residual_consistent = W_T_aligned - W_approx_aligned
    W_full_aligned = W_approx_aligned + Residual_consistent
    W_full_unaligned = W_full_aligned @ R_align.T

    ea = torch.linalg.norm(W_full_aligned - W_T_aligned).item()
    ra = (torch.linalg.norm(W_full_aligned - W_T_aligned) /
          (torch.linalg.norm(W_T_aligned) + 1e-12)).item()
    eu = torch.linalg.norm(W_full_unaligned - W_T_resized).item()
    ru = (torch.linalg.norm(W_full_unaligned - W_T_resized) /
          (torch.linalg.norm(W_T_resized) + 1e-12)).item()
    print(f"Aligned-space full reconstruction: abs={ea:.6f}, rel={ra:.6e}")
    print(f"Unaligned-space full reconstruction: abs={eu:.6f}, rel={ru:.6e}")

    # print the upper 10x10 block of W_super for visual inspection
    print("W_full_unaligned (upper 10x10 block):")
    print(W_full_unaligned[:10, :10].cpu().numpy())  

    # print the upper 10x10 block of W_T_resized for visual inspection
    print("W_T_resized (upper 10x10 block):")
    print(W_T_resized[:10, :10].cpu().numpy())

    # print the total VRAM required for W_super
    total_vram_mb = W_super.element_size() * W_super.nelement() / (1024 ** 2)
    print(f"\nTotal VRAM for W_super: {total_vram_mb:.2f} MB")
    # print the total VRAM for W_all
    total_w_all_vram_mb = sum(w.element_size() * w.nelement() for w in all_W) / (1024 ** 2)
    print(f"Total VRAM for all W_k: {total_w_all_vram_mb:.2f} MB")

    #print the compression ratio of using the super matrix vs storing all W_k
    if total_w_all_vram_mb > 0:
        compression_ratio = total_w_all_vram_mb / total_vram_mb
        print(f"Compression ratio (all W_k vs W_super): {compression_ratio:.2f}x")
    else:
        print("Compression ratio: N/A (no experts extracted)")  

    # ----- Step 4: Functional test (apply W_k on random X) -----
    print("\n[Step 4] Functional test")
    X = torch.randn(FUNC_BATCH, N, device=DEVICE, dtype=torch.float32)
    K_check = min(K_eff, 8)
    for k in range(K_check):
        Wk = all_W[k]
        Wrec_coeff = (W_super * Q_basis[k]).sum()
        Wk_rec = Wrec_coeff * Q_basis[k]
        Y_true = X @ Wk
        Y_rec = X @ Wk_rec
        num = (Y_rec - Y_true).pow(2).mean()
        den = Y_true.pow(2).mean().clamp_min(1e-12)
        relmse = (num / den).item()
        print(f"  LoRA {k+1}: output rel-MSE = {relmse:.3e}")
    print("\nDone.")

if __name__ == "__main__":
    run()
