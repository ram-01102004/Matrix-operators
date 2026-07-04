import numpy as np
import os

# ─────────────────────────────────────────────
#  👇 CHANGE THIS to your .txt file path
FILE_PATH = r"C:\Users\ramsu\ram sundhar\coding projects\matrix operators\matrix.txt"
# ─────────────────────────────────────────────

def load_matrix(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    matrix = np.loadtxt(filepath)   # space/tab separated
    if matrix.ndim != 2:
        raise ValueError("File must contain a 2D matrix.")
    return matrix

def analyze_matrix(matrix):
    rows, cols = matrix.shape

    print("\n" + "=" * 55)
    print("  MATRIX ANALYSIS")
    print("=" * 55)

    print(f"\n📐 Matrix  ({rows} × {cols})")
    print(matrix)

    if rows != cols:
        print(f"\n⚠️  Need a square matrix. Yours is {rows}×{cols}.")
        return

    # Determinant
    det = np.linalg.det(matrix)
    print(f"\n🔢 Determinant")
    print(f"   det(A) = {det:.6f}")
    if abs(det) < 1e-10:
        print("   ⚠️  ≈ 0  →  matrix is SINGULAR (not invertible)")
    else:
        print("   ✅  Matrix is NON-SINGULAR (invertible)")

    # Eigenvalues & Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    print(f"\n🎯 Eigenvalues  (λ)")
    for i, val in enumerate(eigenvalues):
        if np.iscomplex(val):
            print(f"   λ{i+1} = {val.real:.6f} + {val.imag:.6f}i")
        else:
            print(f"   λ{i+1} = {val.real:.6f}")

    print(f"\n📊 Eigenvectors  (column i → λ{chr(8321)})")
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        parts = []
        for v in vec:
            if np.iscomplex(v):
                parts.append(f"{v.real:.4f}+{v.imag:.4f}i")
            else:
                parts.append(f"{v.real:.4f}")
        print(f"   v{i+1} = [ {',  '.join(parts)} ]")

    # Verification
    print(f"\n✔️  Verification  (A·v = λ·v ?)")
    for i in range(len(eigenvalues)):
        lhs = matrix @ eigenvectors[:, i]
        rhs = eigenvalues[i] * eigenvectors[:, i]
        ok = np.allclose(lhs, rhs, atol=1e-8)
        print(f"   λ{i+1}: {'✅ PASS' if ok else '❌ FAIL'}")

    # Extra
    print(f"\n📋 Extra Properties")
    print(f"   Trace = {np.trace(matrix):.6f}   (= sum of eigenvalues)")
    print(f"   Rank  = {np.linalg.matrix_rank(matrix)}")

    print("\n" + "=" * 55 + "\n")

# Run
try:
    matrix = load_matrix(FILE_PATH)
    analyze_matrix(matrix)
except Exception as e:
    print(f"\n❌ Error: {e}")