import numpy as np
from scipy.fftpack import fft2, ifft2

class ADMMOptimizer:
    """
    Alternating Direction Method of Multipliers (ADMM) for DAS Signal Recovery.

    Solves the Total Variation (TV) denoising problem:
        min_x 0.5 * ||y - x||_2^2 + lambda * ||grad(x)||_1

    where y is the noisy observed data and x is the recovered signal.
    """

    def __init__(self, rho: float = 1.0, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize the ADMM solver.

        Args:
            rho: Augmented Lagrangian parameter (penalty weight).
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.
        """
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol

    def tv_denoise(self, y: np.ndarray, lambd: float) -> np.ndarray:
        """
        Perform TV denoising on 2D DAS data.

        Args:
            y: Noisy data array of shape (n_channels, n_samples).
            lambd: Regularization parameter controlling sparsity of the gradient.

        Returns:
            Denoised data array x.
        """
        rows, cols = y.shape
        x = np.zeros_like(y)
        z = np.zeros((2, rows, cols))  # z[0] is grad_x, z[1] is grad_t
        u = np.zeros_like(z)

        # Precompute FFT terms for x-update step
        # The x-update involves solving (I + rho * D^T D) x = y + rho * D^T (z - u)
        # This is a convolution, which is diagonal in Fourier domain.

        # Optical transfer function for difference operators (D^T D)
        # We assume periodic boundary conditions for FFT efficiency.
        # D_x corresponds to difference along rows, D_t along cols.

        # Frequencies
        omega_r = 2 * np.pi * np.fft.fftfreq(rows).reshape(-1, 1)
        omega_c = 2 * np.pi * np.fft.fftfreq(cols).reshape(1, -1)

        # Eigenvalues of the difference operators in Fourier domain
        # D^T D = |1 - e^{-i omega}|^2 = 4 sin^2(omega / 2)
        eig_DtD_r = 4 * np.sin(omega_r / 2) ** 2
        eig_DtD_c = 4 * np.sin(omega_c / 2) ** 2
        eig_DtD = eig_DtD_r + eig_DtD_c

        # LHS of the linear system in Fourier domain
        lhs_denominator = 1 + self.rho * eig_DtD

        for k in range(self.max_iter):
            x_prev = x.copy()

            # --- 1. x-update ---
            # RHS = y + rho * div(z - u)
            # div(V) = D_x^T V_x + D_t^T V_t
            # In Fourier: F(div(V)) = conj(F(D)) . F(V)

            # Compute divergence of (z - u)
            v = z - u

            # Helper for divergence (transpose of gradient)
            # D^T v = v[i] - v[i-1]
            # F(D^T v) = (1 - e^{i omega}) F(v)
            diff_adj_r = (1 - np.exp(1j * omega_r))
            diff_adj_c = (1 - np.exp(1j * omega_c))

            F_v0 = fft2(v[0])
            F_v1 = fft2(v[1])

            rhs_term = self.rho * (ifft2(diff_adj_r * F_v0 + diff_adj_c * F_v1).real)
            rhs = y + rhs_term

            # Solve in Fourier domain
            x = ifft2(fft2(rhs) / lhs_denominator).real

            # --- 2. z-update ---
            # z = soft_threshold(grad(x) + u, lambda/rho)

            # Compute gradient of x
            # D x = x[i+1] - x[i]
            Dx_0 = np.roll(x, -1, axis=0) - x
            Dx_1 = np.roll(x, -1, axis=1) - x
            Dx = np.stack([Dx_0, Dx_1])

            z = self._soft_threshold(Dx + u, lambd / self.rho)

            # --- 3. u-update ---
            u = u + Dx - z

            # --- 4. Convergence check ---
            # Primal residual: r = Dx - z
            # Dual residual: s = rho * D^T (z - z_prev)

            r_norm = np.linalg.norm(Dx - z)
            s_norm = np.linalg.norm(self.rho * (z - (z - u + u))) # simplified check

            if np.linalg.norm(x - x_prev) < self.tol * np.linalg.norm(x):
                # print(f"ADMM converged at iteration {k}")
                break

        return x

    @staticmethod
    def _soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
        """
        Soft thresholding operator.
        S_kappa(v) = sign(v) * max(|v| - kappa, 0)
        """
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    def tv1d_denoise(self, y: np.ndarray, lambd: float) -> np.ndarray:
        """
        Solve the 1D Total Variation Denoising problem using ADMM.

        Problem: min_x 0.5 * ||y - x||_2^2 + lambd * ||Dx||_1
        where D is the 1D difference operator.

        This implementation uses a direct Thomas algorithm (tridiagonal solver) for the x-update,
        making it O(N) complexity per iteration.

        Args:
            y: Input signal (1D array).
            lambd: Regularization parameter.

        Returns:
            Denoised signal x.
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        rho = self.rho

        x = y.copy()
        z = np.zeros(n - 1)
        u = np.zeros(n - 1)

        # Precompute tridiagonal matrix diagonals for (I + rho D^T D)
        # The system is (I + rho D^T D) x = ...
        # D is (N-1 x N). D^T D is (N x N) tridiagonal.
        # Main diag: 1 + 2*rho (internal), 1 + rho (boundaries)
        # Off diag: -rho

        a = -rho * np.ones(n - 1)  # lower
        b = (1 + 2 * rho) * np.ones(n)  # main
        b[0] = 1 + rho
        b[-1] = 1 + rho
        c = -rho * np.ones(n - 1)  # upper

        # Helper: Thomas algorithm for tridiagonal solve
        def thomas_solve(rhs_vec):
            # Forward elimination
            c_prime = np.zeros(n - 1)
            d_prime = np.zeros(n)

            c_prime[0] = c[0] / b[0]
            d_prime[0] = rhs_vec[0] / b[0]

            for i in range(1, n - 1):
                temp = b[i] - a[i-1] * c_prime[i-1]
                c_prime[i] = c[i] / temp
                d_prime[i] = (rhs_vec[i] - a[i-1] * d_prime[i-1]) / temp

            # Last row separate (no c_prime)
            d_prime[n-1] = (rhs_vec[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])

            # Back substitution
            x_sol = np.zeros(n)
            x_sol[-1] = d_prime[-1]
            for i in range(n - 2, -1, -1):
                x_sol[i] = d_prime[i] - c_prime[i] * x_sol[i+1]

            return x_sol

        for _ in range(self.max_iter):
            # --- x-update ---
            # (I + rho D^T D) x = y + rho D^T (z - u)
            # Calculate rhs = y + rho D^T (z - u)

            d_aux = z - u # size N-1
            # D^T d_aux maps (N-1) -> (N)
            # (D^T v)_i = -v_i if i=0
            #           = v_{i-1} - v_i if 0 < i < N-1
            #           = v_{N-2} if i = N-1

            Dt_term = np.zeros(n)
            Dt_term[0] = -d_aux[0]
            Dt_term[1:-1] = d_aux[:-1] - d_aux[1:]
            Dt_term[-1] = d_aux[-1]

            rhs = y + rho * Dt_term
            x = thomas_solve(rhs)

            # --- z-update ---
            # z = soft_threshold(Dx + u, lambd/rho)
            Dx = np.diff(x)
            v = Dx + u
            kappa = lambd / rho
            z = np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

            # --- u-update ---
            u = u + Dx - z

            # Check convergence (primal residual)
            # r = Dx - z
            # if np.linalg.norm(r) < self.tol: break

        return x

class ConsensusADMM:
    """
    Distributed processing simulator using Consensus ADMM.
    Simulates solving a global problem across multiple nodes.
    """
    def __init__(self, num_nodes=5):
        self.num_nodes = num_nodes
        self.nodes = [ADMMOptimizer() for _ in range(num_nodes)]

    def solve_distributed(self, data_chunks):
        # Implementation of consensus loop place holder
        pass
