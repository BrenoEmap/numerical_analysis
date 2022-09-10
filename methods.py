from time import time
import numpy as np
import scipy.sparse as ssp
import pandas as pd
from scipy.sparse.linalg import spsolve

class Matrix():
    def __init__(self, dim: int):
        self.DT = np.float64
        self.dim = dim
        self.A = self.create_A()
        self.b = self.create_b()
        self.x = self.init_guess()
        self.L_plus_D = ssp.tril(self.A, format='csr')
        self.L_plus_U = ssp.tril(self.A, k=-1) + ssp.triu(self.A, k=1)
        self.U = ssp.triu(self.A, k=1)
    
    def create_A(self):
        main_diag = 3 * np.ones((1, self.dim), dtype=self.DT)
        low_up_diag = - np.ones((1, self.dim), dtype=self.DT)
        data = np.concatenate((main_diag, low_up_diag, low_up_diag), axis=0)
        offsets = [0, -1, 1]
        A = ssp.dia_matrix((data, offsets),
                           shape=(self.dim, self.dim))
        A = ssp.dok_matrix(A)
        for i in range(self.dim//2 - 1):
            A[i, self.dim - 1 - i] = 0.5
        for i in range(self.dim//2 + 1, self.dim):
            A[i, self.dim - 1 - i] = 0.5
        return A.tocsr()

    def create_b(self):
        b = 1.5 * np.ones(self.dim, dtype=self.DT)
        b[[0, self.dim-1]] = 2.5
        b[[self.dim//2 - 1, self.dim//2]] = 1
        return b

    def init_guess(self):
        x=np.ones(self.dim, dtype=self.DT)
        return x


class Methods(Matrix):
    def __init__(self, tol: float, dim: int):
        super().__init__(dim)
        self.tol = tol

    def jacobi(self, use_solution: bool = False):
        cur = np.zeros(self.dim, dtype=self.DT)
        nxt = None
        diff = self.tol + 1
        it = 0
        start = time()
        while diff >= self.tol:
            # Use the fact that the diagonal is all 3s
            nxt = ( -self.L_plus_U @ cur + self.b ) / 3
            diff = np.amax(np.abs(nxt - cur))
            cur = nxt
            it += 1
        elapsed = time() - start
        summary =  {
            'result': cur,
            'n_iter': it,
            'run_time': round(float(elapsed),3),
            'error': diff
        }
        return summary


    def gauss_seidel(self, use_solution: bool = False):
        cur = np.zeros(self.dim, dtype=self.DT)
        nxt = None
        diff = self.tol + 1
        it = 0
        start = time()
        while diff >= self.tol:
            nxt =  spsolve(self.L_plus_D, - self.U @ cur + self.b)
            diff = np.amax(np.abs(nxt - cur))
            cur = nxt
            it += 1
        elapsed = time() - start
        summary =  {
            'result': cur,
            'n_iter': it,
            'run_time': round(float(elapsed),3),
            'error': diff
        }
        return summary

    def conjugate_gradient(self, n_iter: int = None):
        zeros = np.zeros(self.dim, dtype=self.DT)
        x = zeros
        r = self.b - self.A @ x
        d = r
        alpha = None
        start = time()
        for it in range(n_iter):
            A_d_prod = self.A @ d
            alpha = d.dot(r) / d.dot(A_d_prod)
            x = x + alpha * d
            r = r - alpha * A_d_prod
            if np.allclose(r, zeros, atol=1e-10):
                elapsed = time() - start
                summary = {
                    'result': x,
                    'n_iter': it,
                    'run_time': round(float(elapsed),3)
                }
                return summary
            else:
                d = r - ( r.dot(A_d_prod) ) / ( d.dot(A_d_prod) ) * d
        elapsed = time() - start
        summary = {
            'result': x,
            'n_iter': it,
            'run_time': round(float(elapsed),3),
        }
        return summary

