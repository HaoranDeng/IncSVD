# Author: Deng Haoran <denghaoran@zju.edu.cn>
#         Li Jiahe
# License: MIT License

import time
import numpy as np
import scipy.sparse

msg_len = 60

class EvolvingMatrix(object):
    """
    Parameters
    ----------
    data : ndarray of shape (m_dim, n_dim)
        The initial matrix.

    sparse : bool, default=True
        True - matrix is sparse: Use the proposed acceleration methods [1].
        False - matrix is dense: Use the original methods (ZhaSimon's [2], Vecharynski's [3], Yamazaki's [4])

    k : int
        Rank of truncated SVD to be calculated

    method : string, default='ZhaSimon'
        method for SVD update
        choose from ['ZhaSimon', 'GKL', 'RPI']

    Attributes
    ----------

    _m : int
        Number of rows in data matrix

    _n : int
        Number of columns in data matrix
    
    _Uk, _sigmak, _Vk : ndarrays of shape (m, k), (k,), (n, k) 
    
    _update_matrix : ndarray of shape (u, n)
        Matrix to be appended directly

        
    References
    ----------
    [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
        The Twelfth International Conference on Learning Representations. 2024.

    [2] Zha, Hongyuan, and Horst D. Simon. "On updating problems in latent semantic indexing." 
        SIAM Journal on Scientific Computing 21.2 (1999): 782-791.

    [3] Vecharynski, Eugene, and Yousef Saad. "Fast updating algorithms for latent semantic indexing." 
        SIAM Journal on Matrix Analysis and Applications 35.3 (2014): 1105-1131.

    [4] Yamazaki, Ichitaro, et al. "Randomized algorithms to update partial singular value decomposition on a hybrid 
        CPU/GPU cluster." Proceedings of the International Conference for High Performance Computing, Networking, 
        Storage and Analysis. 2015.
        
    [5] V. Kalantzis, G. Kollias, S. Ubaru, A. N. Nikolakopoulos, L. Horesh, and K. L. Clarkson,
        “Projection techniquesto update the truncated SVD of evolving matrices with applications,”
        inProceedings of the 38th InternationalConference on Machine Learning,
        M. Meila and T. Zhang, Eds.PMLR, 7 2021, pp. 5236-5246.
    """
    def __init__(self, data, k, sparse = True, method = "RPI"):
        """ Initialize the matrix (not the SVD)

        Parameters
        ---------- 
        matrix : ndarray of shape (m_dim, n_dim)
            Initial input matrix

        network : bool - whether it's a network
            True - An update consists of a row update and a column update
            False - only update rows

        sparse : bool - whether it's a sparse matrix
            True - use our methods
            False - use the original methods(z-s, vecharynski, random)
        """
        
        (self._m, self._n) = np.shape(data) 
        self._sparse = sparse

        assert method in ["ZhaSimon", "GKL", "RPI"]
        self._method_dict = {'ZhaSimon': 1, 'GKL': 2, 'RPI' : 3}
        self._method = self._method_dict.get(method, 1)

        self._k_dim = k
        
        U_true, sigma_true, V_true = scipy.sparse.linalg.svds(data, k)
        V_true = V_true.T

        self._Uk = U_true[:, :self._k_dim]
        self._sigmak = sigma_true[:self._k_dim]
        self._Vk = V_true[:, :self._k_dim]

        if self._sparse == True: # isvd
            self._Ku = np.eye(self._k_dim, dtype=np.float64)
            self._Kv = np.eye(self._k_dim, dtype=np.float64)
        else:
            self._Ku, self._Kv = None, None
            
        self._inverted_index = [0] * max(self._m, self._n)
        
        print(f"{'Initial Uk matrix of evolving matrix set to shape of ':<{msg_len}}{np.shape(self._Uk)}.")
        print(f"{'Initial sigmak array of evolving matrix set to shape of ':<{msg_len}}{np.shape(self._sigmak)}.")
        print(f"{'Initial VHk matrix of evolving matrix set to shape of ':<{msg_len}}{np.shape(self._Vk.T)}.")

        
    @property
    def Uk(self):
        ret = self._Uk
        if self._Ku is not None:
            ret = ret @ self._Ku
        return ret
    
    @property
    def Vk(self):
        ret = self._Vk
        if self._Kv is not None:
            ret = ret @ self._Kv
        return ret
    
    @property
    def Sigmak(self):
        return self._sigmak 
    
    def Uki(self, i):
        ret = self._Uk[i]
        if self._Ku is not None:
            ret = ret @ self._Ku
        return ret


    def Vki(self, i):
        ret = self._Vk[i]
        if self._Kv is not None:
            ret = ret @ self._Kv
        return ret


    def _check_lt(self):
        if self._method == 1:
            assert self._l is None
            assert self._t is None
        elif self._method == 2:
            assert self._l is not None
            assert self._t is None
        elif self._method == 3:
            assert self._l is not None
            assert self._l is not None
        else:
            raise NotImplementedError

    def add_row(self, update_matrix, l=None, t=None):
        """ Perform updates in row
        """
        assert update_matrix.shape[1] == self._n
        self._m += update_matrix.shape[0]
        self._update_matrix = update_matrix
        self._l, self._t = l, t
        self._check_lt()

        if self._sparse == True:
            assert isinstance(self._update_matrix, scipy.sparse._csr.csr_matrix), \
                "The update matrix should be a scipy.sparse._csr.csr_matrix"
            if self._method == 1:
                self._update_svd_isvd1_row()
            elif self._method == 2:
                self._update_svd_isvd2_row()
            else:
                self._update_svd_isvd3_row()
        else:
            if self._method == 1:
                self._update_svd_zhasimon_row()
            elif self._method == 2:
                self._update_svd_vecharynski_row()
            else:
                self._update_svd_yamazaki_row()
        

    def add_column(self, update_matrix, l=None, t=None):
        """ Perform updates in column
        """
        # print(update_matrix)
        assert update_matrix.shape[0] == self._m
        self._n += update_matrix.shape[1]
        self._update_matrix = update_matrix
        self._l, self._t = l, t
        self._check_lt()

        if self._sparse == True:
            assert isinstance(self._update_matrix, scipy.sparse._csc.csc_matrix), \
                "The update matrix should be a scipy.sparse._csc.csc_matrix"
        self._update_matrix = self._update_matrix.T

        self._Uk, self._Vk = self._Vk, self._Uk

        if self._sparse == True:
            self._Ku, self._Kv = self._Kv, self._Ku
            if self._method == 1:
                self._update_svd_isvd1_row()
            elif self._method == 2:
                self._update_svd_isvd2_row()
            else:
                self._update_svd_isvd3_row()
            self._Ku, self._Kv = self._Kv, self._Ku
        else:
            if self._method == 1:
                self._update_svd_zhasimon_row()
            elif self._method == 2:
                self._update_svd_vecharynski_row()
            else:
                self._update_svd_yamazaki_row()

        self._Uk, self._Vk = self._Vk, self._Uk


    def update_weight(self, update_matrix_B, update_matrix_C, l=None, t=None):
        assert update_matrix_B.shape[1] == update_matrix_C.shape[1]
        assert update_matrix_B.shape[0] == self._m
        assert update_matrix_C.shape[0] == self._n

        self._update_B = update_matrix_B
        self._update_C = update_matrix_C
        self._l, self._t = l, t
        self._check_lt()

        if self._sparse is True:
            if self._method == 1:
                self._update_svd_isvd1_weight()
            elif self._method == 2:
                self._update_svd_isvd2_weight()
            elif self._method == 3:
                self._update_svd_isvd3_weight()
            else:
                raise NotImplementedError
        else:
            if self._method == 1:
                self._update_svd_zhasimon_weight()
            elif self._method == 2:
                self._update_svd_vecharynski_weight()
            elif self._method == 3:
                self._update_svd_yamazaki_weight()
            else:
                raise NotImplementedError


    def _tSVD(self, M):
        ''' Return the rank-k truncated SVD of matrix M. '''
        Fk, Tk, Gk = np.linalg.svd(M, full_matrices=False)
        Gk = Gk.T
        # Truncate if necessary
        if self._k_dim < len(Tk):
            Fk = Fk[:, :self._k_dim]
            Tk = Tk[:self._k_dim]
            Gk = Gk[:, :self._k_dim]
        return Fk, Tk, Gk


    def _get_compressed(self, E):
        '''
        This function takes as input a sparse matrix, and outputs a dense matrix with all all-zero rows removed 
        and a list of uid's recording the row number of each row that is not all-zero.
        '''
        print(type(E))
        s = E.shape[1]
        uid = np.unique(E.indices)
        B = np.zeros((len(uid), s), dtype=np.float64)

        while len(self._inverted_index) < E.shape[0]:
            self._inverted_index.append(0)

        for i in range(len(uid)):
            self._inverted_index[ uid[i] ] = i
        cur = 0
        for i in range(s):
            for _ in range( E.indptr[i+1] - E.indptr[i] ):
                B[self._inverted_index[ E.indices[cur] ], i] = E.data[cur]
                cur += 1
        return B, uid
    

    def _sparse_QR(self, B0, C0):
        B, C = B0.copy(), C0.copy()
        s = B.shape[1]
        R = np.zeros((s, s), dtype=np.float64)
        for i in range(s):
            tmp = np.dot(B[:, i], B[:, i]) - np.dot(C[:, i], C[:, i])
            alpha = 0
            if abs(tmp) > 1e-5:
                alpha = np.sqrt(tmp)
                B[:, i] /= alpha
                C[:, i] /= alpha
            R[i, i] = alpha
            for j in range(i+1, s):
                beta =  np.dot(B[:, i], B[:, j]) - np.dot(C[:, i], C[:, j])
                B[:, j] -= beta * B[:, i]
                C[:, j] -= beta * C[:, i]
                R[i, j] = beta
        return B, C, R
    

    def _update_svd_isvd1_row(self):
        '''
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        s = self._update_matrix.shape[0]
        k = self._k_dim

        ''' Step 1. QR-factorization. (Ref: Algorithm 2, [1]; Line 1 in Algorithm 3, [1]) '''
        
        Bs, uid = self._get_compressed(self._update_matrix.T)
        Cs = (self._Kv.T @ self._Vk[uid].T @ Bs)
        B, C, R = self._sparse_QR(Bs, Cs)
 
        ''' Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 3, [1]) '''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, s), dtype=np.float64)), axis=1)
        Md = np.concatenate((Cs.T, R.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        Fk, Tk, Gk = self._tSVD(M)

        ''' Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 3, [1]) '''
        self._Ku = self._Ku @ Fk[:k]
        self._Kv = self._Kv @ (Gk[:k] - C @ Gk[k:])

        delta_Uk = Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk = np.append(self._Uk, delta_Uk, axis=0)
        
        self._sigmak = Tk                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

        delta_Vk = B @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid] += delta_Vk


    def _update_svd_isvd1_weight(self):
        '''
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        s = self._update_B.shape[1]
        k = self._k_dim
        E1 = self._update_B
        E2 = self._update_C
        U = self._Uk
        V = self._Vk

        ''' Step 1. QR-factorization. (Ref: Algorithm 2, [1]; Line 1 in Algorithm 3, [1]) '''
        Bs1, uid1 = self._get_compressed(E1)
        Bs2, uid2 = self._get_compressed(E2)
        Cs1 = (self._Ku.T @ U[uid1].T @ Bs1)
        Cs2 = (self._Kv.T @ V[uid2].T @ Bs2)

        B1, C1, R1 = self._sparse_QR(Bs1, Cs1)
        B2, C2, R2 = self._sparse_QR(Bs2, Cs2)

        ''' Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 3, [1]) '''
        M = np.zeros((k+s, k+s), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((Cs1, R1), axis=0)
        M2 = np.concatenate((Cs2, R2), axis=0)
        M = M + M1 @ M2.T

        Fk, Tk, Gk = self._tSVD(M)

        ''' Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 3, [1]) '''
        self._Ku = self._Ku @ (Fk[:k] - C1 @ Fk[k:])
        self._Kv = self._Kv @ (Gk[:k] - C2 @ Gk[k:])

        delta_Uk = B1 @ Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk[uid1] += delta_Uk
        
        self._sigmak = Tk                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

        delta_Vk = B2 @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid2] += delta_Vk

    
    def _sparse_GKL(self, Bs, Cs, l):
        s = Bs.shape[1]
        Bp = np.zeros((Bs.shape[0], l+1), dtype=np.float64)
        Cp = np.zeros((Cs.shape[0], l+1), dtype=np.float64)

        P = np.zeros((s, l+2), dtype=np.float64)
        P[:, 1] = np.random.randn(s)
        P[:, 1] = P[:, 1] / np.linalg.norm(P[:, 1])
        beta = np.zeros((l+1, ), dtype=np.float64)
        alpha = np.zeros((l+1, ), dtype=np.float64)

        for i in range(1, l+1):
            Bp[:, i] = Bs @ P[:, i] - beta[i-1] * Bp[:, i-1]
            Cp[:, i] = Cs @ P[:, i] - beta[i-1] * Cp[:, i-1]
            tmp = np.dot(Bp[:, i], Bp[:, i]) - np.dot(Cp[:, i], Cp[:, i])
            if abs(tmp) < 1e-9:
                alpha[i] = 0
            else:
                alpha[i] = np.sqrt( tmp )
                Bp[:, i] /= alpha[i]
                Cp[:, i] /= alpha[i]
            
            P[:, i+1] = Bs.T @ Bp[:, i] - Cs.T @ Cp[:, i] - alpha[i] * P[:, i]
            for j in range(1, i+1):
                P[:, i+1] -= np.dot(P[:, i+1], P[:, j]) * P[:, j]
            beta[i] = np.linalg.norm(P[:, i+1])
            if abs(beta[i]) < 1e-9:
                l = i
                break
            P[:, i+1] /= beta[i]
        
        L = np.zeros((l, l+1), dtype=np.float64)
        for i in range(l):
            L[i, i] = alpha[i+1]
            L[i, i+1] = beta[i+1]
        
        Bp, Cp, P = Bp[:, 1:], Cp[:, 1:], P[:, 1:]
        Bp, Cp, P = Bp[:, :l], Cp[:, :l], P[:, :l+1]
        return Bp, Cp, P @ L.T


    def _update_svd_isvd2_row(self):
        '''
        Adding rows.
        Using GKL procedure to approximate the augmented space.
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        E = self._update_matrix
        k = self._k_dim
        s = E.shape[0]
        l = self._l

        ''' Step 1. Approximate the augment space with GKL. (Ref: Algorithm 6, [1]) '''
        Bs, uid = self._get_compressed(self._update_matrix.T)
        Cs = (self._Kv.T @ (self._Vk[uid].T) @ Bs)
        B, C, P = self._sparse_GKL(Bs, Cs, l)       

        ''' 
        Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 9, [1])
        '''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((Cs.T, P), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        Fk, Tk, Gk = self._tSVD(M)

        ''' 
        Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 9, [1])
        '''
        self._Ku = self._Ku @ Fk[:k]
        self._Kv = self._Kv @ (Gk[:k] - C @ Gk[k:])

        delta_Uk = Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk = np.append(self._Uk, delta_Uk, axis=0)
        
        self._sigmak = Tk

        delta_Vk = B @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid] += delta_Vk

    def _update_svd_isvd2_weight(self):
        '''
        Update weight (low-rank update).
        Using GKL procedure to approximate the augmented space.
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        s = self._update_B.shape[1]
        k = self._k_dim
        E1 = self._update_B
        E2 = self._update_C
        U = self._Uk
        V = self._Vk
        l = self._l

        ''' Step 1. Approximate the augment space with GKL. (Ref: Algorithm 6, [1]) '''
        Bs1, uid1 = self._get_compressed(E1)
        Bs2, uid2 = self._get_compressed(E2)
        Cs1 = (self._Ku.T @ U[uid1].T @ Bs1)
        Cs2 = (self._Kv.T @ V[uid2].T @ Bs2)
        B1, C1, P1 = self._sparse_GKL(Bs1, Cs1, l)       
        B2, C2, P2 = self._sparse_GKL(Bs2, Cs2, l)       

        ''' 
        Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 9, [1])
        '''
        M = np.zeros((k+l, k+l), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((Cs1, P1.T), axis=0)
        M2 = np.concatenate((Cs2, P2.T), axis=0)
        M = M + M1 @ M2.T

        Fk, Tk, Gk = self._tSVD(M)

        ''' 
        Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 9, [1])
        '''
        self._Ku = self._Ku @ (Fk[:k] - C1 @ Fk[k:])
        self._Kv = self._Kv @ (Gk[:k] - C2 @ Gk[k:])

        delta_Uk = B1 @ Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk[uid1] += delta_Uk
        
        self._sigmak = Tk                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

        delta_Vk = B2 @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid2] += delta_Vk


    def _sparse_RPI(self, Bs, Cs, l, t):
        s = Bs.shape[1]
        P = np.zeros((s, l), dtype=np.float64)
        for i in range(l):
            P[:, i] = np.random.randn(s)
            P[:, i] = P[:, i] / np.linalg.norm(P[:, i])

        for _ in range(t):
            P, _ = np.linalg.qr(P)
            Bq, Cq = Bs @ P, Cs @ P
            for i in range(l):
                for j in range(i):
                    beta = np.dot(Bq[:, i], Bq[:, j]) - np.dot(Cq[:, i], Cq[:, j])
                    Bq[:, i] -= beta * Bq[:, j]
                    Cq[:, i] -= beta * Cq[:, j]
                tmp = np.dot(Bq[:, i], Bq[:, i]) - np.dot(Cq[:, i], Cq[:, i])
                if abs(tmp) < 1e-9:
                    continue
                alpha = np.sqrt( tmp )
                Bq[:, i] /= alpha
                Cq[:, i] /= alpha
            P = Bs.T @ Bq - Cs.T @ Cq
        return Bq, Cq, P


    def _update_svd_isvd3_row(self):
        '''
        Adding rows.
        Using Random Power Iteration procedure to approximate the augmented space.
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        k = self._k_dim
        l = self._l
        t = self._t

        ''' Step 1. Approximate the augment space with Random Power Iteration. (Ref: Algorithm 8, [1]) '''
        Bs, uid = self._get_compressed(self._update_matrix.T)
        Cs = (self._Kv.T @ (self._Vk[uid].T) @ Bs)
        Bq, Cq, P = self._sparse_RPI(Bs, Cs, l, t)

        ''' 
        Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 9, [1])
        '''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((Cs.T, P), axis=1)
        M = np.concatenate((Mu, Md), axis=0)        
        Fk, Tk, Gk = self._tSVD(M)

        ''' 
        Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 9, [1])
        '''
        self._Ku = self._Ku @ Fk[:k]
        self._Kv = self._Kv @ (Gk[:k] - Cq @ Gk[k:])

        delta_Uk = Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk = np.append(self._Uk, delta_Uk, axis=0)
        
        self._sigmak = Tk

        delta_Vk = Bq @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid] += delta_Vk

    def _update_svd_isvd3_weight(self):
        '''
        Update weight (low-rank update).
        Using Random Power Iteration procedure to approximate the augmented space.
        Reference: 
        [1] Haoran Deng, et al. "Fast Updating Truncated SVD for Representation Learning with Sparse Matrices" 
            The Twelfth International Conference on Learning Representations. 2024.
        '''
        k = self._k_dim
        l = self._l
        t = self._t

        ''' Step 1. Approximate the augment space with Random Power Iteration. (Ref: Algorithm 8, [1]) '''
        Bs1, uid1 = self._get_compressed(self._update_B)
        Bs2, uid2 = self._get_compressed(self._update_C)
        Cs1 = (self._Ku.T @ (self._Uk[uid1].T) @ Bs1)
        Cs2 = (self._Kv.T @ (self._Vk[uid2].T) @ Bs2)
        Bq1, Cq1, P1 = self._sparse_RPI(Bs1, Cs1, l, t)
        Bq2, Cq2, P2 = self._sparse_RPI(Bs2, Cs2, l, t)

        ''' 
        Step 2. Compute a compact Trucated SVD. (Ref: Line 2 in Algorithm 9, [1])
        '''
        M = np.zeros((k+l, k+l), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((Cs1, P1.T), axis=0)
        M2 = np.concatenate((Cs2, P2.T), axis=0)
        M = M + M1 @ M2.T
        Fk, Tk, Gk = self._tSVD(M)

        ''' 
        Step 3. Update Uk, sigmak, and Vk. (Ref: Line 3-7 in Algorithm 9, [1])
        '''
        self._Ku = self._Ku @ (Fk[:k] - Cq1 @ Fk[k:])
        self._Kv = self._Kv @ (Gk[:k] - Cq2 @ Gk[k:])

        delta_Uk = Bq1 @ Fk[k:] @ np.linalg.inv(self._Ku)
        self._Uk[uid1] += delta_Uk
        
        self._sigmak = Tk

        delta_Vk = Bq2 @ Gk[k:] @ np.linalg.inv(self._Kv)
        self._Vk[uid2] += delta_Vk


    def _update_svd_zhasimon_row(self):
        """
        Adding rows.
        Return truncated SVD of updated matrix using the Zha-Simon projection method.
        [2] Zha, Hongyuan, and Horst D. Simon. "On updating problems in latent semantic indexing." 
            SIAM Journal on Scientific Computing 21.2 (1999): 782-791.
        
        """
        
        '''=====Step 1====='''
        E = self._update_matrix
        V = self._Vk

        s = E.shape[0]
        k = self._Uk.shape[1]

        Q, R = np.linalg.qr(E.T - V @ (V.T @ E.T))
        Z = scipy.linalg.block_diag(self._Uk, np.eye(s))
        W = np.concatenate((V, Q), axis=1)

        '''=====Step 2====='''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, s), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, R.T), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk

    def _update_svd_zhasimon_weight(self):
        """
        Update weight (low-rank update).
        Return truncated SVD of updated matrix using the Zha-Simon projection method.
        [2] Zha, Hongyuan, and Horst D. Simon. "On updating problems in latent semantic indexing." 
            SIAM Journal on Scientific Computing 21.2 (1999): 782-791.
        """
        
        '''=====Step 1====='''
        Eb = self._update_B                      # n * s
        Ec = self._update_C                      # m * s
        U = self._Uk
        V = self._Vk
        s = Eb.shape[1]
        k = self._k_dim

        Qb, Rb = np.linalg.qr(Eb - U @ (U.T @ Eb))
        Qc, Rc = np.linalg.qr(Ec - V @ (V.T @ Ec))
        Z = np.concatenate((U, Qb), axis=1)
        W = np.concatenate((V, Qc), axis=1)

        '''=====Step 2====='''
        M = np.zeros((s+k, s+k), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((U.T@Eb, Rb), axis=0)
        M2 = np.concatenate((V.T@Ec, Rc), axis=0)
        M = M + M1 @ M2.T

        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk
    

    def _GKL(self, V, E, l):    
        """
        Return the Lanczos vectors approximate by a Golub-Kahan-Lanczos Bidiagonalization Procedure.
        We found a good reference to learn this process is: https://www.netlib.org/utk/people/JackDongarra/etemplates/node198.html
        Return truncated SVD of updated matrix using the Zha-Simon projection method.
        [2] Zha, Hongyuan, and Horst D. Simon. "On updating problems in latent semantic indexing." 
            SIAM Journal on Scientific Computing 21.2 (1999): 782-791.
        """

        n, s = E.shape
        Q = np.zeros((n, l+1), dtype=np.float64)
        P = np.zeros((s, l+2), dtype=np.float64)
        B = np.zeros((l, l+1), dtype=np.float64)

        P[:, 1] = np.random.randn(s)
        P[:, 1] = P[:, 1] / np.linalg.norm(P[:, 1])
        beta = np.zeros((l+1, ), dtype=np.float64)
        alpha = np.zeros((l+1, ), dtype=np.float64)

        for i in range(1, l+1):
            # Q[:, i] = X @ P[:, i] - beta[i-1] * Q[:, i-1]
            Q[:, i] = E @ P[:, i] - V @ ((V.T @ E) @ P[:, i]) - beta[i-1] * Q[:, i-1]

            alpha[i] = np.linalg.norm(Q[:, i])
            if alpha[i] == 0:
                Q[:, i] = 0
            else:
                Q[:, i] /= alpha[i]

            # P[:, i+1] = X.T @ Q[:, i] - alpha[i] * P[:, i]
            P[:, i+1] = E.T @ Q[:, i] - E.T @ (V @ (V.T @ Q[:, i])) - alpha[i] * P[:, i]
            for j in range(1, i+1):
                P[:, i+1] -= np.dot(P[:, i+1], P[:, j]) * P[:, j]

            beta[i] = np.linalg.norm(P[:, i+1])
            if beta[i] == 0:
                P[:, i+1] = 0
                continue
            P[:, i+1] /= beta[i]


        for i in range(l):
            B[i, i] = alpha[i+1]
            B[i, i+1] = beta[i+1]

        P = P[:, 1:]
        Q = Q[:, 1:]
        P = P @ B.T
        return Q, P


    def _update_svd_vecharynski_row(self):
        """
        Adding rows.
        Return truncated SVD of updated matrix using the GKL process to approximate the augment space.
        
        [3] Vecharynski, Eugene, and Yousef Saad. "Fast updating algorithms for latent semantic indexing." 
            SIAM Journal on Matrix Analysis and Applications 35.3 (2014): 1105-1131.
        """
        E = self._update_matrix
        V = self._Vk
        
        k = self._k_dim
        s = E.shape[0]
        l = self._l
        
        '''=====Step 1====='''
        # X = E.T - V @ ((V.T) @ (E.T))
        Q, P = self._GKL(V, E.T, l)

        Z = scipy.linalg.block_diag(self._Uk, np.eye(s))
        W = np.concatenate((self._Vk, Q), axis=-1)

        '''=====Step 2====='''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, P), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk


    def _update_svd_vecharynski_weight(self):
        """
        Update weight (low-rank update).
        Return truncated SVD of updated matrix using the GKL process to approximate the augment space.
        
        [3] Vecharynski, Eugene, and Yousef Saad. "Fast updating algorithms for latent semantic indexing." 
            SIAM Journal on Matrix Analysis and Applications 35.3 (2014): 1105-1131.
        """
        Eb = self._update_B
        Ec = self._update_C
        U = self._Uk
        V = self._Vk
        k = self._k_dim
        l = self._l

        '''=====Step 1====='''
        Qb, Pb = self._GKL(U, Eb, l)
        Qc, Pc = self._GKL(V, Ec, l)
        Z = np.concatenate((U, Qb), axis=-1)
        W = np.concatenate((V, Qc), axis=-1)

        '''=====Step 2====='''
        M = np.zeros((k+l, k+l), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((U.T @ Eb, Pb.T), axis=0)
        M2 = np.concatenate((V.T @ Ec, Pc.T), axis=0)
        M = M + M1 @ M2.T
        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk


    def _RPI(self, V, E, l, t):
        s = E.shape[1]
        P = np.zeros((s, l), dtype=np.float64)
        for i in range(l):
            P[:, i] = np.random.randn(s)
            P[:, i] = P[:, i] / np.linalg.norm(P[:, i])


        for i in range(t):
            P, _ = np.linalg.qr(P)
            Q = E @ P - V @ (((V.T) @ (E)) @ P)
            Q, _ = np.linalg.qr(Q)
            P = E.T @ Q - E.T @ (V @ (V.T @ Q))
        return Q, P


    def _update_svd_yamazaki_row(self):
        """
        Adding rows.
        Return truncated SVD of updated matrix using Random Power Iterations to approximate the augmented space.

        [4] Yamazaki, Ichitaro, et al. "Randomized algorithms to update partial singular value decomposition on a hybrid 
            CPU/GPU cluster." Proceedings of the International Conference for High Performance Computing, Networking, 
            Storage and Analysis. 2015.
        
        """
        E = self._update_matrix
        V = self._Vk

        s = E.shape[0]
        k = self._Uk.shape[1]
        l = self._l
        l = min(l, s)
        t = self._t

        '''=====Step 1====='''
        Q, P = self._RPI(V, E.T, l, t)

        Z = scipy.linalg.block_diag(self._Uk, np.eye(s))
        W = np.concatenate((self._Vk, Q), axis=-1)
        
        '''=====Step 2====='''
        Mu = np.concatenate((np.diag(self._sigmak), np.zeros((k, l), dtype=np.float64)), axis=1)
        Md = np.concatenate((E@V, P), axis=1)
        M = np.concatenate((Mu, Md), axis=0)
        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk


    def _update_svd_yamazaki_weight(self):
        """
        Update weight (low-rank update).
        Return truncated SVD of updated matrix using Random Power Iterations to approximate the augmented space.

        [4] Yamazaki, Ichitaro, et al. "Randomized algorithms to update partial singular value decomposition on a hybrid 
            CPU/GPU cluster." Proceedings of the International Conference for High Performance Computing, Networking, 
            Storage and Analysis. 2015.
        
        """

        '''=====Step 1====='''
        Eb = self._update_B
        Ec = self._update_C
        U = self._Uk
        V = self._Vk
        k = self._k_dim
        l = min(self._l, Eb.shape[1])
        t = self._t

        '''=====Step 1====='''
        Qb, Pb = self._RPI(U, Eb, l, t)
        Qc, Pc = self._RPI(V, Ec, l, t)
        Z = np.concatenate((U, Qb), axis=-1)
        W = np.concatenate((V, Qc), axis=-1)

        '''=====Step 2====='''
        M = np.zeros((k+l, k+l), dtype=np.float64)
        M[:k, :k] = np.diag(self._sigmak)
        M1 = np.concatenate((U.T @ Eb, Pb.T), axis=0)
        M2 = np.concatenate((V.T @ Ec, Pc.T), axis=0)
        M = M + M1 @ M2.T
        Fk, Tk, Gk = self._tSVD(M)

        '''=====Step 3====='''
        # Calculate updated values for Uk, Sk, Vk
        self._Uk, self._sigmak, self._Vk = Z @ Fk, Tk, W @ Gk
