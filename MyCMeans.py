import numpy as np

class MyCMeans:

    def __init__(self, n_clusters=2, w=2.0, max_iter=1000, eps=1e-6):
        self.k = n_clusters
        self.w = w
        self.max_iter = max_iter
        self.eps = eps

    def init_M(self, n):
        M = np.random.rand(n, self.k)
        M /= M.sum(axis=1, keepdims=True)
        return M

    def calc_centers(self, M):
        # slow code!
        # centers = []
        # for j in range(self.k):
        #     c = np.zeros(self.dim)
        #     q = 0
        #     for i in range(self.n):
        #         c += M[i][j] ** self.w * self.X[i]
        #         q += M[i][j] ** self.w
        #     c /= q
        #     centers.append(c.tolist())
        # return np.array(centers)
        M_pow_w = M ** self.w
        return (M_pow_w[:, :, None] * self.X_[:, None, :]).sum(axis=0) / M_pow_w.sum(axis=0)[:, None]

    def dist(self, x, y):
        return ((x - y) ** 2).sum() ** 0.5

    def recalc_M(self, centers):
        # slow code!
        # p = 2 / (self.w - 1)
        # M = [[0 for _ in range(self.k)] for _ in range(self.n)]
        # for i in range(self.n):
        #     sum = 0
        #     for j in range(self.k):
        #         sum += self.dist(self.X[i], centers[j]) ** -p
        #     for j in range(self.k):
        #         M[i][j] = 1 / (self.dist(self.X[i], centers[j]) ** p * sum)
        # return np.array(M)
        d = np.linalg.norm(self.X_[:, None, :] - centers[None, :, :], axis=2)
        p = 2 / (self.w - 1)
        M = ((d ** p) * (d ** -p).sum(axis=1, keepdims=True)) ** -1
        return M

    def measure(self, A):
        return max(A.max(), -A.min())

    def calc_labels(self):
        max_cols = np.unique(np.argmax(self.M_, axis=1)).tolist()
        self.n_non_empty_centers = len(max_cols)
        return np.array([max_cols.index(np.argmax(self.M_[i])) for i in range(len(self.X_))])

    def calc_center_2_vec(self):
        center_2_vec = {i:[] for i in range(len(self.centers_))}
        [center_2_vec[np.argmax(self.M_[i])].append(i) for i in range(len(self.X_))]
        return center_2_vec

    def fit(self, X):
        self.X_ = np.array(X)
        M = self.init_M(len(X))
        centers = self.calc_centers(M)
        i = 0
        while i < self.max_iter:
            M_new = self.recalc_M(centers)
            if self.measure(M_new - M) < self.eps:
                print(f'total: {i} iters')
                break
            M = M_new
            centers = self.calc_centers(M)
            i += 1
        if i == self.max_iter:
            print(f'{self.max_iter} iters is not enough')

        self.M_ = M
        self.centers_ = centers
        self.labels_ = self.calc_labels()
        self.center_2_vec_ = self.calc_center_2_vec()
        return self
