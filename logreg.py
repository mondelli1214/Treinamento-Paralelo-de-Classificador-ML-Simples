
import numpy as np
from joblib import Parallel, delayed

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot(y, num_classes):
    Y = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y

def chunkify(n, n_chunks):
    size = (n + n_chunks - 1) // n_chunks
    for i in range(0, n, size):
        yield slice(i, min(i+size, n))

class SoftmaxLogReg:
    def __init__(self, input_dim, num_classes, lr=0.1, reg=1e-4, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(input_dim, num_classes)).astype(np.float32)
        self.b = np.zeros((1, num_classes), dtype=np.float32)
        self.lr = lr
        self.reg = reg
        self.num_classes = num_classes
        self.input_dim = input_dim

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X, jobs=1):
        if jobs == 1:
            return np.argmax(self.predict_proba(X), axis=1)
        # inferência paralela por blocos
        blocks = list(chunkify(X.shape[0], jobs))
        parts = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(lambda sl: np.argmax(self.predict_proba(X[sl]), axis=1))(sl) for sl in blocks
        )
        return np.concatenate(parts)

    def _batch_grad(self, Xb, Yb):
        P = self.predict_proba(Xb)     # (B, C)
        N = Xb.shape[0]
        grad_scores = (P - Yb) / N     # (B, C)
        dW = Xb.T @ grad_scores + self.reg * self.W
        db = np.sum(grad_scores, axis=0, keepdims=True)
        return dW, db

    def fit_sequential(self, X, y, epochs=5, batch_size=256, verbose=False):
        Y = one_hot(y, self.num_classes)
        N = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(N)
            X, Y = X[idx], Y[idx]
            for i in range(0, N, batch_size):
                Xb = X[i:i+batch_size]
                Yb = Y[i:i+batch_size]
                dW, db = self._batch_grad(Xb, Yb)
                self.W -= self.lr * dW
                self.b -= self.lr * db
            if verbose:
                loss = self.loss(X[:1000], Y[:1000])
                print(f"[Seq] epoch {ep+1}/{epochs} loss={loss:.4f}")

    def fit_parallel(self, X, y, epochs=5, batch_size=256, jobs=8, verbose=False):
        Y = one_hot(y, self.num_classes)
        N = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(N)
            X, Y = X[idx], Y[idx]
            for i in range(0, N, batch_size*jobs):
                Xchunk = X[i:i+batch_size*jobs]
                Ychunk = Y[i:i+batch_size*jobs]
                # divide o super-minibatch em jobs sub-batches e acumula gradientes em paralelo
                slices = []
                start = 0
                while start < Xchunk.shape[0]:
                    end = min(start + batch_size, Xchunk.shape[0])
                    slices.append(slice(start, end))
                    start = end
                grads = Parallel(n_jobs=jobs, prefer="threads")(
                    delayed(self._batch_grad)(Xchunk[sl], Ychunk[sl]) for sl in slices
                )
                dW = sum(g[0] for g in grads) / len(grads)
                db = sum(g[1] for g in grads) / len(grads)
                self.W -= self.lr * dW
                self.b -= self.lr * db
            if verbose:
                loss = self.loss(X[:1000], Y[:1000])
                print(f"[Par] epoch {ep+1}/{epochs} loss={loss:.4f}")

    def loss(self, X, Y):
        P = self.predict_proba(X)
        eps = 1e-12
        ce = -np.sum(Y * np.log(P + eps)) / X.shape[0]
        reg = 0.5 * self.reg * np.sum(self.W*self.W)
        return ce + reg

    def accuracy(self, X, y, jobs=1):
        preds = self.predict(X, jobs=jobs)
        return float((preds == y).mean())
