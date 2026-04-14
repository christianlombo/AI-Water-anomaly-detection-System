import numpy as np

class IsolationTree:

    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.node = None;

    def fit(self, X, current_height=0):
        if len(X) <= 1 or current_height >= self.height_limit:
            return {"type":  "leaf", "size": len(X)}
        
        num_feats = X.shape[1]
        feat_idx = np.random.randint(0, num_feats)

        f_min = X[:, feat_idx].min()
        f_max = X[:, feat_idx].max()

        if f_min == f_max:
            return {"type": "leaf", "size": len(X)}
        
        split = np.random.uniform(f_min, f_max)
        left_mask = X[:, feat_idx] < split

        return{
            "type": "node",
            "feat": feat_idx,
            "split": split,
            "left": self.fit(X[left_mask], current_height + 1),
            "right": self.fit(X[~left_mask], current_height + 1)
        }
    
class ScratchIsolationForest:
    def __init__(self, n_trees = 100, sam_size=256):
        self.n_trees = n_trees
        self.sam_size = sam_size
        self.trees = []

    def fit(self, X):
        self.trees = []
        h_limit = int(np.ceil(np.log2(self.sam_size)))

        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), min(len(X), self.sam_size), replace=False)
            tree = IsolationTree(h_limit)
            self.trees.append(tree.fit(X[idx]))
        return self
    
    def _path_length(self, x, node, depth):
        if node["type"] == "leaf":
            return depth + self._c_factor(node["size"])
        
        if x[node["feat"]] < node["split"]:
            return self._path_length(x, node["left"], depth + 1)
        return self._path_length(x, node["right"], depth + 1)

    def _c_factor(self, n):
        if n <= 1: return 0
        return 2 * (np.log(n - 1) + 0.5772) - (2 * (n-1)/n)

    def compute_anomaly_score(self, X):
        c = self._c_factor(self.sam_size)
        scores = []
        for x in X:
            avg_path = np.mean([self._path_length(x, t, 0) for t in self.trees])
            scores.append(2 ** - (avg_path / c))
        return np.array(scores)
