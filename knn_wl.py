import networkx as nx
import numpy as np
from grakel import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.neighbors import KNeighborsClassifier


def prepare_distance_matrix(G_all, node_label="constant"):
    """Compute WL distance matrix once, independent of y. Run outside the permutation loop.

    node_label:
        "constant" (default) -- every node gets the same label ("1"). This is the
            standard structure-only baseline (Xu et al., 2019): WL can only exploit
            topology, not any injected structural prior. Use this unless you have a
            specific reason to inject degree.
        "degree"   -- node label = degree. Kept for the degree-informed comparison
            condition; NOT the default, since it primes WL to detect degree-based
            differences specifically (see methodology discussion on circularity).
    """
    if node_label == "constant":
        for G in G_all:
            nx.set_node_attributes(G, "1", "label")
    elif node_label == "degree":
        for G in G_all:
            nx.set_node_attributes(G, dict(G.degree()), "label")
    else:
        raise ValueError(f"Unknown node_label option: {node_label}")

    G_grakel = list(graph_from_networkx(G_all, node_labels_tag="label"))
    wl_kernel = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram, normalize=True)
    K_matrix = wl_kernel.fit_transform(G_grakel)

    return np.sqrt(np.maximum(0, 2 - 2 * K_matrix))


def get_predictions(distance_matrix, y, idx_train, idx_test):
    """Fit ONCE on true labels, predict ONCE. Feeds into the generic permutation test."""
    D_train = distance_matrix[np.ix_(idx_train, idx_train)]
    D_test = distance_matrix[np.ix_(idx_test, idx_train)]

    knn_model = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    knn_model.fit(D_train, y[idx_train])

    y_pred = knn_model.predict(D_test)
    y_true_test = y[idx_test]
    return y_pred, y_true_test

def fit_predict_full(distance_matrix, y):
    """Fit on ALL points, predict on the SAME all points (resubstitution).
    distance_matrix is label-independent and precomputed once outside the
    permutation loop; only the KNN fit is redone per call."""
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    knn_model.fit(distance_matrix, y)
    return knn_model.predict(distance_matrix)