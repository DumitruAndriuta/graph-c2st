import networkx as nx
import numpy as np
from grakel import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC


def prepare_kernel_matrix(G_all, node_label="constant"):
    """Compute WL kernel matrix once, independent of y. Run outside the permutation loop.

    node_label: see prepare_distance_matrix in knn.py for the "constant" vs "degree"
    tradeoff -- "constant" is the default structure-only baseline.
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

    return K_matrix


def get_predictions(kernel_matrix, y, idx_train, idx_test):
    """Fit ONCE on true labels using a precomputed kernel, predict ONCE."""
    K_train = kernel_matrix[np.ix_(idx_train, idx_train)]
    K_test = kernel_matrix[np.ix_(idx_test, idx_train)]

    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(K_train, y[idx_train])

    y_pred = svm_model.predict(K_test)
    y_true_test = y[idx_test]
    return y_pred, y_true_test

def fit_predict_full(kernel_matrix, y):
    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(kernel_matrix, y)
    return svm_model.predict(kernel_matrix)