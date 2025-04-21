import numpy as np
from scipy.spatial import distance_matrix

def random_points_on_sphere(n, d, R):
    """
    Generate n random points on a sphere of dimension d and radius R.
    """
    vec = np.random.normal(size=(n, d))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec * R

def update_positions(points, R, iterations=5000, learning_rate=0.01):
    """
    Optimize the positions of points on a sphere using repulsive forces.
    Works in any dimension d.
    """
    for _ in range(iterations):
        dists = distance_matrix(points, points)
        np.fill_diagonal(dists, np.inf)

        forces = np.zeros_like(points)
        for i in range(points.shape[0]):
            diff_vectors = points[i] - points
            magnitudes = np.where(dists[i] > 0, 1 / dists[i]**2, 0)[:, np.newaxis]
            forces[i] += np.sum(diff_vectors * magnitudes, axis=0)

        forces /= np.linalg.norm(forces, axis=1, keepdims=True) + 1e-6
        points += learning_rate * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= R

    return points

def find_initial_radius(n, d, alpha):
    """
    Compute an initial radius to distribute n points with minimum distance alpha.
    """
    return alpha / (2 * np.sin(np.pi / n))

def optimize_radius(points, R, alpha, shrink_factor=0.99):
    """
    Gradually shrink the radius to satisfy the minimum distance constraint alpha.
    """
    while True:
        dists = distance_matrix(points, points)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        if min_dist < alpha:
            break

        R *= shrink_factor
        points *= shrink_factor

    return R / shrink_factor, points / shrink_factor

def compute_sigma2_per_class(x, y):
    """
    Compute sigma^2 for each class using intra-class variance.
    """
    unique_labels = np.unique(y)
    sigma2_per_class = {}
    for label in unique_labels:
        class_data = x[y == label]
        class_variance = np.var(class_data, axis=0)
        total_variance = np.sum(class_variance)
        sigma2_per_class[label] = total_variance
    return sigma2_per_class

def compute_class_means(x, y):
    """
    Compute the mean feature vector for each class.
    """
    class_means = {}
    for label in np.unique(y):
        class_means[label] = np.mean(x[y == label], axis=0)
    return class_means

def compute_class_distance_matrix(class_means):
    """
    Compute pairwise RMSE distances between class means.
    """
    labels = sorted(class_means.keys())
    n = len(labels)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rmse = np.sqrt(np.mean((class_means[labels[i]] - class_means[labels[j]])**2))
            dist_matrix[i, j] = rmse
            dist_matrix[j, i] = rmse
    return dist_matrix

def assign_labels_to_clusters(class_dist_matrix, cluster_centers):
    """
    Assign class labels to cluster centers using linear sum assignment.
    The goal is to match inter-class distances with cluster distances.
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.optimize import linear_sum_assignment

    cluster_dists = squareform(pdist(cluster_centers))
    D_max = np.max(class_dist_matrix)
    inverted_class_dists = D_max - class_dist_matrix

    cost_matrix = np.abs(cluster_dists - inverted_class_dists)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return {i: cluster_centers[j] for i, j in zip(row_ind, col_ind)}
