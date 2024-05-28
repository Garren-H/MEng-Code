import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.metrics import silhouette_score # type: ignore

def get_subset_data(functional_groups):
    with pd.ExcelFile('All Data.xlsx') as f:
        PureData_df = pd.read_excel(f, sheet_name='Pure compounds')
    if functional_groups[0] == 'all':
        subset_df = PureData_df.copy()
    else:
        subset_df = PureData_df[np.sum((PureData_df['Functional Group'].to_numpy()[:, np.newaxis] == functional_groups[np.newaxis,:]).astype(int), axis=1).astype(bool)].copy()
    X = np.array([subset_df['Boiling temperature [K]'].to_numpy(),
                  subset_df['Density [kg/m3]'].to_numpy(),
                  subset_df['Molecular weight [g/mol]'].to_numpy()]).T
    return X 

def k_means_clustering(functional_groups, K_min, K_max):
    """
    Perform K-means clustering on the input data.

    Parameters:
    - functional_groups: List of functional groups to consider. If 'all', consider all functional groups.
    - K_min: Minimum number of clusters
    - K_max: Maximum number of clusters
    """

    # Get subset Data
    X = get_subset_data(functional_groups)

    # Center and scale
    X_s = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Initialize lists to save results 
    C_K = []
    Silhouette_K = []

    for K in range(K_min, K_max+1):
        kkmeans = KMeans(n_clusters=K, init='k-means++').fit(X_s)
        labels = kkmeans.labels_
        C_K += [(np.arange(K)[:, np.newaxis] == labels[np.newaxis, :]).astype(int)]
        Silhouette_K += [silhouette_score(X_s, labels)]

    idx_best = np.argmax(Silhouette_K)

    C_best = C_K[idx_best]
    K_best = np.arange(K_min, K_max+1)[idx_best]

    return C_K, Silhouette_K, C_best, K_best
    