from sklearn.cluster import SpectralClustering

class spectral():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
    """
    def __init__(self, k):
        self.model = SpectralClustering(n_clusters=k)
        self.name = "Spectral Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.fit_predict(data)