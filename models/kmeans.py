from sklearn.cluster import KMeans

class kMeans():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    """
    def __init__(self, k):
        self.model = KMeans(n_clusters=k)
        self.name = "KMeans Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)