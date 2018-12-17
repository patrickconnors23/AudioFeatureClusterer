from sklearn.cluster import Birch

class birch():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
    """
    def __init__(self, k):
        self.model = Birch(n_clusters=k)
        self.name = "Birch Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)