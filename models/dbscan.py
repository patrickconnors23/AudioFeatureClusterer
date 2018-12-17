from sklearn.cluster import DBSCAN

class dbscan():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
    """
    def __init__(self, k):
        self.model = DBSCAN()
        self.name = "DBSCAN Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.fit_predict(data)