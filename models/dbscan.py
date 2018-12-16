from sklearn.cluster import DBSCAN

class dbscan():
    def __init__(self, k):
        self.model = DBSCAN()
        self.name = "DBSCAN Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.fit_predict(data)