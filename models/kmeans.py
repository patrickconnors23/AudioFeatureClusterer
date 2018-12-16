from sklearn.cluster import KMeans

class kMeans():
    def __init__(self, k):
        self.model = KMeans(n_clusters=k)
        self.name = "KMeans Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)