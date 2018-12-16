from sklearn.cluster import MeanShift

class meanShift():
    def __init__(self, k):
        self.model = MeanShift()
        self.name = "MeanShift Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)