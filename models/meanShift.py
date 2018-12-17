from sklearn.cluster import MeanShift

class meanShift():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
    """
    def __init__(self, k):
        self.model = MeanShift()
        self.name = "MeanShift Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.predict(data)