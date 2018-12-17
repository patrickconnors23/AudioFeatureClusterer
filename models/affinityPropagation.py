from sklearn.cluster import AffinityPropagation

class affinityPropagation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
    """
    def __init__(self, k):
        self.model = AffinityPropagation()
        self.name = "Affinity Propagation Clustering"
    
    def fit(self, data):
        self.model.fit(data)
    
    def predict(self, data):
        return self.model.fit_predict(data)