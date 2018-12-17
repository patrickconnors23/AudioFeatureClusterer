from models.kmeans import kMeans
from models.agglomerativeCluster import agglomerativeCluster
from models.affinityPropagation import affinityPropagation
from models.spectral import spectral
from models.birch import birch
from models.dbscan import dbscan
from models.meanShift import meanShift

class Clusterers():
    """
    Wrapper class for various clustering algos
    Allows all models to be trained and utilized uniformly
    """
    def __init__(self, k):
        self.models = [
            kMeans(k=k),
            agglomerativeCluster(k=k),
            affinityPropagation(k=k),
            spectral(k=k),
            birch(k=k),
            dbscan(k=k),
            meanShift(k=k)
        ]
    
    def fitModels(self, df):
        """
        Fit all models to track data audio features
        """
        def fit(model, data):
            model.fit(data)
            return model
        self.data = df
        self.models = [fit(model, df) for model in self.models]

    def clusterPredict(self, c):
        """
        Create prediction object for specified model
        """
        return { "label": c.name, "predictions": c.predict(self.data) }
    
    def predict(self):
        """
        Generate predictions for all models
        """
        return [self.clusterPredict(model) for model in self.models]
