from models.kmeans import kMeans
from models.agglomerativeCluster import agglomerativeCluster
from models.affinityPropagation import affinityPropagation
from models.spectral import spectral
from models.birch import birch
from models.dbscan import dbscan
from models.meanShift import meanShift

class Clusterers():
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
        def fit(model, data):
            model.fit(data)
            return model
        self.data = df
        self.models = [fit(model, df) for model in self.models]
    
    def predict(self):
        def clusterPredict(c):
            return {
                "label": c.name,
                "predictions": c.predict(self.data)
            }

        return [clusterPredict(model) for model in self.models]
