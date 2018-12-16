from spoticli import SPOTICLI
from pprint import pprint as pp
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from util.helpers import gatherData, gatherPlaylistData
from models.clusterers import Clusterers

class AudioCluster():
    def __init__(self, processData):
        self.playlists = self.readPlaylistData()
        self.audioDF = self.readAudioData(shouldProcess=processData)
        self.clusterLabels = []
        self.models = Clusterers(k=len(self.playlists))
    
    def readPlaylistData(self):
        return gatherPlaylistData(10)
    
    def readAudioData(self, shouldProcess):
        if shouldProcess:
            return gatherData(self.playlists) 
        else:
            return pd.read_pickle("data/audioDF.pkl")

    def predictClusters(self, df, k):
        self.models.fitModels(df)
        return self.models.predict()
    
    def processAndCluster(self):
        intCols = ["acousticness",	
            "danceability",
            "duration_ms",
            "energy",
            "instrumentalness",
            "liveness",
            "loudness",
            "mode",
            "speechiness",
            "tempo",
            "time_signature",
            "valence"]

        def scaleData(df, cols):
            df = df[cols].astype(float)
            scaler = StandardScaler()
            scaler.fit(df)
            df = scaler.transform(df)
            return pd.DataFrame(df, columns=cols)
        
        URIs = self.audioDF["uri"]

        # scale data
        self.audioDF = scaleData(self.audioDF, intCols)
        
        # add cluster tag
        clusterTags = self.predictClusters(self.audioDF, len(self.playlists))
        for ct in clusterTags:
            self.clusterLabels.append(ct["label"])
            self.audioDF[ct["label"]] = ct["predictions"]

        self.audioDF["uri"] = URIs
    
    def analyzeClusterPerformance(self, cluster):
        def calcOverlap(playlist, group):
            urls = set(group["uri"].values)
            playlist = set(playlist)
            return len(urls & playlist)

        groupedDF = self.audioDF.groupby(cluster)
        playlists = list(self.playlists)

        clusters = [group for _, group in groupedDF]
        numSongs = sum([len(p) for p in playlists])

        numOverlap, maxOverlap = 0, 1
        while len(playlists) > 0 and len(clusters) > 0 and maxOverlap > 0:
            maxOverlap, pIDX, gIDX = 0, None, None
            for i, playlist in enumerate(playlists):
                for j, c in enumerate(clusters):
                    overlap = calcOverlap(playlist, c)
                    if overlap > maxOverlap:
                        maxOverlap, pIDX, gIDX = overlap, i, j
            
            if maxOverlap > 0:
                playlists.pop(pIDX)
                clusters.pop(gIDX)
                numOverlap += maxOverlap

        performance = float(numOverlap) / float(numSongs)
        performancePercentage = round(performance) * 100

        print()
        print(f"The {cluster} method resulted in {performancePercentage}% of songs being correctly grouped")
        return {
            "Clustering Algorithm": cluster,
            "Performance": performance
        }
        

    def analyzeResults(self):
        results = [self.analyzeClusterPerformance(c) for c in self.clusterLabels]
        rDF = pd.DataFrame(results)
        rDF.to_csv("data/resultsDF.csv")

if __name__ == "__main__":
    AC = AudioCluster(processData=False)
    print("Clustering data...")
    AC.processAndCluster()
    print("Analyzing results...")
    AC.analyzeResults()