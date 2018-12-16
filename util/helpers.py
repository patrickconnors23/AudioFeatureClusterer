from spoticli import SPOTICLI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def gatherPlaylistData(numPlaylists):
    playlists = pd.read_pickle("playlists.pkl")
    playlists = playlists.iloc[:numPlaylists]

    def addURI(playlist):
        return ["spotify:track:"+track for track in playlist]

    return [addURI(p["tracks"]) for _, p in playlists.iterrows()]

def gatherData(playlist):
    sp = SPOTICLI()
    trackSet = set()

    def getTrackSet(allPlaylists):
        tracksLst = []
        for lst in allPlaylists: tracksLst += lst
        trackLst = list(set(tracksLst))
        return ["spotify:track:"+x for x in tracksLst]
    
    allTracks = getTrackSet(playlists)

    # Get audio features
    i = 0
    j = min(len(allTracks), 100)
    audioMaster = []
    while i < len(allTracks):
        trackSlice = allTracks[i:j]
        audio = sp.getAudioF(trackSlice)
        audioMaster += audio
        i = j
        j = min(j + 100, len(allTracks))

    audioDF = pd.DataFrame.from_dict(audioMaster)

    # intCols = ["acousticness",	
    #     "danceability",
    #     "duration_ms",
    #     "energy",
    #     "instrumentalness",
    #     "liveness",
    #     "loudness",
    #     "mode",
    #     "speechiness",
    #     "tempo",
    #     "time_signature",
    #     "valence"]

    # def scaleData(df, cols):
    #     df = df[cols]
    #     scaler = StandardScaler()
    #     scaler.fit(df)
    #     df = scaler.transform(df)
    #     return pd.DataFrame(df, columns=cols)
    
    # def predictClusters(df, k):
    #     clusterer = KMeans(n_clusters=k)
    #     clusterer.fit(df)
    #     return clusterer.predict(df)

    # URIs = audioDF["uri"]

    # # scale data
    # audioDF = scaleData(audioDF, intCols)
    
    # # add cluster tag
    # clusterTags = predictClusters(audioDF, len(playlists))

    # # Add back columns
    # audioDF["cluster"] = clusterTags
    # audioDF["uri"] = URIs
    audioDF.to_pickle("data/audioDF.pkl")
    return audioDF