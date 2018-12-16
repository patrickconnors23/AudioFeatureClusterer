import random
from spoticli import SPOTICLI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def gatherPlaylistData(numPlaylists):
    # print("Reading data")
    playlists = pd.read_pickle("playlists.pkl")
    i = random.randint(0, len(playlists) - numPlaylists)
    playlists = playlists.iloc[i:i+numPlaylists]

    def addURI(playlist):
        return ["spotify:track:"+track for track in playlist]

    return [addURI(p["tracks"]) for _, p in playlists.iterrows()]

def gatherData(playlists):
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
    audioDF.to_pickle("data/audioDF.pkl")
    return audioDF

def analyzeMetricsDF(dfList):
    df, rest = dfList[0], dfList[1:]
    df = df.set_index("Clustering Algorithm")
    df.columns = ["Results 1"]
    for trialDF in rest:
        df["Results " + str(len(df.columns) + 1)] = list(trialDF["Performance"])
    df["Mean Scores"] = df.mean(axis=1)
    return df