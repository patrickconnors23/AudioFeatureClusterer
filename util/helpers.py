import random
from spoticli import SPOTICLI
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def gatherPlaylistData(numPlaylists):
    """
    Read process playlist data from spotify playlist file
    """
    playlists = pd.read_pickle("playlists.pkl")

    # Select random playlists
    i = random.randint(0, len(playlists) - numPlaylists)
    playlists = playlists.iloc[i:i+numPlaylists]

    def addURI(playlist):
        return ["spotify:track:"+track for track in playlist]

    # Process data for later evaluation
    return [addURI(p["tracks"]) for _, p in playlists.iterrows()]

def gatherData(playlists):
    """
    Gather audio data for all songs in current iteration
    """
    # Init spotify client
    sp = SPOTICLI()

    # Get all unique tracks from playlists
    def getTrackSet(allPlaylists):
        tracksLst = []
        for lst in allPlaylists: tracksLst += lst
        trackLst = list(set(tracksLst))
        return ["spotify:track:"+x for x in tracksLst]
    
    allTracks = getTrackSet(playlists)

    # Get audio features -> batcha requests by 100
    i, j = 0, min(len(allTracks), 100)
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
    """
    Convert list of trial DFs to one master DF
    """
    df, rest = dfList[0], dfList[1:]
    df = df.set_index("Clustering Algorithm")
    df.columns = ["Results 1"]
    for trialDF in rest:
        df["Results " + str(len(df.columns) + 1)] = list(trialDF["Performance"])
    df["Mean Scores"] = df.mean(axis=1)
    return df