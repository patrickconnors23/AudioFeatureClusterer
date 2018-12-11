from spoticli import SPOTICLI
from pprint import pprint as pp
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    sp = SPOTICLI()
    playlists = pd.read_pickle("playlists.pkl")
    tracks = pd.read_pickle("tracks.pkl")
    playlists = playlists.iloc[:10]

    trackSet = set()

    def addURI(playlist):
        return ["spotify:track:"+track for track in playlist]

    playlists = [addURI(p["tracks"]) for _, p in playlists.iterrows()]


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

    URIs = audioDF["uri"]
    audioDF = audioDF[intCols]
    scaler = StandardScaler()
    scaler.fit(audioDF)
    audioDF = scaler.transform(audioDF)
    audioDF = pd.DataFrame(audioDF, columns=intCols)
    
    clusterer = KMeans(n_clusters=len(playlists))
    clusterer.fit(audioDF)
    groups = clusterer.predict(audioDF)
    audioDF["cluster"] = groups
    audioDF["uri"] = URIs
    
    groupedDF = audioDF.groupby("cluster")

    groupSizes, overlaps = [], []
    for _, group in groupedDF:
        def overlap(playlist, group):
            urls = set(group["uri"].values)
            playlist = set(playlist)
            return len(urls & playlist)
        overlap = list(map(lambda x: overlap(x, group), playlists))
        groupSizes.append(len(group))
        overlaps.append(max(overlap))
    for over, total in zip(overlaps, groupSizes):
        print(float(over) / float(total))