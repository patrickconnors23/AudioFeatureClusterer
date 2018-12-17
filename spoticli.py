import numpy as np
import pandas as pd
import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth2

# Hide these in prodiction.. doesn't matter for our purposes
CLIENT_ID = "3a83fd327ec54f3aa3c97b31bdcaa983"
CLIENT_SECRET = "d504bf926fda40c5b311b4a46266ea30"

class SPOTICLI():
    """
    Wrapper for spotify python SDK
    Used to get audio features for tracks
    """
    def __init__(self):
        self.SPOTIFY = self.createClient()
    
    def createClient(self):
        """
        Initialize client with variaus auth pararms
        """
        auth = oauth2.SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )
        token = auth.get_access_token()
        return spotipy.Spotify(auth=token)

    def getAudioF(self, TIDs):
        """
        Return audio features for list of tracks
        [track] -> feature obj
        """
        # print(TIDs)
        return self.SPOTIFY.audio_features(TIDs)
