import pybullet as p
import argparse

from race_rl.race_track import Track

def test(track_path, assets_path):
    clientID=p.connect(p.GUI)
    track=Track(track_path,assets_path, clientID)
    track.segments[0].startPosition()
    print("Set point")

if __name__ == "__main__":
    track_path="assets/tracks/thesis-tracks/long_trakc.csv"
    assets_path="assets"
    test(track_path, assets_path)
    input()
