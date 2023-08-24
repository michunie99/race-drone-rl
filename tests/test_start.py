from race_rl.race_track import Track

import pybullet as p


drone_asset="/home/michunie/projects/magisterka/agh-drone-racing/gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf"
assets_path="assets"
track_path="assets/tracks/circle_track.csv"

def main():
    # Create bullet client
    clientID=p.connect(p.GUI)
    track=Track(track_path,assets_path, clientID)

    # Iterate all segments in the track
    for segment in track.segments:
        pos, ort=segment.startPosition()
        urdf_id=p.loadURDF(
                str(drone_asset),
                pos,
                ort,
                physicsClientId=clientID,
        )

    input("Press to end test")

if __name__ == "__main__":
    main()
