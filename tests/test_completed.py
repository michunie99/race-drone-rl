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
    segment=track.segments[0]
    pos, ort=segment.startPosition()
    urdf_id=p.loadURDF(
            str(drone_asset),
            pos,
            ort,
            physicsClientId=clientID,
    )
    p.applyExternalForce(urdf_id, -1,  [1, 0, -1], [0, 0, 0], p.LINK_FRAME, clientID)

    while True:
        pos, ort = p.getBasePositionAndOrientation(urdf_id, physicsClientId=clientID)
        if segment.segmentFinished(pos):
            print(segment.segmentFinished(pos))
            input()
        p.stepSimulation(clientID)

    input("Press to end test")

if __name__ == "__main__":
    main()
