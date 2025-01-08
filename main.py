from utils import save_video,video_read
from tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimer
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedDistanceEstimator
import numpy as np
import pickle
import cv2
import time
import pandas as pd
from datetime import datetime

def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
    ## Read Video
    video_frames = video_read('input_video/08fd33_4.mp4')  

    ## Create Tracker Object
    tracker = Tracker('models/best.pt')
     
    ## Object Detection and Tracking
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stub.pkl')
    
    ## Add position in Tracks
    tracker.add_position_to_tracks(tracks)
    
    ## Camera Movement Estimator    
    cameramovementestimator = CameraMovementEstimer(video_frames[0])
    camera_movement_per_frames = cameramovementestimator.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_estimator_stub.pkl')    
    cameramovementestimator.add_adjust_position_tracks(tracks,camera_movement_per_frames)
    
    ## Add View Transformed Points
    viewtransformer = ViewTransformer()
    viewtransformer.add_tranformed_position_tracks(tracks)    
    
    ## Add Distance_Speed in Tracks
    speeddistanceestimator = SpeedDistanceEstimator()
    speeddistanceestimator.add_speed_distamce_to_tracks(tracks)
    
    ## Interpolate the ball positions
    tracks['ball'] = tracker.ball_position_interpolation(tracks['ball'])    
    
    ## Assign Team Color
    teamassign = TeamAssigner()
    teamassign.assign_team_color(video_frames[0],tracks['players'][0])

    ## Get Players team_id and color
    for frame_num,player_tracks in enumerate(tracks['players']):
        for track_id,player in player_tracks.items():
            team_id = teamassign.get_players_team(video_frames[frame_num],player['bbox'],track_id)
            tracks['players'][frame_num][track_id]['team'] = team_id
            tracks['players'][frame_num][track_id]['color'] = teamassign.team_colors[team_id]    
            
    ## Assign Player_Ball_ID
    player_assign = PlayerBallAssigner()
    team_ball_controls = []
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assign_player = player_assign.player_ball_assigner(ball_bbox,player_track)
        if assign_player != -1:
            tracks['players'][frame_num][assign_player]['has_ball'] = True
            team_ball_controls.append(tracks['players'][frame_num][assign_player]['team'])
        else:
            team_ball_controls.append(team_ball_controls[-1])
    
    team_ball_controls = np.array(team_ball_controls)
    
    ## Draw Annotations 
    output_frames = tracker.draw_annotation(video_frames,tracks,team_ball_controls)
    
    ## Draw Camera Movement
    output_frames = cameramovementestimator.draw_camear_movement(output_frames,camera_movement_per_frames)
    
    ## Draw Speed Distance Info
    output_frames = speeddistanceestimator.draw_spped_distance(output_frames,tracks)
    
    # Save Video
    save_video(output_frames,'output_videos/output.avi')
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    
if __name__ == '__main__':
    main()
    
    
# Save frames to a file
# with open("frames.pkl", 'wb') as f:
#     pickle.dump(video_frames, f)
# print(f"Video frames saved to frames.pkl")

# with open("frames.pkl", 'rb') as f:
#     video_frames = pickle.load(f)


# ## Save Cropped Image of a Player
# for track_id, player in tracks['players'][0].items():
#     bbox = player['bbox']
#     frame = video_frames[0]
#     cropped_frame = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
#     cv2.imwrite('output_videos/cropped_image_01.jpg',cropped_frame)
#     break