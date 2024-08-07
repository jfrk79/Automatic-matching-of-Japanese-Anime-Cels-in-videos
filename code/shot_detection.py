import os
import csv
import math
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def save_frame(video_path, time_sec, file_path, ffmpeg_exe='./ffmpeg'):
    time_str = f"{time_sec:.2f}"
    command = [
        ffmpeg_exe,
        '-ss', time_str,
        '-i', video_path,
        '-frames:v', '1',
        '-q:v', '2',
        file_path
    ]
    subprocess.run(command, check=True)
    print(f'Saved frame at {time_sec} seconds as {file_path}')


def format_time_for_filename(time_sec, round_type='floor'):
    if round_type == 'floor':
        return math.floor(time_sec * 10) / 10  # 向下取整保留一位小数
    elif round_type == 'round':
        return round(time_sec * 10) / 10  # 四舍五入保留一位小数
    return time_sec


def round_down_to_two_decimals(time_sec):
    return math.floor(time_sec * 100) / 100


def round_up_to_two_decimals(time_sec):
    return math.ceil(time_sec * 100) / 100


def find_scenes(video_path, output_csv_dir='file/csv', output_frames_dir='file/frames', threshold=30.0,
                long_scene_threshold=10, ffmpeg_exe='./ffmpeg'):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 创建保存CSV和帧的目录
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)

    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)

    video_frame_dir = os.path.join(output_frames_dir, video_name)
    if not os.path.exists(video_frame_dir):
        os.makedirs(video_frame_dir)

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print(f'Scene {i + 1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}')

    video_manager.release()

    csv_file_path = os.path.join(output_csv_dir, f'{video_name}_scenes.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Scene Number', 'Start Time (s)', 'End Time (s)'])
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            rounded_start_time = round_down_to_two_decimals(start_time)
            rounded_end_time = round_up_to_two_decimals(end_time)
            csv_writer.writerow([i + 1, f'{rounded_start_time:.2f}', f'{rounded_end_time:.2f}'])

    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        mid_time = (start_time + end_time) / 2

        formatted_start_time = format_time_for_filename(start_time, 'floor')
        formatted_end_time = format_time_for_filename(end_time, 'round')
        formatted_mid_time = format_time_for_filename(mid_time, 'round')

        save_frame(video_path, start_time, os.path.join(video_frame_dir,
                                                        f'{video_name}_scene_{i + 1}_start_{formatted_start_time:.1f}_time_{start_time:.2f}.jpg'),
                   ffmpeg_exe)
        save_frame(video_path, mid_time, os.path.join(video_frame_dir,
                                                      f'{video_name}_scene_{i + 1}_middle_{formatted_mid_time:.1f}_time_{mid_time:.2f}.jpg'),
                   ffmpeg_exe)
        save_frame(video_path, end_time, os.path.join(video_frame_dir,
                                                      f'{video_name}_scene_{i + 1}_end_{formatted_end_time:.1f}_time_{end_time:.2f}.jpg'),
                   ffmpeg_exe)

        if end_time - start_time > long_scene_threshold:
            current_time = start_time
            while current_time < end_time:
                formatted_current_time = format_time_for_filename(current_time, 'round')
                save_frame(video_path, current_time, os.path.join(video_frame_dir,
                                                                  f'{video_name}_scene_{i + 1}_at_{formatted_current_time:.1f}_time_{current_time:.2f}.jpg'),
                           ffmpeg_exe)
                current_time += 1


def process_videos_in_directory(directory, output_csv_dir, output_frames_dir, ffmpeg_exe='./ffmpeg'):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            video_file_path = os.path.join(directory, filename)
            print(f"Processing video: {video_file_path}")
            find_scenes(video_file_path, output_csv_dir, output_frames_dir, ffmpeg_exe=ffmpeg_exe)


# 示例用法
if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.dirname(__file__))
    video_directory = os.path.join(base_directory, 'file', 'animation_frames_extract_videos')
    output_csv_dir = os.path.join(base_directory, 'file', 'csv')
    output_frames_dir = os.path.join(base_directory, 'file', 'frames')

    process_videos_in_directory(video_directory, output_csv_dir, output_frames_dir)
