import os
import shutil
import pandas as pd
import subprocess

# 将 FFmpeg 路径添加到系统 PATH
ffmpeg_path = r'.\ffmpeg.exe'
os.environ["PATH"] += os.pathsep + ffmpeg_path


def extract_info_from_image_name(image_name):
    try:
        # 分离文件名和扩展名
        file_base_name = os.path.splitext(image_name)[0]
        # 根据规则提取信息
        parts = file_base_name.split('_')

        # 寻找_scene的索引
        scene_idx = parts.index('scene')

        if scene_idx >= 1 and len(parts) > scene_idx + 3:
            video_name = '_'.join(parts[:scene_idx])
            scene_number = parts[scene_idx + 1]
            time_code_str = parts[scene_idx + 3].split('time_')[-1]
            try:
                time_code = float(time_code_str)
            except ValueError:
                time_code = 0
            return video_name, scene_number, time_code
        else:
            print(f"Unexpected filename format: {image_name}")
            return None, None, None
    except Exception as e:
        print(f"Error extracting info from image name '{image_name}': {e}")
        return None, None, None


def get_scene_times(csv_path, scene_number):
    try:
        df = pd.read_csv(csv_path)
        scene_row = df[df['Scene Number'] == int(scene_number)]
        if not scene_row.empty:
            start_time = scene_row.iloc[0]['Start Time (s)']
            end_time = scene_row.iloc[0]['End Time (s)']
            return start_time, end_time
    except Exception as e:
        print(f"Error reading scene times from CSV '{csv_path}': {e}")
    return None, None


def extract_frames_with_ffmpeg(video_path, video_name, start_time, end_time, output_dir, scene_number):
    duration = end_time - start_time
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    if duration <= 0:
        print(f"Invalid duration: {duration}")
        return

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入视频的帧率
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fps = result.stdout.decode('utf-8').strip()
    if '/' in fps:
        num, den = fps.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps)

    new_fps = fps  # 使用原始帧率，确保提取所有帧

    print(f"Extracted FPS: {fps}")
    print(f"New FPS: {new_fps}")

    # 确保提取时间段足够长，至少1帧
    min_duration = 1 / fps
    if duration < min_duration:
        print(f"Duration {duration} is less than minimum duration {min_duration}. Adjusting duration to {min_duration}.")
        duration = min_duration

    # 临时输出目录
    temp_output_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_output_dir):
        os.makedirs(temp_output_dir)

    # FFmpeg 命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-ss', f"{start_time:.6f}",  # 使用高精度起始时间
        '-i', video_path,  # 输入视频文件
        '-t', f"{duration:.6f}",  # 使用高精度持续时间
        '-vf', f"fps={new_fps}",  # 设置新的帧率，确保提取所有帧
        os.path.join(temp_output_dir, f"frame_%04d.jpg")
    ]

    print(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFmpeg error with command: {' '.join(ffmpeg_cmd)}")
        print("FFmpeg stderr:", result.stderr.decode('utf-8', errors='ignore'))

    # 单独提取end_time的帧
    end_time_frame_dir = os.path.join(output_dir, "end_time_frame")
    if not os.path.exists(end_time_frame_dir):
        os.makedirs(end_time_frame_dir)

    ffmpeg_end_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-ss', f"{end_time:.6f}",  # 使用高精度end_time
        '-i', video_path,  # 输入视频文件
        '-frames:v', '1',  # 仅提取一帧
        os.path.join(end_time_frame_dir, "end_frame.jpg")
    ]

    print(f"Running FFmpeg end time command: {' '.join(ffmpeg_end_cmd)}")
    result_end = subprocess.run(ffmpeg_end_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result_end.returncode != 0:
        print(f"FFmpeg error with command: {' '.join(ffmpeg_end_cmd)}")
        print("FFmpeg stderr:", result_end.stderr.decode('utf-8', errors='ignore'))

    # 检查提取的帧数
    frame_files = [f for f in os.listdir(temp_output_dir) if f.lower().endswith('.jpg')]
    if not frame_files:
        print("No frames extracted.")
        return

    # 重新命名文件，包含时间戳
    for frame_file in frame_files:
        frame_number = int(frame_file.split('_')[1].split('.')[0])
        frame_time = start_time + (frame_number - 1) / fps
        new_frame_name = f"{video_name}_scene_{scene_number}_time_{frame_time:.2f}_frame_{frame_number:04d}.jpg"
        shutil.move(
            os.path.join(temp_output_dir, frame_file),
            os.path.join(output_dir, new_frame_name)
        )
        print(f"Extracted frame: {new_frame_name}")

    # 移动end_time的帧
    end_frame_path = os.path.join(end_time_frame_dir, "end_frame.jpg")
    if os.path.exists(end_frame_path):
        frame_time = end_time
        new_frame_name = f"{video_name}_scene_{scene_number}_time_{frame_time:.2f}_frame_end.jpg"
        shutil.move(
            end_frame_path,
            os.path.join(output_dir, new_frame_name)
        )
        print(f"Extracted end time frame: {new_frame_name}")

    # 删除临时输出目录
    shutil.rmtree(temp_output_dir)
    shutil.rmtree(end_time_frame_dir)

    print(f"Extracted {len(frame_files)} frames from {start_time} to {end_time}, including the end time frame")

def find_video_file(video_path, video_name):
    possible_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    for ext in possible_extensions:
        video_file_path = os.path.join(video_path, f"{video_name}{ext}")
        if os.path.exists(video_file_path):
            return video_file_path
    return None

def main():
    base_results_path = r'../file/resullts_superglue'
    shot_detection_path = r'../file/csv'
    video_path = r'../file/animation_frames_extract_videos'
    temp_frames_path = r'../file/resullts_frames_extract'
    unprocessed_files = []

    # 清空 temp_frames 目录
    if os.path.exists(temp_frames_path):
        shutil.rmtree(temp_frames_path)
    os.makedirs(temp_frames_path)

    total_files = 0
    processed_files = 0

    for folder_name in os.listdir(base_results_path):
        folder_path = os.path.join(base_results_path, folder_name)
        if os.path.isdir(folder_path):
            for detected_image_name in os.listdir(folder_path):
                if detected_image_name.lower().endswith('.jpg'):
                    total_files += 1
                    video_name, scene_number, time_code = extract_info_from_image_name(detected_image_name)

                    if video_name is None or scene_number is None or time_code is None:
                        unprocessed_files.append(detected_image_name)
                        continue

                    # 输出提取的信息
                    print(f"Detected image name: {detected_image_name}")
                    print(f"Extracted video name: {video_name}")
                    print(f"Extracted scene number: {scene_number}")
                    print(f"Extracted time code: {time_code}")

                    csv_path = os.path.join(shot_detection_path, f"{video_name}_scenes.csv")

                    if not os.path.exists(csv_path):
                        print(f"CSV file not found: {csv_path}")
                        unprocessed_files.append(detected_image_name)
                        continue

                    start_time, end_time = get_scene_times(csv_path, scene_number)
                    if start_time is None or end_time is None:
                        print(f"Scene {scene_number} not found in CSV {csv_path}.")
                        unprocessed_files.append(detected_image_name)
                        continue

                    video_file_path = find_video_file(video_path, video_name)
                    if video_file_path is None:
                        print(f"Video file not found for video name: {video_name}")
                        unprocessed_files.append(detected_image_name)
                        continue

                    frames_dir = os.path.join(temp_frames_path, folder_name)
                    os.makedirs(frames_dir, exist_ok=True)
                    extract_frames_with_ffmpeg(video_file_path, video_name, start_time, end_time, frames_dir, scene_number)
                    processed_files += 1

    print(f"Total files: {total_files}")
    print(f"Processed files: {processed_files}")
    print(f"Unprocessed files: {len(unprocessed_files)}")
    if unprocessed_files:
        print("List of unprocessed files:")
        for file in unprocessed_files:
            print(file)

if __name__ == "__main__":
    main()
