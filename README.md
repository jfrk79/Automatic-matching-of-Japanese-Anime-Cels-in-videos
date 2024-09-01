# Automatic Matching of Japanese Anime Cels in Videos

## Project Overview

This project proposes an integrated method based on multiple deep learning networks for the automatic identification and matching of hand-drawn anime cels in video clips. Addressing the challenges faced in the preservation and digitization of anime cels, this method aids in the preservation and recording of relevant information about these cels.

First, we use the Anime-Segmentation algorithm to segment characters in keyframes extracted from animations, reducing background interference. Next, we utilize the CLIP-ResNet model to extract features from the segmented images and perform image retrieval to narrow down the search scope. Then, the SuperGlue algorithm is employed for multiple feature matching steps to accurately locate keyframes. Our method has been tested on an experimental image set and successfully completed the task of locating anime cel information in animations.

## Features

1. **Keyframe Extraction**:
    - Extract keyframes from animation videos and save scene information to a CSV file.
    - Script: `shot_detection.py`

2. **Character Image Segmentation**:
    - Segment characters in extracted keyframes to reduce background interference.
    - Script: `anime_seg.py`
    - Code Source: [Anime-Segmentation](https://github.com/SkyTNT/anime-segmentation)

3. **Feature Extraction**:
    - Extract features from the segmented images.
    - Script: `clip_resnet.py`

4. **Feature Matching**:
    - Match the extracted features with background images.
    - Script: `match_clipresnet_withbackground.py`

5. **Image Matching**:
    - Use Canny edge detection and the SuperGlue model for image matching to find the best match.
    - Script: `superglue_round1.py`
    - Code Source: [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)

6. **Frame Extraction**:
    - Extract frames from the original video segment based on the information in the CSV file and SuperGlue matching results using FFmpeg.
    - Script: `saveframes_fromvideo.py`

7. **Final Matching**:
    - Perform edge detection and SuperGlue matching again on the extracted original video frames, finding and saving the top 30 matches.
    - Script: `superglue_round2.py`

## Installation

### Requirements

- Python 3.10.10
- Required libraries (see `requirements.txt`)
- FFmpeg

### Installation Steps

1. Clone the repository to your local machine:

    ```sh
    git clone https://github.com/jfrk79/Automatic-matching-of-Japanese-Anime-Cels-in-videos.git
    cd Automatic-matching-of-Japanese-Anime-Cels-in-videos
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Install FFmpeg:

    - Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html), place the executable in the `code` folder, or add it to your system PATH.

5. Download and configure the weight files:

    - Download Anime-Segmentation weight files and place them in the `code/anime-segmentation-mainskytnt/saved_models` directory. Download link: [https://huggingface.co/skytnt/anime-seg/tree/main](https://huggingface.co/skytnt/anime-seg/tree/main)

## Usage

### Step 1: Run `shot_detection.py`

Ensure you are in the virtual environment and the current directory is the project root. Run the following command to process all video files in the `animation_frames_extract_videos` directory and generate a CSV file with scene information in the project root directory.

    ```sh
    python shot_detection.py
    ```

This script will process all video files in the `animation_frames_extract_videos` directory, extract keyframes, and generate a CSV file with scene information.

### Step 2: Run `anime_seg.py`

Before running the `anime_seg.py` script, manually adjust the `data_dir` and `out_dir` parameters in the script:

- `data_dir`: Set to the directory storing the keyframe images.
- `out_dir`: Set to the directory for saving the segmented images.

Edit the `anime_seg.py` file to ensure the paths for `data_dir` and `out_dir` are correct, then run the following command:

    ```sh
    python anime_seg.py
    ```

This script will process the images in the `data_dir` directory and save the segmentation results to the `out_dir` directory.

### Step 3: Run `clip_resnet.py`

Before running the `clip_resnet.py` script, manually adjust the `dataset_path` parameter in the script:

- `dataset_path`: Set to the directory containing the segmented images output by `anime_seg.py` (i.e., `out_dir`).

Edit the `clip_resnet.py` file to ensure the `dataset_path` path is correct, then run the following command:

    ```sh
    python clip_resnet.py
    ```

This script will process the images in the `dataset_path` directory and extract features.

### Step 4: Run `match_clipresnet_withbackground.py`

Before running the `match_clipresnet_withbackground.py` script, manually adjust the following parameters:

- `features_file`: Set to the path of the feature vector file generated by `clip_resnet.py`.
- `new_image_folder`: Set to the directory before image segmentation.
- `simulated_photos_path`: Set to the directory of the background images to be searched.

Edit the `match_clipresnet_withbackground.py` file to ensure all paths are correct, then run the following command:

    ```sh
    python match_clipresnet_withbackground.py
    ```

This script will process the data in `features_file`, `new_image_folder`, and `simulated_photos_path` directories and output the matching results to the `file/withbackground` directory.

### Step 5: Run `superglue_round1.py`

Before running the `superglue_round1.py` script, manually adjust the following parameters:

- `features_file`: Set to the feature vector file corresponding to the anime work.
- `new_image_folder`: Set to the directory before image segmentation.
- `simulated_photos_path`: Set to the directory of the background images to be searched.

Edit the `superglue_round1.py` file to ensure all paths are correct, then run the following command:

    ```sh
    python superglue_round1.py
    ```

This script will perform Canny edge detection and SuperGlue image matching to find the best matching images. The processed Canny images will be saved to `processed_top30` and `processed_cels`, and the final SuperGlue matching output will be saved to `final_results_base_path`.

### Step 6: Run `saveframes_fromvideo.py`

Before running the `saveframes_fromvideo.py` script, manually adjust the relevant parameters in the script:

- `features_file`: Set to the feature vector file containing the most similar image information saved by SuperGlue.
- `new_image_folder`: Set to the directory before image segmentation.
- `simulated_photos_path`: Set to the directory of the background images to be searched.

Edit the `saveframes_fromvideo.py` file to ensure all paths are correct, then run the following command:

    ```sh
    python saveframes_fromvideo.py
    ```

This script will extract frames from the original video segments using FFmpeg based on the information stored in the CSV file and the most similar image information saved by SuperGlue.

### Step 7: Run `superglue_round2.py`

Before running the `superglue_round2.py` script, manually adjust the relevant parameters in the script:

- `features_file`: Set to the feature vector file containing the most similar image information saved by SuperGlue.
- `new_image_folder`: Set to the directory before image segmentation.
- `simulated_photos_path`: Set to the directory of the background images to be searched.

Edit the `superglue_round2.py` file to ensure all paths are correct, then run the following command:

    ```sh
    python superglue_round2.py
    ```

This script will perform edge detection and SuperGlue matching again on the extracted original video frames, finding and saving the top 30 matches to the `results_final` directory.

## Acknowledgements

Special thanks to the following projects for their contributions:
- Thanks to SkyTNT for the Anime-Segmentation project. Parts of our code reference this project. Project link: https://github.com/SkyTNT/anime-segmentation
- Thanks to Magic Leap for the SuperGlue project. Parts of our code reference this project. Project link: https://github.com/magicleap/SuperGluePretrainedNetwork
