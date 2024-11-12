import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from PIL import Image
import random
from torchvision import transforms
from typing import List, Tuple, Dict, Any


def get_transform() -> transforms.Compose:
    """
    Get the transformations to apply to the frames.

    Returns:
        transforms.Compose: Composed transformations.
    """
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

class UCF50Dataset(Dataset):
    def __init__(self, root_dir: str, frames_per_clip: int, transform: Any = None, selection_strategy: str = 'sift'):
        """
        Initialize the UCF50Dataset.

        Args:
            root_dir (str): Root directory containing video files.
            frames_per_clip (int): Number of frames to select per video clip.
            transform (Any, optional): Transform to apply to the frames. Defaults to None.
            selection_strategy (str, optional): Strategy for frame selection ('sift', 'random', 'split'). Defaults to 'sift'.
        """
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.selection_strategy = selection_strategy
        self.video_paths, self.labels, self.class_to_idx = self._load_dataset()

    def _load_dataset(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        Load the dataset from the root directory.

        Returns:
            Tuple[List[str], List[int], Dict[str, int]]: List of video paths, list of labels, and class to index mapping.
        """
        video_paths = []
        labels = []
        class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                class_to_idx[class_name] = idx
                for video_name in os.listdir(class_path):
                    video_path = os.path.join(class_path, video_name)
                    video_paths.append(video_path)
                    labels.append(idx)
        return video_paths, labels, class_to_idx

    def _select_frames_sift(self, video_path: str) -> List[int]:
        """
        Select frames using the SIFT algorithm.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[int]: List of selected frame indices.
        """
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        feature_detector = cv2.SIFT_create()
        feature_matcher = cv2.BFMatcher()

        _, prev_des = None, None
        frame_diffs = []

        for count in range(frame_count):
            ret, frame = vidcap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height))

            kp, des = feature_detector.detectAndCompute(gray, None)

            if prev_des is not None:
                matches = feature_matcher.knnMatch(prev_des, des, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                diff_sum = len(good_matches)
                frame_diffs.append((count, diff_sum, cv2.Laplacian(gray, cv2.CV_64F).var()))

            _, prev_des = kp, des

        vidcap.release()

        # Sort frames by the number of good matches (difference) and sharpness
        frame_diffs.sort(key=lambda x: (x[1], -x[2]))

        # Select top frames_per_clip frames with the least number of good matches
        selected_frames_ids = [frame_diffs[i][0] for i in range(min(self.frames_per_clip, len(frame_diffs)))]

        return selected_frames_ids

    def _select_frames_random(self, video_path: str) -> List[int]:
        """
        Select frames randomly.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[int]: List of selected frame indices.
        """
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        selected_frames_ids = random.sample(range(frame_count), min(self.frames_per_clip, frame_count))
        vidcap.release()
        return selected_frames_ids

    def _select_frames_split(self, video_path: str) -> List[int]:
        """
        Select frames by evenly splitting the video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[int]: List of selected frame indices.
        """
        vidcap = cv2.VideoCapture(video_path)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // self.frames_per_clip)
        selected_frames_ids = list(range(0, frame_count, step))[:self.frames_per_clip]
        vidcap.release()
        return selected_frames_ids

    def _select_frames(self, video_path: str) -> List[int]:
        """
        Select frames based on the specified strategy.

        Args:
            video_path (str): Path to the video file.

        Returns:
            List[int]: List of selected frame indices.
        """
        if self.selection_strategy == 'sift':
            return self._select_frames_sift(video_path)
        elif self.selection_strategy == 'random':
            return self._select_frames_random(video_path)
        elif self.selection_strategy == 'split':
            return self._select_frames_split(video_path)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

    def __len__(self) -> int:
        """
        Get the number of videos in the dataset.

        Returns:
            int: Number of videos.
        """
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a video and its label by index.

        Args:
            idx (int): Index of the video.

        Returns:
            Dict[str, Any]: Dictionary containing the selected frames and the label.
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        selected_frames_ids = self._select_frames(video_path)
        video, _, _ = read_video(video_path, pts_unit='sec')
        
        # Ensure selected frame indices are within bounds
        selected_frames_ids = [i for i in selected_frames_ids if i < video.shape[0]]
        
        # Pad or truncate the selected frames to match frames_per_clip
        if len(selected_frames_ids) < self.frames_per_clip:
            selected_frames_ids += [selected_frames_ids[-1]] * (self.frames_per_clip - len(selected_frames_ids))
        else:
            selected_frames_ids = selected_frames_ids[:self.frames_per_clip]
        
        selected_frames = video[selected_frames_ids]
        if self.transform:
            selected_frames = torch.stack([self.transform(Image.fromarray(frame.numpy())) for frame in selected_frames])
        return {'images': selected_frames, 'labels': label}
    
