import os
import torch
from torch.utils.data import Dataset
from torchvision.io import VideoReader

from .labels import cls2idx


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        filenames: list[str],
        transform=None,
    ):
        """Video dataset implementation.

        Args:
            dataset_dir (str): root dir of dataset
            filenames (list[str]): filenames to include in dataset (used for splitting)
            transform (transforms.Transform | None, optional):
                Transforms to apply to the frames. Defaults to None.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.videos = []
        self.labels = []

        for label in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label)
            if os.path.isdir(label_dir):
                for video_file in os.listdir(label_dir):
                    if video_file in filenames:
                        video_path = os.path.join(label_dir, video_file)

                        # read video by frames and save as big tensor
                        reader = VideoReader(video_path, "video")
                        frames = []
                        for frame_info in reader:
                            frame: torch.Tensor = frame_info.get("data", torch.Tensor())

                            if not frame.size(0):
                                continue

                            # make channel the last channel
                            frame = frame.permute(1, 2, 0)
                            if self.transform:
                                frame = self.transform(frame)

                            frames.append(frame)

                        # convert list of frames to a tensor
                        frames = torch.stack(frames)

                        self.videos.append(frames)
                        self.labels.append(cls2idx[label])

        assert len(self.videos) == len(self.labels)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames = self.videos[idx]
        label = self.labels[idx]
        return frames, label
