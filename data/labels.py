import os
from pathlib import Path

ucf50_path = Path(__file__).parent / "raw" / "UCF50"
idx2cls = {
    idx: class_dir for idx, class_dir in enumerate(sorted(os.listdir(ucf50_path)))
}
cls2idx = {v: k for k, v in idx2cls.items()}
