import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 5 کلاس عمل
ACTION_CLASSES = {
    0: "Static",      # بدون حرکت
    1: "Slow_Move",   # حرکت کند
    2: "Fast_Move",   # حرکت سریع
    3: "Rotation",    # چرخش
    4: "Vibration"    # لرزش
}

NUM_FRAMES = 16  # تعداد فریم در هر ویدیو
FRAME_SIZE = 64  # ابعاد هر فریم
NUM_CLASSES = 5


class SyntheticVideoActionDataset(Dataset):
    """دیتاست مصنوعی: ویدیوهای کوتاه با عمل‌های متفاوت"""
    def __init__(self, num_samples_per_class=20, num_frames=NUM_FRAMES, frame_size=FRAME_SIZE):
        self.num_samples_per_class = num_samples_per_class
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.num_classes = NUM_CLASSES
        self.num_samples = num_samples_per_class * NUM_CLASSES
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        action_id = idx // self.num_samples_per_class
        sample_id = idx % self.num_samples_per_class
        
        # تولید ویدیو مصنوعی
        video = self._generate_video(action_id, sample_id)
        
        video_tensor = torch.from_numpy(video).float() / 255.0
        label = torch.tensor(action_id, dtype=torch.long)
        
        return video_tensor, label
    
    def _generate_video(self, action_id, sample_id):
        """تولید ویدیو بر اساس کلاس عمل"""
        np.random.seed(idx := action_id * 100 + sample_id)
        
        video = np.ones((self.num_frames, self.frame_size, self.frame_size, 3), dtype=np.uint8) * 200
        
        for frame_idx in range(self.num_frames):
            frame = video[frame_idx]
            
            # مربع متحرک در مرکز
            center_x = self.frame_size // 2
            center_y = self.frame_size // 2
            size = 10
            
            # موقعیت بر اساس کلاس عمل
            if action_id == 0:  # Static
                x = center_x
                y = center_y
            elif action_id == 1:  # Slow_Move
                x = center_x + (frame_idx * 1)
                y = center_y
            elif action_id == 2:  # Fast_Move
                x = center_x + (frame_idx * 3)
                y = center_y
            elif action_id == 3:  # Rotation
                angle = (frame_idx / self.num_frames) * 360
                offset = int(10 * np.cos(np.radians(angle)))
                x = center_x + offset
                y = center_y + int(10 * np.sin(np.radians(angle)))
            elif action_id == 4:  # Vibration
                x = center_x + np.random.randint(-5, 5)
                y = center_y + np.random.randint(-5, 5)
            
            x = max(0, min(self.frame_size - size, int(x)))
            y = max(0, min(self.frame_size - size, int(y)))
            
            # رسم مربع
            cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), -1)
            
            # رسم متن
            color = (255, 0, 0) if action_id in [0, 2, 4] else (0, 0, 255)
            cv2.putText(frame, ACTION_CLASSES[action_id], (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return video


def get_dataloaders(batch_size=16):
    full_dataset = SyntheticVideoActionDataset(num_samples_per_class=20)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_set, val_set = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
