import os
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import torch
import numpy as np

mediapipe_dir = '/Users/seanyse/Desktop/GT/FW25/VIP/Popsign/pytorch_dataloader/mediapipe'

"""
mediapipe file structure is

mediapipe
    - word_1
        - word_1_file_1
        - word_1_file_2
    - word_2
        - word_2_file_1

"""
class PopsignDataset(Dataset):
    def __init__(self, mediapipe_dir):
        
        self.mediapipe_dir = mediapipe_dir
        self.mediapipe_labels = self.get_label(mediapipe_dir)

    def __len__(self):
        return len(self.mediapipe_labels)

    def __getitem__(self, idx):
        mediapipe_path, label = self.mediapipe_labels.iloc[idx]

        data = self.load_extract(mediapipe_path)

        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)
        return data, label
    
    def get_label(self, mediapipe_dir):
        labels = []
        for label, word in enumerate(sorted(os.listdir(mediapipe_dir))):
            word_dir = os.path.join(mediapipe_dir, word)
            for file in os.listdir(word_dir):
                if file.endswith('.h5'):
                    labels.append((os.path.join(word_dir, file), label))

        return pd.DataFrame(labels, columns=['mediapipe_dir', 'label'])

    def load_extract(self, mediapipe_dir):
        # extracting left + right hand, and general pose
        # update the hand depth since right now it is anchored to the wrist, use the pose depth which is the distance from nose from wrist
        # anchor the hand xy to the xy of the nose

        with h5py.File(mediapipe_dir, "r") as f:
            # use pose_world_landmarks for the nose and wrist
            pose_landmarks = f["Pose"]["pose_landmarks"][:]
            # scaled 0-1 to find pixel value hands

            # MAKE SURE LEFT AND RIGHT ARE CORRECT
            hand_left = f["Hand2"]["hand_landmarks"][:, 0, :, :]
            hand_right = f["Hand2"]["hand_landmarks"][:, 1, :, :]
        
    
        # relative depth pose wrist - nose
        depth_left = pose_landmarks[:, 0, 15, 2][:, np.newaxis, np.newaxis] - pose_landmarks[:, 0, 0, 2][:, np.newaxis, np.newaxis] # left wrist z  - nose z
        depth_right = pose_landmarks[:, 0, 16, 2][:, np.newaxis, np.newaxis] - pose_landmarks[:, 0, 0, 2][:, np.newaxis, np.newaxis] # right wrist z - nose z
        
        # update depth
        hand_left[:, :, 2] -= depth_left
        hand_right[:, :, 2] -= depth_right

        # subtract nose from wrist then subtract that to the wrist z
        nose = pose_landmarks[:, 0, 0, 0:2][:, np.newaxis, :] # nose

        hand_left_anchor = hand_left.copy()
        hand_right_anchor = hand_right.copy()
        hand_left_anchor[:, :, 0:2] = hand_left[:, :, 0:2] - nose
        hand_right_anchor[:, :, 0:2] = hand_right[:, :, 0:2] - nose

        return np.concatenate([hand_left_anchor, hand_right_anchor], axis=1)
        

        




