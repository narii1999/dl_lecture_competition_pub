import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
#import mne

## デフォルト
# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         if hasattr(self, "y"):
#             return self.X[i], self.y[i], self.subject_idxs[i]
#         else:
#             return self.X[i], self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]


## EEG conformer用
class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", normalize: bool = True):
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.data_dir = data_dir
        self.normalize = normalize
        
        self.X = torch.load(f"{data_dir}/{split}_X.pt")
        self.subject_idxs = torch.load(f"{data_dir}/{split}_subject_idxs.pt")
        
        if split in ["train", "val"]:
            self.y = torch.load(f"{data_dir}/{split}_y.pt")
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        # データ型をfloat32に変換
        self.X = self.X.float()

        if normalize:
            self._normalize_data()

    def _normalize_data(self):
        # チャンネルごとに正規化
        scaler = StandardScaler()
        shape = self.X.shape
        self.X = self.X.view(shape[0], shape[1], -1)
        self.X = scaler.fit_transform(self.X.numpy().reshape(-1, self.X.shape[-1])).reshape(shape)
        self.X = torch.from_numpy(self.X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        # EEGConformerの入力形状に合わせて調整 (1, channels, time)
        x = x.unsqueeze(0)
        
        if hasattr(self, "y"):
            return x, self.y[i], self.subject_idxs[i]
        else:
            return x, self.subject_idxs[i]
    
    @property
    def num_channels(self):
        return self.X.shape[1]
    
    @property
    def seq_len(self):
        return self.X.shape[2]


# ## カスタムモデル1 MEGClassifierを使うパターン
# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data", normalize: bool = True) -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
#         self.data_dir = data_dir
#         self.normalize = normalize
        
#         # .pt ファイルを使用する場合
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#         # データ型をfloat32に変換
#         self.X = self.X.float()

#         if normalize:
#             self._init_normalizer()

#     def _get_data_shape(self):
#         # データの形状を取得（この関数は実際のデータに合わせて実装する必要があります）
#         # 例: return (10000, 271, 200)  # (サンプル数, チャンネル数, 時系列長)
#         pass

#     def _init_normalizer(self):
#         # 最初の1000サンプルを使用して正規化パラメータを計算
#         sample_data = self.X[:1000].reshape(-1, self.X.shape[1])
#         self.scaler = StandardScaler()
#         self.scaler.fit(sample_data)

#     def __len__(self) -> int:
#         return self.X.shape[0]

#     def __getitem__(self, i):
#         # ファイルからデータの一部だけを読み込む
#         with open(os.path.join(self.data_dir, f"{self.split}_X.pt"), 'rb') as f:
#             f.seek(i * self.X[0].numel() * 4)  # float32は4バイト
#             x = torch.from_numpy(np.frombuffer(f.read(self.X[0].numel() * 4), dtype=np.float32)).reshape(self.X[0].shape)

#         if self.normalize:
#             x = torch.from_numpy(self.scaler.transform(x.T).T).float()

#         if self.split == "train":
#             shift = np.random.randint(0, self.seq_len // 10)
#             x = torch.roll(x, shifts=shift, dims=1)
        
#         if hasattr(self, "y"):
#             return x, self.y[i], self.subject_idxs[i]
#         else:
#             return x, self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]


# ##Wav2Vec2を使うパターン
# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
#         self.data_dir = data_dir
        
#         # メモリマップを使用してデータをロード
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"), map_location='cpu')
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"), map_location='cpu')
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"), map_location='cpu')
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#         # データの形状を保存
#         self._num_channels = 271  # チャンネル数を271に設定
#         self._seq_len = 200  # 時間ステップ数を200に設定

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         # スケーリングを getitem 内で行う
#         x = self.X[i][:self._num_channels, :self._seq_len]  # 271チャンネルと200時間ステップにトリミング
        
#         if hasattr(self, "y"):
#             return x, self.y[i], self.subject_idxs[i]
#         else:
#             return x, self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self._num_channels
    
#     @property
#     def seq_len(self) -> int:
#         return self._seq_len


# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#         # データをスケーリング
#         self.X = self.scale_data(self.X)

#     def scale_data(self, data):
#         # データを[-1, 1]の範囲にスケーリング
#         return 2 * (data - data.min()) / (data.max() - data.min()) - 1

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         if hasattr(self, "y"):
#             return self.X[i].squeeze(0), self.y[i], self.subject_idxs[i]  # channel dimension を削除
#         else:
#             return self.X[i].squeeze(0), self.subject_idxs[i]  # channel dimension を削除
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]


# ##以下前処理するver

# class ThingsMEGDataset(torch.utils.data.Dataset):
#     def __init__(self, split: str, data_dir: str = "data") -> None:
#         super().__init__()
        
#         assert split in ["train", "val", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.num_classes = 1854
        
#         preprocessed_file = os.path.join(data_dir, f"{split}_X_preprocessed.pt")
        
#         if os.path.exists(preprocessed_file):
#             self.X = torch.load(preprocessed_file)
#         else:
#             self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
#             self.X = self.preprocess_meg(self.X)
#             # オプション: 前処理済みデータを保存
#             torch.save(self.X, preprocessed_file)
            
        
#         self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
#         if split in ["train", "val"]:
#             self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
#             assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

#     def preprocess_meg(self, data):
#       # データの形状を確認
#       n_trials, n_channels, n_timepoints = data.shape
      
#       sfreq = 1000  # サンプリング周波数（適切な値に置き換えてください）
#       ch_names = [f'ch{i}' for i in range(n_channels)]
#       ch_types = ['mag'] * n_channels  # すべてのチャンネルをマグネトメーターとして扱う
#       info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
      
#       processed_data = []
#       for trial in data:
#           # トランスポーズを削除
#           raw = mne.io.RawArray(trial, info)
          
#           # バンドパスフィルタを適用 (例: 1-40 Hz)
#           raw.filter(1, 40)
          
#           # Z-score normalization
#           raw_data = raw.get_data()
#           normalized_data = (raw_data - np.mean(raw_data, axis=1, keepdims=True)) / np.std(raw_data, axis=1, keepdims=True)
          
#           # エポックごとのベースライン補正
#           # 最初の100msをベースラインとして使用
#           baseline_end = int(0.1 * sfreq)  # 100ms
#           baseline = normalized_data[:, :baseline_end].mean(axis=1, keepdims=True)
#           baseline_corrected_data = normalized_data - baseline
          
#           processed_data.append(baseline_corrected_data)
      
#       return torch.tensor(np.array(processed_data), dtype=torch.float32)

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, i):
#         if hasattr(self, "y"):
#             return self.X[i], self.y[i], self.subject_idxs[i]
#         else:
#             return self.X[i], self.subject_idxs[i]
        
#     @property
#     def num_channels(self) -> int:
#         return self.X.shape[1]
    
#     @property
#     def seq_len(self) -> int:
#         return self.X.shape[2]