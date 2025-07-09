import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import soundfile as sf
import numpy as np
import glob
import pandas as pd

ICBHI_AUDIO_DIR = 'ICBHI_final_database'
ICBHI_FEATURES_PATH = 'ICBHI_feature_fc/icbhi_features_hubert.pt'
RESULT_CSV_PATH = 'ICBHI_feature_fc/result.csv'

class ICBHIDataset(Dataset):
    def __init__(self, audio_dir, result_csv_path, transform=None):
        self.wav_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        self.transform = transform
        
        # 讀取 result.csv 文件
        self.labels_df = pd.read_csv(result_csv_path)
        # 創建文件名到標籤的映射
        self.label_map = {}
        for _, row in self.labels_df.iterrows():
            filename = row['filename']
            # 將 .txt 文件名轉換為對應的 .wav 文件名
            wav_filename = filename.replace('.txt', '.wav')
            self.label_map[wav_filename] = row['disease_label']

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        file_path = self.wav_files[idx]
        filename = os.path.basename(file_path)
        
        # 從 result.csv 中獲取標籤
        label = self.label_map.get(filename, 0)  # 預設為正常 (0)
        
        waveform, sr = sf.read(file_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != 16000:
            import librosa
            waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

def extract_features(model, feature_extractor, dataloader, device, save_interval=100):
    features, labels = [], []
    for i, (waveforms, label_ids) in enumerate(tqdm(dataloader)):
        np_waveforms = waveforms.numpy()
        inputs = feature_extractor(np_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
            features.append(pooled.cpu())
            labels.append(label_ids.cpu())
        if (i + 1) % save_interval == 0:
            part_path = f'ICBHI_feature_fc/icbhi_features_hubert_part_{i+1}.pt'
            torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
            print(f"已保存 {i+1} 筆特徵到 {part_path}")
            features, labels = [], []
    if features:
        part_path = f'ICBHI_feature_fc/icbhi_features_hubert_part_{i+1}.pt'
        torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
        print(f"已保存 {i+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    import glob
    part_files = sorted(glob.glob('ICBHI_feature_fc/icbhi_features_hubert_part_*.pt'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_features, all_labels = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'features': features, 'labels': labels}, ICBHI_FEATURES_PATH)
    print(f'所有分批特徵已合併並保存到 {ICBHI_FEATURES_PATH}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataset = ICBHIDataset(ICBHI_AUDIO_DIR, RESULT_CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    extract_features(model, feature_extractor, dataloader, device)
    merge_feature_parts()
    print(f'特徵已保存到 {ICBHI_FEATURES_PATH}') 