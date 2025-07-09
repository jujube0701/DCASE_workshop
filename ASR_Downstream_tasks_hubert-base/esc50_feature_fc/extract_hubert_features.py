import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import soundfile as sf
import numpy as np

ESC50_META_PATH = 'ESC-50-master/meta/esc50.csv'
ESC50_AUDIO_DIR = 'ESC-50-master/audio'
ESC50_FEATURES_PATH = 'esc50_features_hubert.pt'

class ESC50Dataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = sf.read(file_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)  # 轉單聲道
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != 16000:
            import librosa
            waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        label = row['target']
        return waveform, label  # shape [L]

def extract_features(model, feature_extractor, dataloader, device, save_interval=100):
    features, labels = [], []
    for i, (waveforms, label_ids) in enumerate(tqdm(dataloader)):
        # waveforms: [B, L] (torch tensor)
        np_waveforms = waveforms.numpy()  # 轉 numpy
        inputs = feature_extractor(np_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # [B, T, 768]
            pooled = hidden_states.mean(dim=1)         # [B, 768]
            features.append(pooled.cpu())
            labels.append(label_ids.cpu())
        if (i + 1) % save_interval == 0:
            part_path = f'esc50_features_hubert_part_{i+1}.pt'
            torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
            print(f"已保存 {i+1} 筆特徵到 {part_path}")
            features, labels = [], []
    # 最後一批
    if features:
        part_path = f'esc50_features_hubert_part_{i+1}.pt'
        torch.save({'features': torch.cat(features, dim=0), 'labels': torch.cat(labels, dim=0)}, part_path)
        print(f"已保存 {i+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    import glob
    part_files = sorted(glob.glob('esc50_features_hubert_part_*.pt'), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_features, all_labels = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_labels.append(data['labels'])
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    torch.save({'features': features, 'labels': labels}, ESC50_FEATURES_PATH)
    print(f'所有分批特徵已合併並保存到 {ESC50_FEATURES_PATH}，總共 {features.shape[0]} 筆')

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

    dataset = ESC50Dataset(ESC50_META_PATH, ESC50_AUDIO_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    extract_features(model, feature_extractor, dataloader, device)
    merge_feature_parts()
    print(f'特徵已保存到 {ESC50_FEATURES_PATH}') 