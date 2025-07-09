import os
import json
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np
import glob

# ====== 參數區 ======
json_path = 'nsynth/nsynth-train/examples.json'
audio_dir = 'nsynth/nsynth-train/audio'
output_path = 'nsynth_feature_fc/nsynth_train_features.pt'
batch_size = 1
# ====== END ======

class NSynthTrainDataset(Dataset):
    def __init__(self, json_path, audio_dir):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.audio_dir = audio_dir
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        wav_path = os.path.join(self.audio_dir, key + '.wav')
        waveform, sr = sf.read(wav_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != 16000:
            import librosa
            waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
        return waveform, key

def extract_features(model, feature_extractor, dataloader, device, save_interval=100, resume_idx=0):
    features, keys = [], []
    for i, (waveforms, key_list) in enumerate(tqdm(dataloader)):
        global_idx = i + resume_idx
        np_waveforms = waveforms.numpy()
        inputs = feature_extractor(np_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # [B, T, 768]
            pooled = hidden_states.mean(dim=1)         # [B, 768]
            features.append(pooled.cpu())
            keys.extend(key_list)
        if (global_idx + 1) % save_interval == 0:
            part_path = os.path.join(os.path.dirname(output_path), f'nsynth_train_features_part_{global_idx+1}.pt')
            torch.save({'features': torch.cat(features, dim=0), 'keys': keys}, part_path)
            print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")
            features, keys = [], []
    if features:
        part_path = os.path.join(os.path.dirname(output_path), f'nsynth_train_features_part_{global_idx+1}.pt')
        torch.save({'features': torch.cat(features, dim=0), 'keys': keys}, part_path)
        print(f"已保存 {global_idx+1} 筆特徵到 {part_path}")

def merge_feature_parts():
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'nsynth_train_features_part_*.pt')), 
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    all_features, all_keys = [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_keys.extend(data['keys'])
    features = torch.cat(all_features, dim=0)
    torch.save({'features': features, 'keys': all_keys}, output_path)
    print(f'所有分批特徵已合併並保存到 {output_path}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    # 自動檢測並使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    dataset = NSynthTrainDataset(json_path, audio_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'nsynth_train_features_part_*.pt')),
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    resume_idx = 0
    if part_files:
        last_idx = int(part_files[-1].split('_part_')[-1].split('.pt')[0])
        resume_idx = last_idx  # 從最後一個文件的索引開始
        print(f'偵測到已完成 {resume_idx} 筆，將從第 {resume_idx + 1} 筆繼續')
        dataset = torch.utils.data.Subset(dataset, range(resume_idx, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        print('未偵測到分批檔案，將從頭開始')
    
    extract_features(model, feature_extractor, dataloader, device)
    merge_feature_parts()
    print(f'特徵已保存到 {output_path}') 