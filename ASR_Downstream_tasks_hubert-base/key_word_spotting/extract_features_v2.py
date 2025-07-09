import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel

# ====== 參數區 ======
data_root = 'Speech commands/v2'
output_path = 'key_word_spotting/spc_features_hubert_v2.pt'
batch_size = 1
# ====== END ======

# 掃描所有類別
all_classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('_')]
all_classes.sort()
label2idx = {c: i for i, c in enumerate(all_classes)}

# 收集所有 wav 路徑與 label
samples = []
for c in all_classes:
    class_dir = os.path.join(data_root, c)
    for fname in os.listdir(class_dir):
        if fname.endswith('.wav'):
            samples.append((os.path.join(class_dir, fname), label2idx[c], c + '/' + fname))

print(f'共 {len(samples)} 條樣本，{len(all_classes)} 類')

class SPCDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        wav_path, label, key = self.samples[idx]
        waveform, sr = sf.read(wav_path)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if sr != 16000:
            import librosa
            waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
        return waveform, label, key

def extract_features(model, feature_extractor, dataloader, device):
    features, labels, keys = [], [], []
    for waveforms, label_ids, key_list in tqdm(dataloader):
        # waveforms: [B, L]
        np_waveforms = waveforms.numpy()
        inputs = feature_extractor(np_waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # [B, T, 768]
            pooled = hidden_states.mean(dim=1)         # [B, 768]
            features.append(pooled.cpu())
            labels.append(label_ids)
            keys.extend(key_list)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    features = torch.cat(features, dim=0)
    labels = torch.tensor(labels).view(-1)
    torch.save({'features': features, 'labels': labels, 'keys': keys, 'label2idx': label2idx}, output_path)
    print(f'特徵已保存到 {output_path}')

if __name__ == '__main__':
    device = torch.device('cpu')
    print(f"Device: {device}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    dataset = SPCDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    extract_features(model, feature_extractor, dataloader, device) 