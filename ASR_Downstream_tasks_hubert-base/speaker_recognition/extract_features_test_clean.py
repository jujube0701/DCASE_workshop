import os
import glob
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np

# ====== 參數區 ======
audio_root = '../LibriSpeech/test-clean'
output_path = 'librispeech_test_clean_features.pt'
batch_size = 1
# ====== END ======

def get_wav_list(audio_root):
    wav_list = []
    for speaker_id in os.listdir(audio_root):
        speaker_path = os.path.join(audio_root, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            for wav_file in glob.glob(os.path.join(chapter_path, '*.flac')):
                wav_list.append((wav_file, speaker_id))
            for wav_file in glob.glob(os.path.join(chapter_path, '*.wav')):
                wav_list.append((wav_file, speaker_id))
    return wav_list

def load_audio(wav_path):
    try:
        # 先檢查檔案大小，如果太大就跳過
        file_size = os.path.getsize(wav_path)
        if file_size > 100 * 1024 * 1024:  # 100MB
            print(f"跳過大檔案: {wav_path} ({file_size / 1024 / 1024:.1f}MB)")
            return None
        
        waveform, sr = sf.read(wav_path)
        
        # 檢查音檔長度，如果太長就截斷
        max_length = 30 * sr  # 30秒
        if len(waveform) > max_length:
            waveform = waveform[:max_length]
            print(f"截斷長音檔: {wav_path} ({len(waveform)/sr:.1f}s)")
        
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        
        # 轉換為float32以節省記憶體
        waveform = waveform.astype(np.float32)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        if sr != 16000:
            import librosa
            waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
        
        return waveform
        
    except Exception as e:
        print(f"處理音檔時出錯 {wav_path}: {str(e)}")
        return None

def extract_features(model, feature_extractor, wav_list, device, save_interval=100):
    features, labels, keys = [], [], []
    skipped_count = 0
    
    for i, (wav_path, speaker_id) in enumerate(tqdm(wav_list)):
        waveform = load_audio(wav_path)
        
        if waveform is None:
            skipped_count += 1
            continue
            
        try:
            np_waveform = waveform.numpy()
            inputs = feature_extractor(np_waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state  # [1, T, 768]
                pooled = hidden_states.mean(dim=1)         # [1, 768]
                features.append(pooled.cpu())
                labels.append(speaker_id)
                keys.append(wav_path)
                
        except Exception as e:
            print(f"特徵提取時出錯 {wav_path}: {str(e)}")
            skipped_count += 1
            continue
            
        if (i + 1) % save_interval == 0:
            if features:  # 確保有特徵才保存
                part_path = os.path.join(os.path.dirname(output_path), f'librispeech_test_clean_features_part_{i+1}.pt')
                torch.save({'features': torch.cat(features, dim=0), 'labels': labels, 'keys': keys}, part_path)
                print(f"已保存 {len(features)} 筆特徵到 {part_path}")
                features, labels, keys = [], [], []
    
    if features:
        part_path = os.path.join(os.path.dirname(output_path), f'librispeech_test_clean_features_part_{i+1}.pt')
        torch.save({'features': torch.cat(features, dim=0), 'labels': labels, 'keys': keys}, part_path)
        print(f"已保存 {len(features)} 筆特徵到 {part_path}")
    
    print(f"總共跳過 {skipped_count} 個音檔")

def merge_feature_parts():
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'librispeech_test_clean_features_part_*.pt')), 
                       key=lambda x: int(x.split('_part_')[-1].split('.pt')[0]))
    all_features, all_labels, all_keys = [], [], []
    for pf in part_files:
        data = torch.load(pf)
        all_features.append(data['features'])
        all_labels.extend(data['labels'])
        all_keys.extend(data['keys'])
    features = torch.cat(all_features, dim=0)
    torch.save({'features': features, 'labels': all_labels, 'keys': all_keys}, output_path)
    print(f'所有分批特徵已合併並保存到 {output_path}，總共 {features.shape[0]} 筆')

if __name__ == '__main__':
    # 強制使用CPU
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    wav_list = get_wav_list(audio_root)
    print(f"共找到 {len(wav_list)} 個音檔")
    extract_features(model, feature_extractor, wav_list, device)
    merge_feature_parts()
    print(f'特徵已保存到 {output_path}') 