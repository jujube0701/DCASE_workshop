import os
import glob
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np
import gc  # 添加垃圾回收模組

# ====== 參數區 ======
audio_root = r'G:/surf/hubert-base/LibriSpeech/train-clean-100'
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'librispeech_train_clean_100_features.pt')
batch_size = 1
# ====== END ======

def get_latest_processed_index():
    """自動檢測已經處理到哪個索引"""
    features_dir = os.path.dirname(output_path) or '.'
    part_files = glob.glob(os.path.join(features_dir, 'librispeech_train_clean_100_features_part_*.pt'))
    if not part_files:
        return 0
    
    # 提取所有文件名中的數字並找到最大值
    indices = []
    for pf in part_files:
        try:
            index = int(pf.split('_part_')[-1].split('.pt')[0])
            indices.append(index)
        except:
            continue
    
    return max(indices) if indices else 0

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
    waveform, sr = sf.read(wav_path)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if sr != 16000:
        import librosa
        waveform = torch.tensor(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=16000), dtype=torch.float32)
    
    # 清理記憶體
    gc.collect()
    
    return waveform

def extract_features(model, feature_extractor, wav_list, device, save_interval=100):
    features, labels, keys = [], [], []
    
    # 獲取已經處理到的索引
    start_index = get_latest_processed_index()
    
    # 跳過已經處理過的文件
    wav_list = wav_list[start_index:]
    print(f"從第 {start_index + 1} 個文件開始處理，共剩餘 {len(wav_list)} 個文件")
    
    for i, (wav_path, speaker_id) in enumerate(tqdm(wav_list)):
        waveform = load_audio(wav_path)
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
        
        # 計算實際的全局索引
        global_index = start_index + i + 1
        
        if (i + 1) % save_interval == 0:
            part_path = os.path.join(os.path.dirname(output_path), f'librispeech_train_clean_100_features_part_{global_index}.pt')
            torch.save({'features': torch.cat(features, dim=0), 'labels': labels, 'keys': keys}, part_path)
            print(f"已保存第 {global_index} 筆特徵到 {part_path}")
            features, labels, keys = [], [], []
            
            # 定期清理記憶體
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 保存最後一批
    if features:
        global_index = start_index + len(wav_list)
        part_path = os.path.join(os.path.dirname(output_path), f'librispeech_train_clean_100_features_part_{global_index}.pt')
        torch.save({'features': torch.cat(features, dim=0), 'labels': labels, 'keys': keys}, part_path)
        print(f"已保存第 {global_index} 筆特徵到 {part_path}")

def merge_feature_parts():
    features_dir = os.path.dirname(output_path) or '.'
    part_files = sorted(glob.glob(os.path.join(features_dir, 'librispeech_train_clean_100_features_part_*.pt')), 
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
    start_index = get_latest_processed_index()
    print(f"從第 {start_index + 1} 個開始繼續處理")
    extract_features(model, feature_extractor, wav_list, device)
    merge_feature_parts()
    print(f'特徵已保存到 {output_path}') 