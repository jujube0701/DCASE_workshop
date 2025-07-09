# hubert-base 下游任務評估專案

本專案系統性地評估了 **hubert-base** 音訊表徵模型在6個不同下游任務上的表現，包括環境音分類、醫學音頻分類、城市聲音分類、樂器分類、說話人識別和關鍵詞檢測。

## 📁 專案結構

```
esc50_feature_fc/           # ESC-50 環境音分類
│   ├── extract_hubert_features.py         # 特徵提取腳本
│   ├── train_fc_classifier.py             # 分類器訓練腳本
│   ├── esc50_features_hubert.pt           # 提取的特徵
│   ├── esc50_linear_classifier.pt         # 單層分類器模型
│   └── esc50_three_classifier.pt          # 三層分類器模型

icbhi_feature_fc/           # ICBHI 醫學音頻分類
│   ├── extract_hubert_features.py         # 特徵提取腳本
│   ├── train_fc_classifier.py             # 分類器訓練腳本
│   ├── icbhi_features_hubert.pt           # 提取的特徵
│   ├── icbhi_linear_classifier.pt         # 單層分類器模型
│   ├── icbhi_three_classifier.pt          # 三層分類器模型
│   └── result.csv                        # 標註標籤

urbansound8k_feature_fc/    # UrbanSound8K 城市聲音分類
│   ├── extract_hubert_features.py         # 特徵提取腳本
│   ├── train_fc_classifier.py             # 分類器訓練腳本
│   ├── urbansound8k_features_hubert.pt    # 提取的特徵
│   ├── urbansound8k_linear_classifier.pt  # 單層分類器模型
│   └── urbansound8k_three_classifier.pt   # 三層分類器模型

nsynth_feature_fc/          # NSynth 樂器分類
│   ├── extract_features_train.py          # 訓練集特徵提取
│   ├── extract_features_valid.py          # 驗證集特徵提取
│   ├── extract_features_test.py           # 測試集特徵提取
│   ├── train_fc_classifier.py             # 分類器訓練腳本
│   ├── nsynth_train_features.pt           # 訓練集特徵
│   ├── nsynth_valid_features.pt           # 驗證集特徵
│   └── nsynth_test_features.pt            # 測試集特徵

speaker_recognition/        # LibriSpeech 說話人識別
│   ├── extract_features_train_clean_100.py # 訓練集特徵提取
│   ├── extract_features_test_clean.py      # 測試集特徵提取
│   ├── extract_features_dev_clean.py       # 驗證集特徵提取
│   ├── train_classifier.py                 # 分類器訓練腳本
│   ├── librispeech_trainclean100_features.pt # 訓練集特徵
│   ├── librispeech_testclean_features.pt     # 測試集特徵
│   ├── librispeech_linear_classifier.pt      # 單層分類器模型
│   └── librispeech_three_classifier.pt       # 三層分類器模型

key_word_spotting/          # Speech Commands 關鍵詞檢測
│   ├── extract_features_v1.py              # v1 特徵提取
│   ├── extract_features_v2.py              # v2 特徵提取
│   ├── train_fc_classifier.py              # 分類器訓練腳本
│   ├── spc_features_hubert_v1.pt           # v1 特徵
│   └── spc_features_hubert_v2.pt           # v2 特徵

pretrained_models/          # 預訓練模型
│   └── hubert-base/                        # hubert-base 預訓練權重

README.md                   # 本檔案
```

## 🎯 下游任務說明

### 1. ESC-50 環境音分類
- **資料集**: ESC-50 環境音分類資料集
- **任務**: 50個環境音類別分類
- **樣本數**: 2,000個音頻樣本
- **音頻長度**: 5秒
- **挑戰**: 樣本數量少，類別不平衡
- **hubert-base 優勢**: 強大的語音表徵能力有助於提取環境音特徵

### 2. ICBHI 醫學音頻分類
- **資料集**: ICBHI 呼吸音資料集
- **任務**: 正常/異常呼吸音二分類
- **樣本數**: 920個音頻樣本
- **應用**: 醫學診斷輔助
- **挑戰**: 數據量小，類別不平衡
- **hubert-base 優勢**: 能捕捉細緻呼吸音特徵

### 3. UrbanSound8K 城市聲音分類
- **資料集**: UrbanSound8K 城市聲音資料集
- **任務**: 10個城市聲音類別分類
- **樣本數**: 8,732個音頻樣本
- **應用**: 城市環境監控
- **挑戰**: 音頻長度變化大，背景噪音
- **hubert-base 優勢**: 能提取魯棒的聲音特徵

### 4. NSynth 樂器分類
- **資料集**: NSynth 合成樂器資料集
- **任務**: 10個樂器家族分類
- **樣本數**: 305,979個音頻樣本
- **音頻長度**: 4秒
- **挑戰**: 音頻質量高，需要精細特徵
- **hubert-base 優勢**: 能捕捉樂器音色細節

### 5. LibriSpeech 說話人識別
- **資料集**: LibriSpeech 語音資料集
- **任務**: 248個說話人識別
- **樣本數**: 28,539個音頻樣本
- **應用**: 說話人驗證、語音助手
- **挑戰**: 說話人數量多，需要區分性特徵
- **hubert-base 優勢**: 能提取語者辨識特徵

### 6. Speech Commands 關鍵詞檢測
- **資料集**: Speech Commands 關鍵詞資料集
- **任務**: 35個關鍵詞檢測 (v1) / 36個關鍵詞檢測 (v2)
- **樣本數**: 105,829個音頻樣本
- **音頻長度**: 1秒
- **應用**: 語音助手、智能家居
- **挑戰**: 音頻短，需要快速識別
- **hubert-base 優勢**: 能提取關鍵詞語音特徵

## 🚀 快速開始

### 環境設置
```bash
# 創建虛擬環境
python -m venv venv

# 啟動虛擬環境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安裝依賴
pip install torch torchaudio
pip install librosa
pip install scikit-learn
pip install numpy pandas
pip install tqdm
pip install transformers
```

### 下載預訓練模型
- 預設會自動下載 facebook/hubert-base-ls960 權重
- 或將模型放於 `pretrained_models/` 目錄

### 執行下游任務評估

#### 1. ESC-50 環境音分類
```bash
cd esc50_feature_fc
python extract_hubert_features.py
python train_fc_classifier.py --epochs 100
```

#### 2. ICBHI 醫學音頻分類
```bash
cd icbhi_feature_fc
python extract_hubert_features.py
python train_fc_classifier.py --epochs 100
```

#### 3. UrbanSound8K 城市聲音分類
```bash
cd urbansound8k_feature_fc
python extract_hubert_features.py
python train_fc_classifier.py --epochs 100
```

#### 4. NSynth 樂器分類
```bash
cd nsynth_feature_fc
python extract_features_train.py
python extract_features_valid.py
python extract_features_test.py
python train_fc_classifier.py
```

#### 5. LibriSpeech 說話人識別
```bash
cd speaker_recognition
python extract_features_train_clean_100.py
python extract_features_test_clean.py
python extract_features_dev_clean.py
python train_classifier.py
```

#### 6. Speech Commands 關鍵詞檢測
```bash
cd key_word_spotting
python extract_features_v1.py
python extract_features_v2.py
python train_fc_classifier.py
```

## 📊 模型架構

### hubert-base 特徵提取
- 使用 transformers 的 `HubertModel` 進行特徵提取，取最後一層 hidden state 做平均池化作為音訊特徵。

### 分類器架構

#### 單層分類器
```python
class LinearClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
```

#### 三層分類器
```python
class ThreeLayerClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=None, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)
```

## 🔧 配置說明

### 統一實驗配置
```python
EXPERIMENT_CONFIG = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'epochs': 100,
    'feature_dim': 768,
    'train_test_split': 0.2,
    'random_seed': 42,
    'device': 'cuda'
}
```

### 特徵標準化
所有任務都使用 `StandardScaler` 進行特徵標準化：
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_np = scaler.fit_transform(features.numpy())
features = torch.tensor(features_np, dtype=torch.float32)
```

## 📈 預期結果

| 任務 | 單層分類器 | 三層分類器 | 最佳方法 |
|------|------------|------------|----------|
| ESC-50 | ~85% | ~88% | 三層 |
| ICBHI | ~75% | ~78% | 三層 |
| UrbanSound8K | ~80% | ~83% | 三層 |
| NSynth | ~90% | ~92% | 三層 |
| Speaker Recognition | ~85% | ~88% | 三層 |
| Speech Commands | ~95% | ~96% | 三層 |

*注意：實際性能可能因硬體配置和數據預處理而有所不同*

## 🔍 實驗分析
- **特徵可視化**: 使用 t-SNE 或 UMAP 可視化特徵分佈
- **任務相似性**: 分析不同任務間的特徵相似性
- **消融實驗**: 評估不同組件的貢獻
- **準確率**: 主要評估指標
- **F1分數**: 用於不平衡數據集
- **混淆矩陣**: 詳細分類分析
- **標準差**: 多次實驗的穩定性

## 🎯 研究貢獻
1. **系統性評估**: 全面評估 hubert-base 在6個不同下游任務上的表現
2. **任務覆蓋面廣**: 涵蓋環境音、醫學音頻、城市聲音、樂器、說話人和關鍵詞檢測
3. **實用指南**: 提供基於 hubert-base 的音頻表示學習最佳實踐
4. **開源框架**: 提供完整的實驗代碼和預訓練模型



## 🔗 相關連結
- [hubert-base 官方模型](https://huggingface.co/facebook/hubert-base-ls960)
- [ESC-50 資料集](https://github.com/karolpiczak/ESC-50)
- [ICBHI 資料集](https://bhichallenge.med.auth.gr/)
- [UrbanSound8K 資料集](https://urbansounddataset.weebly.com/)
- [NSynth 資料集](https://magenta.tensorflow.org/datasets/nsynth)
- [LibriSpeech 資料集](https://www.openslr.org/12/)
- [Speech Commands 資料集](https://www.tensorflow.org/datasets/catalog/speech_commands)
