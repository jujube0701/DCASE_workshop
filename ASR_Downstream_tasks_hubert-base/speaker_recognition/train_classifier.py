import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import argparse
import os
from tqdm import tqdm

# ====== 參數區 ======
script_dir = os.path.dirname(os.path.abspath(__file__))
train_features_path = os.path.join(script_dir, 'librispeech_train_clean_100_features.pt')
dev_features_path = os.path.join(script_dir, 'librispeech_dev_clean_features.pt')
test_features_path = os.path.join(script_dir, 'librispeech_test_clean_features.pt')
output_dir = script_dir
# ====== END ======

class SpeakerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=None):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=None, dropout=0.3):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def load_data(train_path, dev_path, test_path):
    print("載入訓練資料...")
    train_data = torch.load(train_path)
    train_features = train_data['features']
    train_labels = train_data['labels']
    
    print("載入驗證資料...")
    dev_data = torch.load(dev_path)
    dev_features = dev_data['features']
    dev_labels = dev_data['labels']
    
    print("載入測試資料...")
    test_data = torch.load(test_path)
    test_features = test_data['features']
    test_labels = test_data['labels']
    
    # 合併所有標籤進行編碼
    all_labels = train_labels + dev_labels + test_labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # 轉換標籤為數字
    train_labels_encoded = label_encoder.transform(train_labels)
    dev_labels_encoded = label_encoder.transform(dev_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    print(f"總共 {len(label_encoder.classes_)} 個唯一說話者")
    print(f"訓練資料: {len(train_features)} 筆, {len(np.unique(train_labels_encoded))} 個說話者")
    print(f"驗證資料: {len(dev_features)} 筆, {len(np.unique(dev_labels_encoded))} 個說話者")
    print(f"測試資料: {len(test_features)} 筆, {len(np.unique(test_labels_encoded))} 個說話者")
    
    # 檢查說話者重疊情況
    train_speakers = set(train_labels)
    dev_speakers = set(dev_labels)
    test_speakers = set(test_labels)
    
    print(f"訓練集獨有說話者: {len(train_speakers - dev_speakers - test_speakers)}")
    print(f"驗證集獨有說話者: {len(dev_speakers - train_speakers - test_speakers)}")
    print(f"測試集獨有說話者: {len(test_speakers - train_speakers - dev_speakers)}")
    print(f"訓練-驗證重疊說話者: {len(train_speakers & dev_speakers)}")
    print(f"訓練-測試重疊說話者: {len(train_speakers & test_speakers)}")
    print(f"驗證-測試重疊說話者: {len(dev_speakers & test_speakers)}")
    
    return train_features, train_labels_encoded, dev_features, dev_labels_encoded, test_features, test_labels_encoded, label_encoder

def train_model(model, train_loader, dev_loader, device, num_epochs=50, lr=0.001, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    dev_losses = []
    dev_accuracies = []
    best_dev_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        # 驗證階段
        model.eval()
        dev_loss = 0.0
        dev_correct = 0
        dev_total = 0
        
        with torch.no_grad():
            for features, labels in dev_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                dev_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                dev_total += labels.size(0)
                dev_correct += (predicted == labels).sum().item()
        
        dev_loss = dev_loss / len(dev_loader)
        dev_acc = 100 * dev_correct / dev_total
        dev_losses.append(dev_loss)
        dev_accuracies.append(dev_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.2f}%')
        
        # 早停機制
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f'保存最佳模型 (Dev Acc: {dev_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停於 epoch {epoch+1} (patience={patience})')
                break
        
        # 每10個epoch保存一次檢查點
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'保存檢查點 epoch {epoch+1}')
    
    # 保存最終模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    print('保存最終模型')
    
    return train_losses, dev_losses, dev_accuracies

def evaluate_model(model, test_loader, device, label_encoder):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='測試中'):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算準確率
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'測試準確率: {accuracy:.4f}')
    
    # 分類報告 - 只包含測試集中實際出現的類別
    unique_test_labels = np.unique(all_labels)
    label_names = [label_encoder.classes_[i] for i in unique_test_labels]
    
    # 過濾預測和標籤，只保留測試集中出現的類別
    filtered_predictions = []
    filtered_labels = []
    for pred, label in zip(all_predictions, all_labels):
        if label in unique_test_labels:
            filtered_predictions.append(pred)
            filtered_labels.append(label)
    
    if filtered_predictions:
        report = classification_report(filtered_labels, filtered_predictions, 
                                     labels=unique_test_labels,
                                     target_names=label_names, 
                                     zero_division=0)
        print("分類報告:")
        print(report)
    else:
        print("警告：測試集中沒有有效的預測結果")
    
    return accuracy, all_predictions, all_labels

def main():
    parser = argparse.ArgumentParser(description='訓練說話者識別分類器')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='訓練輪數')
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--hidden_dim', type=int, default=512, help='MLP隱藏層維度')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--clean', action='store_true', help='清理舊的模型文件')
    
    args = parser.parse_args()
    
    # 清理舊的模型文件
    if args.clean:
        old_files = ['best_model.pth', 'final_model.pth', 'linear_classifier.pt', 'mlp_classifier.pt']
        for file in old_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已刪除舊文件: {file_path}")
        
        # 刪除檢查點文件
        import glob
        checkpoint_files = glob.glob(os.path.join(output_dir, 'model_epoch_*.pth'))
        for file in checkpoint_files:
            os.remove(file)
            print(f"已刪除檢查點文件: {file}")
    
    # 設備設置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 載入資料
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels, label_encoder = load_data(
        train_features_path, dev_features_path, test_features_path
    )
    
    # 創建資料集
    train_dataset = SpeakerDataset(train_features, train_labels)
    dev_dataset = SpeakerDataset(dev_features, dev_labels)
    test_dataset = SpeakerDataset(test_features, test_labels)
    
    # 創建資料載入器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 訓練兩個模型
    models_to_train = [
        ('linear', LinearClassifier),
        ('mlp', MLPClassifier)
    ]
    
    for model_name, model_class in models_to_train:
        print(f"\n{'='*50}")
        print(f"開始訓練 {model_name} 模型")
        print(f"{'='*50}")
        
        # 創建模型
        num_classes = len(label_encoder.classes_)
        if model_name == 'linear':
            model = model_class(input_dim=768, num_classes=num_classes)
        else:
            model = model_class(input_dim=768, hidden_dim=args.hidden_dim, 
                              num_classes=num_classes, dropout=args.dropout)
        
        model = model.to(device)
        print(f"模型: {model_name}, 參數數量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 訓練模型
        train_losses, dev_losses, dev_accuracies = train_model(model, train_loader, dev_loader, device, args.epochs, args.lr, args.patience)
        
        # 載入最佳模型進行測試
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            try:
                model.load_state_dict(torch.load(best_model_path))
                print(f"載入最佳模型: {best_model_path}")
            except RuntimeError as e:
                print(f"載入最佳模型失敗: {e}")
                print("使用當前訓練好的模型進行測試")
        else:
            print("未找到最佳模型，使用最終模型")
        
        # 評估模型
        print("評估模型...")
        test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, device, label_encoder)
        
        # 保存模型和結果
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'model_type': model_name,
            'test_accuracy': test_accuracy,
            'predictions': predictions,
            'true_labels': true_labels,
            'train_losses': train_losses,
            'dev_losses': dev_losses,
            'dev_accuracies': dev_accuracies
        }, os.path.join(output_dir, f'{model_name}_classifier.pt'))
        
        print(f"{model_name} 模型已保存到 {os.path.join(output_dir, f'{model_name}_classifier.pt')}")
        print(f"{model_name} 最終測試準確率: {test_accuracy:.4f}")
        
        # 清理記憶體
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*50}")
    print("所有模型訓練完成！")
    print(f"{'='*50}")

if __name__ == '__main__':
    main() 