import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import joblib  # 用于保存模型和标准化器
import matplotlib.pyplot as plt  # 用于绘制柱状图

# 步骤 1: 加载训练集和测试集
data_train = pd.read_csv('train_data.csv')  # 训练集
data_test = pd.read_csv('test_data.csv')  # 测试集

# 特征和目标列
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
target = 'cnt'

# 标准化特征和目标变量
scaler = StandardScaler()
scaler_target = StandardScaler()

# 对训练数据进行标准化
scaled_features_train = scaler.fit_transform(data_train[features])
scaled_target_train = scaler_target.fit_transform(data_train[target].values.reshape(-1, 1)).flatten()

# 对测试数据进行标准化
scaled_features_test = scaler.transform(data_test[features])
scaled_target_test = scaler_target.transform(data_test[target].values.reshape(-1, 1)).flatten()

# 构建时间序列数据
def create_sequences(data, target, input_window=96, output_window=240):
    sequences, labels = [], []
    for i in range(len(data) - input_window - output_window + 1):
        seq = data[i:i + input_window]
        label = target[i + input_window:i + input_window + output_window]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# 训练集时间序列
X_train, y_train = create_sequences(scaled_features_train, scaled_target_train, input_window=96, output_window=240)
# 测试集时间序列
X_test, y_test = create_sequences(scaled_features_test, scaled_target_test, input_window=96, output_window=240)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 步骤 2: 定义N-BEATS模型
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(NBeatsBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入张量
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NBeatsModel(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=4):
        super(NBeatsModel, self).__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(input_size, output_size) for _ in range(num_blocks)])
    
    def forward(self, x):
        # 多块输出求和
        output = sum(block(x) for block in self.blocks)
        return output

# 步骤 3: 数据集与DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 步骤 4: 初始化模型
input_size = X_train.shape[1] * X_train.shape[2]  # 输入特征维度展开
output_size = y_train.shape[1]  # 预测未来240小时的cnt
epochs = 100
learning_rate = 1e-3

# 运行5轮实验，保存每轮的预测结果
predictions_all_runs = []

for run in range(5):
    model = NBeatsModel(input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 步骤 5: 训练模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            outputs = model(inputs)  # 输出形状 (batch_size, output_window)
            loss = criterion(outputs, targets)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Run {run+1}, Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    
    # 步骤 6: 测试模型
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

        # 反标准化
        predictions_unscaled = scaler_target.inverse_transform(predictions)

        # 将每轮的预测结果保存
        predictions_all_runs.append(predictions_unscaled)

joblib.dump(predictions_all_runs, '240predictions_all_runs.pkl')  # 保存预测结果

# 步骤 7: 计算每个时间步的标准差
predicted_std_per_time_step = []

# 假设预测值按时间步顺序排列
for time_step_idx in range(predictions_all_runs[0].shape[1]):  # 240个时间步
    time_step_predictions = [predictions_all_runs[run_idx][:, time_step_idx] for run_idx in range(5)]
    time_step_std = np.std(time_step_predictions, axis=0)  # 计算每个时间步在五轮实验中的预测标准差
    predicted_std_per_time_step.append(time_step_std)  # 保存每个时间步的标准差


# 转换为数组
predicted_std_per_time_step = np.array(predicted_std_per_time_step)

# 步骤 8: 绘制每个时间步的标准差柱状图
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(predicted_std_per_time_step) + 1), predicted_std_per_time_step.mean(axis=1), color='blue', edgecolor='darkblue')
plt.xlabel('Time Step')
plt.ylabel('Prediction Standard Deviation')
plt.title("240h's Prediction Standard Deviation Across 5 Runs for Each Time Step")
plt.xticks(rotation=90)

# 保存为300 DPI的高质量图片
plt.savefig('prediction_std_per_time_step_240.png', dpi=300)

# 显示图像
plt.show()

# 步骤 9: 保存模型和标准化器
torch.save(model.state_dict(), 'nbeats_model_96h_to_240h_std.pth')  # 保存模型
joblib.dump(scaler, 'scaler_96h_to_240h_std.pkl')  # 保存标准化器
joblib.dump(scaler_target, 'scaler_target_96h_to_240h_std.pkl')  # 保存目标标准化器

print("Model and scalers saved successfully!")