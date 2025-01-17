import torch
import joblib
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 定义 NBeats 模型
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
        output = sum(block(x) for block in self.blocks)  # 多块输出求和
        return output


# 加载模型和标准化器
model = NBeatsModel(input_size=96 * len(['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                         'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']),
                    output_size=240)  # 输出为240小时的cnt预测
model.load_state_dict(torch.load('nbeats_model_96h_to_240h.pth'))  # 加载96h_to_240h模型
model.eval()

scaler = joblib.load('scaler_96h_to_240h.pkl')  # 加载96h_to_240h标准化器
scaler_target = joblib.load('scaler_target_96h_to_240h.pkl')  # 加载96h_to_240h目标标准化器

# 加载测试数据
data_test = pd.read_csv('test_data.csv')
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp',
            'atemp', 'hum', 'windspeed', 'casual', 'registered']
target = 'cnt'

scaled_features_test = scaler.transform(data_test[features])
scaled_target_test = scaler_target.transform(data_test[target].values.reshape(-1, 1)).flatten()

# 构建时间序列
def create_sequences(data, target, input_window=96, output_window=240):
    sequences, labels = [], []
    for i in range(len(data) - input_window - output_window + 1):
        seq = data[i:i + input_window]
        label = target[i + input_window:i + input_window + output_window]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X_test, y_test = create_sequences(scaled_features_test, scaled_target_test, input_window=96, output_window=240)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 预测测试集
with torch.no_grad():
    predictions = model(X_test).numpy()  # 模型预测
    true_values = y_test.numpy()  # 测试集真实值

    # 反标准化
    predictions_unscaled = scaler_target.inverse_transform(predictions)
    true_values_unscaled = scaler_target.inverse_transform(true_values)
    # 计算 MSE
    mse = mean_squared_error(true_values_unscaled, predictions_unscaled)
    print(f"Test MSE: {mse:.4f}")

    # 计算 MAE
    mae = mean_absolute_error(true_values_unscaled, predictions_unscaled)
    print(f"Test MAE: {mae:.4f}")

    # 计算标准差
    prediction_std = np.std(predictions_unscaled, axis=0)  # 预测值的标准差
    true_values_std = np.std(true_values_unscaled, axis=0)  # 真实值的标准差
    print(f"Prediction Standard Deviation: {np.mean(prediction_std):.4f}")
    print(f"True Values Standard Deviation: {np.mean(true_values_std):.4f}")
def plot():
    # 绘制对比曲线（选择3个样本）
    sample_indices = [0, 500, 1000]  # 选择三个样本
    time_axis = np.arange(predictions_unscaled.shape[1])  # 240小时时间轴

    # 创建3个子图
    plt.figure(figsize=(15, 18))

    # 为每个样本选择不同的颜色
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, idx in enumerate(sample_indices):
        # 创建一个子图
        plt.subplot(3, 1, i + 1)  # 3行1列，当前是第i+1个子图
        # 绘制真实值和预测值
        plt.plot(time_axis, true_values_unscaled[idx], label=f"True Sample {idx + 1}", color=colors[i], linestyle='-', linewidth=2)
        plt.plot(time_axis, predictions_unscaled[idx], label=f"Predicted Sample {idx + 1}", color=colors[i], linestyle='--', linewidth=2)
        
        # 设置每个子图的标题和标签
        if i == 0:
            plt.title(f"240h's Prediction vs Ground Truth - Samples", fontsize=16)
        if i == 2:
            plt.xlabel("Time (hours)", fontsize=12)
        plt.ylabel("Bike Rental Count", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片，设置dpi为300
    plt.savefig("240h_prediction_vs_ground_truth_samples_subplots.png", dpi=300)

    # 显示图片
    plt.show()
def plot1():
    # 绘制对比曲线（选择3个样本）
    sample_indices = [0, 500, 1000]  # 选择三个样本
    time_axis = np.arange(predictions_unscaled.shape[1])  # 240小时时间轴

    # 创建3个子图，调整图片大小
    plt.figure(figsize=(10, 12))  

    # 设置字体大小
    plt.rcParams.update({'font.size': 14})  

    # 为每个样本选择不同的颜色
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, idx in enumerate(sample_indices):
        # 创建一个子图
        plt.subplot(3, 1, i + 1)  # 3行1列，当前是第i+1个子图
        # 绘制真实值和预测值
        plt.plot(time_axis, true_values_unscaled[idx], label=f"True Sample {idx + 1}", color=colors[i], linestyle='-', linewidth=2)
        plt.plot(time_axis, predictions_unscaled[idx], label=f"Predicted Sample {idx + 1}", color=colors[i], linestyle='--', linewidth=2)
        
        # 设置每个子图的标题和标签
        if i == 0:
            plt.title(f"240h's Prediction vs Ground Truth - Samples", fontsize=16)
        if i == 2:
            plt.xlabel("Time (hours)", fontsize=14)
        plt.ylabel("Bike Rental Count", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片，设置dpi为300
    plt.savefig("240h_prediction_vs_ground_truth_samples_subplots.png", dpi=300)

    # 显示图片
    plt.show()

plot1()