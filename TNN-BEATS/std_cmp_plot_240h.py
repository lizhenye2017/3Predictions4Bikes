import joblib
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 加载预测结果
    predictions_all_runs = joblib.load('240predictions_all_runs.pkl')  

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
    plt.figure(figsize=(8, 4))  # 调整图片尺寸，宽 8 英寸，高 4 英寸
    plt.rcParams.update({'font.size': 11})  # 增大字体大小

    # 使用浅蓝色绘制柱状图
    plt.bar(range(1, len(predicted_std_per_time_step) + 1), predicted_std_per_time_step.mean(axis=1),  color='lightblue', edgecolor='cornflowerblue')
    plt.xlabel('Time Step')
    plt.ylabel('Prediction Standard Deviation')
    plt.title("240h's Prediction Standard Deviation Across 5 Runs for Each Time Step")
    plt.xticks(rotation=90)

    # 保存为300 DPI的高质量图片
    plt.savefig('prediction_std_per_time_step_240.png', dpi=300)

    # 显示图像
    plt.show()


if __name__ == "__main__":
    main()