import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(model_name):
    # 读取训练历史记录
    file_path = f'model/{model_name} loss.csv'
    df = pd.read_csv(file_path)

    # 获取训练损失和验证损失
    train_loss = df['loss']
    val_loss = df['val_loss']

    # 获取训练轮数
    epochs = range(1, len(train_loss) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{model_name.upper()} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(f'images/{model_name}_loss.png')
    plt.show()

if __name__ == '__main__':
    models = ['lstm', 'gru', 'saes','lstm_transformer','lstm_cnn']
    for model in models:
        plot_loss(model)