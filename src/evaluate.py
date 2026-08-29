"""评估模型:输出二分类/多分类指标、混淆矩阵图,保存到 models/。"""
import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    data = np.load(os.path.join(DATA_DIR, 'preprocessed.npz'), allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test'].astype(int)
    y_test_mc = data['y_test_mc'].astype(int)
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    labels = list(le.classes_)

    print('=' * 60)
    for name in ('rf', 'xgb'):
        model = joblib.load(os.path.join(MODEL_DIR, f'{name}.joblib'))
        pred_mc = model.predict(X_test)
        pred_bin = (pred_mc != np.where(le.classes_ == 'Normal')[0][0]).astype(int)

        print(f'\n----- {name.upper()} 多分类(10类) -----')
        acc = accuracy_score(y_test_mc, pred_mc)
        f1 = f1_score(y_test_mc, pred_mc, average='macro')
        print(f'Accuracy: {acc:.4f}   Macro-F1: {f1:.4f}')
        print(classification_report(y_test_mc, pred_mc, target_names=labels, digits=4))

        print(f'----- {name.upper()} 二分类(Normal vs Attack) -----')
        print(f'Accuracy: {accuracy_score(y_test, pred_bin):.4f}  '
              f'Precision: {precision_score(y_test, pred_bin):.4f}  '
              f'Recall: {recall_score(y_test, pred_bin):.4f}  '
              f'F1: {f1_score(y_test, pred_bin):.4f}')

        plot_confusion(
            y_test_mc, pred_mc, labels,
            f'{name.upper()} 多分类混淆矩阵', os.path.join(MODEL_DIR, f'{name}_confusion.png'),
        )
        print(f'混淆矩阵已保存: models/{name}_confusion.png')


if __name__ == '__main__':
    main()
