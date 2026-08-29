"""Gradio 交互 demo:上传流量特征 CSV 批量检测,或选单条示例查看判定结果。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
import joblib
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from preprocess import clean, MULTICLASS_COL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

model = joblib.load(os.path.join(MODEL_DIR, 'xgb.joblib'))
preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.joblib'))
le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))


def _file_path(f):
    if f is None:
        return None
    if isinstance(f, str):
        return f
    return getattr(f, 'name', None) or getattr(f, 'path', None)


def _load_examples():
    df = clean(pd.read_csv(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv'), low_memory=False))
    pred = le.inverse_transform(model.predict(preprocessor.transform(df[feature_cols])))
    df = df.assign(_pred=pred)
    targets = ['Normal', 'Generic', 'Exploits', 'Shellcode', 'Reconnaissance', 'Fuzzers']
    examples = []
    for target in targets:
        sub = df[df[MULTICLASS_COL] == target]
        correct = sub.index[sub['_pred'] == target]
        idx = int(correct[0]) if len(correct) else int(sub.index[0])
        examples.append((f'{target} 样本 (第 {idx} 行)', idx))
    return df, examples, examples[0][1]


TEST_DF, examples, default_idx = _load_examples()


def predict_batch(file):
    path = _file_path(file)
    if not path:
        return None, None, '请先上传 CSV 文件'
    try:
        df = clean(pd.read_csv(path, low_memory=False))
    except Exception as e:
        return None, None, f'读取失败: {e}'
    X_t = preprocessor.transform(df[feature_cols])
    pred = le.inverse_transform(model.predict(X_t))
    conf = np.max(model.predict_proba(X_t), axis=1)
    result = pd.DataFrame({'预测类别': pred, '置信度': conf.round(4)})
    fig, ax = plt.subplots(figsize=(8, 4))
    pd.Series(pred).value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('预测类别分布')
    ax.set_xlabel('类别')
    ax.set_ylabel('数量')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    return result.head(30), fig, f'共检测 {len(result)} 条记录'


def predict_single(idx):
    row = TEST_DF.iloc[int(idx)]
    X_t = preprocessor.transform(pd.DataFrame([row])[feature_cols])
    probs = model.predict_proba(X_t)
    pred = le.inverse_transform(model.predict(X_t))[0]
    conf = float(np.max(probs))
    top3 = np.argsort(probs[0])[::-1][:3]
    detail = ', '.join(f'{le.classes_[i]} ({probs[0][i]:.3f})' for i in top3)
    return pred, f'{conf:.4f}', row[MULTICLASS_COL], detail


with gr.Blocks(title='网络入侵检测系统 (UNSW-NB15)') as demo:
    gr.Markdown('# 网络入侵检测系统\n基于 UNSW-NB15 数据集 + XGBoost,识别 10 类网络流量(Normal + 9 类攻击)。')
    with gr.Tab('批量检测'):
        with gr.Row():
            inp = gr.File(label='上传流量特征 CSV(测试集格式)')
        btn = gr.Button('开始检测')
        with gr.Row():
            out_df = gr.Dataframe(label='检测结果(前 30 条)')
            out_fig = gr.Plot(label='类别分布')
        out_msg = gr.Textbox(label='状态')

    with gr.Tab('单条检测'):
        gr.Markdown('从测试集中挑选模型能正确识别的真实流量记录,查看判定与置信度。')
        ex = gr.Dropdown(choices=[e[0] for e in examples], value=examples[0][0], label='选择示例')
        idx_box = gr.Number(value=int(default_idx), precision=0, label='行号(可改)')
        s_btn = gr.Button('检测这一条')
        with gr.Row():
            s_pred = gr.Textbox(label='预测类别')
            s_conf = gr.Textbox(label='置信度')
            s_true = gr.Textbox(label='真实标签')
        s_detail = gr.Textbox(label='Top-3 概率')

    btn.click(predict_batch, inputs=inp, outputs=[out_df, out_fig, out_msg])

    def _sync(choice):
        d = {e[0]: e[1] for e in examples}
        return d[choice]

    ex.change(_sync, inputs=ex, outputs=idx_box)
    s_btn.click(predict_single, inputs=idx_box, outputs=[s_pred, s_conf, s_true, s_detail])


if __name__ == '__main__':
    demo.launch()
