# 网络入侵检测系统 (NIDS)

基于 **UNSW-NB15** 数据集,用机器学习方法(随机森林 + XGBoost)对网络流量做多分类检测,识别 1 类正常流量 + 9 类攻击流量,并提供一个可交互的 Gradio demo。全程纯 CPU,无需 GPU。

## 项目背景

网络入侵检测(Network Intrusion Detection)是网络空间安全的核心问题:从海量流量中识别恶意行为。传统基于规则/签名的 IDS 难以应对新型攻击,而机器学习方法可以从流量特征中自动学习攻击模式。

UNSW-NB15(Moustafa & Slay, 2015)是学术界广泛使用的现代入侵检测基准数据集,由新南威尔士大学采集。官方切分:训练集 175,341 条、测试集 82,332 条。每条记录含 42 个特征(3 个类别特征 + 39 个数值特征),标签为 1 类正常 + 9 类攻击:

| 标签 | 含义 |
|---|---|
| Normal | 正常流量 |
| Generic | 通用攻击 |
| Exploits | 漏洞利用 |
| Fuzzers | 模糊测试攻击 |
| DoS | 拒绝服务攻击 |
| Reconnaissance | 侦察/扫描 |
| Backdoor | 后门 |
| Shellcode | Shellcode 注入 |
| Analysis | 分析攻击 |
| Worms | 蠕虫 |

## 环境要求

- Python 3.12(纯 CPU)
- 依赖见 `requirements.txt`

```bash
python -m venv .venv
# Windows:
.venv/Scripts/activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 复现步骤

```bash
# 1. 准备数据集(见下方说明),放入 data/
#    data/UNSW_NB15_training-set.csv   (175,341 条)
#    data/UNSW_NB15_testing-set.csv    (82,332 条)

# 2. 预处理
python src/preprocess.py

# 3. 训练
python src/train.py

# 4. 评估(输出指标 + 混淆矩阵)
python src/evaluate.py

# 5. 启动 demo
python demo/app.py
```

> 若 7860 端口被占用,可用 `GRADIO_SERVER_PORT=8765 python demo/app.py` 指定端口。

## 数据集下载

官方源在国内访问不稳定,本项目数据取自 HuggingFace 镜像(hf-mirror.com):

```bash
curl -L -o data/UNSW_NB15_training-set.csv \
  https://hf-mirror.com/datasets/jharrrry/UNSW-NB15/resolve/main/test.csv
curl -L -o data/UNSW_NB15_testing-set.csv \
  https://hf-mirror.com/datasets/jharrrry/UNSW-NB15/resolve/main/train.csv
```

> 注意:该镜像仓库的 train/test 文件名与官方相反(其 `test.csv` 是官方训练集、`train.csv` 是官方测试集),故上面命令已做对调。下载后请核对行数:训练集 175,341、测试集 82,332。

## 结果

官方训练/测试切分下,XGBoost 为最终 demo 模型:

| 模型 | 多分类准确率 | 多分类 Macro-F1 | 二分类准确率 | 二分类 F1 |
|---|---|---|---|---|
| 随机森林 | 75.5% | 0.46 | 87.8% | 0.90 |
| **XGBoost** | **76.4%** | **0.51** | **87.5%** | **0.90** |

混淆矩阵见 `models/rf_confusion.png` 与 `models/xgb_confusion.png`。

### 类别不平衡的改进尝试

Macro-F1 偏低主要由三个稀有类别拖累(Analysis 677、Worms 44、Backdoor 583 条测试样本)。尝试对类别加平衡权重后:

| 模型 | 多分类准确率 | Macro-F1 | 稀有类召回变化 |
|---|---|---|---|
| XGBoost(无权重) | 76.4% | 0.51 | Worms 0.45 / Shellcode 0.76 |
| XGBoost(平衡权重) | 69.3% | 0.53 | Worms 0.75 / Shellcode 0.94 |

平衡权重能显著提升稀有攻击检出率,但整体准确率与正常流量识别下降——这是典型的**精度-召回权衡**,也是 IDS 中"漏报 vs 误报"的经典取舍。

## 技术要点(面试可展开)

1. **特征工程**:3 个类别特征(`proto`/`service`/`state`)One-Hot 编码,39 个数值特征标准化,共 42 维 → 编码后 194 维;`id` 列剔除。
2. **模型**:随机森林(n_estimators=200)与 XGBoost(n_estimators=300, tree_method='hist'),均为 CPU 友好。
3. **类别不平衡**:稀有类召回率低,可讨论 SMOTE 过采样、代价敏感学习、focal loss 等改进方向。
4. **分布漂移**:DoS 攻击在测试集中为零日变种,与训练集分布差异大,导致其召回率偏低(约 0.1~0.2)——这是真实入侵检测的难点。

## 参考

- Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS*.
- 数据集官网: https://research.unsw.edu.au/projects/unsw-nb15-dataset
