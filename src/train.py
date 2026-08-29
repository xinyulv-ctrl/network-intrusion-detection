"""训练随机森林与 XGBoost 入侵检测模型(多分类:Normal + 9 类攻击)。"""
import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def main():
    data = np.load(os.path.join(DATA_DIR, 'preprocessed.npz'), allow_pickle=True)
    X_train, y_train_mc = data['X_train'], data['y_train_mc']

    os.makedirs(MODEL_DIR, exist_ok=True)

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    t0 = time.time()
    rf.fit(X_train, y_train_mc)
    print(f'[RF] 训练完成, 用时 {time.time() - t0:.1f}s')
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf.joblib'), compress=3)

    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
        eval_metric='mlogloss', random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    xgb.fit(X_train, y_train_mc)
    print(f'[XGB] 训练完成, 用时 {time.time() - t0:.1f}s')
    joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb.joblib'), compress=3)


if __name__ == '__main__':
    main()
