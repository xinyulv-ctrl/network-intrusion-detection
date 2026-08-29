"""UNSW-NB15 数据预处理:清洗、编码、标准化,保存预处理数据与预处理器。"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

CATEGORICAL_COLS = ['proto', 'service', 'state']
TARGET_COL = 'label'          # 二分类:0=正常, 1=攻击
MULTICLASS_COL = 'attack_cat'  # 多分类:Normal + 9 类攻击


def load_raw(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna('unknown').astype(str)
    # attack_cat 缺失:label==0 → Normal; label==1 → Generic
    if MULTICLASS_COL in df.columns:
        empty = df[MULTICLASS_COL].isna() | (df[MULTICLASS_COL].astype(str).str.strip() == '')
        df.loc[empty & (df[TARGET_COL] == 0), MULTICLASS_COL] = 'Normal'
        df.loc[empty & (df[TARGET_COL] == 1), MULTICLASS_COL] = 'Generic'
        df[MULTICLASS_COL] = df[MULTICLASS_COL].astype(str)
    return df


def build_preprocessor(X):
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL_COLS]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
    ])
    return preprocessor


def main():
    train = clean(load_raw(os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')))
    test = clean(load_raw(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')))

    feature_cols = [c for c in train.columns if c not in (TARGET_COL, MULTICLASS_COL)]
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train = train[TARGET_COL].astype(int).values
    y_test = test[TARGET_COL].astype(int).values

    le = LabelEncoder()
    y_train_mc = le.fit_transform(train[MULTICLASS_COL])
    y_test_mc = le.transform(test[MULTICLASS_COL])

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(DATA_DIR, 'preprocessed.npz'),
        X_train=X_train_t, y_train=y_train, y_train_mc=y_train_mc,
        X_test=X_test_t, y_test=y_test, y_test_mc=y_test_mc,
        allow_pickle=True,
    )
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_cols.joblib'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    print(f'X_train: {X_train_t.shape}, X_test: {X_test_t.shape}')
    print('二分类标签分布(训练集):', dict(zip(*np.unique(y_train, return_counts=True))))
    print('多分类标签分布(训练集):', dict(zip(le.classes_, np.bincount(y_train_mc))))


if __name__ == '__main__':
    main()
