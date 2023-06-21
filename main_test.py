import pytest
import os
import pandas as pd
import numpy as np
from main_code import (
    get_load_data,
    get_processed_data,
    get_train_test_split,
    get_fraud_model,
    get_fraud_model_evaluation,
    get_F1_Score,
)
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

@pytest.fixture(scope="module")
def cfg():
    config_dir = os.path.abspath("config")
    with initialize_config_dir(config_dir):
        config = compose(config_name="config")
    return config

def test_get_load_data(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg,file_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df.columns.tolist() == [
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
        "Class",
    ]
    return df


def test_get_processed_data(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg,file_path)
    smote_df = get_processed_data(cfg, df, cfg["random_state"])
    assert isinstance(smote_df, pd.DataFrame)
    assert len(smote_df) > 0

def test_get_train_features_and_labels(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg, file_path)
    smote_df = get_processed_data(cfg, df, cfg["random_state"])
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = get_train_test_split(cfg, smote_df, cfg["test_size_1"], cfg["test_size_2"])
    assert len(train_features) > 0
    assert isinstance(train_labels, np.ndarray)
    assert len(train_labels) == len(train_features)
    assert isinstance(val_features, np.ndarray)
    assert len(val_features) > 0
    assert isinstance(val_labels, np.ndarray)
    assert len(val_labels) == len(val_features)
    assert isinstance(test_features, np.ndarray)
    assert len(test_features) > 0
    assert isinstance(test_labels, np.ndarray)
    assert len(test_labels) == len(test_features)
    assert len(train_features) == len(train_labels)
    assert len(test_features) == len(test_labels)
    assert len(val_features) == len(val_labels)
    assert len(val_features) == len(test_features)
    

def test_get_fraud_model(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg,file_path)
    smote_df = get_processed_data(cfg, df, cfg["random_state"])
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = get_train_test_split(cfg, smote_df, cfg["test_size_1"], cfg["test_size_2"])
    model = get_fraud_model(cfg, train_features, train_labels, val_features, val_labels)
    assert model is not None


def test_get_fraud_model_evaluation(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg,file_path)
    smote_df = get_processed_data(cfg, df, cfg["random_state"])
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = get_train_test_split(cfg, smote_df, cfg["test_size_1"], cfg["test_size_2"])
    model = get_fraud_model(cfg,train_features, train_labels, val_features, val_labels)
    model_eval = get_fraud_model_evaluation(model, test_features, test_labels)
    assert isinstance(model_eval, str)
    assert len(model_eval) > 0


def test_get_F1_Score(cfg):
    file_path=cfg["file_path"]
    df = get_load_data(cfg,file_path)
    smote_df = get_processed_data(cfg, df, cfg["random_state"])
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = get_train_test_split(cfg, smote_df, cfg["test_size_1"], cfg["test_size_2"])
    model = get_fraud_model(cfg, train_features, train_labels, val_features, val_labels)
    score = get_F1_Score(model, test_features, test_labels)
    assert isinstance(score, float)
    assert score > cfg["base_score"] 
