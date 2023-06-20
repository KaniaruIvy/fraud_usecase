import argparse
import hydra
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from omegaconf import DictConfig, OmegaConf

cfg = OmegaConf.load("config/config.yaml")

def get_load_data(file_path: str):
    """
    Loads a dataset from a CSV file and returns a Pandas DataFrame.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
         df (pd.DataFrame): A Pandas DataFrame containing the data from the CSV file.
    """
    y_variable = cfg.columns["y_variable"]
    df = pd.read_csv(file_path)
    mapped_class = {"'0'": 0, "'1'": 1}
    df[y_variable] = df[y_variable].map(lambda x: mapped_class[x])
    return df


def get_processed_data(df: pd.DataFrame, random_state: int):
    """Gets processed data from a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The Pandas DataFrame containing the data.
        random_state (int): The random state for the SMOTE algorithm.

    Returns:
        smote_df (pd.DataFrame): A Pandas DataFrame containing the processed data.
    """
    y_variable, drop_variable = cfg.columns["y_variable"], cfg.columns["drop_variable"]
    sm = SMOTE(random_state=random_state)
    X = df.drop([y_variable, drop_variable], axis=1).values
    y = df[y_variable].values
    X_res, y_res = sm.fit_resample(X, y)
    smote_df = pd.DataFrame(
        X_res, columns=df.drop([y_variable, drop_variable], axis=1).columns
    )
    smote_df[y_variable] = y_res
    smote_df[drop_variable] = df[drop_variable]
    return smote_df


def get_train_test_split(smote_df, test_size_1, test_size_2):
    def create_training_sets(data):
        """
        Convert data frame to train, validation and test
        Args:
            data: The dataframe with the dataset to be split
        Returns:
            train_features: Training feature dataset
            test_features: Test feature dataset
            train_labels: Labels for the training dataset
            test_labels: Labels for the test dataset
            val_features: Validation feature dataset
            val_labels: Labels for the validation dataset
        """
        # Extract the target variable from the dataframe and convert the type to float32
        y_variable, drop_variable = (
            cfg.columns["y_variable"],
            cfg.columns["drop_variable"],
        )
        ys = np.array(data[y_variable]).astype("float32")
        # Drop all the unwanted columns including the target column
        drop_list = [y_variable, drop_variable]
        # Drop the columns from the drop_list and convert the data into a NumPy array of type float32
        xs = np.array(data.drop(drop_list, axis=1)).astype("float32")
        np.random.seed(0)
        # Use the sklearn function train_test_split to split the dataset in the ratio train 80% and test 20%
        train_features, test_features, train_labels, test_labels = train_test_split(
            xs, ys, test_size=test_size_1, stratify=ys
        )
        # Use the sklearn function again to split the test dataset into 50% validation and 50% test
        val_features, test_features, val_labels, test_labels = train_test_split(
            test_features, test_labels, test_size=test_size_2, stratify=test_labels
        )
        return (
            train_features,
            test_features,
            train_labels,
            test_labels,
            val_features,
            val_labels,
        )

    (
        train_features,
        test_features,
        train_labels,
        test_labels,
        val_features,
        val_labels,
    ) = create_training_sets(smote_df)
    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    )


def get_train_test_split_samples(
    test_features, test_labels, train_features, train_labels, val_features, val_labels
):
    """Splits a dataset into three parts i.e train, test, and validation for training, evaluating and tuning the hyperaremters of the model

    Args:
        test_features (np.ndarray): The test features.
        test_labels (np.ndarray): The test labels.
        train_features (np.ndarray): The train features.
        train_labels (np.ndarray): The train labels.
        val_features (np.ndarray): The validation features.
        val_labels (np.ndarray): The validation labels.

    Returns:
        split_samples (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing the train, test, and validation samples.
    """
    split_samples = (
        train_features,
        test_features,
        train_labels,
        test_labels,
        val_features,
        val_labels,
    )
    return split_samples


def get_fraud_model(train_features, train_labels, val_features, val_labels):
    """Defines the model used for training

    Args:
        train_features (np.ndarray): The train features.
        train_labels (np.ndarray): The train labels.
        val_features (np.ndarray): The validation features.
        val_labels (np.ndarray): The validation labels.

    Returns:
        model: The model used for training.
    """
    model = XGBClassifier(
        base_score=cfg.model["base_score"],
        booster=cfg.model["booster"],
        colsample_bylevel=cfg.model["colsample_bylevel"],
        colsample_bynode=cfg.model["colsample_bynode"],
        colsample_bytree=cfg.model["colsample_bytree"],
        gamma=cfg.model["gamma"],
        importance_type=cfg.model["importance_type"],
        interaction_constraints=cfg.model["interaction_constraints"],
        learning_rate=cfg.model["learning_rate"],
        max_delta_step=cfg.model["max_delta_step"],
        max_depth=cfg.model["max_depth"],
        min_child_weight=cfg.model["min_child_weight"],
        monotone_constraints=cfg.model["monotone_constraints"],
        n_estimators=cfg.model["n_estimators"],
        n_jobs=cfg.model["n_jobs"],
        num_parallel_tree=cfg.model["num_parallel_tree"],
        random_state=cfg.model["random_state"],
        reg_alpha=cfg.model["reg_alpha"],
        reg_lambda=cfg.model["reg_lambda"],
        scale_pos_weight=cfg.model["scale_pos_weight"],
        silent=cfg.model["silent"],
        subsample=cfg.model["subsample"],
        tree_method=cfg.model["tree_method"],
        validate_parameters=cfg.model["validate_parameters"],
    )
    model.fit(
        train_features,
        train_labels,
        eval_set=[(val_features, val_labels)],
        verbose=True,
    )
    return model


def get_fraud_model_evaluation(model, test_features, test_labels):
    """Gets the evaluation metrics of the model after training.

    Args:
        model (_type_): _description_
        test_features (np.ndarray): The test features
        test_labels (np.ndarray): _The test labels

    Returns:
        model_eval(str): Returns a summary of evaluation metrics inclusing accuracy, precision, recall and F1-Score
    """
    predictions = model.predict_proba(test_features)[:, 1]
    predictions_1 = []
    for i in predictions:
        if i > 0.5:
            predictions_1.append(1)
        else:
            predictions_1.append(0)
    model_eval = classification_report(test_labels, predictions_1, digits=3)
    return model_eval


def get_F1_Score(model, test_features, test_labels):
    """Gets the F1-Score of the model after evaluating it against the test features.

    Args:
        model (_type_): _description_
        test_features (np.ndarray): The test features
        test_labels (np.ndarray): The test labels

    Returns:
        score(float): The F1 score for the fraud model.
    """
    predictions = model.predict_proba(test_features)[:, 1]
    predictions_1 = []
    for i in predictions:
        if i > 0.5:
            predictions_1.append(1)
        else:
            predictions_1.append(0)
    score = f1_score(test_labels, predictions_1)
    return score


def run_session_including_load_data(
    file_path=cfg["file_path"],
    test_size_1=cfg["test_size_1"],
    test_size_2=cfg["test_size_2"],
    random_state=cfg["random_state"],
):
    """Runs a session of fraud_detection

    Args:
        file_path (str): Path to the CSV that contains the fraud data.
        test_size_1 (float): A fraction that splits the dataset into train and test
        test_size_2 (float):A fraction that splits the test dataset into test and validation
        random_state (int): A random number generator seed


    Returns:
        A dictionary containing the following artifacts:

        * load_data: The loaded data.
        * processed_data: The processed data.
        * train_features: The training features.
        * train_labels: The training labels.
        * val_features: The validation features.
        * val_labels: The validation labels.
        * test_features: The test features.
        * test_labels: The test labels.
        * fraud_model: The trained fraud model.
        * fraud_model_evaluation: The evaluation results for the fraud model.
        * F1_Score: The F1 score for the fraud model.
    """
    # Given multiple artifacts, we need to save each right after
    # its calculation to protect from any irrelevant downstream
    # mutations (e.g., inside other artifact calculations)
    import copy

    artifacts = dict()
    df = get_load_data(file_path)
    artifacts["load_data"] = copy.deepcopy(df)
    smote_df = get_processed_data(df, random_state)
    artifacts["processed_data"] = copy.deepcopy(smote_df)
    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = get_train_test_split(smote_df, test_size_1, test_size_2)
    split_samples = get_train_test_split_samples(
        test_features,
        test_labels,
        train_features,
        train_labels,
        val_features,
        val_labels,
    )
    artifacts["train_test_split_samples"] = copy.deepcopy(split_samples)
    model = get_fraud_model(train_features, train_labels, val_features, val_labels)
    artifacts["fraud_model"] = copy.deepcopy(model)
    model_eval = get_fraud_model_evaluation(model, test_features, test_labels)
    artifacts["fraud_model_evaluation"] = copy.deepcopy(model_eval)
    score = get_F1_Score(model, test_features, test_labels)
    artifacts["F1_Score"] = copy.deepcopy(score)
    return artifacts


def run_all_sessions(
    file_path=cfg["file_path"],
    test_size_1=cfg["test_size_1"],
    test_size_2=cfg["test_size_2"],
    random_state=cfg["random_state"],
):
    """Runs all session of fraud_detection except loading data

    Args:
        file_path (str): Path to the CSV that contains the fraud data.
        test_size_1 (float): A fraction that splits the dataset into train and test
        test_size_2 (float):A fraction that splits the test dataset into test and validation
        random_state (int): A random number generator seed


    Returns:
        A dictionary containing the following artifacts:

        * load_data: The loaded data.
        * processed_data: The processed data.
        * train_features: The training features.
        * train_labels: The training labels.
        * val_features: The validation features.
        * val_labels: The validation labels.
        * test_features: The test features.
        * test_labels: The test labels.
        * fraud_model: The trained fraud model.
        * fraud_model_evaluation: The evaluation results for the fraud model.
        * F1_Score: The F1 score for the fraud model.
    """
    artifacts = dict()
    artifacts.update(
        run_session_including_load_data(
            file_path, test_size_1, test_size_2, random_state
        )
    )
    return artifacts


def main() -> None:
    """Run all sessions."""

    # Edit this section to customize the behavior of artifacts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default=cfg["file_path"],
    )
    parser.add_argument("--test_size_1", type=float, default=cfg["test_size_1"])
    parser.add_argument("--test_size_2", type=float, default=cfg["test_size_2"])
    parser.add_argument("--random_state", type=int, default=cfg["random_state"])
    args = parser.parse_args()

    artifacts = run_all_sessions(
        file_path=args.file_path,
        test_size_1=args.test_size_1,
        test_size_2=args.test_size_2,
        random_state=args.random_state,
    )
    print(artifacts)


if __name__ == "__main__":
    main()
