import numpy as np
import pandas as pd
import pickle
import xgboost
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from matplotlib import pyplot as plt
from pathlib import Path


CLF_NAME = Path(__file__).parent / "models" / "xgboost.pkl"
SCALER_NAME = Path(__file__).parent / "models" / "robust_scaler.pkl"
DATA_FILE_PATH = Path(__file__).parent / "data" / "nba_logreg.csv"


def count_nan_per_column(df):
    """
    Count NaNs for each column of the input DataFrame.
    """
    print(df.isnull().sum())

def score_classifier(df, clf, n_splits=3):

    # names = df["Name"].values.tolist()
    labels = df["TARGET_5Yrs"].values
    # paramset = df.drop(["TARGET_5Yrs", "Name"], axis=1).columns.values
    df_vals = df.drop(["TARGET_5Yrs", "Name"], axis=1).values

    kf = KFold(n_splits=n_splits, random_state=50, shuffle=True)
    confusion_mat = np.zeros((2, 2))
    recall = 0

    for training_ids, test_ids in kf.split(df_vals):
        training_set = df_vals[training_ids]
        training_labels = labels[training_ids]
        test_set = df_vals[test_ids]
        test_labels = labels[test_ids]
        clf.fit(training_set, training_labels)
        predicted_labels = clf.predict(test_set)

        confusion_mat += confusion_matrix(test_labels, predicted_labels)
        recall += recall_score(test_labels, predicted_labels)

    recall /= n_splits

    print("\n→ score_classifier Results:")
    print("- Confusion Matrix:")
    print(confusion_mat)
    print("- Recall: {:.4f}".format(recall))



def load_data(file_path=DATA_FILE_PATH):
    """
    Read data from CSV as a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def save_model(model, filename):
    """
    Save a model to a pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Pkl file saved as: {filename}")


def load_model(filename):
    """
    Load a model from a pickle file.
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def feature_engineering(df):
    """
    Perform feature engineering on the input DataFrame by creating new features based on the existing features.
    
    The new features include total minutes, total points, total turnovers, total assists, total 3 points made, total offensive rebounds, effective field goal percentage (eFG%), and true shooting percentage (TS%).
    """
    df["TOTAL_MIN"] = df["GP"] * df["MIN"]
    df["TOTAL_PTS"] = df["GP"] * df["PTS"]
    df["TOTAL_TOV"] = df["GP"] * df["TOV"]
    df["TOTAL_AST"] = df["GP"] * df["AST"]
    df["TOTAL_3P_Made"] = df["GP"] * df["3P Made"]
    df["TOTAL_OREB"] = df["GP"] * df["OREB"]
    df["eFG%"] = (df["FGM"] + 0.5 * df["3P Made"]) / df["FGA"]
    df["TS%"] = df["PTS"] / (2 * (df["FGM"] + 0.44 * df["FTM"]))
    return df


def preprocess_data(df):
    """
    Preprocess the input DataFrame.

    This preprocessing pipeline include: removing duplicates, filling missing values with zeros, and applying feature engineering.
    """
    df = df.drop_duplicates()
    df = df.fillna(0)
    df = feature_engineering(df)
    return df


def scale_data(df, scaling, scaler_name=None, exclude_cols=["TARGET_5Yrs", "Name"]):
    """
    Scale the input DataFrame using the specified scaling methods: MinMaxScaler, RobustScaler, or MaxAbsScaler. 
    It also allows saving the fitted scaler object to a file using the `save_model` function when `scaler_name` is specified.
    """
    columns_to_scale = [col for col in df.columns if col not in exclude_cols]

    if scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "max_abs":
        scaler = MaxAbsScaler()
    else:
        scaler = MinMaxScaler()

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    if scaler_name:
        save_model(scaler, filename=scaler_name)

    return df


def apply_pca_to_df(df, n_components=None, exclude_cols=["Name", "TARGET_5Yrs"]):
    """
    Apply Principal Component Analysis (PCA) to the input DataFrame when `n_components` is specified.
    """
    base_df = df[exclude_cols]
    columns_to_pca = [col for col in df.columns if col not in exclude_cols]

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df[columns_to_pca])
    columns = [f"PCA_{i}" for i in range(1, n_components + 1)]
    df_pca = pd.DataFrame(X_pca, columns=columns)

    base_df = base_df.reset_index(drop=True)
    df_pca = df_pca.reset_index(drop=True)

    final_df = pd.concat([base_df, df_pca], axis=1)
    return final_df


def get_X_y(df):
    """
    Split the input dataframe into feature matrix X and target vector y.
    """
    X = df.drop(columns=["Name", "TARGET_5Yrs"])
    y = df["TARGET_5Yrs"]
    return X, y


def split_X_y(X, y, test_size=0.2):
    """
    Split the input feature matrix X and target vector y into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def tune_hyperparams_clf(X_train, y_train, cv, score_func):
    """
    Tune hyperparameters for an XGBoost classifier using RandomizedSearchCV, and returns the fitted model. 
    """
    clf = xgboost.XGBClassifier(
        random_state=8,
    )

    param_dist = {
        "max_depth": [3, 4, 5, 6, 7, 8, 9],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "n_estimators": [100, 250, 500, 750, 1000, 3000, 5000, 10000],
        "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "gamma": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5],
        "scale_pos_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "eval_metric": ["aucpr", "map"],
    }

    scorer = make_scorer(score_func=score_func, greater_is_better=True)

    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring=scorer,
        return_train_score=True,
    )

    random_search.fit(X_train, y_train)
    
    best_params = random_search.best_estimator_.get_params()

    print("\n→ Best model found:")
    print(best_params)
    
    unfitted_clf = xgboost.XGBClassifier(**best_params)

    return random_search.best_estimator_, unfitted_clf


def evaluate_clf(clf, X_test, y_test, threshold=0.5):
    """
    Evaluate the performance of a classifier on the test set.
    """

    y_test_prob = clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)

    print("\n→ evaluate_clf Results:")
    print("- Confusion Matrix:")
    print(conf_matrix)
    print("- Number of Positives (target = 1):", np.sum(y_test))
    print("- Number of Negatives (target = 0):", len(y_test) - np.sum(y_test))
    print("- Recall: {:.4f}".format(recall))
    print("- Precision: {:.4f}".format(precision))
    print("- Accuracty: {:.4f}".format(accuracy))


def plot_features_corr_matrix(df, exclude_cols=["Name", "TARGET_5Yrs"]):
    """
    Plot a heatmap of the correlation matrix between all features of the DataFrame.
    """
    concerned_columns = [col for col in df.columns if col not in exclude_cols]

    corr_matrix = df[concerned_columns].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()


def plot_kde_curves(df, target_column="TARGET_5Yrs") -> None:
    """
    Plot KDE curves for each feature in the DataFrame.
    """
    continuous_vars = df.select_dtypes(include="number").columns.drop(target_column)

    n = len(continuous_vars)
    nrows = int(n**0.5) + 1
    ncols = n // nrows + (n % nrows > 0)

    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, var in enumerate(continuous_vars, 1):
        ax = plt.subplot(nrows, ncols, i)
        sns.kdeplot(data=df, x=var, hue=target_column, fill=True, ax=ax)
        plt.ylabel("")
        plt.xlabel(var)

    plt.tight_layout(pad=4)

    plt.show()


def main(enable_plot=False, n_components_pca=None, clf_name=None, scaler_name=None):
    """
    Perform the main tasks of the NBA player career duration prediction pipeline, including data loading, preprocessing, scaling, model hyperparameters tunning, training, evaluation, and saving."
    """
    
    df = load_data()
    df = preprocess_data(df)
    df = scale_data(df, scaling="robust", scaler_name=scaler_name)

    if enable_plot:
        plot_kde_curves(df)
        plot_features_corr_matrix(df)

    if n_components_pca:
        df = apply_pca_to_df(df, n_components=n_components_pca)

    X, y = get_X_y(df)

    X_train, X_test, y_train, y_test = split_X_y(X, y, test_size=0.2)
    fitted_clf, unfitted_clf = tune_hyperparams_clf(X_train, y_train, cv=10, score_func=f1_score)

    evaluate_clf(fitted_clf, X_test, y_test)

    score_classifier(df, unfitted_clf, n_splits=5)

    if clf_name:
        save_model(fitted_clf, filename=clf_name)

if __name__ == "__main__":
    
    main(clf_name=CLF_NAME, scaler_name=SCALER_NAME)
