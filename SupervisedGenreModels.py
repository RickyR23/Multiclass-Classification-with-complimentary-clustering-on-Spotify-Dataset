import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    label_col: str
    metric: str  # accuracy or f1_weighted

#these could honestly be made more dynamic based on user-preferenced dataset, so these are narrowed down to our use case
PROFILES = {
    "spotify": DatasetProfile(name="spotify", label_col="track_genre", metric="f1_weighted"),
    "gtzan": DatasetProfile(name="gtzan", label_col="genre", metric="accuracy"),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

#generic parsing method for the dataset utilized, this is mostly tested utilizing our dataset.csv provided so syntax might differ per other types, unless using a csv
def parse_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def pick_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    feature_mode: str,
    selected_features: Optional[List[str]],
) -> List[str]:
    if feature_mode == "selected":
        if not selected_features:
            raise ValueError("feature_mode='selected' requires --selected_features.")
        missing = [c for c in selected_features if c not in df.columns]
        if missing:
            raise ValueError(f"Selected features missing from dataset: {missing}")
        return selected_features

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c != label_col]


def drop_highly_correlated(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_weighted":
        return float(f1_score(y_true, y_pred, average="weighted"))
    raise ValueError(f"Unsupported metric: {metric}")


def save_confusion_matrix(
    y_true,
    y_pred,
    label_names,
    out_path,
    title,
    top_k=30,
    normalize=None,
):

    n_classes = len(label_names)

    if top_k is not None and top_k > 0 and n_classes > top_k:
        counts = np.bincount(y_true, minlength=n_classes)
        top_idx = np.argsort(counts)[::-1][:top_k]

        y_true_mask = np.isin(y_true, top_idx)
        y_true_f = y_true[y_true_mask]
        y_pred_f = y_pred[y_true_mask]

        remap = {old: new for new, old in enumerate(top_idx)}
        y_true_m = np.array([remap[v] for v in y_true_f])
        y_pred_m = np.array([remap.get(v, -1) for v in y_pred_f])

        keep = y_pred_m != -1
        y_true_m = y_true_m[keep]
        y_pred_m = y_pred_m[keep]

        labels = list(range(len(top_idx)))
        display_labels = [label_names[i] for i in top_idx]
        cm = confusion_matrix(y_true_m, y_pred_m, labels=labels, normalize=normalize)
    else:
        labels = list(range(n_classes))
        display_labels = label_names
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    fig_w = min(18, 6 + 0.35 * len(display_labels))
    fig_h = min(18, 6 + 0.35 * len(display_labels))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, include_values=False, xticks_rotation=45, colorbar=True)

    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()

# defining the pipelines for the three types of modals (K-Nearest Neighbor, Decision Tree, and Random Forest), all from SKlearn
def build_models(profile_name: str) -> Dict[str, Pipeline]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])

    dt = Pipeline([
        ("model", DecisionTreeClassifier(random_state=42)),
    ])

    rf = Pipeline([
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    return {"KNN": knn, "DecisionTree": dt, "RandomForest": rf}


def default_params(profile_name: str) -> Dict[str, Dict[str, object]]:
    if profile_name == "spotify":
        return {
            "KNN": {"model__n_neighbors": 30, "model__metric": "minkowski", "model__p": 1, "model__weights": "uniform"},
            "DecisionTree": {"model__criterion": "gini", "model__splitter": "best", "model__max_features": "log2", "model__class_weight": None},
            "RandomForest": {"model__n_estimators": 230, "model__criterion": "gini", "model__max_features": "log2", "model__bootstrap": True, "model__class_weight": "balanced_subsample"},
        }
    return {
        "KNN": {"model__n_neighbors": 1, "model__metric": "euclidean", "model__weights": "uniform"},
        "DecisionTree": {"model__criterion": "entropy", "model__splitter": "best", "model__max_features": None, "model__class_weight": None},
        "RandomForest": {"model__n_estimators": 230, "model__criterion": "gini", "model__max_features": "sqrt", "model__bootstrap": True, "model__class_weight": None},
    }


def param_grids(profile_name: str) -> Dict[str, Dict[str, List[object]]]:
    if profile_name == "spotify":
        return {
            "KNN": {
                "model__n_neighbors": [15, 30, 45],
                "model__metric": ["minkowski"],
                "model__p": [1, 2],
                "model__weights": ["uniform", "distance"],
            },
            "DecisionTree": {
                "model__criterion": ["gini", "entropy"],
                "model__max_features": ["log2", None],
                "model__class_weight": [None, "balanced"],
            },
            "RandomForest": {
                "model__n_estimators": [180, 230, 300],
                "model__max_features": ["log2", "sqrt"],
                "model__bootstrap": [True],
                "model__class_weight": [None, "balanced_subsample"],
            },
        }

    return {
        "KNN": {
            "model__n_neighbors": [1, 3, 5, 7],
            "model__metric": ["euclidean", "minkowski"],
            "model__p": [2],
            "model__weights": ["uniform", "distance"],
        },
        "DecisionTree": {
            "model__criterion": ["entropy", "gini"],
            "model__max_depth": [None, 10, 20],
            "model__max_features": [None],
        },
        "RandomForest": {
            "model__n_estimators": [100, 180, 230, 300],
            "model__max_features": ["sqrt"],
            "model__bootstrap": [True],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["spotify", "gtzan"], default="spotify")
    ap.add_argument("--data_path", default="dataset.csv")
    ap.add_argument("--label_col", default=None)
    ap.add_argument("--feature_mode", choices=["all_numeric", "selected"], default="all_numeric")
    ap.add_argument("--selected_features", default=None)
    ap.add_argument("--drop_cols", default=None)
    ap.add_argument("--drop_corr_for_knn", action="store_true")
    ap.add_argument("--corr_threshold", type=float, default=0.90)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--out_dir", default="results_supervised")
    args = ap.parse_args()

    profile = PROFILES[args.dataset]
    label_col = args.label_col or profile.label_col

    df = pd.read_csv(args.data_path)

    drop_cols = parse_list(args.drop_cols)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    selected_features = parse_list(args.selected_features)
    feature_cols = pick_feature_columns(df, label_col, args.feature_mode, selected_features)

    needed = [label_col] + feature_cols
    df = df.dropna(subset=needed).copy()

    X = df[feature_cols].copy()
    y_raw = df[label_col].astype(str).copy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_names = le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    X_train_knn, X_test_knn = X_train, X_test
    dropped_corr = []
    if args.drop_corr_for_knn:
        X_train_knn, dropped_corr = drop_highly_correlated(X_train, threshold=args.corr_threshold)
        X_test_knn = X_test.drop(columns=dropped_corr, errors="ignore")

    ensure_dir(args.out_dir)
    models = build_models(profile.name)

    if args.tune:
        grids = param_grids(profile.name)
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        scoring = "accuracy" if profile.metric == "accuracy" else "f1_weighted"
    else:
        params = default_params(profile.name)

    results = {}

    for name, pipe in models.items():
        if name == "KNN":
            Xtr, Xte = X_train_knn, X_test_knn
        else:
            Xtr, Xte = X_train, X_test

        if args.tune:
            search = GridSearchCV(
                estimator=pipe,
                param_grid=grids[name],
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(Xtr, y_train)
            best = search.best_estimator_
            best_params = search.best_params_
        else:
            best = pipe.set_params(**params[name])
            best.fit(Xtr, y_train)
            best_params = params[name]

        y_pred = best.predict(Xte)
        main_score = score(y_test, y_pred, profile.metric)

        report = classification_report(
            y_test, y_pred,
            target_names=label_names,
            zero_division=0,
        )

        report_path = os.path.join(args.out_dir, f"{profile.name}_{name}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        cm_path = os.path.join(args.out_dir, f"{profile.name}_{name}_confusion_matrix.png")
        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            label_names=label_names,
            out_path=cm_path,
            title=f"{profile.name.upper()} - {name} Confusion Matrix",
            top_k=30,
            normalize=None
)

        results[name] = {
            "metric": profile.metric,
            "score": main_score,
            "best_params": best_params,
            "dropped_correlated_features_for_knn": dropped_corr if name == "KNN" else [],
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features": int(Xtr.shape[1]),
        }

        print(f"{name}: {profile.metric}={main_score:.4f}")

    #Saving it directly to a designated area in user system (Honestly have only tested it with windows so this might break with other OS types)
    summary_path = os.path.join(args.out_dir, f"{profile.name}_supervised_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved outputs to: {args.out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
