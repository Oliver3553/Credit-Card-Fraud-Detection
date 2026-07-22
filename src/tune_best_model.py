import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from preparation import preparation


def tune_best_model():

    #Load data
    df = pd.read_csv("data/processed/feature_engineered_transactions.csv")

    #Prepare data from my function
    X_train, X_test, y_train, y_test, prep = preparation(df)

    #Pipeline 
    pipeline = Pipeline(
        steps=[
            ("preprocessing", prep),
            ("model", HistGradientBoostingClassifier())
        ]
    )

    #Parameters to try
    parameters = {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_iter": [100, 200],
        "model__max_depth": [4, 8]
    }

    #Try every combination
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    #Fit the data
    search.fit(X_train, y_train)

    #Save tuned model to the model folder so it can be passed into evaluate.py
    #Save any output data to tuned_model folder for visibility
    output_folder = Path("outputs/models")
    results_folder = Path("outputs/results")
    results_folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        search.best_estimator_,
        output_folder / "Gradient Boosting Tuned.pkl"
    )

    #Save results
    with open(results_folder / "tuning_results.txt", "w") as file:

        file.write("=" * 50 + "\n")
        file.write("BEST PARAMETERS\n")
        file.write("=" * 50 + "\n\n")

        for key, value in search.best_params_.items():

            file.write(f"{key}: {value}\n")

        file.write("\n")

        file.write(
            f"Best Cross Validation ROC-AUC: "
            f"{search.best_score_:.4f}\n"
        )
        

if __name__ == "__main__":
    tune_best_model()