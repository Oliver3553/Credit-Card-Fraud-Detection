import pandas as pd
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from preparation import preparation


def train_models():

    #Create outputs folder if it doesn't exist
    model_folder = Path("outputs/models")
    model_folder.mkdir(parents=True, exist_ok=True)
    results_folder = Path("outputs/results")
    results_folder.mkdir(parents=True, exist_ok=True)

    #Load feature added data
    df = pd.read_csv("data/processed/feature_engineered_transactions.csv")

    #Prepare data using my preparation function
    X_train, X_test, y_train, y_test, prep = preparation(df)

    #Decide what models to use - for my problem I have chosen to look into 
    #Logistic Regression, Random Forrest and Gradient Boosting
    models = {
        
        #1000 iterations so it has more time to find a solution
        #Fraud examples are rare so need to add importance with the class_weight='balanced' 
        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

        #100 trees combined at the end
        "Random Forest":
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            ),

        #Another decision tree based model but learns progressivley
        "Gradient Boosting":
            HistGradientBoostingClassifier()

    }

    
    #Iterate over each model
    for name, model in models.items():

        #Start pipeline 
        pipeline = Pipeline(
            steps=[
            ("preprocessing", prep),
            ("model", model)
            ]
        )

        #Fit the model
        pipeline.fit(X_train, y_train)

        #Save the trained pipeline (preprocessing + model)
        joblib.dump(
            pipeline,
            model_folder / f"{name}.pkl"
        )   

    print("Models saved to outputs/models/")
    
    #Create a training log
    log_file = results_folder / "training_log.txt"

    with open(log_file, "w") as file:

        file.write("=" * 50 + "\n")
        file.write("TRAINING COMPLETE\n")
        file.write("=" * 50 + "\n\n")

        file.write("Successfully trained and saved:\n\n")

        for name in models.keys():
            file.write(f"- {name}.pkl\n")

if __name__ == "__main__":
    train_models()