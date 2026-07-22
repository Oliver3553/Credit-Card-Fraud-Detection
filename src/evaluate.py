import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)
from preparation import preparation


def evaluate_models():

    #Create output folder
    output_folder = Path("outputs/evaluation")
    output_folder.mkdir(parents=True, exist_ok=True)
    results_folder = Path("outputs/results")
    results_folder.mkdir(parents=True, exist_ok=True)

    #Load data
    df = pd.read_csv("data/processed/feature_engineered_transactions.csv")


    #Prepare test data from my function
    X_train, X_test, y_train, y_test, prep = preparation(df)


    #Location of saved models
    models_folder = Path("outputs/models")
    #Finds models in the folder that have a .pkl extension
    models = list(models_folder.glob("*.pkl"))


    #Text output file
    results_file = results_folder / "evaluation_results.txt"

    #Empty list for ROC values later
    roc_curve_results = []

    #Set up for hypertuning best model
    best_model = ""
    best_auc = 0


    with open(results_file, "w") as file:

        for model_file  in models:

            ### REPORT SEGMENT

            #Tidy up the naming convention
            display_name = model_file.stem

            #Load model
            model = joblib.load(model_file)

            #Make predictions
            predictions = model.predict(X_test)

            #Make probability predictions
            probabilities = model.predict_proba(X_test)[:,1]

            #Save report
            report = classification_report(y_test, predictions)

            #Write to file with tidied up name 
            file.write("=" * 50 + "\n")
            file.write(display_name + "\n")
            file.write("=" * 50 + "\n\n")
            file.write(report)
            file.write("\n\n")


            ### CONFUSION MATRIX SEGMENT

            #Add confusion matrix plot the matrix with labels and title and save it to the output folder (per model)
            cm = confusion_matrix(y_test, predictions)

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[
                    "Normal",
                    "Fraud"
                ]
            )

            #Get the values from the confuision matrix
            tn, fp, fn, tp = cm.ravel()

            display.plot()
            plt.title(display_name)
            plt.savefig(
                output_folder /
                f"{display_name}_confusion_matrix.png",
                bbox_inches="tight"
            )
            plt.close()

            file.write("Confusion Matrix\n")
            file.write(f"True Negatives : {tn}\n")
            file.write(f"False Positives: {fp}\n")
            file.write(f"False Negatives: {fn}\n")
            file.write(f"True Positives : {tp}\n\n")


            ### ROC AUC SEGMENT

            #This is the scoring I found on stackoverflow to assist which helps alot with my fraud analysis
            #The fpr and tpr is flase and true positive reults to be plotted later 
            #Essentialy for me it asks - How well can the model rank fraud transactions above normal transactions
            
            fpr, tpr, _ = roc_curve(y_test, probabilities)

            roc_auc = auc(fpr, tpr)

            roc_curve_results.append(
                (
                    display_name,
                    fpr,
                    tpr,
                    roc_auc
                )
            )

            file.write(f"ROC-AUC: {roc_auc:.4f}\n\n")

            #Get best model
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_model = display_name


        #Write in reccomendations for best model based on ROC AUC score
        file.write("=" * 50 + "\n")
        file.write("Recommendation\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Recommended model: {best_model}\n")
        file.write(f"ROC-AUC: {best_auc:.4f}\n")


    ### USE ALL 3 MODELS TO PLOT ROC AUC SEGMENT

    #Plot the ROC results
    plt.figure(figsize=(8,6))

    for name, fpr, tpr, roc_auc in roc_curve_results:

        plt.plot(
            fpr,
            tpr,
            label=f"{name}: {roc_auc:.3f}"
        )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    #Label the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    plt.savefig(
        output_folder / "roc_curve_comparison.png",
        bbox_inches="tight"
    )

    plt.close()



if __name__ == "__main__":
    evaluate_models()