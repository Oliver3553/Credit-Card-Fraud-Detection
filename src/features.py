import numpy as np
import pandas as pd
from pathlib import Path


results_folder = Path("outputs/results")
results_folder.mkdir(parents=True, exist_ok=True)
log_file = Path("outputs/results/features_log.txt")


#Features for fraud detection
def engineer_features(df):


    #Current collumns
    original_columns = df.columns.tolist()

    #Sort the dataframe
    df = df.sort_values(["customer_id", "post_ts"])

    #Time based features - ideas are fraud may be more likley at night, can analyse time between uses
    df["post_ts"] = pd.to_datetime(df["post_ts"])

    df["hour"] = df["post_ts"].dt.hour

    df["day_of_week"] = df["post_ts"].dt.dayofweek

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["night_transaction"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 6)
    ).astype(int)

    
    #Spending based features - ideas are average amount user spends  
    df["amount_difference"] = (
        df["amt"] - df["mean_amount"]
    )

    df["amount_ratio"] = (
        df["amt"] / df["mean_amount"]
    )

    df["amount_zscore"] = (
        (df["amt"] - df["mean_amount"]) /
        df["std_amount"]
    )

    # Avoid division by zero if std_amount is 0
    df["amount_zscore"] = (
        df["amount_zscore"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    
    #Customer based features - ideas on how fast a transactions been made 
    previous_transaction = (
        df.groupby("customer_id")["post_ts"]
        .shift(1)
    )

    df["minutes_since_last_transaction"] = (
        (
            df["post_ts"] - previous_transaction
        ).dt.total_seconds() / 60
    )

    # First transaction has no previous transaction
    df["minutes_since_last_transaction"] = (
        df["minutes_since_last_transaction"]
        .fillna(-1)
    )

    #Terminmal based features - main idea is if a different terminal is used too fast (impossible travel)
    previous_terminal = (
        df.groupby("customer_id")["terminal_id"]
        .shift(1)
    )

    df["terminal_changed"] = (
        previous_terminal != df["terminal_id"]
    ).astype(int)

    df.loc[
        previous_terminal.isna(),
        "terminal_changed"
    ] = 0

    df["rapid_terminal_change"] = (
        (df["terminal_changed"] == 1)
        &
        (df["minutes_since_last_transaction"] <= 5)
        &
        (df["minutes_since_last_transaction"] >= 0)
    ).astype(int)

    
    #Distance from customer to terminal 
    df["distance_between_customer_and_terminal"] = np.sqrt(

        (df["lat_customer"] - df["lat_terminal"]) ** 2 +

        (df["log_customer"] - df["log_terminal"]) ** 2

    )

    with open(log_file, "w") as file:

        file.write("=" * 50 + "\n")
        file.write("FEATURE EGINEERING COMPLETE\n")
        file.write("=" * 50 + "\n\n")

        file.write(f"Features added: \n")

        new_columns = [col for col in df.columns if col not in original_columns]

        for feature in new_columns:
            file.write(f"- {feature}\n")

        file.write("\n")
        file.write(f"Final dataset shape: {df.shape}\n")


    return df


def main():

    df = pd.read_csv("data/processed/clean_transactions.csv")

    df = engineer_features(df)

    df.to_csv("data/processed/feature_engineered_transactions.csv", index=False)

    print("Feature engineering complete.")

    
if __name__ == "__main__":
    main()