import pandas as pd
from pathlib import Path

results_folder = Path("outputs/results")
results_folder.mkdir(parents=True, exist_ok=True)
log_file = Path("outputs/results/data_processing_log.txt")


def load_data():

    #Load in the 3 datasets
    transactions_df = pd.read_csv("data/raw/transactions_df.csv")
    customer_df = pd.read_csv("data/raw/customer_profiles_table.csv")
    terminal_df = pd.read_csv("data/raw/terminal_profiles_table.csv")

    return transactions_df, customer_df, terminal_df


def inspect_data(df, name):
    
    #Show if theres any duplicates, missing values, what the data types are, and shape for the df

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")

    print("\nShape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def clean_data(transactions_df, customer_df, terminal_df):
    
    #Merge the datasets together keep necassary columns and clean the data

    #Convert timestamp
    transactions_df["post_ts"] = pd.to_datetime(transactions_df["post_ts"])

    #This is a duplicate of the fraud column only true if fraud is true
    transactions_df = transactions_df.drop(columns=["fraud_scenario"])

    # Remove duplicate transactions
    transactions_df = transactions_df.drop_duplicates()

    #Remove duplicate BIN column from customer table
    customer_df = customer_df.drop(columns=["bin"])

    #Merge customer information
    merged_df = transactions_df.merge(
        customer_df,
        on="customer_id",
        how="left"
    )

    #Merge terminal information
    merged_df = merged_df.merge(
        terminal_df,
        on="terminal_id",
        how="left"
    )

    #Final duplicate check
    merged_df = merged_df.drop_duplicates()

    #Check for missing values
    print("\nMissing values after merge:")
    print(merged_df.isnull().sum())

    return merged_df


def save_data(df):
   
    #Save the new df to the processed output for use in notebooks etc

    output_path = "data/processed/clean_transactions.csv"

    df.to_csv(output_path, index=False)

    print(f"\nClean dataset saved to:\n{output_path}")


def main():

    transactions_df, customer_df, terminal_df = load_data()

    inspect_data(transactions_df, "Transactions")
    inspect_data(customer_df, "Customers")
    inspect_data(terminal_df, "Terminals")

    merged_df = clean_data(
        transactions_df,
        customer_df,
        terminal_df
    )

    save_data(merged_df)

    with open(log_file, "w") as file:

        file.write("=" * 50 + "\n")
        file.write("DATA CLEANING COMPLETE\n")
        file.write("=" * 50 + "\n\n")

        file.write("Datasets merged:\n")
        file.write("- transactions_df\n")
        file.write("- customer_profiles_table\n")
        file.write("- terminal_profiles_table\n\n")

        file.write(f"Final dataset shape: {merged_df.shape}\n")

        file.write("\nColumns:\n")

        for column in merged_df.columns:
            file.write(f"- {column}\n")


if __name__ == "__main__":
    main()