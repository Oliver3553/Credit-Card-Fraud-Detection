from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def preparation(df):

    #Drop unneeded columns
    df = df.drop(
        columns=[
            "transaction_id",
            "customer_id",
            "terminal_id",
            "available_terminals",
        ]
    )

    #mcc being read as type int needs to be str as a categorical variable
    df["mcc"] = df["mcc"].astype(str)

    #X = the dataframe without the binary fraud factor
    X = df.drop(columns=["fraud"])
    #y = the binary fraud factor
    y = df["fraud"]

    #Identify categorical columns
    categorical = [
        "entry_mode",
        "network_id",
        "mcc"
    ]

    #Identify numerical columns that arent the date
    numerical = [
        col
        for col in X.columns
        if col not in categorical
        and col != "post_ts"
    ]

    #Different types of columns go down different routes 
    #For all numerical columns use StandardScaler() and categorical columns use OneHotEncoder()
    #Model can focus on relationship and not size 
    prep = ColumnTransformer(transformers=[

            (
                "num",
                StandardScaler(),
                numerical
            ),

            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical
            )

        ]

    )

    #Train Test Split - stratify=y allows us to keep preserve teh fraud ratio
    #As there is not alot of fraud we want to make sure its balanced (eg if 100 fraud reports then 80 in train and 20 in test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        prep
    )