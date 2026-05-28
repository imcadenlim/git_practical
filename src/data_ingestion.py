import sqlite3
import pandas as pd


def load_data(database_path):
    """
    Load gas monitoring dataset from SQLite database.
    """

    connection = sqlite3.connect(database_path)

    query = "SELECT * FROM gas_monitoring"

    dataframe = pd.read_sql(query, connection)

    connection.close()

    return dataframe


if __name__ == "__main__":

    db_path = "../data/gas_monitoring.db"

    df = load_data(db_path)

    print(df.head())
    print("\nDataset Shape:")
    print(df.shape)