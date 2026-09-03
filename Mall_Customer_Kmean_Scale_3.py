import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    #---------------------------------------------------
    #Step 1 : Load the dataset
    #---------------------------------------------------

    print("\nStep 1 : Load the dataset")
    df = pd.read_csv("Mall_Customers.csv")


    print("First Few records : ")
    print(df.head())

    print("Shape of Dataset : ")
    print(df.shape)

    print("Missing values : ")
    print(df.isnull().sum())

    #---------------------------------------------------
    #Step 2 : Select Features
    #---------------------------------------------------

    print("\nStep 2 : Select Features")

    X = df[["AnnualIncome","SpendingScore"]]

    print("Selected Features : ")
    print(X.head())

    print("Shape of Selected Features : ")
    print(X.shape)

    #---------------------------------------------------
    #Step 3 : Scale the data
    #---------------------------------------------------

    print("\nStep 3 : Scale the data")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nData after scaling : ")
    print(X_scaled[:5])


if __name__ == "__main__":
    main()