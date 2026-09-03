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

    #---------------------------------------------------
    #Step 3 : Use elbow method
    #---------------------------------------------------

    print("\nStep 3 : Use elbow method")

    WCSS = []

    for i in range(1,11):
        model = KMeans(n_clusters=i, random_state=42,n_init=10)
        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,11),WCSS,marker = 'o')
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow method")
    plt.grid(True)
    plt.show()

    #---------------------------------------------------
    #Step 5 : Train the model
    #---------------------------------------------------

    model = KMeans(n_clusters=4,random_state=42,n_init=10)

    clusters = model.fit_predict(X_scaled)

    df["clusters"] = clusters

    print("Dataset with cluster : ")
    print(df.head(30))

if __name__ == "__main__":
    main()