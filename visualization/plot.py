import matplotlib.pyplot as plt

def plot_data(X, y):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y==0,0], X[y==0,1], s=10, label="Class 1")
    plt.scatter(X[y==1,0], X[y==1,1], s=10, label="Class 2")
    plt.scatter(X[y==2,0], X[y==2,1], s=10, label="Class 3")

    plt.title("Generated Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()
