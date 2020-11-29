import matplotlib.pyplot as plt


def plotting(data1, title, x_label="Date", y_label=None, data2=None):
    plt.figure(figsize=(10, 6))
    plt.plot(data1, color="blue")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if data2 is not None:
        plt.plot(data2, color="red")
        plt.legend(["Real values", "Predicted values"], loc="upper left")
    return plt.show()