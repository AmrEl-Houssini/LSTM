import matplotlib.pyplot as plt


def plotting(data1, title, x_label="Date", y_label=None, data2=None, legend_d1=None, legend_d2=None):
    """
    Plotting the given data into 10 X 6 figure

    Args:
        data1 - data to be plotted
        (str) title - the figure title
        (str) x_label - x axis label
        (str) y_label - y axis label (if given)
        data2 - data to be plotted (if given)
        (str) legend_d1 - legend specifies data1
        (str) legend_d2 - legend specifies data2

    Returns:
        a plot with the specified args
    """

    plt.figure(figsize=(10, 6))
    plt.plot(data1, color="blue")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if data2 is not None:
        plt.plot(data2, color="red")
        plt.legend([legend_d1, legend_d2], loc="upper left")
    return plt.show()
