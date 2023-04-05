import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def comparing_result_and_actual(df):
    sns.set(rc={'figure.figsize': (20, 13)})
    graph = sns.lineplot(data=df[["sample", 'Aligned_0']], palette=["#40B0A6", "#E1BE6A"])
    return graph


def master_and_sample(df):
    sns.set(rc={'figure.figsize': (20, 13)})
    graph = sns.lineplot(data=df[["master_chronology", 'Aligned_0']], palette=["#40B0A6", "#E1BE6A"])
    plt.legend(['master chronology', 'sample'], fontsize=15)
    plt.ylabel('detrended sample value', fontsize=25)
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.xlabel('year', fontsize=25)
    plt.title("Detrended values of a master chronology and a sample.", fontsize=25)
    plt.show()
    return graph


if __name__ == "__main__":
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('y=sin(x)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()
