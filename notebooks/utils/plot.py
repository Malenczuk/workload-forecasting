import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def plot_corr(df: pd.DataFrame, size=10, fname=None) -> None:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, format='pdf')



