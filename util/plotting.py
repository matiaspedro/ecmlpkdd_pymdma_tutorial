import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List
from math import pi
from pandas.api.types import is_numeric_dtype


# helper to find the best subplot structure
def subplot_dim_optm(no_signals):
    """
    Find the best subplot structure, trying to equalize, as much as possible,
    the plotting matrix's number of rows and columns. It avoids having a plots
    of (n, 1) or (1, n) type (without setting them previously by hand).
    """
    import math
    matrix_n, matrix_m = int(np.sqrt(no_signals)), int(np.sqrt(no_signals))
    matrix_n += math.ceil((no_signals - matrix_m ** 2) / matrix_n)
    return matrix_n, matrix_m

# helper to determine if a feature should be treated as discrete
def is_discrete(series: pd.Series, threshold: int = 6) -> bool:
    return not is_numeric_dtype(series) or series.nunique(dropna=True) <= threshold

# main function to plot real vs synthetic distributions
def plot_distributions(
        real_dset: pd.DataFrame, 
        syn_dsets: List[pd.DataFrame],
        cols: List[str], 
        names: List[str], 
        real_color: str = 'blue', 
        syn_color: str = 'orange',
        single_dataset: bool = False,
        max_cols: int = None
    ) -> plt.Figure:

    # get columns
    cols_ = cols if max_cols is None else cols[:max_cols]

    # initialize subplots grid
    if single_dataset:
        # find the best subplot structure (most square possible)
        n_rows, n_cols = subplot_dim_optm(len(cols_))
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * n_rows))
    else:
        # calculate number of synthetic datasets and columns to compare
        num_syn = len(syn_dsets)
        n_cols = num_syn + 1  # 1 for real + syn comparisons

        # adjust number of rows to the number of columns to compare
        num_rows = len(cols_)
        fig, axes = plt.subplots(nrows=num_rows, ncols=n_cols, figsize=(6 * n_cols, 4 * num_rows))

    if single_dataset:
        axes = axes.flatten()
    elif num_rows == 1:
        axes = axes[None, :]  # ensure 2d shape
    elif n_cols == 1:
        axes = axes[:, None]
    else:
        pass

    # loop through each feature (row direction)
    for i, col in enumerate(cols_):
        name = names[i] if i < len(names) else col
        ax_real = axes[i] if single_dataset else axes[i, 0]
        is_disc = is_discrete(real_dset[col])

        # for discrete case, determine global y-axis max for row
        y_max = 0
        if is_disc:
            real_counts = real_dset[col].value_counts(normalize=True)
            syn_counts_list = [syn[col].value_counts(normalize=True) for syn in syn_dsets]
            all_cats = sorted(set(real_counts.index).union(*[set(syn_counts.index) for syn_counts in syn_counts_list]))
            real_vals = [real_counts.get(cat, 0) for cat in all_cats]
            y_max = max(y_max, max(real_vals))
            for syn_counts in syn_counts_list:
                syn_vals = [syn_counts.get(cat, 0) for cat in all_cats]
                y_max = max(y_max, max(syn_vals))

            # plot real bar chart
            x = range(len(all_cats))
            bar_width = 0.4
            ax_real.bar(list(x), real_vals, width=bar_width, align='center', alpha=0.7, color=real_color)
            ax_real.set_ylim(0, y_max * 1.1)
        else:
            sns.kdeplot(real_dset[col].dropna(), ax=ax_real, fill=True, color=real_color)

        # set title only on the first row
        ax_real.set_title("Baseline" if i == 0 else "", fontsize=30)
        #ax_real.set_xticks([])
        #ax_real.set_xticklabels([])
        ax_real.set_yticks([])
        ax_real.set_ylabel("")
        ax_real.set_xlabel(f"{col.split('__')[-1]}")

        # plot synthetic comparisons
        for j, syn_dset in enumerate(syn_dsets):
            ax = axes[i, j + 1]

            if is_disc:
                syn_counts = syn_dset[col].value_counts(normalize=True)
                syn_vals = [syn_counts.get(cat, 0) for cat in all_cats]
                x = range(len(all_cats))
                bar_width = 0.4
                ax.bar([xi - bar_width / 2 for xi in x], real_vals, width=bar_width, align='center', alpha=0.7, color=real_color)
                ax.bar([xi + bar_width / 2 for xi in x], syn_vals, width=bar_width, align='center', alpha=0.7, color=syn_color)
                ax.set_ylim(0, y_max * 1.1)
            else:
                sns.kdeplot(real_dset[col].dropna(), ax=ax, fill=True, color=real_color)
                sns.kdeplot(syn_dset[col].dropna(), ax=ax, fill=True, color=syn_color)

            # set title only on the first row
            ax.set_title(f"{names[j+1]}" if i == 0 else "", fontsize=30)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel(f"{col.split('__')[-1]}")

            # share x-axis across the row
            ax.get_shared_x_axes().join(ax_real, ax)

    # adjust layout
    plt.tight_layout()
    return fig


# function to plot correlation matrices for real and synthetic datasets
def plot_corr_matrices(real_dset: pd.DataFrame, syn_dsets: List[pd.DataFrame], names: List[str], cmap: str = 'vlag') -> plt.Figure:
    # combine real and synthetic into one list for iteration
    datasets = [real_dset] + syn_dsets
    num = len(datasets)
    # create subplots horizontally
    fig, axes = plt.subplots(nrows=1, ncols=num, figsize=(5 * num, 5))
    if num == 1:
        axes = [axes]

    # loop through each dataset
    for i, (df, name) in enumerate(zip(datasets, names)):
        ax = axes[i]

        # convert to dataframe if necessary
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # compute correlation matrix
        corr = df.corr().abs()

        # plot heatmap without ticks
        sns.heatmap(
            corr, 
            ax=ax, 
            cmap=cmap, 
            cbar=False, 
            square=True,
            vmin=0,
            vmax=1,
            xticklabels=False, 
            yticklabels=False
        )
        # set title
        ax.set_title(name, fontsize=19)
        # remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    return fig


def plot_barchart(data: pd.DataFrame, x: str, y: str, hue: str, **kwargs) -> plt.Figure:
    fig = sns.barplot(
        x=x, 
        y=y, 
        hue=hue, 
        data=data, 
        palette='Set1', 
        saturation=0.5, 
        orient='h', 
        **kwargs
    )
    plt.xlim(0, int(np.maximum(max(data[x]), 1.2) * 1.2))
    for container in fig.containers:
        fig.bar_label(container)
    return fig


def plot_2d_embeddings(embeds: np.ndarray) -> plt.Figure:
    # figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter plot
    scatter = ax.scatter(embeds[:, 0], embeds[:, 1], c=range(len(embeds)), cmap='viridis', alpha=0.7)
    
    # params
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title("2D Embeddings")
    plt.tight_layout()
    
    return fig


def plot_radar_charts(df: pd.DataFrame, group_col: str):
    def make_spider(ax, row_data, angles, category_labels, color, title):
        values = row_data.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.tick_params(axis='x', pad=12)
        ax.set_xticklabels(category_labels, color='grey', size=12)
        ax.set_rlabel_position(0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color="grey", size=10)
        ax.set_ylim(0, 1)
        ax.set_title(title, size=16, color=color, y=1.1)

    # separate group labels and feature columns
    categories = [col for col in df.columns if col != group_col]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # figure setup
    my_dpi = 40
    n_rows = 1
    n_cols = len(df)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1000/my_dpi, 1000/my_dpi), subplot_kw={'polar': True})
    axs = axs.flatten()  # Flatten in case it's a 2D array of axes

    # color palette
    my_palette = plt.cm.get_cmap("Set2", n_cols)

    for idx, row in df.iterrows():
        make_spider(
            ax=axs[idx],
            row_data=row.drop(group_col),
            angles=angles,
            category_labels=categories,
            color=my_palette(idx),
            title=f"{group_col}: {row[group_col]}"
        )

    # hide any unused subplots
    for j in range(len(df), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    return fig