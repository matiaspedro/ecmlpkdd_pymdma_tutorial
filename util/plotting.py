import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path

from .data import read_image

value_bbox_style = dict(boxstyle='round', facecolor='white', edgecolor="none", alpha=0.5)
def image_grid(image_paths: List[Path | str], n_cols: int, title: str = None, values: List[float] = None) -> None:
    n_rows = (len(image_paths) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4+2))
    
    values = values if values is not None else [None] * len(image_paths)
    for image, ax, value in zip(image_paths, axes.flatten(), values):
        ax.imshow(read_image(image, "pil"))
        ax.axis("off")
        if value is not None:
            ax.text(
                0.95, 0.95, f"{value:.2f}", transform=ax.transAxes, 
                ha="right", va="top", bbox=value_bbox_style, fontsize=14
            )
        
    plt.tight_layout()
    fig.suptitle(title)
    return fig

def plot_distribution(df: pd.DataFrame, column: str, n_bins: int = 100) -> None:
    df[column].plot.hist(bins=n_bins)
    text = f"mean: {df[column].mean():.2f}\nstd: {df[column].std():.2f}\nmin: {df[column].min():.2f}\nmax: {df[column].max():.2f}"
    bbox_style = dict(boxstyle='round', facecolor='tab:blue', edgecolor="none", alpha=0.5)
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, ha="right", va="top", bbox=bbox_style)
    plt.show()
    
def plot_image_grid_extremes(df: pd.DataFrame, column: str, n_cols: int = 5, n_samples: int = 10) -> None:
    low = df.sort_values(column).head(n_samples)
    mid = df.sort_values(column).iloc[len(df) // 2 - n_samples // 2 : len(df) // 2 + n_samples // 2]
    high = df.sort_values(column, ascending=False).head(n_samples)
    metric_name = " ".join(sub.capitalize() for sub in column.split("_"))
    image_grid(low["path"], n_cols=n_cols, title=f"Low {metric_name}", values=low[column])
    image_grid(mid["path"], n_cols=n_cols, title=f"Medium {metric_name}", values=mid[column])
    image_grid(high["path"], n_cols=n_cols, title=f"High {metric_name}", values=high[column])