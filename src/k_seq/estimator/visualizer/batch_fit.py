"""Visualization for mutliple fitting results"""

import numpy as np
import pandas as pd

__all__ = ['scatter_plot_2d_plotly']


def scatter_plot_2d_plotly(x, y, s=10, color='#4C78A8', note=None,
                           xlog=False, ylog=False,
                           xlabel=None, ylabel=None,
                           title=None):
    """Interaction scatter plot"""

    assert len(x) == len(y)
    if isinstance(s, (int, float)):
        s = np.repeat(s, len(x))
    if not isinstance(color, (list, np.ndarray, pd.Series)):
        color = np.repeat(color, len(x))

    import plotly.graph_objs as go

    if note is None and isinstance(x, pd.Series):
        note = list(x.index)

    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers',
                    marker_size=s, marker_color=color, text=note))

    if xlabel is None and isinstance(x, pd.Series):
        xlabel = x.name
    if ylabel is None and isinstance(y, pd.Series):
        ylabel = y.name

    fig.update_layout(
        title="" if title is None else title,
        xaxis_type='log' if xlog else 'linear', xaxis_title=xlabel,
        yaxis_type='log' if ylog else 'linear', yaxis_title=ylabel
    )

    return fig


#################### to organize


def sample_curves(exp, seq_list, repeated_fitting=True, bootstrap=True, sample_n=10):
    """Plot a list of sample curves on convergence or bootstrap results"""

    if len(seq_list) > sample_n:
        seq_list = np.random.choice(seq_list, size=sample_n, replace=False)

    if repeated_fitting:
        fig, axes = plt.subplots(2, sample_n, figsize=(4 * sample_n, 8))

        for seq, ax in zip(seq_list, axes[0]):
            plot_fitting_curve(seq=seq, exp=exp, plot_on='convergence', ax=ax)
        for seq, ax in zip(seq_list, axes[1]):
            plot_heatmap(seq=seq, exp=exp, ax=ax, plot_on='convergence', colorbar=(ax == axes[1][-1]), add_lines=[0.1, 1, 10, 100])

        plt.tight_layout()
        plt.show()


    if bootstrap:
        fig, axes = plt.subplots(2, sample_n, figsize=(4 * sample_n, 8))

        for seq, ax in zip(seq_list, axes[0]):
            plot_fitting_curve(seq=seq, exp=exp, plot_on='bootstrap', ax=ax)
        for seq, ax in zip(seq_list, axes[1]):
            plot_heatmap(seq=seq, ax=ax, exp=exp, plot_on='bootstrap', colorbar=(ax == axes[1][-1]), add_lines=[0.1, 1, 10, 100])

        plt.tight_layout()
        plt.show()