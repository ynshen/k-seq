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

