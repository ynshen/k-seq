"""Visualization for single fit results
TODO: code clean up
"""


def get_single_fitter(seq, exp):
    """Load single fitter from exp with label seq if seq is not FitResult"""
    from ...utility import func_tools

    single_fitter = exp.results.get_FitResult(seq=seq)
    if not hasattr(single_fitter, 'data'):
        single_fitter.data = func_tools.AttrScope(
            x=exp.x_values[exp.y_values.columns],
            y=exp.y_values.loc[seq],
            sigma=None
        )

    return single_fitter



def bootstrap_params_dist_plot(fitting_res, model=None, seq_ix=None, params_to_plot=['k', 'A'],
                               subsample_num=500, seq_name=None, save_fig_to=None):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if seq_ix is None:
        seq_ix = None
    if seq_name is None:
        seq_name = None
    if model is None:
        model = None

    model, data_to_plot = parse_fitting_results(fitting_res=fitting_res,
                                                model=model,
                                                seq_ix=seq_ix,
                                                seq_name=seq_name,
                                                num_bootstrap_records=subsample_num)

    for seq_name, data in data_to_plot.items():
        jp = sns.jointplot(x=params_to_plot[0], y=params_to_plot[1], data=data['bs_params'],
                           kind='scatter')
        if save_fig_to:
            jp.savefig(save_fit_to, bbox_inches='tight', dpi=300)


def param_value_plot(fitting_res, param, seq_to_show=None, ax=None,
                     line_postfix='', line_color='#2C73B4', line_marker='',
                     show_shade=True, upper_postfix='_97.5%', lower_postfix='_2.5%',
                     shade_color='#2C73B4', shade_alpha=0.3,
                     sort_by=None, y_log=False, save_fig_to=None, **kwargs):
    """Plot of given estimated parameter values across selected sequences"""

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from .least_squares import BatchFitResults, BatchFitter
    from ..data.seq_data import SeqData

    def get_res_col():
        """parse col name to use"""

        def check_col(param, postfix):
            if (param + postfix) in param_values.columns:
                return param + postfix
            elif postfix in param_values.columns:
                return postfix
            else:
                print(param_values.columns)
                raise IndexError(f"Column '{param + postfix}' or '{postfix}' not found in result")

        line_col = check_col(param, line_postfix)
        if show_shade:
            lower_col = check_col(param, lower_postfix)
            upper_col = check_col(param, upper_postfix)
        else:
            lower_col = None
            upper_col = None
        return line_col, lower_col, upper_col

    def parse_fitting_res():
        """Parse various fitting result input type and get the `pd.DataFrame` output of summary"""

        if isinstance(fitting_res, pd.DataFrame):
            return fitting_res
        elif isinstance(fitting_res, BatchFitter):
            return fitting_res.results.summary()
        elif isinstance(fitting_res, BatchFitResults):
            return fitting_res.summary()
        elif isinstance(fitting_res, SeqData):
            return fitting_res.fitter.results.summary
        else:
            raise Exception('Error: fitting_res has to be an either SeqData or BatchFitting instance')

    param_values = parse_fitting_res()
    line_col, lower_col, upper_col = get_res_col()

    if seq_to_show is not None:
        if isinstance(seq_to_show, list):
            param_values = param_values.loc[seq_to_show]
        elif callable(seq_to_show):
            param_values = param_values[param_values.index.to_series().apply(seq_to_show)]

    if isinstance(sort_by, str):
        if sort_by in param_values.columns:
            param_values = param_values.sort_values(by=sort_by)
        else:
            raise IndexError(f'Unknown column name {sort_by} to sort')
    elif callable(sort_by):
        new_index = param_values.apply(sort_by, axis=1).index
        param_values = param_values.reindex(new_index)
    else:
        if sort_by is not None:
            raise TypeError('Unknown sorting method')

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=kwargs.pop('figsize', (12, 6)))

    pos = np.arange(param_values.shape[0])
    ax.plot(pos, param_values[line_col], marker=line_marker, color=line_color, ls='-')
    if show_shade:
        ax.fill_between(pos, y1=param_values[upper_col], y2=param_values[lower_col],
                        facecolor=shade_color, alpha=shade_alpha)
    if y_log:
        ax.set_yscale('log')
    ax.set_xlabel('Sequences', fontsize=14)
    ax.set_ylabel(param, fontsize=14)

    if save_fig_to:
        fig.savefig(save_fig_to, bbox_inches='tight', dpi=300)

    return ax
