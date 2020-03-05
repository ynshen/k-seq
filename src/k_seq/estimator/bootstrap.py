"""Uncertainty estimation through bootstrap"""


class Bootstrap:
    """Class to perform bootstrap for fitting uncertainty estimation
    Three types of bootstrap supported:
      - `pct_res`: resample the percent residue, based on the assumption that variance are proportional to the mean
         (from data property)
      - `data`: directly resample data points
      - `stratified`: resample within groups, `grouper` is needed

    Attributes:
        fitter (`EstimatorBase` type): proxy to the associated fitter

    {}
    """.format(doc_helper.get(['bs_method', 'bootstrap_num', 'bs_record_num', 'grouper']))

    def __repr__(self):
        return f"Bootstrap method using {self.bs_method} (n = {self.bootstrap_num})"

    def __init__(self, fitter, bootstrap_num, bs_record_num, bs_method, grouper=None):
        """
        Args:
            fitter (EstimatorType): the fitter generates the results
        {}
        """.format(doc_helper.get(['bootstrap_num', 'bs_record_num', 'bs_method', 'grouper']))

        self.bs_method = bs_method
        if bs_method == 'stratified':
            try:
                from ..data.grouper import Group
                if isinstance(grouper, Group):
                    grouper = grouper.group
                if isinstance(grouper, dict):
                    self.grouper = grouper
                else:
                    raise TypeError('Unsupported grouper type for stratified bootstrap')
            except KeyError:
                raise Exception('Please indicate grouper when using stratified bootstrapping')
        self.fitter = fitter
        self.bootstrap_num = bootstrap_num
        self.bs_record_num = bs_record_num

    @property
    def bs_method(self):
        return self._bs_method

    @bs_method.setter
    def bs_method(self, bs_method):
        implemented_methods = {
            'pct_res': 'pct_res',
            'resample percent residues': 'pct_res',
            'resample data points': 'data',
            'data': 'data',
            'stratified': 'stratified',
        }
        if bs_method in implemented_methods.keys():
            self._bs_method = bs_method
        else:
            raise NotImplementedError(f'Bootstrap method {bs_method} is not implemented')

    def _percent_residue(self):
        """Bootstrap percent residue"""
        try:
            y_hat = self.fitter.model(
                self.fitter.x_data, **self.fitter.results.point_estimation.params[self.fitter.parameters].to_dict()
            )
        except AttributeError:
            # if could not find point estimation, do another fit
            params = self.fitter.point_estimate()['params'][self.fitter.parameters]
            y_hat = self.fitter.model(self.fitter.x_data, **params.to_dict())

        pct_res = (self.fitter.y_data - y_hat) / y_hat
        for _ in range(self.bootstrap_num):
            pct_res_resample = np.random.choice(pct_res, size=len(pct_res), replace=True)
            yield self.fitter.x_data, y_hat * (1 + pct_res_resample)

    def _data(self):
        """Apply data based bootstrap"""
        indices = np.arange(len(self.fitter.x_data))
        for _ in range(self.bootstrap_num):
            indices_resample = np.random.choice(indices, size=len(indices), replace=True)
            yield self.fitter.x_data[indices_resample], self.fitter.y_data[indices_resample]

    def _stratified(self):
        """Apply stratified bootstrap, need grouper assigned
        x_data and y_data needs to be `Series` or the grouper key should be index
        {}
        """
        for _ in range(self.bootstrap_num):
            ix_resample = []
            for member_ix in self.grouper.values():
                ix_resample += list(np.random.choice(member_ix, size=len(member_ix), replace=True))
            yield self.fitter.x_data[ix_resample], self.fitter.y_data[ix_resample]

    def _bs_sample_generator(self):
        if self.bs_method == 'pct_res':
            return self._percent_residue()
        elif self.bs_method == 'data':
            return self._data()
        elif self.bs_method == 'stratified':
            return self._stratified()
        else:
            return None

    def run(self):
        """Perform bootstrap with arguments indicated in instance attributes
        Returns
           summary, results
        """

        bs_sample_gen = self._bs_sample_generator()
        ix_list = pd.Series(np.arange(self.bootstrap_num))

        def fitting_runner(_):
            x_data, y_data = next(bs_sample_gen)
            result = self.fitter.point_estimate(x_data=x_data, y_data=y_data)
            res_series = pd.Series(data=result['params'], index=self.fitter.parameters)
            if result['metrics'] is not None:
                for key, value in result['metrics'].items():
                    res_series[key] = value
            res_series['x_data'] = x_data
            res_series['y_data'] = y_data
            return res_series

        results = ix_list.apply(fitting_runner)
        summary = results.describe(percentiles=[0.025, 0.5, 0.975], include=np.number)
        allowed_stats = ['mean', 'std', '2.5%', '50%', '97.5%']
        from ..utility.func_tools import dict_flatten
        summary = pd.Series(dict_flatten(summary.loc[allowed_stats].to_dict()))
        return summary, results