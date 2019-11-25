"""This submodule contains different sampler to
  - subsample dataset
  - resample dataset (e.g. bootstrap)
"""


class SamplerBase:
    """Base sampler class"""

    def __init__(self):
        pass

    def get_sample(self):
        pass


class RandomSampler(SamplerBase):

    def __init__(self, target, table=None, sample_n=None, rnd_seed=23, axis='seq'):
        super().__init__()





