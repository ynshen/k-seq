
# Getting Started with `k-seq` package

This is the quick-start manual for `k-seq` package to analyze kinetic sequencing data. This page is the static version for notebook [`examples/getting_started.ipynb`](https://github.com/ynshen/k-seq/tree/master/examples/getting_started.ipynb). For the details of the package, see [k-seq documentation](https://ynshen.github.io/k-seq/).


#### Example project: estimate the kinetic coefficients for selected ribozymes catalyzing self-aminoacylcation with BFO (biotinyl-phenylalanine-oxazolone)

This tutorial uses data from Evan Janzen's kinetic sequencing experiments on ribozymes that are selected to catalyze self-aminoacylation with BFO. In the experiment design, each unique sequence (unqiue type ribozyme) *s* follows the pseudo first-order reaction kinetics:

![eq1-0](http://www.sciweavers.org/tex2img.php?eq=F_%7Bs%2C%20%5Ctext%7BReacted%7D%7D%20%3D%20%5Cfrac%7Bm_%7Bs%2C%20t%7D%7D%7Bm_%7Bs%2C%20t_0%7D%7D%3D%20A%281-exp%28-%5Calpha%20k%20c_%7BBFO%7D%20%28t-t_0%29%29%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

where

![eq1-1](http://bit.ly/2GqmciP)


In this tutorial, we will use the python package `k-seq` to conduct step-by-step analysis and visualization from count file for each sample to final estimated kinetic coefficients. For examples of command line tool of pipelines see [GitHub:k_seq/examples/clt (to do)](https://github.com/ynshen/k-seq/tree/master/examples/clt)

### Requirements

To run this notebook, make sure:
  - `Python 3` is install (most python version works)
  - `k-seq` package and its dependencies are installed (See [Installation](https://github.com/ynshen/k-seq))
  - Count files from k-seq experiment is obtained

### Contents of this tutorial: 
  - Use `k_seq.data.SeqSampleSet` to parse count files and analyze k-seq data
  - Use `k_seq.data.SeqTable` to obtain and analyze a collection of "valid sequences"
  - Use `k_seq.data.SeqTable` to fit the kinetic model and estimate the kinetic coefficients with uncertainty estimation

## Initialize the workspace

We first setup the workspace by loading the core modules from `k_seq` package, assign the path to count file, and path to working space for data or figure saving


```python
from k_seq.data import SeqSampleSet, SeqTable, SeqFilter

# Optional, set default screen dpi for jupyter notebook for download figures with correct resolution directly
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

# Set path to count files and current working space
from pathlib import Path

COUNT_FILES = '/mnt/storage/projects/k-seq/input/bfo_counts/counts'  # Directory to count files
WORKING_PATH = '/mnt/storage/projects/k-seq/working/bfo_evan/2019-6-28'
output_dir = Path(WORKING_PATH)
if not output_dir.exists():
    output_dir.mkdir(parents=True)
```

## Sequencing sample analysis

In this section, we use `SeqSampleSet` to load a batch of count files from given path (folder storing count files) and analyze the data based on each sequencing samples.

### load sample count files
We can create a `SeqSampleSet` object and named it as `sample_set` by linking the object to multiple count files autmatically scan and extracted. We can use

```notebook
?SeqSampleSet()
```
To list the docstring for modules/classes/methods to see the usage details, here are the common parameters to create a `SeqSampleSet` instance:

| Parameter | Note |
|:-----|:-----|
|`file_root`| Path to the folder containing the count files|
|`count_file_pattern`| Optional. Only files strictly contain the pattern will be considered as count files, useful when there are other file formats under the same folder|
|`name_pattern`| Optional. A built-in function to automatically extract infomation about the sample from its file name and assign the info as a metadata for the sample. Use `[]` to define the region used as sample name and use `{attribute name (,int/float)}` to define the attribute that is at the position and its data type|
|`x_values`| The 'x values' associate with the samples, for example, substrate concentration, time, etc. It can be assign with the attribution extracted using `name_pattern`|
|`sort_by`| Recommended. Always keep the samples in known order|
|`load_data`| Default False. If load the data while creating the sample_set object, useful for large files|
|`silent` | Limit extra printings if True|


```python
sample_set = SeqSampleSet(
    file_root=COUNT_FILES,
    count_file_pattern='_counts.txt',
    x_values='bfo',
    name_pattern='R4[{exp_rep}-{bfo, float}{seq_rep}_S{id, int}]_counts.txt',
    sort_by = 'id',
    load_data=True,
    silent=True
)
```


### Calculate quantification factors
As different sequencing sample has different amount of DNA and sequencing depth, raw count number can not reflect a sequence's absolute abundance. Here, in this experiment, we use a method of spike in that adding a non-reactive RNA `AAAACAAAACAAAACAAA` with known amount to normalize each sample.

We define the quantification factor as the effective DNA amount to sequence, which can be calucalted as
$$
q_i = \frac{\text{Spike in amount (mol)}}{\text{Spike in counts}} \times \text{Total counts *N*}
$$

Thus, the absolute amount of Seq *s* with count *n* is $\frac{n}{N} \times q_i$

Due to the synthsis and sequencing error, we will see the spike-in sequence not only as the exact spike-in sequence but also as some similar sequences. Surveying the sequence peak around the spike-in can help us assess the sequencing error and determine the cutoff distance to count a sequence as spike in.

We can use `get_quant_factors` methods of `SeqSampleSet` to survey spike-in and calculate quantification factors. `max_dist_to_survey` is the argument control the maximal distance to survey around spike in.


```python
sample_set.survey_spike_in_peak(spike_in_seq='AAAAACAAAAACAAAAACAAA',
                                max_dist_to_survey=10,
                                silent=True)
```

After survey of the spike-in sequence, we can visualize the peak around the spike-in sequence to determine a maximal distance to count a sequence as spike-in. In `SeqSampleSet`, there is a class `SeqSampleSet.visualizer` wrapping the visualizations functions from `k_seq.data.visualizer`. Here, show an example of programmable (customizable) plotting based on our visualizers to quickly create a bit more complex figures. We use `matplotlib.pyplot` to create a master figure with two sub-axes, and in each axis, we plot a `spike_in_peak_plot` with different configurations. Some core arguments of `spike_in_peak_plot`:

| Parameter | Note |
|:-----|:-----|
|`accumulate`| the counts will be accumulated counts from center to current distance if True|
|`max_dist`| maximal distance to plot|
|`norm_on_center`| if the abundance is normalized on the center |
|`log_y`| if the y-axis is on log scale|


```python
import numpy as np

# Manual marker/color list for each reps
marker_list = np.repeat(['-o', '->', '-+', '-s'], 7)
color_list = np.tile(['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE'], reps=4).reshape([28])

# create a master figure side by side
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=[16, 6])
# add left figure as accumulated
sample_set.visualizer.spike_in_peak_plot(accumulate=True,
                                         max_dist=15, norm_on_center=True,
                                         color_list=color_list, marker_list=marker_list,
                                         legend_off=True,  ax=axes[0])
# add right figure as average peak shape, with legend
sample_set.visualizer.spike_in_peak_plot(max_dist=15, norm_on_center=True,log_y=True,
                                         color_list=color_list, marker_list=marker_list,
                                         legend_off=False,  ax=axes[1])
plt.show()
```


![png](./_README_meta/output_7_0.png)


From the plots, we can see that by setting the maximal edit distance to count as a spike-in as 2, the error can be in general controlled within 1%. Thus, we set `max_dist=2`


```python
# amount of spike-in DNA in unit fmol. We can also use a dictionary to avoid order difference
import numpy as np
spike_in_amounts = np.array([[4130, 1240, 826, 413, 207, 82.6, 41.3]])
spike_in_amounts = np.repeat(spike_in_amounts, repeats=4, axis=0).reshape(28)

sample_set.get_quant_factors(from_spike_in_amounts=spike_in_amounts, max_dist=2)
```

### Sample overview
We can use `SeqSampleSet.sample_overview` to get a `pd.DataFrame` object as a summary for each sample. As it is a `pd.DataFrame` object, it can be directly saved as `.csv` files by using `SeqSampleSet.sample_overview.to_csv(args)`.

Another useful visualizer of total counts, number of unique sequences, fraction of spike-in sequences is `SeqSampleSet.visualizer.count_file_info_plot()`. See examples as follows:


```python
sample_set.sample_overview
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample type</th>
      <th>name</th>
      <th>total counts</th>
      <th>unique sequences</th>
      <th>x_value</th>
      <th>spike-in amount</th>
      <th>spike-in counts (dist=2)</th>
      <th>spike-in percent</th>
      <th>quantification factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>input</td>
      <td>A-inputA_S1</td>
      <td>2165970</td>
      <td>330565</td>
      <td>NaN</td>
      <td>0.004130</td>
      <td>416438</td>
      <td>0.192264</td>
      <td>0.021481</td>
    </tr>
    <tr>
      <th>1</th>
      <td>reacted</td>
      <td>A-1250A_S2</td>
      <td>2006578</td>
      <td>29455</td>
      <td>1250.0</td>
      <td>0.001240</td>
      <td>322730</td>
      <td>0.160836</td>
      <td>0.007710</td>
    </tr>
    <tr>
      <th>2</th>
      <td>reacted</td>
      <td>A-250A_S3</td>
      <td>1845900</td>
      <td>23911</td>
      <td>250.0</td>
      <td>0.000826</td>
      <td>267257</td>
      <td>0.144784</td>
      <td>0.005705</td>
    </tr>
    <tr>
      <th>3</th>
      <td>reacted</td>
      <td>A-50A_S4</td>
      <td>2617260</td>
      <td>43687</td>
      <td>50.0</td>
      <td>0.000413</td>
      <td>453121</td>
      <td>0.173128</td>
      <td>0.002386</td>
    </tr>
    <tr>
      <th>4</th>
      <td>reacted</td>
      <td>A-10A_S5</td>
      <td>1515552</td>
      <td>26410</td>
      <td>10.0</td>
      <td>0.000207</td>
      <td>351337</td>
      <td>0.231821</td>
      <td>0.000893</td>
    </tr>
    <tr>
      <th>5</th>
      <td>reacted</td>
      <td>A-2A_S6</td>
      <td>1580503</td>
      <td>24362</td>
      <td>2.0</td>
      <td>0.000083</td>
      <td>460933</td>
      <td>0.291637</td>
      <td>0.000283</td>
    </tr>
    <tr>
      <th>6</th>
      <td>reacted</td>
      <td>A-0A_S7</td>
      <td>2948173</td>
      <td>2825</td>
      <td>0.0</td>
      <td>0.000041</td>
      <td>2839946</td>
      <td>0.963290</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <th>7</th>
      <td>input</td>
      <td>A-inputB_S8</td>
      <td>1257100</td>
      <td>174225</td>
      <td>NaN</td>
      <td>0.004130</td>
      <td>304332</td>
      <td>0.242091</td>
      <td>0.017060</td>
    </tr>
    <tr>
      <th>8</th>
      <td>reacted</td>
      <td>A-1250B_S9</td>
      <td>3451114</td>
      <td>42691</td>
      <td>1250.0</td>
      <td>0.001240</td>
      <td>601219</td>
      <td>0.174210</td>
      <td>0.007118</td>
    </tr>
    <tr>
      <th>9</th>
      <td>reacted</td>
      <td>A-250B_S10</td>
      <td>10273787</td>
      <td>120423</td>
      <td>250.0</td>
      <td>0.000826</td>
      <td>1312643</td>
      <td>0.127766</td>
      <td>0.006465</td>
    </tr>
    <tr>
      <th>10</th>
      <td>reacted</td>
      <td>A-50B_S11</td>
      <td>2544354</td>
      <td>56718</td>
      <td>50.0</td>
      <td>0.000413</td>
      <td>329774</td>
      <td>0.129610</td>
      <td>0.003186</td>
    </tr>
    <tr>
      <th>11</th>
      <td>reacted</td>
      <td>A-10B_S12</td>
      <td>2401143</td>
      <td>65885</td>
      <td>10.0</td>
      <td>0.000207</td>
      <td>354022</td>
      <td>0.147439</td>
      <td>0.001404</td>
    </tr>
    <tr>
      <th>12</th>
      <td>reacted</td>
      <td>A-2B_S13</td>
      <td>1913930</td>
      <td>41178</td>
      <td>2.0</td>
      <td>0.000083</td>
      <td>398710</td>
      <td>0.208320</td>
      <td>0.000397</td>
    </tr>
    <tr>
      <th>13</th>
      <td>reacted</td>
      <td>A-0B_S14</td>
      <td>2545559</td>
      <td>2201</td>
      <td>0.0</td>
      <td>0.000041</td>
      <td>2462237</td>
      <td>0.967268</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <th>14</th>
      <td>input</td>
      <td>B-inputA_S15</td>
      <td>2546171</td>
      <td>491652</td>
      <td>NaN</td>
      <td>0.004130</td>
      <td>731109</td>
      <td>0.287141</td>
      <td>0.014383</td>
    </tr>
    <tr>
      <th>15</th>
      <td>reacted</td>
      <td>B-1250A_S16</td>
      <td>1616904</td>
      <td>27376</td>
      <td>1250.0</td>
      <td>0.001240</td>
      <td>286251</td>
      <td>0.177036</td>
      <td>0.007004</td>
    </tr>
    <tr>
      <th>16</th>
      <td>reacted</td>
      <td>B-250A_S17</td>
      <td>3186048</td>
      <td>43505</td>
      <td>250.0</td>
      <td>0.000826</td>
      <td>424314</td>
      <td>0.133179</td>
      <td>0.006202</td>
    </tr>
    <tr>
      <th>17</th>
      <td>reacted</td>
      <td>B-50A_S18</td>
      <td>3661612</td>
      <td>67469</td>
      <td>50.0</td>
      <td>0.000413</td>
      <td>571166</td>
      <td>0.155988</td>
      <td>0.002648</td>
    </tr>
    <tr>
      <th>18</th>
      <td>reacted</td>
      <td>B-10A_S19</td>
      <td>3801335</td>
      <td>67576</td>
      <td>10.0</td>
      <td>0.000207</td>
      <td>798440</td>
      <td>0.210042</td>
      <td>0.000986</td>
    </tr>
    <tr>
      <th>19</th>
      <td>reacted</td>
      <td>B-2A_S20</td>
      <td>2569754</td>
      <td>30430</td>
      <td>2.0</td>
      <td>0.000083</td>
      <td>830313</td>
      <td>0.323110</td>
      <td>0.000256</td>
    </tr>
    <tr>
      <th>20</th>
      <td>reacted</td>
      <td>B-0A_S21</td>
      <td>1971971</td>
      <td>1861</td>
      <td>0.0</td>
      <td>0.000041</td>
      <td>1910791</td>
      <td>0.968975</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <th>21</th>
      <td>input</td>
      <td>B-inputB_S22</td>
      <td>2340997</td>
      <td>496360</td>
      <td>NaN</td>
      <td>0.004130</td>
      <td>479406</td>
      <td>0.204787</td>
      <td>0.020167</td>
    </tr>
    <tr>
      <th>22</th>
      <td>reacted</td>
      <td>B-1250B_S23</td>
      <td>3102130</td>
      <td>47268</td>
      <td>1250.0</td>
      <td>0.001240</td>
      <td>536628</td>
      <td>0.172987</td>
      <td>0.007168</td>
    </tr>
    <tr>
      <th>23</th>
      <td>reacted</td>
      <td>B-250B_S24</td>
      <td>3230772</td>
      <td>37131</td>
      <td>250.0</td>
      <td>0.000826</td>
      <td>435436</td>
      <td>0.134778</td>
      <td>0.006129</td>
    </tr>
    <tr>
      <th>24</th>
      <td>reacted</td>
      <td>B-50B_S25</td>
      <td>2878311</td>
      <td>54080</td>
      <td>50.0</td>
      <td>0.000413</td>
      <td>487128</td>
      <td>0.169241</td>
      <td>0.002440</td>
    </tr>
    <tr>
      <th>25</th>
      <td>reacted</td>
      <td>B-10B_S26</td>
      <td>2101362</td>
      <td>38602</td>
      <td>10.0</td>
      <td>0.000207</td>
      <td>488872</td>
      <td>0.232645</td>
      <td>0.000890</td>
    </tr>
    <tr>
      <th>26</th>
      <td>reacted</td>
      <td>B-2B_S27</td>
      <td>2562117</td>
      <td>42324</td>
      <td>2.0</td>
      <td>0.000083</td>
      <td>794099</td>
      <td>0.309939</td>
      <td>0.000267</td>
    </tr>
    <tr>
      <th>27</th>
      <td>reacted</td>
      <td>B-0B_S28</td>
      <td>2204073</td>
      <td>19959</td>
      <td>0.0</td>
      <td>0.000041</td>
      <td>1391111</td>
      <td>0.631155</td>
      <td>0.000065</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_set.visualizer.count_file_info_plot(plot_total_counts=True,
                                           plot_unique_seq=True,
                                           plot_spike_in_frac=True,
                                           sep_plot=True)
```


![png](./_README_meta/output_12_0.png)


We can see that for samples `A-0A_S7`, `A-0B_S14`, `B-0A_S21`, and `B-0B_S29` (negative standards), most of reads belong to spike-in sequences, and minimal passing of RNA was observed. Here we choose to exclude these samples in further analysis.

### Replicates repeatability at sample level
In this experiment, we have four replicates for each of BFO concentration (2 sequencing reps for each of 2 experimental reps). We have a visualizer to examine the repeatablity of replicates in terms of its spike-in fractions and entropy efficiency (a measure of population distribution)


```python
sample_set.visualizer.rep_spike_in_plot(group_by='bfo')
```


![png](./_README_meta/output_14_0.png)


###  Distribution of sequences length and sequence populations
There are some built-in functions to visualize other properties of sequences in each sample, e.g. sequence length and population hetrogeneity


```python
sample_set.visualizer.length_dist_plot_all(y_log=False)
```


![png](./_README_meta/output_16_0.png)



```python
sample_set.visualizer.sample_count_cut_off_plot_all()
```


![png](./_README_meta/output_17_0.png)


## Valid sequences analysis
In this section, we will pool and extract valid sequences from all the k-seq samples we have in `sample_set`, which can be further used for per sequence kinetic fitting. Extract valid seuqence (which has been detected in at least one input sample and one reacted sample) and convert to a `SeqTable` object can be easily done by using `SeqSampleSet.to_SeqTable()` function, and we can choose to remove the spike in the table.

There are some core attributes and methods in `SeqTable`:
| Attributes | Note |
|:-----|:-----|
|`x_values`| x values corresponding to each sample |
|`metadata`| some metadata related to the dataset |
|`count_table_react`/`count_table_input`| `pd.DataFrame` of count table for input samples and reacted samples|

| Methods | Note |
|:-----|:-----|
|`get_reacted_frac` | method to convert the `count_table_reacted` to a table of reacted fraction|
|`visualizer` | multiple visualization tools |


```python
seq_table = sample_set.to_SeqTable(remove_spike_in=True)
```

We choose to calculate the reacted fraction of all samples except the 0 BFO concentration samples. As we have multiple input samples, choose to use the median of input sequences amount as the initial sequence for the calculation of reacted fraction


```python
zero_samples = [sample_name for sample_name in seq_table.sample_info.keys() if '-0' in sample_name]
seq_table.get_reacted_frac(inplace=True, input_average='median', black_list=zero_samples)
```

### Valid sequence characterization
#### Sequence diversity and distribution
We can next analyze the valid sequences of their diversity and distribution. Here we use `visualizer.seq_occurrence_plot()` to show the number of unique sequences and total counts with respect to number of times detected in samples:


```python
seq_table.visualizer.seq_occurrence_plot()
```


![png](./_README_meta/output_23_0.png)


Here we can see that this dataset is obviously heterogenous that most of unique sequences detected are only detected in limited number of samples while a smaller number of unique sequences are very abundant and has been detected in all the samples, including the sample with zero concentration BFO.
#### Measure variability across replicates

The variablity among replicates of experiments can be further evaluated by the reacted value of valid sequences across different replicates. The built-in method `visualizer.rep_variablity_plot` can draw violin plots indicating the standard deviation (`var_method='SD'`) or median absolute deviation (`var_method='MAD'`), with original value or at percent on mean (PSD) or median (PMAD): 


```python
seq_table.visualizer.rep_variability_plot(group_by='bfo', percentage=False)
seq_table.visualizer.rep_variability_plot(group_by='bfo', percentage=True)
```


![png](./_README_meta/output_25_0.png)



![png](./_README_meta/output_25_1.png)


We can see that, with different initial concentration of BFO, the variability varies; however, the percent variability is similar across different BFO concentration. This gives us the sense of exisitance of heteroscedasticity in fitting.
## Fit the sequences to kinetic model
After we calculated the reacted fraction for each sequence, we can fit data into kinetic model in parallel. We first need to define the kinetic model as a function


```python
def bfo_model(x, A, k):
    return A * (1 - np.exp(-0.3371 * 90 * k * x * 1e-6))
```

Then, we can use `add_fitting` function to add a fitter to our table. Core arguments for this functions

| Arguments | Note |
|:-----|:-----|
|`bounds`| a 2 by k list indicate the lower bound and higher bound for parameters to estimate|
|`weights`| for weighted fitting, all data are weighted same if it is None|
|`metrics`| a list of callables (function) indicate extra metrics to calculate from data, in this case, 'kA'|
|`bootstrap_depth`| assign a number of bootstraps to dataset to estimate the confidence interval, recommand 1000 or more|
|`bs_return_size` | to save memory, we can choose only return part of the bootstrap records, after calculating relavent statistics|
|`missing_data_as_zero`| treat missing data as 0 if True|
|`random_init`|if randomly draw a number from [0, 1] as initial value for each parameter|
|`resample_pct_res`| there are two approaches to perform bootstrap: 1) bootstrap percent residues (from previous violin plots we conclude that percdent error are similarity distributed across data with different BFO concentration) or 2) bootstrap data point directly|
|`seq_to_fit`| pass a list of sequences and only fit those sequences|

In `k_seq` package, we also preset some commonly used filter in `SeqFilter` that can quick generate the list of sequences that pass the filter, for example, here we looked at sequences that:
  - detected in minimal 24 reacted samples
  - detected in minimal 4 input samples
  - minimal average relative abundance in input samples greater than 1%
  
And then only fit on these sequences 


```python
def kA_fn(param):
    return param['k'] * param['A']

filters = SeqFilter(seq_table=seq_table, min_occur_reacted=24, min_occur_input=4, min_rel_abun_input=0.01)
filters.apply_filters()
seq_table.add_fitting(model=bfo_model,
                      bounds=[[0, 0], [1, np.inf]],
                      weights=None,
                      metrics={'kA': kA_fn},
                      bootstrap_depth=200,
                      bs_return_size=200,
                      missing_data_as_zero=False,
                      random_init=True,
                      resample_pct_res=False,
                      seq_to_fit=filters.seq_to_keep)
```

Then a `fitting` attribute is added to `seq_table` and we can check `fitting.config` to fitting configurations. (The fitting is not performed yet)

Before conduct actual fitting, we can use `SeqTable.save_as_dill()` and `SeqTable.load_from_dill()` to save and load data, configuration, and functions at hard drive to conduct parallel fitting on cluster server.

Here we will just run the fitting on the local machine with 6 parallel threads


```python
seq_table.fitting.fitting(parallel_cores=6)
```

## Visualize the fitting results
### Single sequence results
There are some built-in function to visualize the fitting result for single sequence
  - `fitting_curve_plot`: show the fitting curve with data and bootstrapped curves
  - `bootstrap_params_dist_plot`: show the distribution of two estimated parameters and their correlation


```python
seq_table.fitting.visualizer.fitting_curve_plot(seq_ix=['CTCTTCAAACAATCGGTCTTC'])
```


![png](./_README_meta/output_33_0.png)



```python
seq_table.fitting.visualizer.bootstrap_params_dist_plot(params_to_plot=['k', 'A'], seq_ix=['CTCTTCAAACAATCGGTCTTC'])
```

    /home/yuning/.pyenv/versions/anaconda3-5.0.1/lib/python3.6/site-packages/seaborn/distributions.py:214: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      color=hist_color, **hist_kws)



![png](./_README_meta/output_34_1.png)


### All sequences results
We can show the fitting results for sequences using `fitting.summary` to return a `df.DataFrame` object, which can be further saved as `.csv` using `.to_csv()` method


```python
seq_table.fitting.summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A_point_est</th>
      <th>k_point_est</th>
      <th>kA_point_est</th>
      <th>A_mean</th>
      <th>A_std</th>
      <th>A_2.5</th>
      <th>A_median</th>
      <th>A_97.5</th>
      <th>k_mean</th>
      <th>k_std</th>
      <th>k_2.5</th>
      <th>k_median</th>
      <th>k_97.5</th>
      <th>kA_mean</th>
      <th>kA_std</th>
      <th>kA_2.5</th>
      <th>kA_median</th>
      <th>kA_97.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CTCTTCAAACAATCGGTCTTC</th>
      <td>0.236052</td>
      <td>165.386232</td>
      <td>39.039720</td>
      <td>0.235701</td>
      <td>0.011815</td>
      <td>0.212657</td>
      <td>0.237246</td>
      <td>0.251351</td>
      <td>168.579116</td>
      <td>32.938176</td>
      <td>123.721049</td>
      <td>166.743608</td>
      <td>223.489928</td>
      <td>39.458586</td>
      <td>5.797215</td>
      <td>30.786389</td>
      <td>38.971530</td>
      <td>50.397497</td>
    </tr>
    <tr>
      <th>CCACACTTCAAGCAATCGGTC</th>
      <td>0.140131</td>
      <td>277.722522</td>
      <td>38.917565</td>
      <td>0.139650</td>
      <td>0.009200</td>
      <td>0.120568</td>
      <td>0.141253</td>
      <td>0.153023</td>
      <td>292.919409</td>
      <td>83.059320</td>
      <td>176.712397</td>
      <td>277.066869</td>
      <td>500.226230</td>
      <td>40.302786</td>
      <td>9.040099</td>
      <td>26.265296</td>
      <td>39.296391</td>
      <td>60.705757</td>
    </tr>
    <tr>
      <th>ACCCACTTCAAACAATCGGTC</th>
      <td>0.203435</td>
      <td>207.199326</td>
      <td>42.151687</td>
      <td>0.203673</td>
      <td>0.016003</td>
      <td>0.168918</td>
      <td>0.202967</td>
      <td>0.239031</td>
      <td>219.533854</td>
      <td>62.329118</td>
      <td>141.415098</td>
      <td>210.292199</td>
      <td>368.095859</td>
      <td>43.985614</td>
      <td>9.452043</td>
      <td>31.472603</td>
      <td>42.927537</td>
      <td>66.085833</td>
    </tr>
    <tr>
      <th>CCGCTTCAAACAATCGGTTTG</th>
      <td>0.471540</td>
      <td>312.937193</td>
      <td>147.562322</td>
      <td>0.470079</td>
      <td>0.036867</td>
      <td>0.403135</td>
      <td>0.473646</td>
      <td>0.534286</td>
      <td>321.340872</td>
      <td>72.146995</td>
      <td>191.900185</td>
      <td>318.054021</td>
      <td>453.251629</td>
      <td>149.054975</td>
      <td>25.745743</td>
      <td>93.365382</td>
      <td>148.801883</td>
      <td>193.673831</td>
    </tr>
    <tr>
      <th>ATTCACCTAGGTCATCGAGTGT</th>
      <td>0.416950</td>
      <td>542.638441</td>
      <td>226.253006</td>
      <td>0.416844</td>
      <td>0.010700</td>
      <td>0.396379</td>
      <td>0.416742</td>
      <td>0.436115</td>
      <td>552.118955</td>
      <td>66.959350</td>
      <td>431.625532</td>
      <td>546.777537</td>
      <td>707.607330</td>
      <td>229.828594</td>
      <td>25.852821</td>
      <td>185.908576</td>
      <td>227.834102</td>
      <td>290.477151</td>
    </tr>
    <tr>
      <th>ATTACCCTGGTCATCGAGTGT</th>
      <td>1.000000</td>
      <td>404.836926</td>
      <td>404.836926</td>
      <td>0.997755</td>
      <td>0.005523</td>
      <td>0.978665</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>412.540401</td>
      <td>33.100813</td>
      <td>361.256461</td>
      <td>408.779645</td>
      <td>490.248217</td>
      <td>411.584658</td>
      <td>32.686758</td>
      <td>359.297150</td>
      <td>408.692856</td>
      <td>490.171210</td>
    </tr>
    <tr>
      <th>CTACTTCAAACAATCGGTCTG</th>
      <td>0.498193</td>
      <td>264.077118</td>
      <td>131.561334</td>
      <td>0.499020</td>
      <td>0.011561</td>
      <td>0.477144</td>
      <td>0.499080</td>
      <td>0.519871</td>
      <td>264.468386</td>
      <td>28.539579</td>
      <td>215.729852</td>
      <td>260.460842</td>
      <td>324.468799</td>
      <td>131.821886</td>
      <td>12.997788</td>
      <td>109.389124</td>
      <td>130.057152</td>
      <td>161.377738</td>
    </tr>
    <tr>
      <th>CACACTTCAAGCAATCGGTC</th>
      <td>0.110547</td>
      <td>292.540645</td>
      <td>32.339520</td>
      <td>0.111657</td>
      <td>0.008871</td>
      <td>0.093616</td>
      <td>0.112071</td>
      <td>0.125952</td>
      <td>300.480994</td>
      <td>92.845298</td>
      <td>164.615496</td>
      <td>286.304874</td>
      <td>512.269630</td>
      <td>32.966212</td>
      <td>8.397737</td>
      <td>19.926049</td>
      <td>31.927276</td>
      <td>50.535988</td>
    </tr>
    <tr>
      <th>ATTACCCTGGTCATCGAGTGA</th>
      <td>0.897218</td>
      <td>354.569393</td>
      <td>318.126027</td>
      <td>0.895628</td>
      <td>0.027472</td>
      <td>0.840204</td>
      <td>0.900448</td>
      <td>0.942822</td>
      <td>361.876921</td>
      <td>46.103495</td>
      <td>291.348216</td>
      <td>357.000858</td>
      <td>462.387550</td>
      <td>323.222030</td>
      <td>34.705990</td>
      <td>266.685072</td>
      <td>318.434124</td>
      <td>401.878968</td>
    </tr>
    <tr>
      <th>ATTCACCTAGGTCATCGGGTGT</th>
      <td>0.298714</td>
      <td>794.942275</td>
      <td>237.460002</td>
      <td>0.298744</td>
      <td>0.011136</td>
      <td>0.275917</td>
      <td>0.299711</td>
      <td>0.317083</td>
      <td>815.041323</td>
      <td>128.544128</td>
      <td>614.517868</td>
      <td>798.928682</td>
      <td>1069.952883</td>
      <td>242.772975</td>
      <td>33.809142</td>
      <td>188.196417</td>
      <td>239.763774</td>
      <td>306.786715</td>
    </tr>
    <tr>
      <th>ATTCACCTAGGTCATCGGGTG</th>
      <td>0.304345</td>
      <td>165.495376</td>
      <td>50.367642</td>
      <td>0.303263</td>
      <td>0.030287</td>
      <td>0.247403</td>
      <td>0.302807</td>
      <td>0.366285</td>
      <td>177.677420</td>
      <td>53.061528</td>
      <td>111.562565</td>
      <td>169.916342</td>
      <td>318.845307</td>
      <td>52.701257</td>
      <td>11.411497</td>
      <td>38.213291</td>
      <td>51.240526</td>
      <td>83.757301</td>
    </tr>
  </tbody>
</table>
</div>



We can also show one estimated parameter as a line plot with shade indicating 95% confidence intervals using `fitting.visualizer.param_value_plot`:


```python
seq_table.fitting.visualizer.param_value_plot(param='kA', show_point_est=True)
```


![png](./_README_meta/output_38_0.png)


### Summary
Here concludes the core components of `k-seq` package for kinetic sequencing experiment analysis. For detail usage and other functions, please refer to [k-seq documentation](https://ynshen.github.io/k-seq/).
