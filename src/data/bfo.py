import numpy as np
import matplotlib.pyplot as plt


##### FOLLOWS ARE BASIC CLASSES USED IN BFO ANALYSIS #####
class Sample:

    """
    This class defines and describe the samples from k-seq experiments
    """

    def __init__(self):
        pass

    def import_count_file(self, dirc, sampleType='reacted'):
        """
        :param dirc:
        :param sampleType:
        :return:
        """
        with open(dirc, 'r') as file:
            self.uniqueSeq = int([elem for elem in next(file).strip().split()][-1])
            self.totalSeq = int([elem for elem in next(file).strip().split()][-1])
            next(file)
            self.seqs = {}
            for line in file:
                seq = line.strip().split()
                self.seqs[seq[0]] = int(seq[1])
        self.sampleType = sampleType

    def survey_ext_std(self, stdSeq='AAAAACAAAAACAAAAACAAA', maxDist=10):
        """
        This method will survey the spike in sequences in each sample
        :param stdSeq:
        :return:
        """
        import Levenshtein
        self.stdCounts = np.array([0 for _ in range(maxDist+1)])
        for seq in self.seqs.keys():
            dist = Levenshtein.distance(stdSeq, seq)
            if (dist <= maxDist)and(len(seq)==21):
                self.stdCounts[dist] += self.seqs[seq]

    def get_seq_length(self, minCounts=1, blackList=None):
        if not blackList:
            blackList = []
        return np.array([len(seq[0]) for seq in self.seqs.items() if (seq[0] not in blackList)and(seq[1] >= minCounts)])

    def get_seq_composition(self, fraction=False, minCounts=1, blackList=None):
        if not blackList:
            blackList = []
        seqsNew = np.array([seq[1] for seq in self.seqs.items() if (seq[0] not in blackList)and(seq[1] >= minCounts)])
        if fraction:
            return seqsNew/np.sum(seqsNew)
        else:
            return seqsNew

###### FOLLOWS ARE BASIC UTILITY FUNCTIONS ######

def basic_info(countFileRoot='/mnt/storage/projects/k-seq/input/bfo_counts/'):
    from os import listdir
    from os.path import isfile, join

    sampleList = [f for f in listdir(countFileRoot) if isfile(join(countFileRoot, f))]
    sort_fn = lambda s: int(s.split('_')[1][1:])
    sampleList.sort(key=sort_fn)
    return countFileRoot, sampleList


root, sampleList = basic_info()


def load_all_samples(sampleSetDirc=None, sampleList=sampleList):
    import util

    if sampleSetDirc:
        sampleSet = util.load_pickle(sampleSetDirc)
    else:
        sampleSet = []
        for sample in sampleList:
            currentSample = Sample()
            currentSample.id = sample[sample.find('-')-1:sample.find('_counts')]
            if sample.find('input') >= 0:
                currentSample.import_count_file(dirc=root+sample, sampleType='input')
                currentSample.survey_ext_std(maxDist=15)
            else:
                currentSample.import_count_file(dirc=root+sample, sampleType='reacted')
                currentSample.survey_ext_std(maxDist=15)
            sampleSet.append(currentSample)

    return sampleSet


def print_sample_overview(sampleSet, table=False, figures=True, figSaveDirc=None):
    if table:
        from IPython.display import HTML
        tableHTML = """
        <table>
        <tr>
        <th>{}</th>
        <th>{}</th>
        <th>{}</th>
        <th>{}</th>
        <th>{}</th>
        <th>{}</th>
        </tr>
        """.format(
            'index',
            'sample name',
            'total counts',
            'unique counts',
            'ext. std. counts',
            'ext. std. percent'
        )
        for ix,sample in enumerate(sampleSet):
            tableHTML += """
            <tr>
            <td>{}</td>
            <td>{}</td>
            <td>{:,}</td>
            <td>{:,}</td>
            <td>{:,}</td>
            <td>{:.3f}</td>
            </tr>
            """.format(
                ix + 1,
                sample.id,
                sample.totalSeq,
                sample.uniqueSeq,
                sample.stdCounts[0],
                sample.stdCounts[0] / sample.totalSeq
            )
        display(HTML(tableHTML))

    if figures:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatch
        import plot

        sampleNum = len(sampleSet)
        fig = plt.figure(figsize=[sampleNum * 0.5, 6])
        # Plot bar for total seqs
        ax = fig.add_subplot(111)
        ax.bar(x=[i - 0.2 for i in range(sampleNum)], height=[sample.totalSeq for sample in sampleSet], align='center',
               width=0.4, color='#2C73B4')
        # plot bar for unique seqs
        ax2 = ax.twinx()
        ax2.bar(x=[i + 0.2 for i in range(sampleNum)], height=[sample.uniqueSeq for sample in sampleSet],
                align='center', width=0.4, color='#FC820D')
        # plot scatter for spike-in percentage
        ax3 = ax.twinx()
        ax3.scatter([i for i in range(sampleNum)], [sample.stdCounts[0] / sample.totalSeq for sample in sampleSet],
                    color='#B2112A', marker='x')
        ax3.plot([-0.5, sampleNum - 0.5], [0.2, 0.2], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.4, 0.4], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.6, 0.6], '#B2112A', ls='--', alpha=0.3)
        ax3.plot([-0.5, sampleNum - 0.5], [0.8, 0.8], '#B2112A', ls='--', alpha=0.3)
        ax3.set_ylim([0, 1])
        ax3.set_yticks([])

        # Aesthetic adjustment
        ax.set_ylabel('Number of total reads in the sample', fontsize=14)
        ax2.set_ylabel('Number of unique sequences in the sample', fontsize=14)
        ax.set_xticks([i for i in range(sampleNum)])
        ax.set_xticklabels([sample.id[:sample.id.find('_counts')] for sample in sampleSet], rotation=90)
        plot.set_ticks_size(ax)
        plot.set_ticks_size(ax2)
        lgd = [mpatch.Patch(color='#2C73B4', label='Total Seqs'), mpatch.Patch(color='#FC820D', label='Unque Seqs'),
               plt.plot([], [], lw=0, marker='x', color='#B2112A', label='Percent of spike-in')[0]]
        plt.legend(handles=lgd)

        if figSaveDirc:
            fig.savefig(figSaveDirc, dpi=300)
        plt.show()


def print_length_dist(sampleSet, unique=True, total=True, minCounts=1, figSaveDirc=None, blackList=None):

    fig, axes = plt.subplots(7, 4, figsize=[12, 16])
    for ix in range(len(sampleSet)):
        ax = axes[ix % 7, int(ix / 7)]
        lengths = sampleSet[ix].get_seq_length(minCounts=minCounts, blackList=blackList)
        bins = np.linspace(0, np.max(lengths), np.max(lengths) + 1)
        if total:
            weights = sampleSet[ix].get_seq_composition(fraction=False, minCounts=minCounts, blackList=blackList)
            ax.hist(lengths, weights=weights, bins=bins, color='#FC820D')
        if unique:
            ax.hist(lengths, bins=bins, color='#2C73B4')
        ax.set_yscale('log')
        ax.set_title(sampleSet[ix].id)
    fig.text(s='Sequence length (nt)', x=0.5, y=0, ha='center', va='top', fontsize=16)
    if unique and total:
        fig.text(s='Number of total sequences (Orange)\n Number of unique sequences (Blue)',
                 x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    elif unique:
        fig.text(s='Number of unique sequences', x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    else:
        fig.text(s='Number of total sequences', x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    plt.tight_layout()
    if figSaveDirc:
        fig.savefig(figSaveDirc, dpi=300)
    plt.show()


def print_composition_dist(sampleSet, unique=True, total=True, minCounts=1, fraction=False, figSaveDirc=None, blackList=None):
    fig, axes = plt.subplots(7, 4, figsize=[12, 16])
    for ix in range(len(sampleSet)):
        ax = axes[ix % 7, int(ix / 7)]
        compositions = sampleSet[ix].get_seq_composition(fraction=fraction, minCounts=minCounts, blackList=blackList)
        bins = np.logspace(np.log10(np.min(compositions) * 0.8), np.log10(np.max(compositions) * 1.1), 50)
        if total:
            weights = compositions
            ax.hist(compositions, weights=weights, bins=bins, color='#FC820D')
        if unique:
            ax.hist(compositions, bins=bins, color='#2C73B4')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(sampleSet[ix].id)
    fig.text(s='Counts in the sample', x=0.5, y=0, ha='center', va='top', fontsize=16)
    if unique and total:
        fig.text(s='Number of total sequences (Orange)\n Number of unique sequences (Blue)',
                 x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    elif unique:
        fig.text(s='Number of unique sequences', x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    else:
        fig.text(s='Number of total sequences', x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    plt.tight_layout()
    if figSaveDirc:
        fig.savefig(figSaveDirc, dpi=300)
    plt.show()


def print_cutoff_changes(sampleSet, cutoffList=None, figSaveDirc=None, blackList=None):

    def get_cutoff_seq_counts(compositions, cutoff):
        compNew = compositions[compositions >= cutoff]
        return [len(compNew), np.sum(compNew)]

    fig, axes = plt.subplots(7, 4, figsize=[12, 16])
    if not cutoffList:
        cutoffList = [1, 5, 10, 20, 50, 100, 500, 1000, 10000, 100000, 200000]
    for ix in range(len(sampleSet)):
        ax = axes[ix % 7, int(ix / 7)]
        compositions = sampleSet[ix].get_seq_composition(fraction=False, blackList=blackList)
        cutoffCounts = np.array([get_cutoff_seq_counts(compositions, cutoff) for cutoff in cutoffList]).T
        ax.plot(cutoffList, cutoffCounts[0], 'o-', color='#2C73B4')
        ax.plot(cutoffList, cutoffCounts[1], 'o-', color='#FC820D')
        ax.set_title(sampleSet[ix].id)
        ax.set_yscale('log')
    fig.text(s='Minimal counts of unique sequences', x=0.5, y=0, ha='center', va='top', fontsize=16)
    fig.text(s='Number of total sequences (Orange)\n Number of unique sequences (Blue)',
             x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    plt.tight_layout()
    if figSaveDirc:
        fig.savefig(figSaveDirc, dpi=300)
    plt.show()

def read_count_file(dirc):
    """
    utility function to read the count files
    :param dirc:
    :param sampleType:
    :return:
    """
    sample = {
        'uniqueSeq': 0,
        'totalSeq': 0,
        'seqs': {}
    }
    with open(dirc, 'r') as file:
        sample['uniqueSeq'] = int([elem for elem in next(file).strip().split()][-1])
        sample['totalSeq'] = int([elem for elem in next(file).strip().split()][-1])
        next(file)
        for line in file:
            seq = line.strip().split()
            sample['seqs'][seq[0]] = int(seq[1])
    return sample


def plot_std_peak_dist(sampleSet, norm=True, maxDist=15):

    markerList = ['-o', '->', '-+', '-s']  # different marker for different replicates
    colorList = ['#FC820D', '#2C73B4', '#1C7725', '#B2112A', '#70C7C7', '#810080', '#AEAEAE']  # different color for different type of samples
    symbolList = []
    for marker in markerList:
        symbolList += [marker for i in range(7)]

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=[24, 8])

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsNormed = counts/counts[0]
        ax[0].plot([i for i in range(maxDist + 1)], countsNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)

    # add binomial distribution guide line
    pList = [0.1, 0.01, 0.001, 0.0005, 0.0001]
    from scipy.stats import binom
    for p in pList:
        rv = binom(21, p)
        pmfs = np.array([rv.pmf(x) for x in range(7)])
        pmfsNormed = pmfs/pmfs[0]
        ax[0].plot([i for i in range(7)], pmfsNormed, color='k', ls = '--', alpha=0.3)
    ax[0].text(s='p=0.1', x=6, y=1e-1, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.01', x=6, y=1e-6, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.001', x=3.8, y=5e-7, ha='left', va='center', fontsize=12)
    ax[0].text(s='p=0.0001', x=0.8, y=3e-7, ha='left', va='center', fontsize=12)

    ax[0].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-7, 5])
    ax[0].tick_params(labelsize=12)
    ax[0].set_ylabel('Sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)

    for sampleIx, sample in enumerate(sampleSet):
        counts = sample.stdCounts[:maxDist+1]
        countsAccumulated = np.array([np.sum(counts[:i+1]) for i in range(maxDist + 1)])
        countsAccumulatedNormed = countsAccumulated/countsAccumulated[0]
        ax[1].plot([i for i in range(maxDist + 1)], countsAccumulatedNormed,
                   symbolList[sampleIx], color=colorList[sampleIx % 7],
                   label=sample.id, alpha=0.5)
    ax[1].set_xlabel('Edit distance to spike-in sequence', fontsize=16)
    ax[1].tick_params(labelsize=12)
    ax[1].set_ylim([0.8, 1.5])
    ax[1].set_ylabel('Accumulated sequence counts\n(normalized on exact spike-in sequence)', fontsize=16)
    ax[1].legend(loc=[1.02, 0], fontsize=14, frameon=False, ncol=2)
    plt.tight_layout()
    # fig.savefig('/home/yuning/Work/ribozyme_pred/fig/extStdErr.jpeg', dpi=300)
    plt.show()

def get_sequence_cluster(sample, center, maxDist=2):
    import Levenshtein
    return {seq[0]:seq[1] for seq in sample.seqs.items() if Levenshtein.distance(center, seq[0]) <= maxDist}