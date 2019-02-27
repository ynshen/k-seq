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
            if dist <= maxDist:
                self.stdCounts[dist] += self.seqs[seq]

    def get_seq_length(self):
        return np.array([len(seq) for seq in self.seqs.keys()])

    def get_seq_fraction(self, blackList=None):
        if blackList:
            totalSeq = self.totalSeq - np.sum([self.seqs[seq] for seq in blackList])
            return (np.array([seq[1]/totalSeq for seq in self.seqs.items() if not(seq[0] in blackList)]), totalSeq)
        return (np.array([seqCount/self.totalSeq for seqCount in self.seqs.values()]), self.totalSeq)


###### FOLLOWS ARE BASIC UTILITY FUNCTIONS ######

def basic_info(countFileRoot='/mnt/storage/projects/k-seq/input/bfo_counts/'):
    from os import listdir
    from os.path import isfile, join

    sampleList = [f for f in listdir(countFileRoot) if isfile(join(countFileRoot, f))]
    sort_fn = lambda s: int(s.split('_')[1][1:])
    sampleList.sort(key=sort_fn)
    return countFileRoot, sampleList


root, sampleList = basic_info()


def load_all_samples(sampleSetDirc=None):
    import util

    if sampleSetDirc:
        sampleSet = util.load_pickle(sampleSetDirc)
    else:
        sampleSet = []
        for sample in sampleList:
            currentSample = Sample()
            currentSample.id = sample
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


def print_length_dist(sampleSet, figSaveDirc=None):

    fig, axes = plt.subplots(7, 4, figsize=[12, 16])
    for ix in range(len(sampleSet)):
        ax = axes[ix % 7, int(ix / 7)]
        lengths = sampleSet[ix].get_seq_length()
        bins = np.linspace(0, np.max(lengths), np.max(lengths) + 1)
        ax.hist(lengths, bins=bins)
        ax.set_yscale('log')
        ax.set_title(sampleSet[ix].id)
    fig.text(s='Sequence length (nt)', x=0.5, y=0, ha='center', va='top', fontsize=16)
    fig.text(s='Number of unique sequences', x=0, y=0.5, ha='right', va='center', fontsize=16, rotation=90)
    plt.tight_layout()
    if figSaveDirc:
        fig.savefig(figSaveDirc, dpi=300)
    plt.show()




