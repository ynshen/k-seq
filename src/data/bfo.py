import numpy as np

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



###### FOLLOWS ARE BASIC UTILITY FUNCTIONS ######

def basic_info():
    from os import listdir
    from os.path import isfile, join

    root = '/mnt/storage/projects/k-seq/input/bfo_counts/'
    sampleList = [f for f in listdir(root) if isfile(join(root, f))]
    sort_fn = lambda s: int(s.split('_')[1][1:])
    sampleList.sort(key=sort_fn)
    return root, sampleList


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
        print('|index|sample name| total counts | unique counts | ext. std. counts | ext. std. percent|')
        print('|:--:|:-----:|:-----:|:----:|:----:|:-----:|')
        for ix,sample in enumerate(sampleSet):
            print('|%i|%s|%i|%i|%i|%.3f|' %(ix+1, sample.id, sample.totalSeq, sample.uniqueSeq, sample.stdCounts[0],
                                            sample.stdCounts[0]/sample.totalSeq))
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

root, sampleList = basic_info()