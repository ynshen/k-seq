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

def print_sample_overview(sampleSet, table=False, figures=True):
    if table:
        print('|index|sample name| total counts | unique counts | ext. std. counts | ext. std. percent|')
        print('|:--:|:-----:|:-----:|:----:|:----:|:-----:|')
        for ix,sample in enumerate(sampleSet):
            print('|%i|%s|%i|%i|%i|%.3f|' %(ix+1, sample.id, sample.totalSeq, sample.uniqueSeq, sample.stdCounts[0],
                                            sample.stdCounts[0]/sample.totalSeq))
    if figures:
        ### insert figures for overviewing



root, sampleList = basic_info()