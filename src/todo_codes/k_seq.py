'''
This module contain methods for k-seq calculation
'''




def get_seqToFit(initDirc, sampleDircList, validSampleList):
    """
    This function generate a list of sequences that will pass to estimator
    Three steps: 1) create candidate list; 2) align each sample to candidate list; 3) select sequences with interests
    :param initDirc: directory to the count file of initial pool
    :param sampleDircList: FULL list of directories to all k-seq samples
    :param validSampleList: a list of ids of valid samples that will be considered in this function, PlEASE NOTE: 1) the
    id MUST coordinate with the order of samples in sampleDircList, starting from 1; 2) the list should format in
    replicate sets:
    [[replicates in time/concentration point 1],
     [replicates in time/concentration point 2],
     ...]
    :return seqToFit: a list of sequences will be used in estimator
    """

    import numpy as np

    def read_initPool(initDirc):
        '''
        Read the sequences detected in the initial pool, and return a dictionary of sequences
        :param initDirc: directory to the count file of initial pool
        :return: candidList: a dictionary of all sequences detected in the initial pool
        '''
        candidList = {}
        with open(initDirc) as file:
            next(file)
            next(file)
            next(file)
            for line in file:
                seq = line.strip().split()
                candidList[seq[0]] = {
                    'rawCounts': [int(seq[1])] + [np.nan for i in sampleDircList]
                }
        print('Candidate sequences imported from initial pool')
        return candidList

    def align_samples(candidList, sampleDircList, validSampleList):
        """
        Survey sequence counts in each k-seq sample and add raw counts of sequences in candidList to the list
        :param candidList: sequences detected in initial pool
        :param sampleDircList: directories to the k-seq sample count file
        :param validSampleList: A list of valid samples
        :return: an updated candidList with raw counts from valid k-seq samples
        """
        validSampleList = [rep for repSet in validSampleList for rep in repSet]
        for sampleIx, sampleDirc in enumerate(sampleDircList):
            if (sampleIx + 1) in validSampleList:
                print('Surveying sequences in sample %s ...' %sampleDirc)
                with open(sampleDirc) as file:
                    next(file)
                    next(file)
                    next(file)
                    for line in file:
                        seq = line.strip().split()
                        if seq[0] in candidList.keys():
                            candidList[seq[0]]['rawCounts'][sampleIx + 1] = int(seq[1])
        return candidList

    def filter_seq(candidList):
        """
        Filter out sequences that are not detected in any valid k-seq samples
        :param candidList
        :return validSeq: a list of valid sequences; number of seqs in original candidList, number of seqs in validSeq
        """
        def get_config(seq):
            return [sum([1 for rep in repSet if ~np.isnan(seq['rawCounts'][rep])]) for repSet in validSampleList]

        validSeq = []
        for seq in candidList.items():
            if np.nansum(seq[1]['rawCounts'][1:]) > 0:
                validSeq.append({
                    'seq': seq[0],
                    'rawCounts': seq[1]['rawCounts'],
                    'config': get_config(seq[1]),
                    'id': len(validSeq)
                })
        return (validSeq, len(candidList), len(validSeq))

    candidList = align_samples(candidList=read_initPool(initDirc),
                               sampleDircList=sampleDircList,
                               validSampleList=validSampleList)

    return filter_seq(candidList)

def get_normalized_fraction(seqToFit, sampleTotals, qFactors):
    """
    :param seqToFit: list of sequences will pass to estimator
    :param sampleTotals: list of total counts of reads in samples
    :param qFactors:
    :return:
    """
    import numpy as np

    for seq in seqToFit:
        frac = np.array(seq['rawCounts'])/sampleTotals*qFactors
        seq['reactedFrac'] = frac/frac[0]
    return seqToFit


