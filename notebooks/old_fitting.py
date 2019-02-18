import numpy as np
from ribo import load_pickle, dump_pickle
from k_seq import data_fitting

if __name__=='__main__':
    countTable = load_pickle('/home/yuning/Work/ribozyme_pred/data/k_seq/labeled_count_table_non_empty_in_r5_INT_STD.pkl')
    print('Data table loaded, %i sequences to be fitted...' %len(countTable))    
    rndsToAvg = [[1, 2, 3, 4, 5, 6],
                 [7, 8, 10, 11, 12],
                 [13, 14, 15],
                 [19, 20, 21, 23]]
    devEstimate = True
    rndsToCount = [rnd for rndBatch in rndsToAvg for rnd in rndBatch]
   
    fittedSeqs = data_fitting(countTable, rndsToAvg, devEstimate=devEstimate, eqnType=2)
    
    if devEstimate:
        dataToDump = [[seq[0], seq[1][0], seq[3], sum([1 for rnd in rndsToCount if seq[1][rnd]>0]),
                       seq[4], seq[5], seq[6], seq[7], seq[8], seq[9], seq[10]] for seq in fittedSeqs]
    else:
        dataToDump = [[seq[0], seq[1][0], seq[3], sum([1 for rnd in rndsToCount if seq[1][rnd]>0]),
                       seq[4], seq[5], seq[6]] for seq in fittedSeqs]
    outDirc = '/home/yuning/Work/ribozyme_pred/data/k_seq/fitted_non_empty_r5_INT_STD_excNone_raw.csv'
    dump_pickle(dataToDump, outDirc)
    print('Fitted data has been saved to %s' %outDirc)
