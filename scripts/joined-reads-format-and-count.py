"""
Created on Feb 11, 2017

@author: abepres
"""
from __future__ import print_function
import Levenshtein
import argparse

# Requires the Levenshtein package

# Hey y'all, here's Abe's script for turning Illumina files into lists of what sequences are present and by how much.
# This is a fairly hectic script! The standard way to run it is to find and count all subsequences between two flanking
# regions, which should look like:
# 'input_path/file' 'output_path/file' -o count -u +'upstream_seq' -d +'downstream_seq'
# where 'upstream_seq' is e.g. a forward primer, with a '+' as the first character,
# and 'downstream_seq' is e.g. the reverse complement of a reverse primer, with a '+' as the first character


def main():
    parser = argparse.ArgumentParser(
        description="remove sequences or subsequences (primers, etc) based on various criteria, "
                    "and/or convert sequence file formats")
    parser.add_argument("input", help="input file location/name")
    parser.add_argument("output", help="output file location/name")
    parser.add_argument("-i", "--inType", default=["automatic"], nargs='+',
                        help="set input file type; default attempts to read file extension, and if that fails assumes "
                             "a text file of single-sequence lines; if .tab, .tsv, or .csv, "
                             "a second argument is encouraged, e.g. -i tsv fasta")
    parser.add_argument("-o", "--outType", default=["automatic"], nargs='+',
                        help="set output file type; default is to use the same type as input file type; "
                             "if .tab, .tsv, or .csv, a second argument is encouraged, e.g. -o tsv fasta")

    parser.add_argument("-u", "--removeUpstream", nargs='+',
                        help='remove all nucleotides upstream of a specific string; first argument is either a file '
                             'location or a direct sequence denoted as e.g. +ACTG; optional arguments are "hamming" '
                             'or "edit", which sets distance type, optionally followed by an integer maximum distance, '
                             'and "include" which retains the upstream sequence (instead of trimming it to); '
                             'defaults to hamming distance of 0, and no inclusion')
    parser.add_argument("-d", "--removeDownstream", nargs='+',
                        help='remove all nucleotides downstream of a specific string; first argument is either a file '
                             'location or a direct sequence denoted as e.g. +ACTG; optional arguments are "hamming" '
                             'or "edit", which sets distance type, optionally followed by an integer maximum distance, '
                             'and "include" which retains the downstream sequence (instead of trimming it to); '
                             'defaults to hamming distance of 0, and no inclusion')
    parser.add_argument("-k", "--keepMatches", nargs='+',
                        help='keep only sequences containing a substring, first argument is either a file location '
                             'or a direct sequence denoted as e.g. +ACTG; '
                             'optional arguments are "hamming" or "edit", which sets distance type, '
                             'optionally followed by an integer maximum distance; defaults to hamming distance of 0')
    parser.add_argument("-r", "--removeMatches", nargs='+',
                        help='remove only sequences containing a substring, first argument is either a file location '
                             'or a direct sequence denoted as e.g. +ACTG; optional arguments are "hamming" or "edit", '
                             'which sets distance type, optionally followed by an integer maximum distance; defaults '
                             'to hamming distance of 0')
    parser.add_argument("--minRead", type=int,
                        help='filter sequences by minimum (read) length before all other processing')
    parser.add_argument("--maxRead", type=int,
                        help='filter sequences by maximum (read) length before all other processing')
    parser.add_argument("--min", type=int,
                        help='filter sequences by minimum length after all other processing')
    parser.add_argument("--max", type=int,
                        help='filter sequences by maximum length after all other processing')
    parser.add_argument("-t", "--translate", action='store_true',
                        help='convert DNA/RNA sequences into amino acids')
    parser.add_argument("-p", "--placeHolder", default='',
                        help='give a placeholder character, e.g. to keep sequences lined up for joining')
    parser.add_argument("-s", "--startChars", action='store_true',
                        help='enforce line start characters (to account for unexpected line breaks in e.g. fasta files)')
    # (currently does nothing)

    args = parser.parse_args()

    operationList = []

    if args.minRead:
        operationList.append(LengthFilter('min', args.minRead))

    if args.maxRead:
        operationList.append(LengthFilter('max', args.maxRead))

    if args.removeUpstream:

        upSeq = ''

        # if the first character in the argument is a + we handle it as an argument instead of loading a file
        if args.removeUpstream[0][0] == '+':
            # this works even if they forget to put a quotation mark at the end
            upSeq = args.removeUpstream[0][1:]
        else:

            with open(args.removeUpstream[0]) as upFile:
                upFileRead = next(upFile)

            if upFileRead == '':
                raise ValueError(
                    'Upstream sequence file should contain one line of sequences, but is empty')

            upSeq = upFileRead

        distType = 'hamming'
        distMax = 0
        includeThis = False

        for arg in args.removeUpstream[1:]:
            if arg == 'edit':
                distType = 'edit'
            elif arg == 'hamming':
                distType = 'hamming'
            elif arg == 'include':
                includeThis = True
            elif arg.isdigit():
                distMax = int(arg)
            else:
                raise ValueError(
                    'Argument flag ' + arg + ' is not recognized. Note that a sequence or sequence location '
                                             'should only be the first argument following -u')

        operationList.append( EndFilter('upstream', distType, upSeq, errors=distMax, include=includeThis))

    if args.removeDownstream:

        downSeq = ''

        if args.removeDownstream[0][0] == '+':
            # this works even if they forget to put a quotation mark at the end
            downSeq = args.removeDownstream[0][1:]
        else:
            with open(args.removeDownstream[0]) as downFile:
                downFileRead = next(downFile)

            if downFileRead == '':
                raise ValueError(
                    'Downstream sequence file should contain one line of sequences, but is empty')

            downSeq = downFileRead

        distType = 'hamming'
        distMax = 0
        includeThis = False

        for arg in args.removeDownstream[1:]:
            if arg == 'edit':
                distType = 'edit'
            elif arg == 'hamming':
                distType = 'hamming'
            elif arg == 'include':
                includeThis = True
            elif arg.isdigit():
                distMax = int(arg)
            else:
                raise ValueError('Argument flag ' + arg + ' is not recognized. Note that a sequence or '
                                                          'sequence location should only be the first '
                                                          'argument following -u')

        operationList.append(EndFilter(
            'downstream', distType, downSeq, errors=distMax, include=includeThis))

    if args.keepMatches:

        keepList = []

        if args.keepMatches[0][0] == '+':
            keepList.append(args.keepMatches[0][1:])
        else:
            with open(args.keepMatches[0]) as keepFile:
                for seq in keepFile:
                    keepList.append(seq)

            if keepList == []:
                raise ValueError(
                    'Keep sequence file should contain one line of sequences, but is empty')

        distType = 'hamming'
        distMax = 0

        for arg in args.keepMatches[1:]:
            if arg == 'edit':
                distType = 'edit'
            elif arg == 'hamming':
                distType = 'hamming'
            elif arg.isdigit():
                distMax = int(arg)
            else:
                raise ValueError(
                    'Argument flag ' + arg + ' is not recognized. Note that a sequence or sequence location '
                                             'should only be the first argument following -u')

        for seq in keepList:
            operationList.append(SubseqFilter(
                'keep', distType, seq, errors=distMax))

    if args.removeMatches:

        removeList = []

        if args.removeMatches[0][0] == '+':
            removeList.append(args.removeMatches[0][1:])
        else:

            with open(args.removeMatches[0]) as removeFile:
                for seq in removeFile:
                    removeList.append(seq)

            if removeList == []:
                raise ValueError('Remove sequence file should contain one line of sequences, but is empty')

        distType = 'hamming'
        distMax = 0

        for arg in args.removeMatches[1:]:
            if arg == 'edit':
                distType = 'edit'
            elif arg == 'hamming':
                distType = 'hamming'
            elif arg.isdigit():
                distMax = int(arg)
            else:
                raise ValueError(
                    'Argument flag ' + arg + ' is not recognized. Note that a sequence or sequence location '
                                             'should only be the first argument following -u')

        for seq in removeList:
            operationList.append(SubseqFilter(
                'remove', distType, seq, errors=distMax))

    if args.min:
        operationList.append(LengthFilter('min', args.min))

    if args.max:
        operationList.append(LengthFilter('max', args.max))

    if args.translate:
        operationList.append(AminoTranslator())

    # below, parsing input/output instructions

    inType = 'automatic'
    inTabular = False
    inSpacer = ''

    if args.inType[0] != 'automatic':
        if (args.inType[0] == 'tab') or (args.inType[0] == 'tsv') or (args.inType[0] == 'tabular'):
            inTabular = True
            inSpacer = '\t'
            if len(args.inType) > 1:
                inType = args.inType[1]
        elif args.inType[0] == 'csv':
            inTabular = True
            inSpacer = ','
            if len(args.inType) > 1:
                inType = args.inType[1]
        else:
            inType = args.inType[0]
    else:
        if (args.input.split('.')[-1] == 'tab') or (args.input.split('.')[-1] == 'tsv') or (args.input.split('.')[-1] == 'tabular'):
            inTabular = True
            inSpacer = '\t'
        elif args.input.split('.')[-1] == 'csv':
            inTabular = True
            inSpacer = ','

    outType = 'automatic'
    outTabular = False
    outSpacer = ''
    countEm = False

    if args.outType[0] == 'count':
        countEm = True
    elif args.outType[0] != 'automatic':
        if (args.outType[0] == 'tab') or (args.outType[0] == 'tsv') or (args.inType[0] == 'tabular'):
            outTabular = True
            outSpacer = '\t'
            if len(args.outType) > 1:
                outType = args.outType[1]
        elif args.outType[0] == 'csv':
            outTabular = True
            outSpacer = ','
            if len(args.inType) > 1:
                outType = args.outType[1]
        else:
            outType = args.outType[0]

        operationList.append(FormatConverter(
            outType, tabular=outTabular, columnChar=outSpacer))
    else:
        if inTabular:
            # if automatic, we preserve tabular
            operationList.append(FormatConverter(
                outType, tabular=True, columnChar=inSpacer))

    outFileName = args.output

    if len(outFileName.split('.')) == 1:
        # cleaning up file names
        if outType == 'automatic':
            if (inType == 'automatic' and len(args.input.split('.')) > 1):
                outFileName = outFileName + '.' + args.input.split('.')[-1]
        else:
            outFileName = outFileName + '.' + args.outType[0]

    extractAndPrint(args.input, outFileName, operationList, fileTypeIn=inType, deletionMarker=args.placeHolder,
                    tabular=inTabular, columnChar=inSpacer, keyChars=args.startChars, countSeqs=countEm)


# function purpose: trim sequences (extracting primers, deleting sequences containing or lacking subsequences,
# all-in-one as a combined multipurpose function

def extractAndPrint(inFileName, outFileName, operations, fileTypeIn='seqs', deletionMarker='', tabular=False,
                    columnChar='\t', keyChars=False, countSeqs=False):
    # To save on memory, we'll be reading + writing one line of our output file at a time
    # deletionMarker: do we delete sequences that don't fit our parameters, or replace them with a dummy
    # sequence for joining?

    if fileTypeIn == 'automatic':
        if len(fileTypeIn.split('.')) > 0:
            fileTypeIn = inFileName.split('.')[1]
        else:
            fileTypeIn = 'seqs'

    lineStarts = ['']
    seqLine = 0
    truncLines = [0]
    titleLines = []

    totalSeqs = 0
    uniqSeqs = 0
    seqDict = {}

    # for each file type, we set which lines are kept, used as the "sequence," or truncated
    # seqs file is the default type consisting of simply a list of sequences
    # other file types, like tsv and csv, behave like a "seqs" file unless otherwise noted

    if fileTypeIn == 'fastq':
        lineStarts = ['@', '', '+', '']
        seqLine = 1
        truncLines = [1, 3]
        titleLines = [0, 2]

    if fileTypeIn == 'fasta':
        lineStarts = ['>', '']
        seqLine = 1
        truncLines = [1]
        titleLines = [0]

    if tabular:
        lineStarts = ['']*len(lineStarts)

    sequence_reader = read_sequence(
        inFileName, lineStarts, tabular=tabular, columnChar=columnChar, keyChars=keyChars)
    # initializes the generator and read buffer

    linesIn = next(sequence_reader)
    if tabular:
        numTabs = len(linesIn)  # necessary for handling tabulars
        lineStarts = ['']*numTabs
        if numTabs == 4:  # FASTq kludge; need to handle better eventually
            seqLine = 1
            truncLines = [1, 3]
            titleLines = [0, 2]
        elif numTabs == 2:
            seqLine = 1
            truncLines = [1]
            titleLines = [0]
        else:
            seqLine = 0
            truncLines = [0]

    linesOut = []

    with open(outFileName, 'w') as outFile:

        while linesIn[seqLine]:
            # will break the while when giving an empty string
            linesOut = linesIn[:]

            for op in operations:
                #print (linesOut, seqLine, truncLines, titleLines, lineStarts)
                linesOut = op.operate(
                    linesOut, seqLine, truncLines, titleLines, lineStarts)
                # operations are the things we're doing to our sequences

            if countSeqs:
                totalSeqs += 1
                if linesOut[0] != '':
                    if len(linesOut[0].split("N")) < 2:
                        if linesOut[0] in seqDict:
                            seqDict[linesOut[0]] += 1
                        else:
                            seqDict[linesOut[0]] = 1
                            uniqSeqs += 1

            else:
                if deletionMarker != '':
                    if linesOut[seqLine] == '':
                        for i in truncLines:
                            linesOut[i] = deletionMarker

                elif len(linesOut) == 1:
                    if linesOut[0] != '':
                        for line in linesOut:
                            outFile.write(line + '\n')

                elif linesOut[seqLine] != '':
                    for line in linesOut:
                        outFile.write(line + '\n')

            linesIn = next(sequence_reader, ['']*len(lineStarts))
            # returns None if file is exhausted

        if countSeqs:
            counter2 = 0
            print('hello world')
            allSeqs = []

            for seq in seqDict:
                counter2 += 1
                if counter2 % 100000 == 0:
                    print(counter2)
                allSeqs.append((seq, seqDict[seq]))

            print('goodbye world?')
            sortSeqs = sorted(allSeqs, key=lambda tup: tup[1], reverse=True)

            print('well this is awkward')

            outFile.write('number of unique sequences = ' +
                          str(uniqSeqs) + '\n')
            outFile.write('total number of molecules = ' +
                          str(totalSeqs) + '\n' + '\n')

            print(allSeqs[0][0])
            print(allSeqs[0])
            print(allSeqs[1])
            print(allSeqs[10])
            lineLen = len(allSeqs[0][0]) + 20

            for seqCount in sortSeqs:
                outFile.write(
                    seqCount[0] + ' '*max(1, lineLen-len(seqCount[0])) + str(seqCount[1]) + '\n')

    # note: we assume that a blank line means end of file. This means our files
    # can't have any blank lines in them!


class LengthFilter(object):
    # a class to store metadata for our sequence operations, for ease of readability

    def __init__(self, min_max, length):
        self.operation = min_max
        self.length = length

    def operate(self, lines, seqLine, truncLines, titleLines, *args):
        if self.operation == 'min':
            if len(lines[seqLine]) < self.length:
                for i in truncLines:
                    # creates blank sequences if deletion occurs
                    lines[i] = ''

        elif self.operation == 'max':
            if len(lines[seqLine]) > self.length:
                for i in truncLines:
                    # creates blank sequences if deletion occurs
                    lines[i] = ''

        else:
            raise ValueError(
                'Inappropriate operation name; should be min or max')

        return lines


class AminoTranslator(object):
    # a class to store metadata for our sequence operations, for ease of readability

    def __init__(self):
        self.operation = 'translate'
        self.codeDict = standard_translation()

    def operate(self, lines, seqLine, truncLines, titleLines):

        lines[seqLine] = translate(lines[seqLine], self.codeDict)

        return lines


class EndFilter(object):

    def __init__(self, upstream_downstream, edit_hamming, subseq, errors=0, include=False):
        self.operation = upstream_downstream
        self.distance = edit_hamming
        self.includeSelf = include
        self.errors = errors

        if upstream_downstream == 'downstream':
            self.subseq = subseq[::-1]
        else:
            self.subseq = subseq

    def operate(self, lines, seqLine, truncLines, titleLines, lineStarts):

        if self.operation == 'downstream':

            for i in truncLines:
                lines[i] = lines[i][::-1]
            # here we invert the sequence and subsequence, to be de-inverted later

        subsSearch = subsequence_search(
            lines[seqLine], self.subseq, self.distance, self.errors)

        if subsSearch[0]:
            # only if the subsequence is found
            if self.includeSelf:
                slicePoint = subsSearch[1][0]
            else:
                slicePoint = subsSearch[1][1]

            for i in truncLines:
                lines[i] = lines[i][slicePoint:]

            if self.operation == 'downstream':
                for i in truncLines:
                    lines[i] = lines[i][::-1]

            return lines

        else:
            for i in truncLines:
                # creates blank sequences if deletion occurs
                lines[i] = ''
            return lines


class SubseqFilter(object):

    def __init__(self, keep_remove, edit_hamming, subseq, errors=0):
        self.operation = keep_remove
        self.distance = edit_hamming
        self.subseq = subseq
        self.errors = errors

    def operate(self, lines, seqLine, truncLines, titleLines, lineStarts):
        subsSearch = subsequence_search(
            lines[seqLine], self.subseq, self.distance, self.errors)

        if self.operation == 'keep':
            if subsSearch[0]:

                return lines
        elif self.operation == 'remove':
            if subsSearch[0] == False:
                return lines
        else:
            raise ValueError(
                'Inappropriate operation name; should be keep or remove')

        for i in truncLines:
            # creates blank sequences if deletion occurs
            lines[i] = ''
        return lines


class FormatConverter(object):
    # unlike the other operations, this one can change which lines are title/truncation lines
    # and should be the LAST operation called in a list of operations

    def __init__(self, outFormat, tabular=False, columnChar='\t'):
        self.operation = 'convert'
        self.outFormat = outFormat
        self.tabular = tabular
        self.columnChar = columnChar

    def operate(self, lines, seqLine, truncLines, titleLines, lineStarts):

        for i in range(len(lines)):
            if len(lines[i]) > 0:

                if lines[i][0] == lineStarts[i]:
                    # remove special characters for format conversion
                    lines[i] = lines[i][1:]

        if self.outFormat == 'fasta':

            linesOut = ['']*2

            if len(titleLines) > 0:
                linesOut[0] = '>' + lines[titleLines[0]]
                linesOut[1] = lines[seqLine]
            else:
                linesOut[0] = '>Sequence_Title'
                linesOut[1] = lines[seqLine]

        elif self.outFormat == 'fastq':
            linesOut = ['']*4

            if len(titleLines) > 1:
                linesOut[0] = '@' + lines[titleLines[0]]
                linesOut[2] = '+' + lines[titleLines[1]]
            elif len(titleLines) == 1:
                linesOut[0] = '@' + lines[titleLines[0]]
                linesOut[2] = '+' + lines[titleLines[0]]
            else:
                linesOut[0] = '@Sequence_Title'
                linesOut[2] = '+Sequence_Title'

            if len(truncLines) > 1:
                linesOut[1] = lines[seqLine]

                if truncLines[1] != seqLine:
                    linesOut[3] = lines[truncLines[1]]

            elif len(truncLines) == 1:
                linesOut[1] = lines[seqLine]
                length = len(lines[seqLine])
                linesOut[3] = 'FakeQualityScore' * \
                    (length/16)+'FakeQualityScore'[:(length % 16)]

            else:
                raise ValueError(
                    'Inappropriate output type; currently supported: fasta, fastq, seqs')

        else:
            linesOut = [lines[seqLine]]

        if self.tabular:
            if self.outFormat == 'fasta':
                linesOut[0] = linesOut[0][1:]
            if self.outFormat == 'fastq':
                linesOut[0] = linesOut[0][1:]
                linesOut[2] = linesOut[2][1:]

            singleLine = linesOut[0]

            for i in range(1, len(linesOut)):
                singleLine += self.columnChar
                singleLine += linesOut[i]

        return linesOut


def read_sequence(inFileName, lineStarts, tabular=False, columnChar='\t', keyChars=False, skipHeader=0, skipChars=[]):
    # This subroutine reads a different number of lines based on requested file type
    # output is itself a list of lines, used to reconstruct a FASTX file if necessary

    # planned/accepted file types are fasta, fastq, csv, and seqs
    # (seqs simply a txt file consisting of a list of seqs)

    # inFileLoc is a file handler

    # yield instead of return
    # readline
    try:
        inFile = open(inFileName)

    except IOError:
        print("Could not read file: {}".format(inFileName))

    nextStart = []
    numLines = len(lineStarts)

    for i in range(numLines):
        nextStart.append(lineStarts[(i+1) % numLines])
        # reshuffles the list to correspond to the expected "next" start char, so we can peek ahead

    with inFile:
        for i in range(0, skipHeader):
            next(inFile)

        while True:
            seqLines = ['']*numLines
            if tabular:
                cols = next(inFile).split(columnChar)
                seqLines = []
                for col in cols:
                    if col != '':
                        seqLines.append(col.split('\n')[0])
            else:
                for i in range(numLines):
                    seqLines[i] = next(inFile).split('\n')[0]

                '''   if keyChars:
                        #do we require key lines to start with key characters?
                
                        if nextStart[i] != '':
                                                        
                            pos = inFile.tell()
                            peek = next(inFile).split('\n')[0]
                            
                            
                            while peek[0] != nextStart[i]:
                                #tests if the next first character is correct
                                seqLines[i]+=peek
                                pos = inFile.tell()
                                peek = next(inFile).split('\n')[0]
                                
                            inFile.seek(pos)
                            
                            (*errors were occurring with next start line check functionality; deprecated for now*)
               '''

            yield seqLines


def subsequence_search(sequence, subsequence, distanceType='hamming', maxErrors=0, matchLength=0):
    # we can use one function to search for primers and internal functions using the same algorithm;
    # we can search LtR or RtL, though in the case of internal sequences we don't care either way

    # this function simply gives the locations of matching substrings; other functions will remove or otherwise handle them

    # it may be possible to optimize this into L1*L2 time with a better dynamic programming array to match substrings

    seqLength = len(sequence)
    subLength = len(subsequence)

    '''
    #Planned extension will allow partial matches
    partialMatch = True' 
    if matchLength == 0:
        matchLength = subLength
        partialMatch = False
        #setting matchLength to zero means we want to match the whole subsequence (else, we allow partial matches)
    '''
    matchLength = subLength

    searchPos = range(seqLength-subLength)
    # if direction == 'RtL'
    #    searchPos.reverse()
    # currently only LtR searching is implemented/necessary, though we leave space to optionally add this later

    matchFound = False
    fewestErrors = maxErrors + 1
    # note that we're looking for the *first best* appearance of the subsequence, not necessarily the best
    # (finding a match will mean we only prefer future matches with *better* distance, letting us "slide"
    # the window specifically from left to right)

    startPos = 0
    endPos = 0

    # this is important in case we need to pass to another function the *position* of our match;
    # we'll pass back both end positions of our subsequence, for other functions to handle

    if distanceType == 'hamming':
        for position in searchPos:
            numErrors = Levenshtein.hamming(
                subsequence, sequence[position:position+matchLength])
            if numErrors < fewestErrors:
                matchFound = True
                fewestErrors = numErrors
                startPos = position
                endPos = position+matchLength

                '''
                Part of planned extension to allow a shorter subsequence than that given
                for slidingEnd in range (position+matchLength,position+subLength+1):
                    if numErrors == Levenshtein.hamming(subsequence,sequence(position:slidingEnd)):
                        endPos = slidingEnd
                    
                
                #Hamming distance *can* match to the minimum match length, but we extend it all the way to subsequence
                #length as long as it doesn't increase (and halt it as soon as it does)
                '''

    elif distanceType == 'edit':

        flexWindow = range(matchLength-maxErrors, matchLength+maxErrors)

        for position in searchPos:
            for window in flexWindow:
                numErrors = Levenshtein.distance(
                    subsequence, sequence[position:position+window])
                if numErrors < fewestErrors:
                    matchFound = True
                    fewestErrors = numErrors
                    startPos = position
                    endPos = position+window

    else:
        raise ValueError('distance type should be edit or hamming')

    return (matchFound, (startPos, endPos))
    # function returns whether or not a match is found, and if so, what segment it corresponds to

    # Note: we prefer the last nucleotide, if mismatched, to be a deletion
    # rather than a replacement, and will thus prefer shorter matches with the same
    # number of errors


'''
    #The original Levenshtein comparator, while algorithmically faster, also
    #makes mistakes in a small minority of cases. A better comparator has been used,
    #but the original code (somewhat unfinished at the end) included
    #in this big comment in case it becomes useful later
    
    changeOps = Levenshtein.editops(subsequence,sequence[position:position+subLength+maxErrors])
    
    #Known bugs: difficulty precisely counting errors in insertions among sequences that end with repeated chars
    #(e.g. 'ACTGAA' and 'ACTGTAAA' counts an extra error by mistake)
    
    
    #Future extension for smaller sub-matches?
    droppedLength = 0
    #"dropped" is a temporary, adjustible length window. We can remove nucleotides from the beginning
    #of our subsequence; doing so, however, increases the point at which the last error is allowed to occur,
    #up to a maximum of subLength (the length of the subsequence)
    #Effectively, this lets us ignore up to the first n nucleotides of our subsequence with out a mismatch penalty,
    #where n is the difference between subsequence length and minimum match window length
    
    #Known bugs: this doesn't search for the "perfect" substring, just a "good enough" one?
    
    for i in range (subLength-matchLength):
        if len(changes) > 0:
            if changes[0] == ('delete', 1, 0):
                changes = changes[1:]
                droppedLength += 1
'''
'''
    removeChanges = True
    
    while removeChanges:
        if changeOps[-1][1] >= subLength:
            #only triggers on change operations resulting from extraneous
            #nucleotides at the end of the sequence, which are not in the subsequence
            changeOps = changeOps[:-1]
        else:
            removeChanges = False
    
    if len(changeOps) == 0:
        matchFound = True
        startPos = position
        endPos = seqLength
        
    elif len(changeOps) < fewestErrors:
        matchFound = True
        
        #if the last edit operation is a replace or delete, at the last position,
        #we don't need to excise that nucleotide
        startPos = position
        endPos = startPos + subLength
                        
        #We use this to track the "end" of the sequence; +1 for insertion, -1 for deletion
        #and -1 for replacement only if we're at the "end"
        
        removeEndMismatch = True 
        
        while removeEndMismatch:
            #first, we pull off and count all changes at the "end"
            if len(changeOps) > 0:
                if changeOps[-1][0] != 'insert':
                    if changeOps[-1][1] == endPos-1:
                        changeOps = changeOps[:-1]
                        endPos -= 1
                    else removeEndMismatch = False
                else removeEndMismatch = False
            else removeEndMismatch = False
        
        for change in changeOps:
            #now we count all the remaining changes' effect on sequence end point
            if change[0] == 'insert':
                endPos += 1
            if change[0] == 'delete':
                endPos -= 1

        
    elif changes[maxErrors][0]<

'''


if __name__ == "__main__":
    main()
