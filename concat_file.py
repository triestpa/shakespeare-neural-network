import itertools

filenames = ['textdatasets/lotr1.txt', 'textdatasets/lotr2.txt', 'textdatasets/lotr3.txt']
with open('textdatasets/lotr_combined.txt', 'w') as outfile:
    for line in itertools.chain.from_iterable(itertools.imap(open, filenames)):
        outfile.write(line)