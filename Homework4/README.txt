NAME: Qianyuan Chen   UNI: qc2200

Part A:

3) AER:

IBMModel1: 0.665
IBMModel2: 0.650

Consider this sentence pair:

[u'Ich', u'bitte', u'Sie', u',', u'sich', u'zu', u'einer', u'Schweigeminute', u'zu', u'erheben', u'.']
[u'Please', u'rise', u',', u'then', u',', u'for', u'this', u'minute', u"'", u's', u'silence', u'.']

The alignment results of IBMModel1 and IBMModel2 are the following:

IBMModel1:
0-1 1-1 2-1 3-4 4-10 5-10 6-10 7-10 8-10 9-1
IBMModel2:
0-0 1-1 2-0 4-10 3-2 5-10 6-10 7-7 8-10 9-0
Correct:
0-0 1-0 2-0 10-11 5-5 6-6 4-1 7-7 7-8 8-10 9-10 7-10 3-4 7-9 

The first three words are examples cases of IBMModel2 performs better than Model1. Here 'Ich' and ’Sie' are aligned to their uncommon meanings, so in both models the t(language) probability will be low. But Model2 takes the location information into account, and (0 —> 0) (2 —> 0) are in high probability. So Model2 gives the correct alignments.


4) AER convergence:

IBMModel1: 0.626
IBMModel2: 0.642

The following is the iteration times and corresponding AER. The best iteration times for IBMModel1 and IBMModel2 is 6 and 4 in my implementation. The relationship between the number of iterations and AER is the same for both model: when iteration times increase from 0, the AER decrease until reaching the bottom, then AER slightly rise and fall again when increasing the iteration time, and finally converge after iteration times larger than 40.

IBMModel1:
2:0.684 4:0.630 5:0.627 6:0.626 7:0.629 10:0.665 20:0.661 30:0.660 40:0.657 50:0.658

IBMModel2:
2:0.644 4:0.642 5:0.644 6:0.647 7:0.646 10:0.650 20:0.648 30:0.649 40:0.650 50:0.654



Part B:

4)
AER: 0.550

Let’s take the 20th input sentence as example, the source and target sentence are as follows:

[u'All', u'dies', u'entspricht', u'den', u'Grunds\xe4tzen', u',', u'die', u'wir', u'stets', u'verteidigt', u'haben', u'.']
[u'This', u'is', u'all', u'in', u'accordance', u'with', u'the', u'principles', u'that', u'we', u'have', u'always', u'upheld', u'.']

And the alignment results of IBMModel2 and BerkeleyAligner are:

IBMModel2:
0-12 1-4 2-7 3-4 4-12 5-10 6-10 7-9 8-7  9-12 10-7
BerkeleyAligner:
0-5  1-1 2-2 3-6 4-4  5-6  6-6  7-9 8-7  9-12 10-10 11-13
Correct:
0-2  1-0 2-3 3-6 4-7  5-8  6-8  7-9 8-11 9-12 10-10 11-13 2-1 2-4 2-5

In this case BerkeleyAligner performs better than IBM models. Take a look at 3rd, 6th, 10th words’ alignment in source sentence and these alignments show the reason for the better performance.

The 3rd word 'den' is aligned to 'accordance' in IBM model, which is incorrect, so in this case the q(location) probability may dominate the t(language) probability in IBM model. However, since both 'den' —> 'accordance' and 'den' <— 'accordance' have low probability, in BerkeleyAligner it considers both translation directions and determines that this is a low probability alignment, and thus align 'den' to 'the' which is more reasonable.

The 5&6th words ‘, die' is aligned to 'have' in IBM while to 'the' in Berkeley. Though both are incorrect, latter is more reasonable, since ‘die' —> 'the' and ‘die' <— 'the' are possible translation between source and target language, but ‘die' <—> ‘have’ is quite rare.

The 10th word is similar to 3rd, where 'principles' —> 'haben' is quite rare in English to German. But since IBM model just consider the translation model from one direction, it cannot catch the information like this.

