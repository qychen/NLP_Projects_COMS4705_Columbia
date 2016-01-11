UNI: qc2200  name: Qianyuan Chen

PA.1)
UNIGRAM natural -13.766408817
BIGRAM natural that -4.05889368905
TRIGRAM natural that he -1.58496250072

PA.2)
The perplexity is 1052.4865859
The perplexity is 53.8984761198
The perplexity is 5.7106793082

PA.3)
The perplexity is 12.5516094886

PA.4)
Yes. Since we score the model using training data here, the Trigram model performs much better than the Unigram model, but they are treated equally when doing the linear interpolation, so the result is worse than trigram. Linear interpolation is expected to have better performance when there’re more non-exist N-grams, which will decrease the performance of trigram model.

PA.5)
The perplexity is 11.1670289158
The perplexity is 1627571078.54

Sample1 belongs to the Brown dataset and Sample2 does not, since the model performs much better in Sample1 than Sample2.



PB.2)
TRIGRAM CONJ ADV ADP -2.9755173148
TRIGRAM DET NOUN NUM -8.9700526163
TRIGRAM NOUN PRT PRON -11.0854724592

PB.4)
* * 0.0
Night NOUN -13.8819025994
Place VERB -15.4538814891
prime ADJ -10.6948327183
STOP STOP 0.0
_RARE_ VERB -3.17732085089

PB.5)
Percent correct tags: 93.3249946254

PB.6)
Percent correct tags: 87.9985146677


Part A time: 14.61 sec
Part B time: 153.94 sec
