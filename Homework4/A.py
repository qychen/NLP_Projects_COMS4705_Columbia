import nltk
from nltk.align import IBMModel1
from nltk.align import IBMModel2

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    ibm = IBMModel1(aligned_sents, 10)
    return ibm

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    ibm = IBMModel2(aligned_sents, 10)
    return ibm

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    aer_sum = 0.0
    for sent in aligned_sents[:n]:
        aer_sum += sent.alignment_error_rate(model.align(sent))
    return aer_sum/n

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    f = open(file_name,'w')
    for sent in aligned_sents[:20]:
        res = model.align(sent)
        #s1 = '['+' '.join(res.words)+']\n'
        #s2 = '['+' '.join(res.mots)+']\n'
        f.write(str(res.words)+'\n')
        f.write(str(res.mots)+'\n')
        f.write(str(res.alignment)+'\n')
        #for t in res.alignment:
        #    f.write(str(t[0])+'-'+str(t[1])+' ')
        #f.write('\n')
        #for t in sent.alignment:
        #    f.write(str(t[0])+'-'+str(t[1])+' ')
        #f.write('\n')
        f.write('\n')
    f.close()

def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
