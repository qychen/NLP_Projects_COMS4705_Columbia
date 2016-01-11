import nltk
from collections import defaultdict
from nltk.align import Alignment
from nltk.align import AlignedSent
from nltk.align import IBMModel1
import A

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        MIN_PROB = 1.0e-12
        best_alignment = []

        l = len(align_sent.mots)
        m = len(align_sent.words)

        for j, trg_word in enumerate(align_sent.words):
            # Initialize trg_word to align with the NULL token
            best_prob = (self.t[trg_word][None] * self.q[0][j + 1][l][m])
            best_prob = max(best_prob, MIN_PROB)
            best_alignment_point = 0
            for i, src_word in enumerate(align_sent.mots):
                align_prob = (self.t[trg_word][src_word] * self.q[i + 1][j + 1][l][m])
                if align_prob >= best_prob:
                    best_prob = align_prob
                    best_alignment_point = i
            #print j, best_alignment_point
            best_alignment.append((j, best_alignment_point))

        #print align_sent, best_alignment
        new_sent = AlignedSent(align_sent.words, align_sent.mots, Alignment(best_alignment))
        return new_sent
    
    # TODO: Implement the EM algorithm. num_iters is the number of num_iters. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
        MIN_PROB = 1.0e-12

        t = {}
        q = {}
        t = defaultdict(
            lambda: defaultdict(lambda: MIN_PROB))
        q = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: MIN_PROB))))

        #train model1: target --> source
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in aligned_sents:
            trg_vocab.update(aligned_sentence.words)
            src_vocab.update(aligned_sentence.mots)
        # Add the NULL token
        src_vocab.add(None)

        translation_table1 = defaultdict(
            lambda: defaultdict(lambda: MIN_PROB))

        alignment_table1 = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: MIN_PROB))))

        # Initialize translation probability distribution
        #initial_value = 1.0 / len(trg_vocab)
        #for wt in trg_vocab:
        #    for ws in src_vocab:
        #        translation_table1[wt][ws] = initial_value
        ibm1 = IBMModel1(aligned_sents, 6)
        translation_table1 = ibm1.probabilities

        # Initialize alignment probability distribution,
        # a(i | j,l,m) = 1 / (l+1) for all i, j, l, m
        for aligned_sentence in aligned_sents:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            initial_value = 1.0 / (l + 1)
            #!!!
            for i in range(l + 1):
                for j in range(1, m + 1):
                    alignment_table1[i][j][l][m] = initial_value


        for i in range(num_iters):
            count_t_given_s1 = defaultdict(lambda: defaultdict(float))
            count_any_t_given_s1 = defaultdict(float)

            # count of i given j, l, m
            alignment_count1 = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))
            alignment_count_for_any_i1 = defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0)))

            for aligned_sentence in aligned_sents:
                src_sentence = [None] + aligned_sentence.mots
                trg_sentence = ['UNUSED'] + aligned_sentence.words  # 1-indexed
                l = len(aligned_sentence.mots)
                m = len(aligned_sentence.words)
                total_count = defaultdict(float)

                # E step (a): Compute normalization factors to weigh counts
                for j in range(1, m + 1):
                    wt = trg_sentence[j]
                    total_count[wt] = 0
                    for i in range(l + 1):
                        ws = src_sentence[i]
                        #print wt, translation_table1[wt][ws], alignment_table1[i][j][l][m]
                        count = (translation_table1[wt][ws] * alignment_table1[i][j][l][m])
                        total_count[wt] += count

                # E step (b): Collect counts
                for j in range(1, m + 1):
                    wt = trg_sentence[j]
                    for i in range(l + 1):
                        ws = src_sentence[i]
                        count = (translation_table1[wt][ws] * alignment_table1[i][j][l][m])
                        #print total_count
                        normalized_count = count / total_count[wt]

                        count_t_given_s1[wt][ws] += normalized_count
                        count_any_t_given_s1[ws] += normalized_count
                        alignment_count1[i][j][l][m] += normalized_count
                        alignment_count_for_any_i1[j][l][m] += normalized_count

            # M step: Update probabilities with maximum likelihood estimates
            for ws in src_vocab:
                for wt in trg_vocab:
                    estimate = count_t_given_s1[wt][ws] / count_any_t_given_s1[ws]
                    translation_table1[wt][ws] = max(estimate, MIN_PROB)

            for aligned_sentence in aligned_sents:
                l = len(aligned_sentence.mots)
                m = len(aligned_sentence.words)
                for i in range(l + 1):
                    for j in range(1, m + 1):
                        estimate = (alignment_count1[i][j][l][m] / alignment_count_for_any_i1[j][l][m])
                        alignment_table1[i][j][l][m] = max(estimate, MIN_PROB)


        #train model2: source --> target
        src_vocab = set()
        trg_vocab = set()
        for aligned_sentence in aligned_sents:
            trg_vocab.update(aligned_sentence.mots)
            src_vocab.update(aligned_sentence.words)
        # Add the NULL token
        src_vocab.add(None)

        translation_table2 = defaultdict(
            lambda: defaultdict(lambda: MIN_PROB))

        alignment_table2 = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                lambda: MIN_PROB))))

        # Initialize translation probability distribution
        #initial_value = 1.0 / len(trg_vocab)
        #for wt in trg_vocab:
        #    for ws in src_vocab:
        #        translation_table2[wt][ws] = initial_value
        ibm1 = IBMModel1(aligned_sents, 6)
        translation_table2 = ibm1.probabilities


        # Initialize alignment probability distribution,
        # a(i | j,l,m) = 1 / (l+1) for all i, j, l, m
        for aligned_sentence in aligned_sents:
            l = len(aligned_sentence.words)
            m = len(aligned_sentence.mots)
            initial_value = 1.0 / (l + 1)
            #!!!
            for i in range(l + 1):
                for j in range(1, m + 1):
                    alignment_table2[i][j][l][m] = initial_value


        for i in range(num_iters):
            count_t_given_s2 = defaultdict(lambda: defaultdict(float))
            count_any_t_given_s2 = defaultdict(float)

            # count of i given j, l, m
            alignment_count2 = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0))))
            alignment_count_for_any_i2 = defaultdict(
                lambda: defaultdict(lambda: defaultdict(
                    lambda: 0.0)))

            for aligned_sentence in aligned_sents:
                src_sentence = [None] + aligned_sentence.words
                trg_sentence = ['UNUSED'] + aligned_sentence.mots  # 1-indexed
                l = len(aligned_sentence.words)
                m = len(aligned_sentence.mots)
                total_count = defaultdict(float)

                # E step (a): Compute normalization factors to weigh counts
                for j in range(1, m + 1):
                    wt = trg_sentence[j]
                    total_count[wt] = 0
                    for i in range(l + 1):
                        ws = src_sentence[i]
                        count = (translation_table2[wt][ws] * alignment_table2[i][j][l][m])
                        total_count[wt] += count

                # E step (b): Collect counts
                for j in range(1, m + 1):
                    wt = trg_sentence[j]
                    for i in range(l + 1):
                        ws = src_sentence[i]
                        count = (translation_table2[wt][ws] * alignment_table2[i][j][l][m])
                        normalized_count = count / total_count[wt]

                        count_t_given_s2[wt][ws] += normalized_count
                        count_any_t_given_s2[ws] += normalized_count
                        alignment_count2[i][j][l][m] += normalized_count
                        alignment_count_for_any_i2[j][l][m] += normalized_count

            # M step: Update probabilities with maximum likelihood estimates
            for ws in src_vocab:
                for wt in trg_vocab:
                    #print count_any_t_given_s
                    estimate = count_t_given_s2[wt][ws] / count_any_t_given_s2[ws]
                    translation_table2[wt][ws] = max(estimate, MIN_PROB)

            for aligned_sentence in aligned_sents:
                l = len(aligned_sentence.words)
                m = len(aligned_sentence.mots)
                for i in range(l + 1):
                    for j in range(1, m + 1):
                        estimate = (alignment_count2[i][j][l][m] / alignment_count_for_any_i2[j][l][m])
                        alignment_table2[i][j][l][m] = max(estimate, MIN_PROB)

        # average two models
        for ws in trg_vocab:
            for wt in src_vocab:
                #translation_table1[wt][ws]=translation_table1[wt][ws]*0.45+translation_table2[ws][wt]*0.55
                translation_table1[wt][ws]=(count_t_given_s1[wt][ws]+count_t_given_s2[ws][wt])/ (count_any_t_given_s1[ws]+count_any_t_given_s2[wt])

        for aligned_sentence in aligned_sents:
            l = len(aligned_sentence.mots)
            m = len(aligned_sentence.words)
            for i in range(l + 1):
                for j in range(1, m + 1):
                    alignment_table1[i][j][l][m]=alignment_table1[i][j][l][m]*0.57+alignment_table2[j][i][m][l]*0.43
                    #alignment_table1[i][j][l][m]=(alignment_count1[i][j][l][m]+alignment_count2[j][i][m][j])/(alignment_count_for_any_i1[j][l][m]+alignment_count_for_any_i2[i][m][l])

        t = translation_table1
        q = alignment_table1

        return (t,q)

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 250)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
