import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_tuples = list()
    bigram_tuples = list()
    trigram_tuples = list()
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    totalwords = 0
    # start creating N-gram tuples for Language Model
    for line in training_corpus:
        words = line.strip().split(" ")+[STOP_SYMBOL]
        totalwords += len(words)
        for word in words:
            unigram_tuples += ((word,),)
        words = [START_SYMBOL]+ words
        bigram_tuples +=list(nltk.bigrams(words))
        words = [START_SYMBOL]+words+[STOP_SYMBOL]
        trigram_tuples +=list(nltk.trigrams(words))

    #Count each N-gram in training corpus
    for unituple in unigram_tuples:
        unigram_p[unituple] = 0
    for bituple in bigram_tuples:
        bigram_p[bituple] = 0
    for trituple in trigram_tuples:
        trigram_p[trituple] = 0
    for line in training_corpus:
        words = line.strip().split(" ")+[STOP_SYMBOL]
        bigram_p[(START_SYMBOL,words[0])] += 1
        trigram_p[(START_SYMBOL,START_SYMBOL,words[0])] += 1
        trigram_p[(START_SYMBOL,words[0],words[1])] += 1
        unigram_p[(words[len(words)-2],)] += 1
        unigram_p[(words[len(words)-1],)] += 1
        bigram_p[(words[len(words)-2],words[len(words)-1])] += 1
        trigram_p[(words[len(words)-2],words[len(words)-1],STOP_SYMBOL)] += 1
        for i in range(len(words)-2):
            unigram_p[(words[i],)] += 1
            bigram_p[(words[i], words[i+1])] += 1
            trigram_p[(words[i],words[i+1],words[i+2])] += 1
    
    #Calculate the log-probability using the count number
    for trigram in trigram_p.keys():
        if trigram[0:2]==(START_SYMBOL,START_SYMBOL):
            trigram_p[trigram]=math.log((trigram_p[trigram]/(len(training_corpus)*1.0)),2)
        else:
            trigram_p[trigram]=math.log((trigram_p[trigram]/(bigram_p[trigram[0:2]]*1.0)),2)
    for bigram in bigram_p.keys():
        if bigram[0]==START_SYMBOL:
            bigram_p[bigram]=math.log((bigram_p[bigram]/(len(training_corpus)*1.0)),2)
        else:
            bigram_p[bigram]=math.log((bigram_p[bigram]/(unigram_p[bigram[0:1]]*1.0)),2)
    for unigram in unigram_p.keys():
        unigram_p[unigram]=math.log((unigram_p[unigram]/(totalwords*1.0)),2)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for line in corpus:
        words_tuples = list()
        #Create N-gram tuples based on the model being used
        if n == 1: 
            words = line.strip().split(" ") + [STOP_SYMBOL]
            for word in words:
                words_tuples += ((word,),) 
        elif n == 2:
            words = [START_SYMBOL]+ line.strip().split(" ") +[STOP_SYMBOL]
            words_tuples = list(nltk.bigrams(words)) 
        else :
            words = [START_SYMBOL]+[START_SYMBOL]+line.strip().split(" ")+[STOP_SYMBOL]+[STOP_SYMBOL]
            words_tuples = list(nltk.trigrams(words))

        #Score the sentence by combining the log probability of each N-gram 
        #Use flag to deal with non-exist N-gram, change it to 0 when happens  
        score = 0
        flag = 1
        for word_tuple in words_tuples:
            if word_tuple in ngram_p:
                score += ngram_p[word_tuple]
            else:
                flag = 0
                break
        if flag == 1:
            scores.append(score)
        else:
            scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    for line in corpus:
        words =[START_SYMBOL]+[START_SYMBOL]+line.strip().split(" ")+[STOP_SYMBOL]
        #Count the last trigram (Wn, STOP, STOP) which do not have unigram or bigram
        #Use flag to deal with non-exist N-gram, change it to 0 when happens
        flag = 1
        if (words[len(words)-2],words[len(words)-1],STOP_SYMBOL) in trigrams:
            score = trigrams[(words[len(words)-2],words[len(words)-1],STOP_SYMBOL)]
            if score > 0: print score
        else:
            flag = 0
      
        #Count the score of linear interpolation using N-gram models
        if (flag==1):
            for i in range(2,(len(words))):
                if ((words[i-2],words[i-1],words[i]) in trigrams) and ((words[i-1],words[i]) in bigrams) and ((words[i],) in unigrams):
                    score += math.log((1/3.0),2)+math.log((math.pow(2,trigrams[(words[i-2],words[i-1],words[i])])+math.pow(2,bigrams[(words[i-1],words[i])])+math.pow(2,unigrams[(words[i],)])),2)
                else:
                    flag = 0
                    break
        if (flag == 1):
            scores.append(score)
        else:
            scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
