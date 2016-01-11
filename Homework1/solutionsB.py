import sys
import nltk
import math
import time

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for line in brown_train:
        words = [START_SYMBOL+'/'+START_SYMBOL]+[START_SYMBOL+'/'+START_SYMBOL]+line.strip().split(" ")+[STOP_SYMBOL+'/'+STOP_SYMBOL]
        sen_words = []
        sen_tags = []
        #split WORD/TAG by the first right of '/'
        for word in words:
            index = word.rindex('/')
            sen_words.append(word[:index])
            sen_tags.append(word[(index+1):])
        brown_words.append(sen_words)
        brown_tags.append(sen_tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    bigram_tuples = list()
    trigram_tuples = list()
    bigram_p = {}
    trigram_p = {}
    # start creating N-gram tuples for Language Model
    for line in brown_tags: 
        words = line[1:len(line)]
        bigram_tuples +=list(nltk.bigrams(words))
        words = line+[STOP_SYMBOL]
        trigram_tuples +=list(nltk.trigrams(words))

    #Count each N-gram in tag corpus
    for bituple in bigram_tuples:
        bigram_p[bituple] = 0
    for trituple in trigram_tuples:
        trigram_p[trituple] = 0
    for line in brown_tags:
        words = line[1:]+[STOP_SYMBOL]
        trigram_p[(START_SYMBOL,words[0],words[1])] += 1
        for i in range(len(words)-2):
            bigram_p[(words[i], words[i+1])] += 1
            trigram_p[(words[i],words[i+1],words[i+2])] += 1
    
    #Calculate the log-probability using the count number
    for trigram in trigram_p.keys():
        if trigram[0:2]==(START_SYMBOL,START_SYMBOL):
            trigram_p[trigram]=math.log((trigram_p[trigram]/(len(brown_tags)*1.0)),2)
        else:
            trigram_p[trigram]=math.log((trigram_p[trigram]/(bigram_p[trigram[0:2]]*1.0)),2)
    q_values = trigram_p
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    words_count = {}
    for line in brown_words:
        for word in line:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1
    known_list = []
    for word in words_count.keys():
        if words_count[word] > RARE_WORD_MAX_FREQ:
            known_list.append(word)

    known_words = set(known_list)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for line in brown_words:
        newline = []
        for word in line:
            if word in known_words:
                newline.append(word)
            else:
                newline.append(RARE_SYMBOL)
        brown_words_rare.append(newline)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    count_tuples = {}
    count_tags = {}
    
    #Count the number of appearance for each tag tuple and each tag
    for i in range(len(brown_words_rare)):
        for j in range(len(brown_words_rare[i])):
            tuples = (brown_words_rare[i][j],brown_tags[i][j])
            if tuples in count_tuples:
                count_tuples[tuples] += 1
            else:
                count_tuples[tuples] = 1
            if brown_tags[i][j] in count_tags:
                count_tags[brown_tags[i][j]] +=1
            else:
                count_tags[brown_tags[i][j]] =1
    for tuples in count_tuples:
        count_tuples[tuples] = math.log((count_tuples[tuples]/(count_tags[tuples[1]]*1.0)),2)

    e_values = count_tuples
    taglist = set(count_tags.keys())
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    brown_rare =replace_rare(brown_dev_words, known_words)
    tagged = []
    taglist = list(taglist)
    for line_index in range(len(brown_rare)):
        line = brown_rare[line_index]
        #initialization step
        v_state = {}
        for i in range(len(taglist)):
            for j in range(len(taglist)):
                v_state[(taglist[i],taglist[j])] = -3000
        v_state[(START_SYMBOL,START_SYMBOL)] = 0
        t_state = []
        t_state.append(v_state)
        bp_state = []
        bp_state.append({})

        #recuision step
        #loop over each word in sentence
        for t in range(1,len(line)+1):
            #consider the trigram of state(tag) (s1,s2,s3)
            v_state = {}
            bp_current = {}
            for s3 in range(len(taglist)):
                #Only consider tags with emission probability greater than zero
                if (line[t-1],taglist[s3]) in e_values:
                    for s2 in range(len(taglist)):
                        maxvalue = 3 * LOG_PROB_OF_ZERO 
                        bp = taglist.index('NOUN')
                        for s1 in range(len(taglist)):
                            if (taglist[s1],taglist[s2],taglist[s3]) in q_values:    
                                newvalue = t_state[t-1][(taglist[s1],taglist[s2])] + q_values[(taglist[s1],taglist[s2],taglist[s3])]
                                if newvalue > maxvalue:
                                    maxvalue = newvalue
                                    bp = s1
                            else:
                                newvalue = t_state[t-1][(taglist[s1],taglist[s2])] + LOG_PROB_OF_ZERO
                                if newvalue > maxvalue:
                                    maxvalue = newvalue
                                    bp = s1

                        v_state[(taglist[s2],taglist[s3])] = maxvalue + e_values[(line[t-1],taglist[s3])]
                        bp_current[(taglist[s2],taglist[s3])] = taglist[bp]
                else:
                    for s2 in range(len(taglist)):
                        v_state[(taglist[s2],taglist[s3])] = 3 * LOG_PROB_OF_ZERO 
            t_state.append(v_state)
            bp_state.append(bp_current)

        #termination step
        bps1 = taglist.index('NOUN')
        bps2 = taglist.index('NOUN')
        maxvalue = 3 * LOG_PROB_OF_ZERO
        for s2 in range(len(taglist)):
            for s1 in range(len(taglist)):
                if (taglist[s1],taglist[s2],STOP_SYMBOL) in q_values:
                    newvalue = t_state[len(line)][(taglist[s1],taglist[s2])] + q_values[(taglist[s1],taglist[s2],STOP_SYMBOL)]
                    if newvalue > maxvalue:
                        maxvalue = newvalue
                        bps1 = s1
                        bps2 = s2
        bp_tags = []
        bp_tags.append(taglist[bps2])
        bp_tags.append(taglist[bps1])
        i = len(line)
        s3 = taglist[bps2]
        s2 = taglist[bps1]

        while i > 2:
            if (s2,s3) in bp_state[i]:
                s1 = bp_state[i][(s2,s3)]
#            else:
#                state = (taglist[0],taglist[0])
#                flag = 0
#                maxvalue = LOG_PROB_OF_ZERO
#                for v in t_state[i].keys():
#                    if t_state[i][v] > LOG_PROB_OF_ZERO:
#                        maxvalue = t_state[i][v]
#                        state = v
#                        flag = 1
#                if flag == 1:
#                    s1 = bp_state[i][state]
#                    s2 = state[0]
#                    s3 = state[1]
#                    bp_tags[-2] = state[1]
#                    bp_tags[-1] = state[0]
#                else:
#                    s1 = taglist[taglist.index('NOUN')]
            bp_tags.append(s1)
            s3 = s2
            s2 = s1
            i = i - 1
        bp = bp_tags[::-1]
        ori_sentence = brown_dev_words[line_index]

        output_sentence = ""
        for i in range(len(bp)):
            output_sentence = output_sentence + ori_sentence[i] + "/" + bp[i] + " "
        output = output_sentence.strip() + "\n"
        tagged.append(output)
#        print output_sentence
#        break

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

    tagged = []
    for line in brown_dev_words:
        output = ""
        tuples = trigram_tagger.tag(line)
        for t in tuples:
            output = output + t[0] + "/" + t[1] + " "
        output += " \n"
        tagged.append(output)
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
