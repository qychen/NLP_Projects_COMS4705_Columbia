import A
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
import nltk
from nltk.data import load
from nltk.corpus import cess_esp
from nltk.corpus import cess_cat
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# You might change the window size
wordbag_size = 10
window_size = 10
bigram_size = 2
pos_size = 2
base_count = 2
top_count = 9
#top_count = 4
infinite = 1000

eng_tagger = None
spa_tagger = None
cat_tagger = None


def build_s(data, lang):
    s = {}
    for lex in data.keys():
        words = []
        for ins in data[lex]:
            left = nltk.word_tokenize(ins[1])[-1*window_size:]
            right = nltk.word_tokenize(ins[3])[:window_size]
            if lang != "English":
                for w in left:
                    if w in string.punctuation:
                        left.remove(w)
                for w in right:
                    if w in string.punctuation:
                        right.remove(w)
            for w in left:
                if w not in words:
                    words.append(w)
            for w in right:
                if w not in words:
                    words.append(w)

        s[lex] = words

    return s


#a. pos train pos tagger
def get_tagger(lang):
    if lang == "English":
        global eng_tagger
        if eng_tagger:
            return eng_tagger
        else:
            _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
            eng_tagger = load(_POS_TAGGER)
            return eng_tagger
    elif lang == "Spanish":
        global spa_tagger
        if spa_tagger:
            return spa_tagger
        else:
            print 111
            training = cess_esp.tagged_sents()
            default_tagger = nltk.DefaultTagger('NN')
            unigram_tagger = nltk.UnigramTagger(training,backoff=default_tagger)
            bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
            spa_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
            print 555
            return spa_tagger
    else:
        global cat_tagger
        if cat_tagger:
            return cat_tagger
        else:
            training = cess_cat.tagged_sents()
            default_tagger = nltk.DefaultTagger('NN')
            unigram_tagger = nltk.UnigramTagger(training,backoff=default_tagger)
            bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
            cat_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
            return cat_tagger

#c. relevance score
def build_feas_set(data, lang):
    #count Ns,c and Nc
    ns = {}
    nc = {}
    for ins in data:

        left_context = nltk.word_tokenize(ins[1])
        right_context = nltk.word_tokenize(ins[3])

        #b. remove punctuations
        #for w in left_context:
        #    if w in string.punctuation:
        #        left_context.remove(w)
        #if lang == "Spanish":
        #    for w in right_context:
        #        if w in string.punctuation:
        #            right_context.remove(w)

        if ins[-1] not in ns.keys():
            ns[ins[-1]] = {}
        for w in left_context[-1*window_size:]:
            if w in nc.keys():
                nc[w] += 1
            else:
                nc[w] = 1
            if w in ns[ins[-1]].keys():
                ns[ins[-1]][w] += 1
            else:
                ns[ins[-1]][w] = 1
        for w in right_context[:window_size]:
            if w in nc.keys():
                nc[w] += 1
            else:
                nc[w] = 1
            if w in ns[ins[-1]].keys():
                ns[ins[-1]][w] += 1
            else:
                ns[ins[-1]][w] = 1

    #chooes top words as feature set
    #give up words appearing less than top_count times
    feas_set = []
    for sense in ns.keys():
        top = {}
        for w in ns[sense].keys():
            if nc[w] > base_count:
                if nc[w] == ns[sense][w]:
                    p = infinite
                else:
                    p = ns[sense][w]*1.0/(nc[w]-ns[sense][w])
                top[w] = p
        for w in sorted(top, key=top.get, reverse=True)[:top_count]:
            if w not in feas_set:
                feas_set.append(w)
    #print top
    #print feas_set

    return feas_set


# B.1.a,b,c,d
def extract_features(data, lang, feas_set, wordsbag):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    tagger = get_tagger(lang)


    for ins in data:
        labels[ins[0]] = ins[-1]
        feas = {}

        left_context = nltk.word_tokenize(ins[1])
        right_context = nltk.word_tokenize(ins[3])

        #Part A bag of words
        words = []
        left = left_context[-1*wordbag_size:]
        right = right_context[:wordbag_size]
        if lang != "English":
            for w in left:
                if w in string.punctuation:
                    left.remove(w)
            for w in right:
                if w in string.punctuation:
                    right.remove(w)
        for w in left:
            if w not in words:
                words.append(w)
        for w in right:
            if w not in words:
                words.append(w)

        count = []
        for word in wordsbag:
             #count.append(words.count(word))
             feas[word] = words.count(word)
        #vectors[ins[0]] = count

        
        #a. POS tags
        if len(left_context) > pos_size:
            left = left_context[-1*pos_size:]
        else:
            left = left_context
        if len(right_context) > pos_size:
            right = right_context[:pos_size]
        else:
            right = right_context
        pos = tagger.tag(left+[ins[2]]+right)

        #if len(left) >= 2:
        #    feas["pos-2"] = pos[len(left)-2][1]
        if len(left) >= 1:
            feas["pos-1"] = pos[len(left)-1][1]
        feas["pos0"] = pos[len(left)][1]
        #if len(right) >= 2:
        #    feas["pos2"] = pos[len(left)+2][1]
        if len(right) >= 1:
            feas["pos1"] = pos[len(left)+1][1]
        
        
        #c. relevance score
        for w in feas_set:
            feas[w] = 0
        for w in left_context[-1*window_size:]:
            if w in feas_set:
                feas[w] += 1
        for w in right_context[:window_size]:
            if w in feas_set:
                feas[w] += 1
        
        #b. do stemming
        if lang == "English":
            stemmer = SnowballStemmer("english", ignore_stopwords=True)
        else:
            stemmer = SnowballStemmer("spanish", ignore_stopwords=True)
        
        #b. remove punctuations
        for w in left_context:
            if w in string.punctuation:
                left_context.remove(w)
        if lang == "Spanish":
            for w in right_context:
                if w in string.punctuation:
                    right_context.remove(w)
        
        #bi-gram
        if len(left_context) > bigram_size:
            left = left_context[-1*bigram_size:]
        else:
            left = left_context
        if len(right_context) > bigram_size:
            right = right_context[:bigram_size]
        else:
            right = right_context

        for i in range(len(left)-1):
            index = -1*(i+1)
            feas["bi-"+str(i+1)] = stemmer.stem(left[index-1])+" "+stemmer.stem(left[index])
        for i in range(len(right)-1):
            index = i+1
            feas["bi"+str(i+2)] = stemmer.stem(right[index-1])+" "+stemmer.stem(right[index])
        if left:
            feas["bi0"] = stemmer.stem(left[-1])+" "+stemmer.stem(ins[2])
        if right:
            feas["bi1"] = stemmer.stem(ins[2])+" "+stemmer.stem(right[0])
      

        '''
        #tri-gram
        #if len(left_context) >= 3:
        #    feas["tri-3"] = stemmer.stem(left_context[-3])+" "+stemmer.stem(left_context[-2])+" "+stemmer.stem(left_context[-1])
        if len(left_context) >= 2:
            feas["tri-2"] = stemmer.stem(left_context[-2])+" "+stemmer.stem(left_context[-1])+" "+stemmer.stem(ins[2])
        if len(left_context) >= 1 and len(right_context) >= 1:
            feas["tri-1"] = stemmer.stem(left_context[-1])+" "+stemmer.stem(ins[2])+" "+stemmer.stem(right_context[0])
        if len(right_context) >= 2:
            feas["tri0"] = stemmer.stem(ins[2])+" "+stemmer.stem(right_context[0])+" "+stemmer.stem(right_context[1])
        #if len(right_context) >= 3:
        #    feas["tri1"] = stemmer.stem(right_context[0])+" "+stemmer.stem(right_context[1])+" "+stemmer.stem(right_context[2])
        '''
        '''
        

        #a. surrounding words
        #if len(left_context) >= 4:
        #    feas["w-4"] = left_context[-4]
        if len(left_context) >= 2:
            feas["w-2"] = left_context[-2]
        if len(left_context) >= 1:
            feas["w-1"] = left_context[-1]
        feas["w0"] = ins[2]
        #if len(right_context) >= 4:
        #    feas["w4"] = right_context[3]
        if len(right_context) >= 2:
            feas["w2"] = right_context[1]
        if len(right_context) >= 1:
            feas["w1"] = right_context[0]
        if lang == "English":
            if len(left_context) >= 3:
                feas["w-3"] = left_context[-3]
            if len(right_context) >= 3:
                feas["w3"] = right_context[2]

        

        if len(left_context) > window_size:
            left = left_context[-1*window_size:]
        else:
            left = left_context
        if len(right_context) > window_size:
            right = right_context[:window_size]
        else:
            right = right_context

        for i in range(len(left)):
            feas["uni-"+str(i+1)] = stemmer.stem(left[-1*(i+1)])
        for i in range(len(right)):
            feas["uni"+str(i+1)] = stemmer.stem(right[i])
        feas["uni0"] = stemmer.stem(ins[2])
        '''

        #d. wordnet: 
        left_hypers = []
        if len(left_context) >= 1:
            synset = wn.synsets(stemmer.stem(left_context[-1]))
            for syn in synset:
                hypers = syn.hypernyms()
                for hyper in hypers:
                    name = hyper.name().split(".")[0]
                    if name not in left_hypers:
                        left_hypers.append(name)
        right_hypers = []
        if len(right_context) >= 1:
            synset = wn.synsets(stemmer.stem(right_context[0]))
            for syn in synset:
                hypers = syn.hypernyms()
                for hyper in hypers:
                    name = hyper.name().split(".")[0]
                    if name not in right_hypers:
                        right_hypers.append(name)

        for hyper in left_hypers:
            feas[hyper] = hyper
        for hyper in right_hypers:
            feas[hyper] = hyper
        
        #if len(left_context) >= 2:
        #    synset = wn.synsets(left_context[-2])
        #    for syn in synset:
        #        hypers = syn.hypernyms()
        #        for hyper in hypers:
        #            name = hyper.name().split(".")[0]
        #            if name not in left_hypers:
        #                left_hypers.append(name)
        #if len(right_context) >= 2:
        #    synset = wn.synsets(right_context[1])
        #    for syn in synset:
        #        hypers = syn.hypernyms()
        #        for hyper in hypers:
        #            name = hyper.name().split(".")[0]
        #            if name not in right_hypers:
        #                right_hypers.append(name)
        '''
        #sentence location
        stop_punctuation = [',','.','?','!',';']
        if left_context:
            index = -1
            while index > (-1)*len(left_context) and left_context[index] not in stop_punctuation:
                index -= 1
            l_len = index*(-1)
        else:
            l_len = 1
        if right_context:
            index = 0
            while index < len(right_context)-1 and right_context[index] not in stop_punctuation:
                index += 1
            feas["dis"] = l_len*1.0/(index + 1 + l_len)
        '''






        #w0 pos
     #   feas["pos0"] = ins[0].split(".")[1]
        #print feas["pos0"]

        features[ins[0]] = feas
    # implement your code here

    #print features

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    #transform to list of lists
    X_train_trans = []
    X_test_trans = []
    y_train_trans = []
    index = {}
    count = 0

    for instance_id in X_train.keys():
        X_train_trans.append(X_train[instance_id])
        y_train_trans.append(y_train[instance_id])
        index[instance_id] = count
        count += 1
    #if len(X_train_trans) > 100:
    #    selector = SelectKBest(chi2, k=100)
    #else:
    #    selector = SelectKBest(chi2, k='all')
    selector = SelectKBest(chi2, k=len(X_train_trans[0])*0.9)
    X_train_new_array = selector.fit_transform(X_train_trans, y_train_trans)

    #print len(X_train_trans)
    X_train_new = {}
    X_test_new = {}
    for instance_id in X_train.keys():
        feas_train = list(X_train_new_array[index[instance_id]])
        X_train_new[instance_id] = feas_train
    selected_feas = selector.get_support(True)
    for instance_id in X_test.keys():
        feas_test_old = X_test[instance_id]
        feas_test = []
        for ind in selected_feas:
            feas_test.append(feas_test_old[ind])
        X_test_new[instance_id] = feas_test

    # implement your code here

    return X_train_new, X_test_new
    # or return all feature (no feature selection):
    #return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []
    svm_clf = svm.LinearSVC()
    #svm_clf = svm.SVC(gamma=0.001, C=100)
    data = []
    target = []  
    for w in X_train.keys():
        data.append(X_train[w])
        target.append(y_train[w])

    svm_clf.fit(data, target)
    for ins in X_test.keys():
        results.append((ins, svm_clf.predict(X_test[ins])[0]))

    # implement your code here

    return results

# run part B
def run(train, test, language, answer):
    results = {}
    s = build_s(train, language)

    for lexelt in train:

        feas_set = build_feas_set(train[lexelt], language)
        #feas_set = None
        train_features, y_train = extract_features(train[lexelt], language, feas_set, s[lexelt])
        test_features, _ = extract_features(test[lexelt], language, feas_set, s[lexelt])
    #    print train_features
        X_train, X_test = vectorize(train_features,test_features)
    #    print X_train
    #    X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
    #    results[lexelt] = classify(X_train_new, X_test_new,y_train)
        results[lexelt] = classify(X_train, X_test,y_train)

    A.print_results(results, answer)






