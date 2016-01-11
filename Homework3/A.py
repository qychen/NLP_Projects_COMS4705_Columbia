from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import main
import nltk
import string

# don't change the window size
window_size = 10

# A.1
def build_s(data, lang):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    for lex in data.keys():
        '''
        for ins in data[lex]:
            left = nltk.word_tokenize(ins[1])
            index = -1
            count = 0
            while count < window_size and index >= -1*len(left):

                if left[index] not in string.punctuation:
                    count += 1
                    if left[index] not in words:
                        words.append(left[index])
                index -= 1       
            right = nltk.word_tokenize(ins[3])

            index = 0
            count = 0
            while count < window_size and index < len(right):
                if right[index] not in string.punctuation:
                    count += 1
                    if right[index] not in words:
                        words.append(right[index])
                index += 1
        for ins in data[lex]:
            left = nltk.word_tokenize(ins[1])[-1*window_size:]
            for w in left:
                if w not in string.punctuation and w not in words:
                    words.append(w)
            right = nltk.word_tokenize(ins[3])[:window_size]
            for w in right:
                if w not in string.punctuation and w not in words:
                    words.append(w)
        '''

        words = []
        for ins in data[lex]:
            left = nltk.word_tokenize(ins[1])[-1*window_size:]
            right = nltk.word_tokenize(ins[3])[:window_size]
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


# A.1
def vectorize(data, s, lang):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here

    '''
    for ins in data:
        labels[ins[0]] = ins[4]

        l = []
        left = nltk.word_tokenize(ins[1])
        index = -1
        count = 0
        while count < window_size and index >= -1*len(left):

            if left[index] not in string.punctuation:
                count += 1
                l.append(left[index])
            index -= 1

        r = []
        right = nltk.word_tokenize(ins[3])
        index = 0
        count = 0
        while count < window_size and index < len(right):

            if right[index] not in string.punctuation:
                count += 1
                r.append(right[index])  
            index += 1
             
        words = l + r
        count = []
        for word in s:
             count.append(words.count(word))
        vectors[ins[0]] = count
    '''
    for ins in data:
        labels[ins[0]] = ins[4]

        words = []
        left = nltk.word_tokenize(ins[1])[-1*window_size:]
        right = nltk.word_tokenize(ins[3])[:window_size]
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
        for word in s:
             count.append(words.count(word))
        vectors[ins[0]] = count



    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier(7)

    # implement your code here
    data = []
    target = []
    for w in X_train.keys():
        data.append(X_train[w])
        target.append(y_train[w])
    svm_clf.fit(data, target)
    for ins in X_test.keys():
        svm_results.append((ins, svm_clf.predict(X_test[ins])[0]))

    data = []
    target = []
    for w in X_train.keys():
        data.append(X_train[w])
        target.append(y_train[w])
    knn_clf.fit(data, target)
    for ins in X_test.keys():
        knn_results.append((ins, knn_clf.predict(X_test[ins])[0]))

#    print svm_results
#    print knn_results

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    outfile = open(output_file, 'w')
    for lexelt, instances in sorted(results.iteritems(), key=lambda d: main.replace_accented(d[0].split('.')[0])):
        for instance in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            instance_id = instance[0]
            sid = instance[1]
            outfile.write(main.replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))
    outfile.close()


# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train, language)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt], language)
        X_test, _ = vectorize(test[lexelt], s[lexelt], language)
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



