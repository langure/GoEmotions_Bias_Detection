import re
from os import system, name
import csv
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Hyperparameters
destination_file = 'data/res_biased.csv'
ngram_len = 3
file_name = 'data/goemotions_1_csv_full.csv'
labels = ['admiration','amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
          'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
          'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

def ngram(token, n): 
    output = []
    for i in range(n-1, len(token)): 
        ngram = ' '.join(token[i-n+1:i+1])
        output.append(ngram) 
    return output

def ngram_tokenize(text, nrange=(1, ngram_len)):
    text_features = [] 
    text = text.lower() 
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)         # Preprocessing, just removing characters that are not text or numbers
    for n in range(nrange[0], nrange[1]+1): 
        text_features += ngram(text_alphanum.split(), n)    
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def train_test(clf, x_train, x_test, y_train, y_test):
    try:
        clf.fit(x_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(x_train))
        test_acc = accuracy_score(y_test, clf.predict(x_test))
        return train_acc, test_acc
    except:
        return 0,0

def process_by_rater_id(biased_df):

    print(f'Biased_DF shape {biased_df.shape}')
    
    text_examples = biased_df[biased_df.columns[0]].values.tolist()
    raw_labels = biased_df[labels].astype(str).apply(lambda row: row[row == '1'].index, axis=1)
    text_labels = []

    for key, value in raw_labels.iteritems():
        try:
            text_labels.append(value[0])
        except IndexError:
            text_labels.append('unclear')
    
    #split into test and train
    y_all = text_labels
    x_all = []
    
    for sample in text_examples:
        x_all.append(ngram_tokenize(sample, nrange=(1,ngram_len)))
    
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=.2, train_size = .8, random_state=123, shuffle=False)

    vectorizer = DictVectorizer(sparse = True)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    svc = SVC()
    lsvc = LinearSVC(random_state=123)
    rforest = RandomForestClassifier(random_state=123)
    dtree = DecisionTreeClassifier()
    knn = KNeighborsClassifier(n_neighbors=10)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=123, max_iter=400)

    clifs = [svc, lsvc, rforest, dtree, knn, clf]
    results = []
    for clf in clifs: 
        clf_name = clf.__class__.__name__
        print(f'training: {clf_name}')
        train_acc, test_acc = train_test(clf, x_train, x_test, y_train, y_test)
        results.append(test_acc)
        
    return results, biased_df.shape[0]
        

print("Starting biased")

# Read the file
df = pd.read_csv(file_name)
exit_file = open(destination_file, 'w')
writer = csv.writer(exit_file)
header = ['rater_id', 'SVC', 'LinearSVC', 'RandomForestClassifier', 'DecisionTreeClassifier', 'KNeighborsClassifier', 'MLPClassifier', 'Examples']
writer.writerow(header)

for rater_id in range(47,82):
    clear()
    print(f'Processing: {rater_id}')
    biased_df = df[df['rater_id'] == rater_id]
    if rater_id != 68:
        r, size = process_by_rater_id(biased_df)
        csv_row = [rater_id]
        csv_row.extend(r)
        csv_row.append(size)

        writer.writerow(csv_row)

exit_file.close()
print("done")