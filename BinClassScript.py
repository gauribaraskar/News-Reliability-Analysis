'''

Simple Binary Classifier using Neural Networks

Model used for creating a baseline

Authors:

Jay Satish Shinde
Ayush Kumar
Gauri Baraskar

'''

import pandas as pd

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from keras import layers,models,optimizers


train_data = pd.read_csv('/Users/ayush/train.csv')

train_DF = pd.DataFrame()

train_DF["text"] = train_data["text"]
train_DF["label"] = train_data["label"]

    # Splitting into training and testing
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_DF['text'], train_DF['label'])



def vectorization(code):
    if(code == "count"):
        count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        count_vect.fit(train_DF['text'].values.astype('str'))

        return count_vect

    elif(code == "word"):
        tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        tfidf_vect.fit(train_DF["text"].values.astype('str'))

        return tfidf_vect


def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    hidden_layer1 = layers.Dense(50, activation="relu")(hidden_layer)

    hidden_layer2 = layers.Dense(25, activation="relu")(hidden_layer1)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer2)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adagrad(), loss='binary_crossentropy')
    return classifier


def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label, epochs=500, batch_size=2048, validation_split=0.20)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    res = []

    for i in predictions:
        if i >= 0.5:
            res.append(1)
        else:
            res.append(0)
    #
    clarity = list(test_y)
    sum = 0
    for i in range(len(test_y)):
        if (res[i] == clarity[i]):
            sum += 1
    test_accuracy = float(sum / len(test_y))

    res = []

    predictions = classifier.predict(feature_vector_train)

    for i in predictions:
        if i >= 0.5:
            res.append(1)
        else:
            res.append(0)
    #
    clarity = list(label)
    sum = 0
    for i in range(len(test_y)):
        if (res[i] == clarity[i]):
            sum += 1
    train_accuracy = float(sum / len(label))

    accuracies = [train_accuracy, test_accuracy]

    return classifier, accuracies

try:
    choice = int(input("Choose a vectorization method :\nPress 1 for Count or 2 for Word tfidf \n"))
except:
    print("Invalid choice")

if choice == 1:
    vectorizer = vectorization("count")
    xtrain = vectorizer.transform(train_x.astype('str'))
    xtest = vectorizer.transform(test_x.astype('str'))
elif choice == 2:
    vectorizer = vectorization("word")
    xtrain = vectorizer.transform(train_x.astype('str'))
    xtest = vectorizer.transform(test_x.astype('str'))
else:
    print( "Invalid Choice")

classifier = create_model_architecture(xtrain.shape[1])
accuracy = train_model(classifier,xtrain,train_y,xtest)

classifier.save("model.hd5")

print(accuracy)


