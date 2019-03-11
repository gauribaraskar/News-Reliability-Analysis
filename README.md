# News-Reliability-Analysis
Algorithms used  to analyse news articles to determine if they are fake or real

## Requirements:
1. keras
2. tensorflow
3. textblob
4. numpy
5. pandas

## Progress:
1. Currently at a basic neural network<br/><br/>
   The features used at this stage are: Title,Author and the articel itself.<br/>
   These features are passed to the neural network are passed after vectorization.<br/>
   The structure of the model:<br/>
   ![Model](https://i.imgur.com/sXR8Zjs.png)
   
   This dataset scores 97% accuracy on testing(Done through kaggle,as the dataset was part of a kaggle competeiton)
2. Adding More Features<br/><br/>
   The following features have to be added for a better performance
   1. Word Score (A numerical score for the words in the article, based on which words occur in real news)
   2. Source Checking (Verifying of source through a google search)
   3. Fact Checking (Looking up the authenticity through the title by using DBPedia
3. Comparison of various Classification techniques<br/><br/>
   Various classification techniques like HAN,RNN and our own binary classifier have to be compared
   

## Data:

The data is inside Archive.zip labelled as test.csv and train.csv, Extract to proper location for it to work
