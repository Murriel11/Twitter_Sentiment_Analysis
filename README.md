# Twitter Sentiment Analysis
## A Sentiment Analysis Project using Python, Machine Learning and Flask.
This Project was done using Natural Language Processing (NLP) Techniques. In December 2023, I felt it would be a good idea to obtain insights into how Twitter users felt about the year. Twitter receives over 500 million tweets per day from its users across the globe, so I only had to find a way to retrieve the data. Python libraries like Pandas (for Data Cleaning/Manipulation), Selenium (for Tweets Mining), NLTK (Natural Language Toolkit), TextBlob (for Sentiment Analysis), MatPlotlib & WordCloud (for Data Exploration), Emot (for Emojis identification), Plotly (for some Data Visualisation) were used for this project. I have trained and tested the data using some Machine Learning Algorithms such as Logistic Regression, Support vector Machine (SVM), Bernoulli Naive Bayes Classifier and Deep Learning Algorithm Long Short Term Memory (LSTM) by which i am getting an accuracy of 75%.

## Here, you will leearn how I carried out the following steps for the project:
1. Import Libraries
2. Tweets Mining
3. Data Cleaning
4. Tweets Processing
5. Data Exploration
6. Sentiment Analysis
7. Training and Testing Data
8. Creating GUI

## Tweets Processing Steps
To reach the ultimate goal, there was a need to clean up the individual tweets. To make this easy, I created a function "preProcessTweets" in my Python program which I further applied to the "Tweets" to produce the desired results. This user-defined function was used to remove punctuations, links, emojis, and stop words from the tweets in a single run. Additionally, I used a concept known as "Tokenization" in NLP. It is a method of splitting a sentence into smaller units called "tokens" to remove unnecessary elements. Another technique worthy of mention is "Lemmatization". This is a process of returning words to their "base" form. A simple illustration is shown below.

## Word Cloud Generation
To get the most common words used to describe 2020, I made use of the POS-tag (Parts of Speech tagging) module in the NLTK library. Using the WordCloud library, one can generate a Word Cloud based on word frequency and superimpose these words on any image. In this case, I used the Twitter logo and Matplotlib to display the image. The Word Cloud shows the words with higher frequency in bigger text size while the "not-so" common words are in smaller text sizes.

![image](https://github.com/Murriel11/Twitter_Sentiment_Analysis/assets/129143386/86f418f8-3b7e-4bd3-92c9-088cb3cb10e0)

## Sentiment Analysis
For this analysis, I went with TextBlob. Text Blob analyzes sentences by giving each tweet a Subjectivity and Polarity score.  Based on the Polarity scores, one can define which tweets were Positive, Negative, or Neutral. A Polarity score of < 0 is Negative, 0 is Neutral while > 0 is Positive. I used the "apply" method on the "Polarity" column in my data frame to return the respective Sentiment Category. The distribution of the Sentiment categories is shown below.

![image](https://github.com/Murriel11/Twitter_Sentiment_Analysis/assets/129143386/05f1753d-be85-46fc-b57d-d2c913719487) ![image](https://github.com/Murriel11/Twitter_Sentiment_Analysis/assets/129143386/5f0a7473-7c7e-43d1-bab4-e92c1c99e5c9)

## Creating GUI
Created User Interface (UI) using Flask and the trained data in the previous step.

![image](https://github.com/Murriel11/Twitter_Sentiment_Analysis/assets/129143386/f4b2a328-f10f-4bd9-8bab-10fb71d78205)
