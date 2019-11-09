""" 
This script scans through the nom and using Naive Bayes classifer
to assign the gen. 

Revision history:
4-Nov-2019 AGP - Originally when the program was created.
5-Nov-2019 AGP - Modified code to write dataframes to excel spreadsheet.
"""

from os import path
from textblob import TextBlob as tb
from scipy import stats

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import re
import os
import sys
import subprocess

#import p1_stream_tweets
#import p1_twitter_credentials
#import p2_streaming_tweets
import p3_analyzing_tweet_data
import p4_sentiment_analysis

#Global Variables - currdir
currdir = os.path.dirname(__file__)
gendir = os.path.join(currdir, "genderizer-master", "genderizer-master", "genderizer")
print(gendir)

class GenderIdentifier():
    def tweets_to_df(self):
        os.chdir(currdir)

        df = pd.read_csv("data.csv")
        
        droplist = ['ID', 'url', 'is_reply', 'is_retweet']#,'has_media','media']
        df.drop(droplist, axis=1, inplace=True)
            
        df = df[['user_id', 'usernameTweet', 'datetime', 'text', 'nbr_retweet', 'nbr_favorite', 'nbr_reply']]
        
        print(df.head(10))

        df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))

        df[['text','word_count']].head()

        return df

    #Function to pass to the classifier 
    def get_gender(self):

        genscript = os.path.join(currdir, "genderizer-master", "genderizer")
        envscript = "C:\\Users\\Ajay\\Anaconda3\\envs\\py27\\python.exe"
        splcmd = envscript + " " + genscript + "\g_test.py"

        result = subprocess.getoutput(splcmd)

        df_gen = result.split("\n")
        print(result)

    def genderdata_to_df(self):
        os.chdir(currdir)

        print(currdir)

        df = pd.read_csv("data_gender.csv")
        df['gender'].replace('', np.nan, inplace=True)
        df.dropna(subset=['gender'], inplace=True)

        return df
    
    #def sentiment analysis()
    def get_sentiment_level(self, text):
        analysis = tb(p3_analyzing_tweet_data.TweetAnalyzer.clean_tweet(text))
        return analysis.sentiment.polarity

    def analyze_sentiment(self, df_sentiment):
        positive = 0
        negative = 0
        neutral = 0
        totaltweets = 0

        for num in df_sentiment:
            totaltweets += 1
            if num > 0:
                positive += 1
            elif num < 0:
                negative += 1
            elif num == 0:
                neutral += 1

        positive = self.percentage(positive, totaltweets)
        negative = self.percentage(negative, totaltweets)
        neutral = self.percentage(neutral, totaltweets)

        return [positive, negative, neutral, totaltweets]
    
    def pie_plot_sentiment(self, list):
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [list[0], list[2], list[1]]
        colors = ['yellowgreen', 'gold', 'red']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)
        #ax1.legend(labels, loc="best")
        ax1.axis('equal')
        plt.show()

    def percentage (self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')
    
    def chi_square_test(self, list_m, list_f):
        #list_m = map(int, list_m)
        #list_f = map(int, list_f)

        list_m = list(map(float, list_m))
        list_f = list(map(float, list_f))
        
        tot_m = ((list_m[0]/100)*list_m[3]) + ((list_m[1]/100)*list_m[3])
        tot_f = ((list_f[0]/100)*list_f[3]) + ((list_f[1]/100)*list_f[3])

        tot = tot_m + tot_f

        tot_p = ((list_m[0]/100)*list_m[3]) + ((list_f[0]/100)*list_f[3])
        tot_n = ((list_m[1]/100)*list_m[3]) + ((list_f[1]/100)*list_f[3])

        exp_m_p = tot_p*(float(tot_m)/float(tot))
        exp_m_n = tot_n*(float(tot_m)/float(tot))
        exp_f_p = tot_p*(float(tot_f)/float(tot))
        exp_f_n = tot_n*(float(tot_f)/float(tot))

        act_m_p = ((list_m[0]/100)*list_m[3])
        act_m_n = ((list_m[1]/100)*list_m[3])
        act_f_p = ((list_f[0]/100)*list_f[3])
        act_f_n = ((list_f[1]/100)*list_f[3])

        score_m_p = ((act_m_p-exp_m_p)*(act_m_p-exp_m_p))/(exp_m_p)
        score_m_n = ((act_m_n-exp_m_n)*(act_m_n-exp_m_n))/(exp_m_n)
        score_f_p = ((act_f_p-exp_f_p)*(act_f_p-exp_f_p))/(exp_f_p)
        score_f_n = ((act_f_n-exp_f_n)*(act_f_n-exp_f_n))/(exp_f_n)

        score = score_m_p + score_m_n + score_f_p + score_f_n

        print(score)

        p = 1 - stats.chi2.cdf(score, 1)

        print(p)

        return score

 #This basically calls the functions to be executed   
if __name__ == '__main__':
    #Starting an initiation of the class
    GenderIdentifier = GenderIdentifier()

    #Calling the Part 3 to extract tweets. Needed if scripts are not run in sequence
    #df = p3_analyzing_tweet_data.TweetAnalyzer.tweets_to_df()
    df = GenderIdentifier.tweets_to_df()

    #Calling clean_tweets function in Part 3 to cleanse the text
    df['modtext'] = np.array([p3_analyzing_tweet_data.TweetAnalyzer.clean_tweet(text) for text in df['text']])
    
    #Storing the variables needed to a different data frame
    df_users = df[['usernameTweet', 'modtext']]

    #This is a basically checkpoint to look into the results
    print(df_users.head(10))
    
    GenderIdentifier.get_gender()
        
    df_gender = GenderIdentifier.genderdata_to_df()
    
    df_tweets = df_gender['text']
    df_gender['sentiment'] = np.array([GenderIdentifier.get_sentiment_level(text) for text in df_tweets])
    #print(df_gender.head(10))

    df_male = df_gender[df_gender['gender'] == 'male']
    #print(df_male.head(10))

    df_female = df_gender[df_gender['gender'] == 'female']
    print(df_female.head(10))
    
    df_m_sentiment = df_male['sentiment']
    list_m = []
    list_m = GenderIdentifier.analyze_sentiment(df_m_sentiment)
    GenderIdentifier.pie_plot_sentiment(list_m)

    df_f_sentiment = df_female['sentiment']
    list_f = []
    list_f = GenderIdentifier.analyze_sentiment(df_f_sentiment)
    GenderIdentifier.pie_plot_sentiment(list_f)

    score = GenderIdentifier.chi_square_test(list_m, list_f)