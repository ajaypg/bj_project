import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from textblob import TextBlob as tb
import re
import datetime

#from TweetAnalyzer import clean_tweet
import p3_analyzing_tweet_data

class TweetSentiment():
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

    def clean_tweet(self, text):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    
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
        ax1.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.legend(labels, loc="best")
        ax1.axis('equal')
        plt.savefig("plot_pie_all.png")
        plt.show()

    def split_df(self, df):
        df_positive = df[['user_id', 'datetime', 'text', 'nbr_retweet', 'nbr_favorite', 'nbr_reply']].copy()
        df_negative = df[['user_id', 'datetime', 'text', 'nbr_retweet', 'nbr_favorite', 'nbr_reply']].copy()
        df_neutral = df[['user_id', 'datetime', 'text', 'nbr_retweet', 'nbr_favorite', 'nbr_reply']].copy()

        df_positive['sentiment'] = df['sentiment'].where(df['sentiment'] > 0)
        df_negative['sentiment'] = df['sentiment'].where(df['sentiment'] < 0)
        df_neutral['sentiment'] = df['sentiment'].where(df['sentiment'] == 0)
        
        df_positive['positive_sentiment'] = df_positive['sentiment'].cumsum()
        df_negative['negative_sentiment'] = df_negative['sentiment'].cumsum()
        df_neutral['neutral_sentiment'] = df_neutral['sentiment'].cumsum()
        df['total_sentiment'] = df['sentiment'].cumsum()
        
        print(df_positive.head(10))
        print(df.head(10))
        
        df['datetime_new'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['datetime_new'], ascending = True)

        df_positive['datetime_new'] = pd.to_datetime(df['datetime'])
        df_positive = df_positive.sort_values(by=['datetime_new'], ascending = True)

        df_negative['datetime_new'] = pd.to_datetime(df['datetime'])
        df_negative = df_negative.sort_values(by=['datetime_new'], ascending = True)

        ax = df_positive.plot.line(x = 'datetime_new', fontsize = 12, 
                                    y = 'positive_sentiment', title = "Sentiment Analysis")
        
        df_negative.plot(ax=ax, x = 'datetime_new', y = 'negative_sentiment')
        df.plot(ax=ax, x= 'datetime_new', y = 'total_sentiment')

        ax.set_xlabel("datetime",fontsize = 16)
        ax.set_ylabel("sentiment",fontsize = 16)
        ax.set_title("Sentiment Analysis", fontsize = 20)
        ax.set_xlim([datetime.date(2019, 11, 8), datetime.date(2019, 11, 10)])
        plt.savefig("plot_sentiment.png")
        plt.show()
        

        return [df_positive, df_negative, df_neutral]

    def percentage (self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

if __name__ == '__main__':
    TweetSentiment = TweetSentiment()
    #df = p3_analyzing_tweet_data.TweetAnalyzer.tweets_to_df()
    df = TweetSentiment.tweets_to_df()
    df_tweets = df['text']
      
    df['sentiment'] = np.array([TweetSentiment.get_sentiment_level(text) for text in df_tweets])
       
    print(df.head(10))

    df_sentiment = df['sentiment']
    print(df_sentiment.head(10))

    sentiment = pd.Series(data=df['sentiment'].values)
    sentiment.plot(figsize=[16,4], color='r')
    plt.savefig("plot_timeseries.png")
    plt.show()

    list = TweetSentiment.analyze_sentiment(df_sentiment)
    TweetSentiment.pie_plot_sentiment(list)

    df_array = []
    df_array = TweetSentiment.split_df(df)