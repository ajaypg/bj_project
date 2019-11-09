from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import re
import os
import datetime

currdir = os.path.dirname(__file__)

class TweetAnalyzer():
    def tweets_to_df():
        os.chdir(currdir)

        df = pd.read_csv("data.csv")
        
        droplist = ['ID', 'url', 'is_reply', 'is_retweet']#,'has_media','media']
        df.drop(droplist, axis=1, inplace=True)
            
        df = df[['user_id', 'usernameTweet', 'datetime', 'text', 'nbr_retweet', 'nbr_favorite', 'nbr_reply']]
        
        print(df.head(10))

        df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))

        df[['text','word_count']].head()

        return df
    
    def data_visualization(df):
        df['datetime_new'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by=['datetime_new'], ascending = True)
        df.insert(1, 'count_tweets', 1)
        df['count_tweets'] = df['count_tweets'].cumsum()
        df['sum_retweets'] = df['nbr_retweet'].cumsum()
        df['sum_favorites'] = df['nbr_favorite'].cumsum()
        df['sum_replies'] = df['nbr_reply'].cumsum()

        print(df.head(10))

        fig, axes = plt.subplots(nrows = 2, ncols =  2)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

        ax = df.plot(ax = axes[0,0], x = 'datetime_new', fontsize = 6, 
                                    y = 'count_tweets', title = "Tweets")
        
        ax.set_xlabel("",fontsize = 16)
        ax.set_ylabel("Retweets",fontsize = 16)
        ax.set_title("Tweets", fontsize = 20)
        ax.set_xlim([datetime.date(2019, 11, 8), datetime.date(2019, 11, 10)])
        ax.xaxis.set_major_formatter(plt.NullFormatter())

        ax = df.plot(ax = axes[0,1], x = 'datetime_new', fontsize = 6, 
                                    y = 'sum_retweets', title = "Retweets")
        
        ax.set_xlabel("",fontsize = 16)
        ax.set_ylabel("",fontsize = 16)
        ax.set_title("Retweets", fontsize = 20)
        ax.set_xlim([datetime.date(2019, 11, 8), datetime.date(2019, 11, 10)])
        ax.xaxis.set_major_formatter(plt.NullFormatter())

        ax = df.plot(ax = axes[1,0], x = 'datetime_new', fontsize = 6, 
                                    y = 'sum_favorites', title = "Favourites")
        
        ax.set_xlabel("datetime",fontsize = 16)
        ax.set_ylabel("Favourites",fontsize = 16)
        ax.set_title("Favourites", fontsize = 20)        
        ax.set_xlim([datetime.date(2019, 11, 8), datetime.date(2019, 11, 10)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax = df.plot(ax = axes[1,1], x = 'datetime_new', fontsize = 6, 
                                    y = 'sum_replies', title = "Replies")
        
        ax.set_xlabel("datetime",fontsize = 16)
        ax.set_ylabel("Replies",fontsize = 16)
        ax.set_title("Replies", fontsize = 20)
        ax.set_xlim([datetime.date(2019, 11, 8), datetime.date(2019, 11, 10)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


        plt.savefig("plot_retweets.png")

    def clean_tweet(text):
        corp = []

        #Removes punctuations
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text)
        p = re.compile(r'\<http.+?\>', re.DOTALL)
        text = re.sub(p, '', text)

        #Removes words less than three letters
        q = re.compile(r'\W*\b\w{1,3}\b')
        text = re.sub(q, ' ', text)

        #Converts to lowercase
        text = text.lower()

        #Removes tags
        text=  re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

        #Removes digits and special characters
        text = re.sub("(\\d|\\W)+"," ",text)

        #Creates a list from string
        text = text.split()
        
        #Import stop words from package
        stop_words = set(stopwords.words("english"))
        #Custom stopwords - have no value in the analysis
        new_words = ["com", "html", "http", "twitter", "using", "show", "result", "large", "also", "one", "two", "nshe","new", "previously", "shown",'http',"oct","amp","ever"]

        #Lemmatisation
        lemm = WordNetLemmatizer()
        text = [lemm.lemmatize(word) for word in text if not word in stop_words] 
        text = [lemm.lemmatize(word) for word in text if not word in new_words] 
        text = " ".join(text)
        
        corp.append(text)
        '''
        wc2 = WordCloud(width=400, height=100).generate(text)
        plt.figure(figsize=(20,10),facecolor = 'k')
        plt.imshow(wc2)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        '''
        return text
    
    def create_wordcloud(text):
	    mask = np.array(Image.open(path.join(currdir, "cloud.png")))
        
	    wc = WordCloud(background_color="white",
					max_words=200, 
					mask=mask)
        
	    wc.generate(text)
	    
	    wc.to_file(path.join(currdir, "plot_wc.png"))
    
    def word_frequency(text):
        
        #Breaks the string into a list of words
        text = text.split()
        wordlist = []
        wordfreq = []

        #Loop till string values present in text list
        for i in text:

            #Checking for duplicacy
            if i not in wordlist:

                #Insert value in tlist
                wordlist.append(i)
            
                wordfreq.append(text.count(i))
        '''
        print(wordlist)
        print(wordfreq)
        '''
        df_word_freq = pd.DataFrame({'word': wordlist,
                                'count':wordfreq})
        
        df_plt_wf = df_word_freq.sort_values(by='count', ascending = False).head(15)
        df_plt_wf['word'] = df_plt_wf['word'].astype(str)
        
        #Breakpoint to show data
        #print(df_plt_wf)

        my_colors = ['xkcd:salmon']*1

        ax = df_plt_wf.plot.barh(x = 'word', fontsize = 12, 
                                    y = 'count', title = "Frequency Analysis")
                                    #figsize = (20,10), color = my_colors, legend = False)
        
        ax.set_xlabel("Word",fontsize = 16)
        ax.set_ylabel("Count",fontsize = 16)
        ax.set_title("Frequency Analysis", fontsize = 20)

        ax = ax.invert_yaxis()

        plt.savefig("plot_wordfreq.png") 

        return df_word_freq
    
if __name__ == '__main__':
    df = TweetAnalyzer.tweets_to_df()
    TweetAnalyzer.data_visualization(df)
    
    #print(df.shape)
    
    #print(df.head())
    '''
    tweets = pd.Series(data=df['nbr_favorite'].values)
    tweets.plot()
    plt.show()
    '''
    #text = 'APPLE a ab cd ef efff mango apple orange orange apple guava mango mango'
    text = ' '.join(df['text'])

    #print(text)
    
    text = TweetAnalyzer.clean_tweet(text)

    TweetAnalyzer.create_wordcloud(text)
    
    df_word_freq = TweetAnalyzer.word_frequency(text)

    #TweetAnalyzer.word_vectors(text)

    #vector analysis based on cluster bi-gram
    