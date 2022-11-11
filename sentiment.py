from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
pd.set_option('max_colwidth', None) # show full width of showing cols
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from newsapi import NewsApiClient
from datetime import date, datetime, timedelta
from GoogleNews import GoogleNews
from newspaper import Article
import os

today = date.today()
print("Today's date: ", today, "\n")

finviz_url = 'https://finviz.com/quote.ashx?t='
newsapi = NewsApiClient(api_key='54dfae66dc4d468e92ca81f586d5af5f')
vader = SentimentIntensityAnalyzer()

stock_input = input("Enter the stock Name (Ticker Symbol): ")
tickers = [stock_input]
keyword = input("Enter a keyword: ")
#keyword = 'apple'
#tickers = ['AAPL']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'test_agent'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    
parsed_data = []
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])


df_finviz = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df_finviz['date'] = pd.to_datetime(df_finviz.date).dt.date

f = lambda title: vader.polarity_scores(title)['compound']
df_finviz['compound'] = df_finviz['title'].apply(f)

mean_df_finviz = df_finviz.groupby(['ticker', 'date']).mean().unstack()
mean_df_finviz = mean_df_finviz.xs('compound', axis="columns").transpose()
print (df_finviz.head(10))

old_date = today - timedelta(days=10)
#NEWSAPI News Headlines Sentiments ================================================================================
print ("\n Other News on the Stock: \n")
top_headlines = newsapi.get_top_headlines(q=keyword, category='business', language='en', country='us')
all_articles = newsapi.get_everything(q=keyword, sources='bbc-news,the-verge', domains='bbc.co.uk,techcrunch.com', from_param=old_date, to=today, language='en', sort_by='relevancy', page=2)
sources = newsapi.get_sources()

newsapi_data = []
for headline in top_headlines['articles']:
	dateTime = datetime.strptime(headline['publishedAt'],"%Y-%m-%dT%H:%M:%SZ")
	date = dateTime.strftime("%Y-%m-%d")
	time = dateTime.strftime("%H:%M")
	title = headline['title']
	ticker = tickers[0]
	newsapi_data.append([ticker, date, time, title])
	parsed_data.append([ticker, date, time, title])
	
if (len(newsapi_data) > 0):
	df_newsapi = pd.DataFrame(newsapi_data, columns=['ticker', 'date', 'time', 'title'])
	df_newsapi['compound'] = df_newsapi['title'].apply(f)
	mean_df_newsapi = df_newsapi.groupby(['ticker', 'date']).mean().unstack()
	mean_df_newsapi = mean_df_newsapi.xs('compound', axis="columns").transpose()
	print (df_newsapi.head(10))
	
#GOOGLE News Headlines Sentiments ================================================================================
today_slash_format = today.strftime('%-m/%-d/%Y')
old_date_slash_format = old_date.strftime('%-m/%-d/%Y')
googlenews=GoogleNews(start=today_slash_format,end=old_date_slash_format)
googlenews.search(keyword)

for i in range(1,5):
	googlenews.getpage(i)
	googlenews_result=googlenews.result()
	df_googlenews=pd.DataFrame(googlenews_result)
print(df_googlenews.head(10))

news_list=[]
for ind in df_googlenews.index:
    dict={}
    article = Article(df_googlenews['link'][ind])
    article.download()
    article.parse()
    article.nlp()
    dict['Date']=df_googlenews['date'][ind]
    dict['Media']=df_googlenews['media'][ind]
    dict['Title']=article.title
    dict['Article']=article.text
    dict['Summary']=article.summary
    news_list.append(dict)
news_df=pd.DataFrame(news_list)
news_df.to_csv("articles.csv")


# All News in a single cumulative data frame #
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
df['date'] = pd.to_datetime(df.date).dt.date
df['compound'] = df['title'].apply(f)

mean_df = df.groupby(['ticker', 'date']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()

#print(df)


data_range = 5
# Plot the Scores ===========================================================================================
plt.figure(figsize=(10,6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.plot(mean_df.tail(data_range), 'o-')
plt.xlabel("Date", fontsize=20)
plt.ylabel("Sentiment Score", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=8)
plt.axhline(y=0, color="green", linestyle='--')
plt.title(ticker, color="Red",  fontsize=20)
#plt.gcf().autofmt_xdate()

plt.show()






































