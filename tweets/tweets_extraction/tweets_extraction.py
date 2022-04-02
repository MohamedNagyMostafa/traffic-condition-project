import requests
import pandas as pd
import time

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(tweets_ids):
    
    search_url = "https://api.twitter.com/2/tweets?ids=" + tweets_ids  #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'expansions': 'author_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params, next_token = None):
    while True:
        params['next_token'] = next_token   #params object received from create_url function
        response = requests.request("GET", url, headers = headers, params = params)
        print("Endpoint Response Code: " + str(response.status_code))
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429: # Too many requests
            time.sleep(60)
        else:
            raise Exception(response.status_code, response.text)

## PUT YOUR KEYS HERE ##
api_key = 'HcFxd8Pe5tcHdB0owInjAhBzZ'
api_secret = '3RCPCEcTxE2bqFkWm8hlFbUSuCJvlKH9zjHC3mVHWbPoREt195'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAK%2FoaAEAAAAARuAAnAYp9o79LvDQo%2B80HU3qk28%3DKI3hrfwP7C4Uwh7RgVt2sXdeZ2hnUosQbHZMRCpNC5dtaW6zIR'

headers = create_headers(bearer_token)

# dataset headers
data = pd.DataFrame(columns=['tweet_id', 'text', 'created_at', 'user_id', 'user_name', 'user_description', 'followers_count', 'verified'])

tweets_ids = pd.read_csv('tweets_ids.tsv', sep='\n')

# config
batch_size = 100
start_batch = 10295
end_batch = start_batch + batch_size
last_idx = len(tweets_ids) - 1

while True:
    req_ids = ''
    for i in range(start_batch, end_batch):
        req_ids += str(tweets_ids.iloc[i].values[0]) + ','

    req_ids = req_ids[:-1]

    url = create_url(req_ids)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    users_by_id = {}

    for user in json_response['includes']['users']:
        users_by_id[user['id']] = user

    for tweet in json_response['data']:
        row = []
        author_id = tweet['author_id']

        # clean text to save as csv
        tweet_text = tweet['text'].replace(',', '')
        author_name = users_by_id[author_id]['name'].replace(',', '')
        author_description = users_by_id[author_id]['description'].replace(',', '')

        row.append(tweet['id'])
        row.append(tweet['text'])
        row.append(tweet['created_at'])
        row.append(author_id)
        row.append(author_name)
        row.append(author_description)
        row.append(users_by_id[author_id]['public_metrics']['followers_count'])
        row.append(users_by_id[author_id]['verified'])

        data.loc[len(data)] = row

    data.to_csv('tweets_dataset3.csv', index=False)

    start_batch = end_batch

    if end_batch == last_idx:
        break

    end_batch = min(end_batch + batch_size, last_idx)