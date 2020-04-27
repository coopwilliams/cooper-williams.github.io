---
title: "Building a Collaborative Filtering Movie Recommender with Word2Vec"
last_modified_at: 2020-04-26T16:20:02-05:00
categories:
  - Blog
tags:
  - nlp
  - ml
  - recommender systems
  - ipynb
  - projects 
---
__A guided tour of the most cultured open source movie recommender to date.__ 

# The Backstory

In 2020, I got to pick up a proverbial $20 bill off the sidewalk. That is, I helped execute on a good idea that you'd think would have been tried already, but surprisingly hadn't been. The idea came from an audacious classmate who noticed that movie lovers are generating tons of data on their moviegoing habits, but they can't systematically use it to discover new movies. Letterboxd and IMDb allow users to keep a log of all the movies they've ever seen, write reviews, and leave ratings.

From a machine learning perspective, a few hundred movie ratings should be pretty indicative of someone's taste. It's at least enough to generate some strong candidates for your next movie to watch. Of course, there are plenty of movie recommender systems out there already. But they're coupled with IP silos and not really user-driven. Amazon Prime will steer you towards movies they have rights to. Netflix optimizes for time spent on the platform. These platforms can't learn from your whole movie history, nor help you find a diamond in the rough.

So our goal was simple: build the movie recommender that we would want to use. Let me export my Letterboxd data, upload it to a web app, and get bottomless movie recommendations from all 538,353 movies on IMDb. To do that, we built a scraper that collected every IMDb movie review into our database (about 3.4 million reviews). Then I set about training models on the data, using multiple metrics to score the models' performance on my friends' watch histories. The result was incredible. Once we had deployed the best model in a web app, I had a blast interacting with the system and exploring movie recommendations. The model recommended movies that I was actually interested in, and responded to positive and negative feedback with updated recommendations. 

The code below is an edited version of the iPython notebook I put together for the team that would take over the project. Though it's only been a few months, I recognize a lot of things I would do differently now. The predictor itself is a kind of antipattern called a "God object" where all the useful methods are stuffed in. That's a nightmare for code readability and maintainability. Someday I'll go back and rewrite this thing with actual design patterns. For now, I post it as example of what kind of recommender systems can be achieved using Word2Vec. I try to explain my decisions as I go and offer guidance for those who would try to replicate the results. The original .ipynb file can be found on GitHub [here](https://github.com/coopwilliams/Groa/blob/master/SageMaker/Word2Vec_train_test.ipynb).

# The Theory

The core model for our app was built around Gensim's Word2Vec algorithm. I won't try to explain Word2Vec in-depth, since there are so many good [articles](https://www.knime.com/blog/word-embedding-word2vec-explained) (and [this video](https://youtu.be/QyrUentbkvw)) readily available. 

The simple version is this: Word2Vec is a neural network technique for relating words to their context. Once it's trained on a corpus of natural language text, it can either predict the next word in a sentence, or predict the sentence a word came from. If you have data on the products users like (e.g. movies rated positively), you can pass each user's history to the network, treating each product as a word. The network learns, in shocking detail, which products tend to be liked by the same user.

As always with machine learning models, the details make all the difference. In the notebook, I discuss the construction of training data, the importance of tuning hyperparameters, and the various tests one can throw at the model. I had to iterate through many configurations to get a model that gave consistently good recommendations.

# The Notebook

This notebook can be used to train and test user-based collaborative filtering models using Word2Vec. Before you start, a couple things are needed:

1. A database with scraped movie reviews and movie data. See the Groa/web_scraping folder for a [scraper](https://github.com/coopwilliams/Groa/tree/master/web_scraping) you can use to get the reviews. Movie data can be found [here](https://datasets.imdbws.com/). You'll want the files 'title.basics.tsv.gz' and 'title.ratings.tsv.gz'.

2. Folders in the current working directory titled:
    - /models
        - This is where trained models are saved.
    - /training_data
        - This is where training data is saved.
    - /exported_data
         - This contains a small version of the IMDb title.basics.tsv file that we use to quickly retrieve movie info.
         - It also contains /letterboxd and /imdb, where test data is saved for scoring various models. I have included my own data as an example of how to format any test data you can collect. The letterboxd data is just the unzipped Letterboxd export folder, exactly as it comes from the site.


## Connect to Database
Do this at the start of every session.


```python
! pip3 install psycopg2-binary --user
import pandas as pd
import psycopg2
import numpy as np
from getpass import getpass

# connect to database. Never save your password to a notebook.
connection = psycopg2.connect(
    database  = "postgres",
    user      = "postgres",
    password  = getpass(), # secure password entry. Enter DB password in the prompt and press Enter.
    host      = "mydatabase.us-east-1.rds.amazonaws.com",
    port      = '5432'
)

# create cursor that is used throughout
try:
    c = connection.cursor()
    print("Connected!")
except Exception as e:
    print("Connection problem chief!\n")
    print(e)
```

    Requirement already satisfied: psycopg2-binary in c:\users\cooper\appdata\roaming\python\python37\site-packages (2.8.4)
    

    WARNING: You are using pip version 20.0.1; however, version 20.0.2 is available.
    You should consider upgrading via the 'c:\users\cooper\anaconda3\python.exe -m pip install --upgrade pip' command.
    

     ················
    

    Connected!
    

## Data preparation plan:

1. Get the list of reviewers whose reviews we want (about 17k).
    - We only want to train on users who have positively rated a minimum number of movies, otherwise their watch histories will give us poor-quality associations between movies. So we set the minimum number as the variable `n` below.
2. Create the dataframe of reviewers, movie IDs with positive reviews
3. Inner join the above two dataframes to remove positive reviews whose reviewers don't meet our criteria.
4. Run the list constructor on the join table to construct the training data.
    - Training data is of this format: [['movieid1', 'movieid2', ...],['movieid3', 'movieid4', ...], ...]
                                      ^user1 watch history     ^user2 watch history 
5. Save the training data.
6. Train Word2Vec on the list of watch histories (which are themselves lists of movie IDs).


```python
# Get reviewers with at least n positive reviews (rating m-10 inclusive)

n = 5 # minimum number of positive reviews for a watch history to be included
m = 7 # minimum star rating for inclusion (IMDb rating scale)

c.execute(f"""
SELECT username
FROM reviews
WHERE user_rating BETWEEN {m} AND  10
GROUP BY username
HAVING COUNT(username) >= {n}
ORDER BY COUNT(username) DESC
""")

'''
Minimum rating for training data has been increased to 8 stars in v3_LimitingFactor.

Explanation: v2_MistakeNot is returning movies with an average rating of 7.66, 
which is towards the low end of the distribution in the training data. It might be 
near the mean, but we want our model to give the user an above-average movie experience.


'''
reviewers = c.fetchall()
```


```python
# how many reviewers qualify for inclusion?
len(reviewers)
```




    44438



#### Training note: 

The following query currently returns reviews in no discernible order.
This is because the reviews were inserted into the database by multiple scrapers
running in parallel.

Future users of this notebook should take care to note whether their database gives
the same result. The reason the order is important is that Word2Vec trains by learning to predict the movies within a 10-movie window. A non-random ordering may introduce a bias. That bias might improve the model, e.g. in the case where training data is sorted by review date. For all our initial models,however, we have elected not to pursue that approach. 

There are two reasons for this:

1. Ultimately, the user "taste vector" that is the input for making recommendations is a vector average of all the movies the user has watched, so it's not perfectly analogous to finding similar movies to a single movie.

2. More importantly, sorting the training data by review date would influence the model to associate movies according to the order people watch them in. This has pros and cons. We don't want to provide recommendations that lead a user down a path that others have trodden, and this would seem to be one potential drawback. But further testing is needed. It might be a great idea.


```python
# Get IDs of all positive reviews from database with minimum star rating m
# Note: we didn't scrape Letterboxd reviews so we only have 10-scale ratings

# query for IDs
c.execute(f"""SELECT movie_id, username 
            FROM reviews 
            WHERE user_rating BETWEEN {m} and 10
            ORDER BY review_date""")
result = c.fetchall()

# create reviews dataframe
df = pd.DataFrame(result, columns = ['movieid', 'userid'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieid</th>
      <th>userid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0070047</td>
      <td>robh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0120201</td>
      <td>robh</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0084787</td>
      <td>robh</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0120746</td>
      <td>jreeves</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0117571</td>
      <td>robh</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create reviewers dataframe
df_reviewers = pd.DataFrame(reviewers, columns = ['userid'])
```


```python
# number of reviewers
df_reviewers.shape
```




    (44438, 1)




```python
# merge to get only the IDs relevant to training
df = df.merge(df_reviewers, how='inner', on='userid')
df.shape
```




    (986835, 2)




```python
# ! sudo su
# ! yum update -y
# ! yum -y install python-pip
! python -V #should be python 3.7
```

    Python 3.7.3
    


```python
# ! which pip
```

### Install gensim


```python
! python -m pip install tqdm # this is just a terminal loading bar, not necessary
# ! python -c 'import tqdm'
! python -m pip install gensim
```

    ERROR: Invalid requirement: '#'
    WARNING: You are using pip version 20.0.1; however, version 20.0.2 is available.
    You should consider upgrading via the 'C:\Users\Cooper\Anaconda3\python.exe -m pip install --upgrade pip' command.
    

    Requirement already satisfied: gensim in c:\users\cooper\anaconda3\lib\site-packages (3.8.1)
    ....
    


```python
# list to capture watch history of the users
watched_train = []

# populate the list with the movie codes
for i in reviewers:
    temp = df[df["userid"] == i[0]]["movieid"].tolist()
    watched_train.append(temp)
    
len(watched_train) # number of watch histories
```




    44438




```python
# save the data so we don't lose all that hard work
import pickle
pickle.dump(watched_train, open(f'training_data/watched_train_{m}-10_{n}reviews_ordered.sav', 'wb'))
```


```python
# # uncomment if you want to save the model in protocol 2 so it can be opened in python 2.7
# import pickle
# temp = pickle.load(open('watched_train.sav', 'rb'))
# pickle.dump(temp, open('watched_train.sav', 'wb'), protocol=2)
```

## Train the Model

We use Gensim to train a Word2Vec model.


```python
# should be 3.6.0 or above
gensim.__version__
```


```python
# load training data
import pickle
watched_train = pickle.load(open(f'training_data/watched_train_{m}-10_{n}reviews_ordered.sav', 'rb'))
len(watched_train)
```

### Tuning Hyperparameters

Much insight can be gained from reading [Word2vec applied to Recommendation: Hyperparameters Matter](https://www.groundai.com/project/word2vec-applied-to-recommendation-hyperparameters-matter/1).

- 4 hyperparameters that can significantly improve results:
    - negative sampling distribution
        - not included yet.
        - (negative) 0.5 in best results
    - number of epochs
        - 90-150 in the best results.
    - subsampling parameter
        - 10^-4 in all the best results.
    - window-size
        - Set to 6000 to capture whole history?
        - Perhaps it shouldn't be so large, to capture movies watched around the same time. But that requires ordering the training data by review date. Great experiment to try. 
        - Best results set this 3-7. Surprisingly small.
- `Tuning the parameters seems to drive the algorithm towards organizing the space in a way or another (e.g. better positioning the top items in the space, or pushing away less demanded items). Furthermore, the homogeneity of popularity between items of a same sequence, the shape of the popularity distribution, or the heterogeneity of the items in the catalog have a direct impact on the task evaluation.`
    - Probably do this after the first try at optimizing the above.
- `Some authors claim without empirical or theoretical verification that it is best to use a ”infinite” window-size (Barkan and
Koenigstein, [2016](https://www.groundai.com/project/word2vec-applied-to-recommendation-hyperparameters-matter/1#bib.bib2)), meaning that the whole sessions is considered as one context, but most arbitrarily used a fixed value without further discussion.`



```python
import random
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
%matplotlib inline
import warnings;
warnings.filterwarnings('ignore')

# train word2vec model
model = Word2Vec(
                 size = 100, # vector size
                 window = 10, # perhaps increase this
                 sg = 1, # sets to skip-gram
                 hs = 0, # must be set to 0 for negative sampling
                 negative = 10, # for negative sampling
                 ns_exponent = 0.3, # 0.5 in best results
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14,
                 sample = 0.0001 # 10^-4 in best results
                )
model.min_rating = m # store the minimum rating of training data
model.min_pos_reviews = n # store the minimum number of positive reviews
model.training_order = "review_date"
model.build_vocab(watched_train, progress_per=200)
model.train(watched_train, total_examples = model.corpus_count, 
            epochs=1500, # best results set this 90-150
            report_delay=60, compute_loss=True)

# save word2vec model
model.save("models/w2v_limitingfactor_v3.36.model")
```

## Use the model


```python
# !pip install gensim==3.8.1
```

### Build the predictor

**Note: This code does the heavy lifting in the end application. As I noted above, it's not maintainable or pythonic to have a single class containing all your prediction methods (a "God object"). If you're a beginner like me, study design patterns that break the functionality down into extensible pieces. But the worst hubris I committed here is hardcoding a polynomial to weight movies with extreme ratings.**

First we define some helper functions that standardize the data the user gives us. We had to deal with both IMDb data and Letterboxd data, which provide different fields. Of particular difficulty was the fact that Letterboxd doesn't provide the IMDb movie IDs that comprise our model's vocabulary. So we use the function `df_to_id_list()` to match a movie's title and year to an ID in the database. `prep_data()` separates the user's data into lists of IDs (tokens) that the model can recognize and make inferences on. 

The `Recommender` object gets its own database connection. The `predict()` method does most of the work, using the lists generated by `prep_data()` to improve the quality of recommendations. We can subtract the average of disliked movies from the average of liked movies. We can give extra importance to movies with extreme ratings. Crucially, we can ensure that the user doesn't get recommendations they've already seen.


```python
import gensim
import pandas as pd
import re
import warnings;
warnings.filterwarnings('ignore')
    
def fill_id(id):

    """Adds leading zeroes back if necessary. This makes the id match the database."""

    if len(str(id)) < 7:
        length = len(str(id))
        id = "0"*(7 - length) + str(id)

    return str(id)
    
def df_to_id_list(df, id_book):

    """Converts dataframe of movies to a list of the IDs for those movies.

    Every title in the input dataframe is checked against the local file, which
    includes all the titles and IDs in our database. For anything without a match,
    replace the non-alphanumeric characters with wildcards, and query the database
    for matches.
    """

    df['Year'] = df['Year'].astype(int).astype(str)
    matched = pd.merge(df, id_book,
                left_on=['Name', 'Year'], right_on=['primaryTitle', 'startYear'],
                how='inner')
    ids = matched['tconst'].astype(str).tolist()
    final_ratings = []
    names = df.Name.tolist()
    years = [int(year) for year in df.Year.tolist()]

    if 'Rating' in df.columns:
        stars = [int(rating) for rating in df.Rating.tolist()]
        info = list(zip(names, years, stars))
        final_ratings = matched['Rating'].astype(int).tolist()
    else:
        info = list(zip(names, years, list(range(len(years)))))

    missed = [x for x in info if x[0] not in matched['primaryTitle'].tolist()]

    for i, j, k in missed:
        i = re.sub('[^\s0-9a-zA-Z\s]+', '%', i)

        try:
            cursor_dog.execute(f"""
                SELECT movie_id, original_title, primary_title
                FROM movies
                WHERE primary_title ILIKE '{i}' AND start_year = {j}
                  OR original_title ILIKE '{i}' AND start_year = {j}
                ORDER BY runtime_minutes DESC
                LIMIT 1""")
            id = cursor_dog.fetchone()[0]
            ids.append(id)
            final_ratings.append(k)	
        except:
            continue

    ids = [fill_id(id) for id in ids]
    final_ratings = [x*2 for x in final_ratings]
    ratings_dict = dict(zip(ids, final_ratings))

    return tuple([ids, ratings_dict])
    
def prep_data(ratings_df, watched_df=None, watchlist_df=None, 
                   good_threshold=4, bad_threshold=3):
    """Converts dataframes of exported Letterboxd data to lists of movie_ids.

    Parameters
    ----------
    ratings_df : pd dataframe
        Letterboxd ratings.

    watched_df : pd dataframe
        Letterboxd watch history.

    watchlist_df : pd dataframe
        Letterboxd list of movies the user wants to watch.
        Used in val_list for scoring the model's performance.

    good_threshold : int
        Minimum star rating (10pt scale) for a movie to be considered "enjoyed" by the user.

    bad_threshold : int
        Maximum star rating (10pt scale) for a movie to be considered "disliked" by the user.


    Returns
    -------
    tuple of lists of ids.
        (good_list, bad_list, hist_list, val_list)
    """
    try:
        # try to read Letterboxd user data
        # drop rows with nulls in the columns we use
        ratings_df = ratings_df.dropna(axis=0, subset=['Rating', 'Name', 'Year'])
        # split according to user rating
        good_df = ratings_df[ratings_df['Rating'] >= good_threshold]
        bad_df = ratings_df[ratings_df['Rating'] <= bad_threshold]
        neutral_df = ratings_df[(ratings_df['Rating'] > bad_threshold) & (ratings_df['Rating'] < good_threshold)]
        # convert dataframes to lists
        good_list, good_dict = df_to_id_list(good_df, id_book)
        bad_list, bad_dict = df_to_id_list(bad_df, id_book)
        neutral_list, neutral_dict = df_to_id_list(neutral_df, id_book)

    except KeyError:
        # Try to read IMDb user data
        # strip ids of "tt" prefix
        ratings_df['movie_id'] = ratings_df['Const'].str.lstrip("tt")
        # drop rows with nulls in the columns we use
        ratings_df = ratings_df.dropna(axis=0, subset=['Your Rating', 'Year'])
        # split according to user rating
        good_df = ratings_df[ratings_df['Your Rating'] >= good_threshold*2]
        bad_df = ratings_df[ratings_df['Your Rating'] <= bad_threshold*2]
        neutral_df = ratings_df[(ratings_df['Your Rating'] > bad_threshold*2) & (ratings_df['Your Rating'] < good_threshold*2)]
        # convert dataframes to lists
        good_list = good_df['movie_id'].to_list()
        bad_list = bad_df['movie_id'].to_list()
        neutral_list = neutral_df['movie_id'].to_list()

    except Exception as e:
        # can't read the dataframe as Letterboxd or IMDb user data
        print("This dataframe has columns:", ratings_df.columns)
        raise Exception(e)
        
    ratings_dict = dict(list(good_dict.items()) + list(bad_dict.items()) + list(neutral_dict.items()))

    if watched_df is not None:
        # Construct list of watched movies that aren't rated "good" or "bad"
        # First, get a set of identified IDs.
        rated_names = set(good_df.Name.tolist() + bad_df.Name.tolist() + neutral_list)
        # drop nulls from watched dataframe
        full_history = watched_df.dropna(axis=0, subset=['Name', 'Year'])
        # get list of watched movies that haven't been rated
        hist_list = df_to_id_list(full_history[~full_history['Name'].isin(rated_names)], id_book)[0]
        # add back list of "neutral" movies (whose IDs we already found before)
        hist_list = hist_list + neutral_list

    else: hist_list = neutral_list

    if watchlist_df is not None:

        try:
            watchlist_df = watchlist_df.dropna(axis=0, subset=['Name', 'Year'])
            val_list = df_to_id_list(watchlist_df, id_book)[0]

        except KeyError:
            watchlist_df = watchlist_df.dropna(axis=0, subset=['Const', 'Year'])
            watchlist_df['movie_id'] = watchlist_df['Const'].str.lstrip("tt")
            val_list = watchlist_df['movie_id'].tolist()

    else: val_list = []

    return (good_list, bad_list, hist_list, val_list, ratings_dict)


class Recommender(object):

    def __init__(self, model_path):
        """Initialize model with name of .model file"""

        self.model_path = model_path
        self.model = None
        self.cursor_dog = c # set to this notebook's connection
        self.ratings_book = pd.read_csv('exported_data/title_ratings.tsv', delimiter='\t')
        self.ratings_book['id'] = self.ratings_book['movie_id'].str.lstrip('tt').str.lstrip('0').astype(int)                            
        self.id_book = pd.read_csv('exported_data/title_basics_filtered.csv').merge(
                       self.ratings_book, left_on='tconst', right_on='id')

    def connect_db(self):
        """connect to database, create cursor.
        In the notebook, this isn't used, and all connections are 
        handled through the notebook's global connection at the top."""

        # connect to database
        connection = psycopg2.connect(
            database  = "postgres",
            user      = "postgres",
            password  = os.getenv('DB_PASSWORD'),
            host      = "movie-rec-scrape.cvslmiksgnix.us-east-1.rds.amazonaws.com",
            port      = '5432'
        )
        # create cursor that is used throughout

        try:
            self.cursor_dog = connection.cursor()
            print("Connected!")
        except:
            print("Connection problem chief!")

    def _get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""

        if self.model == None:
            model_path = self.model_path
            w2v_model = gensim.models.Word2Vec.load(model_path)
            # Keep only the normalized vectors.
            # This saves memory but makes the model untrainable (read-only).
            w2v_model.init_sims(replace=True)
            self.model = w2v_model

        return self.model

    def _get_info(self, id, score=None):
        """Takes an id string and returns the movie info with a url."""

        try:
            info_query = f"""
            SELECT m.primary_title, m.start_year, r.average_rating, r.num_votes
            FROM movies m
            JOIN ratings r ON m.movie_id = r.movie_id
            WHERE m.movie_id = '{id}'"""
            self.cursor_dog.execute(info_query)

        except Exception as e:
            return tuple([f"Movie title unknown. ID:{id}", None, None, None, None, None, id])

        t = self.cursor_dog.fetchone()

        if t:
            title = tuple([t[0], t[1], f"https://www.imdb.com/title/tt{id}/", t[2], t[3], score, id])
            return title
        else:
            return tuple([f"Movie title not retrieved. ID:{id}", None, None, None, None, None, id])

    def get_most_similar_title(self, id, id_list):
        """Get the title of the most similar movie to id from id_list"""

        clf = self._get_model()
        vocab = clf.wv.vocab

        if id not in vocab:
            return ""

        id_list = [id for id in id_list if id in vocab] # ensure all in vocab
        id_book = self.id_book
        match = clf.wv.most_similar_to_given(id, id_list)

        return id_book['primaryTitle'].loc[id_book['tconst'] == int(match)].values[0]

    def predict(self, input, bad_movies=[], hist_list=[], val_list=[],
                ratings_dict = {}, checked_list=[], rejected_list=[],
                n=50, harshness=1, rec_movies=True,
                show_vibes=False, scoring=False, return_scores=False):
        """Returns a list of recommendations and useful metadata, given a pretrained
        word2vec model and a list of movies.

        Parameters
        ----------

            input : iterable
                List of movies that the user likes.

            bad_movies : iterable
                List of movies that the user dislikes.

            hist_list : iterable
                List of movies the user has seen.

            val_list : iterable
                List of movies the user has already indicated interest in.
                Example: https://letterboxd.com/tabula_rasta/watchlist/
                People really load these up over the years, and so they make for 
                the best validation set we can ask for with current resources.

            ratings_dict : dictionary
                Dictionary of movie_id keys, user rating values.

            checked_list : iterable
                List of movies the user likes on the feedback form.

            rejected_list : iterable
                List of movies the user dislikes on the feedback form.

            n : int
                Number of recommendations to return.

            harshness : int
                Weighting to apply to disliked movies.
                Ex:
                    1 - most strongly account for disliked movies.
                    3 - divide "disliked movies" vector by 3.

            rec_movies : boolean
                If False, doesn't return movie recommendations (used for scoring).

            show_vibes : boolean
                If True, prints out the dupes as a feature.
                These movies are closest to the user's taste vector, 
                indicating some combination of importance and popularity.

            scoring : boolean
                If True, prints out the validation score.
            
            return_scores : boolean
                If True, skips printing out

        Returns
        -------
        A list of tuples
            (Title, Year, IMDb URL, Average Rating, Number of Votes, Similarity score)
        """

        clf = self._get_model()
        dupes = []                 # list for storing duplicates for scoring

        def _aggregate_vectors(movies, feedback_list=[]):
            """Gets the vector average of a list of movies."""

            movie_vec = []

            for i in movies:
                try:
                    m_vec = clf[i]  # get the vector for each movie

                    if ratings_dict:
                        try:
                            r = ratings_dict[i] # get user_rating for each movie
                            # Use a polynomial to weight the movie by rating.
                            # This equation is somewhat arbitrary. I just fit a polynomial
                            # to some weights that look good. The effect is to raise
                            # the importance of 1, 2, 9, and 10 star ratings to about 1.8.
                            w = ((r**3)*-0.00143) + ((r**2)*0.0533) + (r*-0.4695) + 2.1867
                            m_vec = m_vec * w
                        except KeyError:
                            continue

                    movie_vec.append(m_vec)

                except KeyError:
                    continue

            if feedback_list:
                for i in feedback_list:
                    try:
                        f_vec = clf[i]
                        movie_vec.append(f_vec*1.8) # weight feedback by changing multiplier here
                    except KeyError:
                        continue

            return np.mean(movie_vec, axis=0)

        def _similar_movies(v, bad_movies=[], n=50):
            """Aggregates movies and finds n vectors with highest cosine similarity."""

            if bad_movies:
                v = _remove_dislikes(bad_movies, v, harshness=harshness)

            self.my_vec = v

            return clf.similar_by_vector(v, topn= n+1)[1:]

        def _remove_dupes(recs, input, bad_movies, hist_list=[], feedback_list=[]):
            """remove any recommended IDs that were in the input list"""

            all_rated = input + bad_movies + hist_list + feedback_list
            nonlocal dupes
            dupes = [x for x in recs if x[0] in input]

            return [x for x in recs if x[0] not in all_rated]

        def _remove_dislikes(bad_movies, good_movies_vec, rejected_list=[], harshness=1):
            """Takes a list of movies that the user dislikes.
            Their embeddings are averaged,
            and subtracted from the input."""

            bad_vec = _aggregate_vectors(bad_movies, rejected_list)
            bad_vec = bad_vec / harshness

            return good_movies_vec - bad_vec

        def _score_model(recs, val_list):
            """Returns the number of recs that were already in the user's watchlist. Validation!"""

            ids = [x[0] for x in recs]

            return len(list(set(ids) & set(val_list)))

        aggregated = _aggregate_vectors(input, checked_list)
        recs = _similar_movies(aggregated, bad_movies, n=n)
        recs = _remove_dupes(recs, input, bad_movies, hist_list, checked_list + rejected_list)
        formatted_recs = [self._get_info(x[0], x[1]) for x in recs]

        if val_list:
            if return_scores:
                return tuple([_score_model(recs, val_list), sum([i[3] for i in formatted_recs if i[3] is not None])/len(formatted_recs)])
            elif scoring:
                print(f"The model recommended {_score_model(recs, val_list)} movies that were on the watchlist!\n")
                print(f"\t\t Average Rating: {sum([i[3] for i in formatted_recs if i[3] is not None])/len(formatted_recs)}\n")

        if show_vibes:
            print("You'll get along with people who like: \n")
            for x in dupes:
                print(self._get_info(x[0], x[1]))
            print('\n')

        if rec_movies:
            return formatted_recs
```

### Prep generic test data

This test data comes from my own Letterboxd export data. To get your exported Letterboxd data, go to [the export page for your account](https://letterboxd.com/settings/data/) and click "Export your data". 

```python
# df to lookup ids from titles
id_book = pd.read_csv('exported_data/title_basics_small.csv')

# import user Letterboxd data
ratings = pd.read_csv('exported_data/letterboxd/cooper/ratings.csv')
watched = pd.read_csv('exported_data/letterboxd/cooper/watched.csv')
watchlist = pd.read_csv('exported_data/letterboxd/cooper/watchlist.csv')

# note: if you import IMDb data, it's currently encoded 'cp1252' (but they may someday switch to utf-8)

# prep user data
good_list, bad_list, hist_list, val_list, ratings_dict = prep_data(
                                    ratings, watchlist_df=watchlist, good_threshold=4, bad_threshold=3)
```


```python
print(len(good_list), len(bad_list), len(hist_list), len(val_list))
```

    389 194 146 1062
    


```python
#-------------------------------------------------------------#
# To inspect a model, enter its path here and run the cell
model_path = "models/w2v_limitingfactor_v3.36.model"
#------------------------------------------------#

model = gensim.models.Word2Vec.load(model_path)

def _get_info(id):
            """Takes an id string and returns the movie info with a url."""

            try:
                c.execute(f"""
                select m.primary_title, m.start_year, r.average_rating, r.num_votes
                from movies m
                join ratings r on m.movie_id = r.movie_id
                where m.movie_id = '{id[0]}'""")
            except:
                return tuple([f"Movie title unknown. ID:{id[0]}", None, None, None, None, None])

            t = c.fetchone()

            if t:
                title = tuple([t[0], t[1], f"https://www.imdb.com/title/tt{id[0]}/", t[2], t[3], id[1]])
                return title
            else:
                return tuple([f"Movie title unknown. ID:{id[0]}", None, None, None, None, None])

print(model)
print(f"""\t
            corpus_count: {model.corpus_count}
            corpus_total_words: {model.corpus_total_words}
            window: {model.window}
            sg: {model.sg}
            hs: {model.hs}
            negative: {model.negative}
            ns_exponent: {model.ns_exponent}
            alpha: {model.alpha}
            min_alpha: {model.min_alpha}
            sample: {model.sample}
            epochs: {model.epochs}
            """)
print("Most similar movies to Porco Rosso minus Kiki's Delivery Service?")
movies = model.similar_by_vector(model['0104652']
                                -model['0097814']
                                 ,topn=11)
for i in movies:
    print(_get_info(i))
```

    Word2Vec(vocab=24691, size=100, alpha=0.03)
    	
                corpus_count: 35371
                corpus_total_words: 933359
                window: 10
                sg: 1
                hs: 0
                negative: 10
                ns_exponent: 0.75
                alpha: 0.03
                min_alpha: 0.0007
                sample: 0.0001
                epochs: 1500
                
    Most similar movies to Porco Rosso minus Kiki's Delivery Service?
    ('Porco Rosso', 1992, 'https://www.imdb.com/title/tt0104652/', 7.8, 64422, 0.5604802370071411)
    ('Kaleidoscope', 1966, 'https://www.imdb.com/title/tt0060581/', 6.0, 613, 0.41448774933815)
    ('Capote', 2005, 'https://www.imdb.com/title/tt0379725/', 7.3, 119883, 0.39984023571014404)
    ('Uncertainty', 2008, 'https://www.imdb.com/title/tt1086216/', 5.8, 6290, 0.39574727416038513)
    ("Von Ryan's Express", 1965, 'https://www.imdb.com/title/tt0059885/', 7.1, 12038, 0.3899157643318176)
    ('How to Steal a Million', 1966, 'https://www.imdb.com/title/tt0060522/', 7.6, 22981, 0.379239022731781)
    ('Psycho-Circus', 1966, 'https://www.imdb.com/title/tt0060865/', 5.4, 1026, 0.37854233384132385)
    ('Gambit', 1966, 'https://www.imdb.com/title/tt0060445/', 7.1, 4834, 0.3746405243873596)
    ('The Taming of the Shrew', 1967, 'https://www.imdb.com/title/tt0061407/', 7.2, 6912, 0.36873605847358704)
    ('The Kennel Murder Case', 1933, 'https://www.imdb.com/title/tt0024210/', 7.0, 2750, 0.3628532886505127)
    ('Red Rock West', 1993, 'https://www.imdb.com/title/tt0105226/', 7.0, 19059, 0.36147841811180115)
    


```python
z = Recommender('models/w2v_limitingfactor_v3.36.model')
z.predict(good_list, bad_list, hist_list, val_list, ratings_dict, n=100, harshness=1, rec_movies=False, scoring=True,)
```

    The model recommended 28 movies that were on the watchlist!
    
    		 Average Rating: 7.985106382978725
    
    

## Best model so far

LimitingFactor_v3.51 is the model to beat. It performs consistently well across various tests, which are detailed below.


```python
s = Recommender('models/w2v_limitingfactor_v3.36.model')
s.predict(good_list, bad_list, hist_list, val_list, ratings_dict, n=100, harshness=1, rec_movies=False, scoring=True,)
# s.predict(aj2, n=100, harshness=1)
```

    The model recommended 19 movies that were on the watchlist!
    
    		 Average Rating: 8.156666666666666
    
    

### Add test cases

These test cases are small lists of movies that various people said they enjoyed. I ran them through the model and shared the results with most of these people to hear their feedback.


```python
# Early test cases

# A list of some Coen Bros movies.
coen_bros = ['116282', '2042568', '1019452', 
             '1403865', '190590', '138524', 
             '335245', '477348', '887883', '101410']

# Data scientist's recent watches.
cooper_recent = ['0053285', '0038650', '0046022', 
                 '4520988', '1605783', '6751668', 
                 '0083791', '0115685', '0051459', 
                 '8772262', '0061184', '0041959',
                 '7775622']

# dirkh public letterboxd recent watches.
dirkh = ['7975244', '8106534', '1489887', 
         '1302006', '7286456', '6751668', 
         '8364368', '2283362', '6146586', 
         '2194499', '7131622', '6857112']

# Marvin watches
marvin = ['7286456', '0816692', '2543164', '2935510', 
          '2798920', '0468569', '5013056', '1375666', 
          '3659388', '0470752', '0266915', '0092675', 
          '0137523', '0133093', '1285016']  

# Gabe watches
gabe = ['6292852','0816692','2737304','3748528',
        '3065204','4154796','1536537','1825683',
        '1375666','8236336','2488496','1772341',
        '0317705','6857112','5052448']

# Eric watches
eric = ['2974050','1595842','0118539','0093405',
        '3216920','1256535','5612742','3120314',
        '1893371','0046248','0058548','0199481',
        '2296777','0071198','0077834']

chuckie = ['4263482','0084787','3286052','5715874','1172994','4805316','3139756','8772262','7784604','1034415',]

harlan = ['1065073','5052448','0470752','5688932','1853728','1596363','0432283','6412452','4633694','9495224','0443453','0063823',
          '0066921','0405296','1130884','1179933','0120630','0268126','0137523','0374900','8772262','0116996','0107290','7339248']

ryan = ['0166924','2866360','0050825','2798920','3416742','0060827','1817273','0338013','0482571','5715874','2316411','4550098']

karyn = ['4425200','0464141','1465522','0093779','0099810','0076759','3748528','6763664','0317740','2798920','0096283','0258463','0118799','0058092','0107290','0045152','0106364']

richard = ['0074119','0064115','0070735','0080474','0061512','0067774','0057115','0070511','0081283',
           '0065126','0068421','0078227','0079100','0078966','0081696','0082085','0072431','0075784',
           '0093640','0098051','0094226','0097576','0099810','0081633','0080761','0077975','0085244','0095159','0101969']

joe = ['6335734','0291350','0113568','0208502','0169858','0095327','0097814','0983213','0094625','7089878']

lena = ['1990314','3236120','1816518','0241527','0097757','0268978','0467406','2543164','2245084','3741834']

wade = ['0118665','0270846','0288441','2287250','2287238','8668804','9448868','1702443','1608290','5519340']

aj1 = ['0087995','0118694','0181689','0061184','0063032','2402927','4633694','0058946','0103074','0060196',
       '2543164','0109445','0245429','5105250','0088846','0370986','0246578','0053114','0014429','0047478',
       '0081505','2396224','0054215','1259521','0096283','0095159','0093779','0087544']

aj2 = ['0173716','0086541','0119809','0109445','0112887','0120879','0081455','0079813','0087995','0156610',
       '0097940','0089886','0088846','0090967','1523483','0109424','0102536','0105793','0246578','0370986']
```

# Optional: Plot my movies

I used this code to generate vectors and metadata for the Tensorflow projector. It lets you view a 3D projection of up to 10k movies and reduce their dimensionality with several algorithms. I even inserted my own taste vector in there so I could view the nearest movies in the projection.

[Here is the interactive visualization of our movie embeddings, specifically the top 10k popular movies.](https://projector.tensorflow.org/?config=https%3A%2F%2Fraw.githubusercontent.com%2Fcoopwilliams%2Fw2v_movie_projector%2Fmaster%2Fprojector_config_top_10k.json)

[This one includes all movies, so a lot of the points are really obscure stuff.](https://projector.tensorflow.org/?config=https%3A%2F%2Fraw.githubusercontent.com%2Fcoopwilliams%2Fw2v_movie_projector%2Fmaster%2Fprojector_config_all.json)

I went on to plot the UMAP projection myself, with middling results.

```python
!pip install umap
```

    Requirement already satisfied: umap in c:\users\cooper\anaconda3\lib\site-packages (0.1.1)
    

    WARNING: You are using pip version 20.0.1; however, version 20.0.2 is available.
    You should consider upgrading via the 'c:\users\cooper\anaconda3\python.exe -m pip install --upgrade pip' command.
    


```python
import numpy as np
import umap
reducer = umap.UMAP(random_state=42)
```


```python
s = Recommender('models/w2v_limitingfactor_v3.36.model')
len(s._get_model().wv.vocab.keys())
```




    24691




```python
all_movies = np.array(s._get_model().wv.vectors)
my_movies = np.array([s._get_model()[i] for i in hist_list+good_list+bad_list if i in s._get_model()])
histed = np.array([1 for i in hist_list if i in s._get_model()])
gooded = np.array([2 for i in good_list if i in s._get_model()])
badded = np.array([3 for i in bad_list if i in s._get_model()])

legend = np.concatenate((histed, gooded, badded))
```


```python
all_movies.shape
```




    (24691, 100)




```python
legend.shape
```




    (711,)




```python
np.savetxt('all_movie_vectors.tsv', all_movies, delimiter="\t")
```


```python
reducer = reducer.fit(all_movies)
```


```python
embedding = reducer.transform(all_movies)
```


```python

```




    (9505, 13)




```python
s = Recommender('models/w2v_limitingfactor_v3.36.model')
# s.id_book = s.id_book[s.id_book['num_votes']>8000]
clf = s._get_model()

id_list = s.id_book['id'].tolist()
title_list = s.id_book['primaryTitle'].tolist()
year_list = s.id_book['startYear'].tolist()
runtime_list = s.id_book['runtimeMinutes'].tolist()
genres_list = s.id_book['genres'].tolist()
rating_list = s.id_book['average_rating'].tolist()
votes_list = s.id_book['num_votes'].tolist()

included = []
metadata = [['title', 'year', 'avg_rating', '#votes', 'genres', 'runtime_min', 'id']]
for pos, i in enumerate(id_list):
    i = fill_id(i)
    if str(i) in clf.wv.vocab:
        included.append(clf[i])
        metadata.append([title_list[pos], year_list[pos], rating_list[pos], votes_list[pos], genres_list[pos], runtime_list[pos] ,fill_id(id_list[pos])])

included.append(z.my_vec)
metadata.append(['Coop Williams taste vector', '2020', '4', '389', 'action', '25', '0000000'])
        
all_movies = np.array(id_list)
all_metadata = np.array(metadata)
np.savetxt('vectors_all.tsv', included, delimiter="\t")
np.savetxt('metadata_all.tsv', all_metadata, delimiter="\t", fmt='%1s', encoding='utf-8')
```


```python
print(len(all_metadata), len(included))
```

    24692 24691
    


```python
np.array(included).shape
```




    (24691, 100)




```python
s.id_book.columns
```


```python
test = s._get_info(x) for x in 
```


```python
!pip install bokeh
```

    Requirement already satisfied: bokeh in c:\users\cooper\anaconda3\lib\site-packages (1.2.0)
    Requirement already satisfied: numpy>=1.7.1 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (1.16.4)
    Requirement already satisfied: packaging>=16.8 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (19.0)
    Requirement already satisfied: tornado>=4.3 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (6.0.3)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (2.8.0)
    Requirement already satisfied: PyYAML>=3.10 in c:\users\cooper\appdata\roaming\python\python37\site-packages (from bokeh) (3.13)
    Requirement already satisfied: six>=1.5.2 in c:\users\cooper\appdata\roaming\python\python37\site-packages (from bokeh) (1.11.0)
    Requirement already satisfied: pillow>=4.0 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (6.1.0)
    Requirement already satisfied: Jinja2>=2.7 in c:\users\cooper\anaconda3\lib\site-packages (from bokeh) (2.10.1)
    Requirement already satisfied: pyparsing>=2.0.2 in c:\users\cooper\anaconda3\lib\site-packages (from packaging>=16.8->bokeh) (2.4.0)
    Requirement already satisfied: MarkupSafe>=0.23 in c:\users\cooper\anaconda3\lib\site-packages (from Jinja2>=2.7->bokeh) (1.1.1)
    

    WARNING: You are using pip version 20.0.1; however, version 20.0.2 is available.
    You should consider upgrading via the 'c:\users\cooper\anaconda3\python.exe -m pip install --upgrade pip' command.
    


```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bokeh.plotting import figure, show, output_file

# fig = px.scatter([1, 2]) 
# #             c=[sns.color_palette()[x] for x in legend])
# # plt.gca().set_aspect('equal', 'datalim')
# fig.show()

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

p = figure(tools=TOOLS)

p.scatter(embedding[:, 0], embedding[:, 1], fill_alpha=0.6,
          line_color=None)

output_file("color_scatter.html", title="color_scatter.py example")

show(p)  # open a browser
```

# Score the models on various test cases

The below cell is your one-stop shop for scoring the various trained models and comparing their performance. Before using it, check the following:

    - The database connection is live (top cell of this notebook).
    
    - You have run the two cells under the headings "Define inferencing functions" and "prep generic data"
    
    - `id_book` is reading a CSV containing data from the IMDb title.basics.tsv document found [here](https://datasets.imdbws.com/).
    
    - `test_users` list contains the names of folders containing users' Letterboxd data that you want to use for scoring.
    
    - `model_list` contains the names of the versions you want to score.
    
Scoring is acheived using two metrics:
    1. Watchlist validation: How many movies were found that the user has already indicated interest in?
    2. Avg. Rating: What is the average rating of the movies recommended?
    
The cell below scores all models with these metrics and records the results in three dataframes. The first two dataframes, match_test_results and rating_test_results, use only different configurations of the 'cooper' data. This is meant to demonstrate various scenarios where more or less data are provided from the user, and different settings are selected. The third dataframe records scoring from both metrics across all users you care to test. For the sake of privacy, I have not included my test data with this repo, so future experimenters will have to gather their own Letterboxd data.


```python
# log test results for 3 tests.

# df to lookup ids from titles
id_book = pd.read_csv('exported_data/title_basics_small.csv')

# define "user-test" test cases. set 'cooper' to last so it defines good_list, bad_list etc. for the cooper test cases
# each name must correspond to a folder of letterboxd data under "exported_data"
'''EXAMPLE file structure
current directory
    |
    thisnotebook.ipynb
    w2v_someversion.model
    ...
    |
    /exported_data
        |
        /eric
        ... other folders
        /cooper
            |
            ratings.csv
            watched.csv
            watchlist.csv
'''
test_users = ['eric', 'wade', 'aj', 'kelly', 'cooper', 'thomas'] 

# these names must match the file names of the versions you want to test, without prefix or extensions
model_list = ['mistakenot',
              'limitingfactor_v1', 'limitingfactor_v2', 
              'limitingfactor_v3', 'limitingfactor_v3.36', 'limitingfactor_v3.5', 'limitingfactor_v3.51', 'limitingfactor_v3.6', 
              'limitingfactor_v4', 'limitingfactor_v4.1', 'limitingfactor_v4.12']

###########################################################################################
# Nothing below this needs to be configured, unless you want to change the tests themselves.
###########################################################################################

# import user Letterboxd data
test_users_data = {}
for user in test_users:
    user_data = {}
    path = f"exported_data/letterboxd/{user}/"
    ratings = pd.read_csv(f'{path}ratings.csv')
    watched = pd.read_csv(f'{path}watched.csv')
    watchlist = pd.read_csv(f'{path}watchlist.csv')
    good_list, bad_list, hist_list, val_list, ratings_dict = prep_data(
                                    ratings, watched_df=watched, watchlist_df=watchlist, good_threshold=4, bad_threshold=3)
    user_data.update([("good_list", good_list), ("bad_list", bad_list), 
                      ("hist_list", hist_list), ("val_list", val_list), 
                      ("ratings_dict", ratings_dict)])
    test_users_data.update([(str(user), user_data)])

# empty dictionary to use when user elects not to give extra weight to extreme ratings
no_weights = {}

# define "cooper" test cases
params = [
    (cooper_recent, [], 1, ratings_dict), #1
    (cooper_recent, bad_list, 1, ratings_dict),
    (cooper_recent, bad_list, 2, ratings_dict),
    (cooper_recent, bad_list, 3, ratings_dict),
    (good_list, [], 1, ratings_dict), #5
    (good_list, bad_list, 1, ratings_dict),
    (good_list, bad_list, 2, ratings_dict),
    (good_list, bad_list, 3, ratings_dict),
    (cooper_recent, [], 1, no_weights),
    (cooper_recent, bad_list, 1, no_weights), #10
    (cooper_recent, bad_list, 2, no_weights),
    (cooper_recent, bad_list, 3, no_weights),
    (good_list, [], 1, no_weights),
    (good_list, bad_list, 1, no_weights),
    (good_list, bad_list, 2, no_weights), #15
    (good_list, bad_list, 3, no_weights),
#     (good_3p5_list, [], 1),
#     (good_3p5_list, bad_list, 1), #10
#     (good_3p5_list, bad_list, 2),
#     (good_3p5_list, bad_list, 3),
#     (cooper_recent, karyn, 1),
#     (good_list, karyn, 1),
#     (good_3p5_list, karyn, 1), #15
]

mods = dict(zip(model_list, ["models/w2v_" + x + ".model" for x in model_list])) # zip model paths with their names

# attributes/training hyperparameters we want to tabulate
attr_list = ['vector_size', 'corpus_count', 'corpus_total_words', 
             'window', 'sg', 'hs', 'negative', 
             'alpha', 'min_alpha', 
             'sample', 'epochs']

# the three tests we will run
match_test_results = pd.DataFrame({'model':model_list}) # Test different settings for "cooper", score on watchlist items found
rating_test_results = pd.DataFrame({'model':model_list}) # Test different settings for "cooper", score on avg. rating of recs
user_test_results = pd.DataFrame({'model':model_list}) # Test different users on the same settings, score on watchlist items found


# Get model training parameters for models
for attr in attr_list:
    match_test_results[str(attr)] = match_test_results['model'].apply(lambda x: getattr(Recommender(mods[x])._get_model(), attr))
    rating_test_results[str(attr)] = match_test_results[str(attr)]

# add ns_exponent parameter, which is buried in the model.vocabulary (can't be gotten with getattr())
for df in [match_test_results, rating_test_results]:
    df['ns_exponent'] = df['model'].apply(lambda x: Recommender(mods[x])._get_model().vocabulary.ns_exponent)

# get match scores, average ratings, attributes for all "cooper" test cases
count = 0
for i, j, k, m in params:
        count+=1
        print(count, "\t")
        match_test_results[str(count)] = match_test_results['model'].apply(
                                                lambda x: Recommender(mods[x]).predict(input=i, bad_movies=j, 
                                                                                          val_list=val_list, ratings_dict=m, 
                                                                                          n=100, harshness=k, 
                                                                                          rec_movies=False, show_vibes=False, 
                                                                                          scoring=True, return_scores=True))
        rating_test_results[str(count)] = match_test_results[str(count)].apply(lambda x: x[1])
        match_test_results[str(count)] = match_test_results[str(count)].apply(lambda x: x[0])
        
for user, data in test_users_data.items():
    print(str(user))
    user_test_results[str(user)] = user_test_results['model'].apply(
                                                lambda x: Recommender(mods[x]).predict(input=data['good_list'], 
                                                                                        bad_movies=data['bad_list'], 
                                                                                        val_list=data['val_list'], 
                                                                                        ratings_dict=data['ratings_dict'], 
                                                                                        n=100, harshness=1, 
                                                                                        rec_movies=False, show_vibes=False, 
                                                                                        scoring=True, return_scores=True))
        
```

    1 	
    2 	
    3 	
    4 	
    5 	
    6 	
    7 	
    8 	
    9 	
    10 	
    11 	
    12 	
    13 	
    14 	
    15 	
    16 	
    eric
    wade
    aj
    kelly
    cooper
    thomas
    

## Test cases legend:

```
1 	(cooper_recent, [], 1, ratings_dict)

2 	(cooper_recent, bad_list, 1, ratings_dict)

3 	(cooper_recent, bad_list, 2, ratings_dict)

4 	(cooper_recent, bad_list, 3, ratings_dict)

5 	(good_list, [], 1, ratings_dict)

6 	(good_list, bad_list, 1, ratings_dict) 
        # hardcore movie fan looking for something crazy

7 	(good_list, bad_list, 2, ratings_dict)

8 	(good_list, bad_list, 3, ratings_dict)

9 	(cooper_recent, [], 1, no_weights) 
        # small list of movies with only a few good movie ratings

10 	(cooper_recent, bad_list, 1, no_weights) 
        # small list of movies with some bad movies too

11 	(cooper_recent, bad_list, 2, no_weights)

12 	(cooper_recent, bad_list, 3, no_weights)

13 	(good_list, [], 1, no_weights) 
        # Unlikely case

14 	(good_list, bad_list, 1, no_weights) 
        # Hardcore movie fan

15 	(good_list, bad_list, 2, no_weights)

16 	(good_list, bad_list, 3, no_weights)
```

### Show test results

Overall, the best models scored seem to be limitingfactor_v1 and limitingfactor_v3.51. Unfortunately I did not have the presence of mind to document the settings for the former's training data. But I believe it may have been `m=7, n=10`.


```python
pd.set_option('display.max_columns', 50)
match_test_results # for each model: training params, validation score out of 100 for 16 test cases
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>vector_size</th>
      <th>corpus_count</th>
      <th>corpus_total_words</th>
      <th>window</th>
      <th>sg</th>
      <th>hs</th>
      <th>negative</th>
      <th>alpha</th>
      <th>min_alpha</th>
      <th>sample</th>
      <th>epochs</th>
      <th>ns_exponent</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mistakenot</td>
      <td>100</td>
      <td>17812</td>
      <td>904140</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0010</td>
      <td>10</td>
      <td>0.75</td>
      <td>5</td>
      <td>9</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>11</td>
      <td>19</td>
      <td>10</td>
      <td>5</td>
      <td>11</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>13</td>
      <td>18</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>limitingfactor_v1</td>
      <td>100</td>
      <td>17812</td>
      <td>817687</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>13</td>
      <td>14</td>
      <td>13</td>
      <td>14</td>
      <td>18</td>
      <td>21</td>
      <td>29</td>
      <td>26</td>
      <td>16</td>
      <td>20</td>
      <td>22</td>
      <td>17</td>
      <td>18</td>
      <td>19</td>
      <td>28</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>limitingfactor_v2</td>
      <td>100</td>
      <td>13641</td>
      <td>766970</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>15</td>
      <td>17</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>21</td>
      <td>28</td>
      <td>23</td>
      <td>13</td>
      <td>19</td>
      <td>20</td>
      <td>17</td>
      <td>13</td>
      <td>19</td>
      <td>27</td>
      <td>22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>limitingfactor_v3</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>10</td>
      <td>13</td>
      <td>14</td>
      <td>13</td>
      <td>18</td>
      <td>26</td>
      <td>32</td>
      <td>29</td>
      <td>20</td>
      <td>21</td>
      <td>23</td>
      <td>24</td>
      <td>17</td>
      <td>25</td>
      <td>31</td>
      <td>28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>limitingfactor_v3.36</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>1500</td>
      <td>0.30</td>
      <td>13</td>
      <td>16</td>
      <td>14</td>
      <td>12</td>
      <td>19</td>
      <td>35</td>
      <td>31</td>
      <td>28</td>
      <td>19</td>
      <td>16</td>
      <td>20</td>
      <td>21</td>
      <td>17</td>
      <td>35</td>
      <td>33</td>
      <td>28</td>
    </tr>
    <tr>
      <th>5</th>
      <td>limitingfactor_v3.5</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.35</td>
      <td>6</td>
      <td>12</td>
      <td>10</td>
      <td>10</td>
      <td>20</td>
      <td>27</td>
      <td>30</td>
      <td>28</td>
      <td>17</td>
      <td>21</td>
      <td>25</td>
      <td>20</td>
      <td>18</td>
      <td>24</td>
      <td>31</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>limitingfactor_v3.51</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>500</td>
      <td>0.35</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>32</td>
      <td>32</td>
      <td>26</td>
      <td>19</td>
      <td>20</td>
      <td>27</td>
      <td>27</td>
      <td>16</td>
      <td>33</td>
      <td>32</td>
      <td>25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>limitingfactor_v3.6</td>
      <td>100</td>
      <td>44438</td>
      <td>986835</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>1500</td>
      <td>0.30</td>
      <td>13</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>27</td>
      <td>47</td>
      <td>44</td>
      <td>41</td>
      <td>20</td>
      <td>20</td>
      <td>34</td>
      <td>35</td>
      <td>27</td>
      <td>48</td>
      <td>43</td>
      <td>41</td>
    </tr>
    <tr>
      <th>8</th>
      <td>limitingfactor_v4</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>30</td>
      <td>0.50</td>
      <td>2</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>limitingfactor_v4.1</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>100</td>
      <td>0.50</td>
      <td>6</td>
      <td>11</td>
      <td>8</td>
      <td>9</td>
      <td>3</td>
      <td>19</td>
      <td>14</td>
      <td>8</td>
      <td>3</td>
      <td>12</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>21</td>
      <td>12</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10</th>
      <td>limitingfactor_v4.12</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>500</td>
      <td>0.50</td>
      <td>5</td>
      <td>15</td>
      <td>10</td>
      <td>8</td>
      <td>4</td>
      <td>28</td>
      <td>15</td>
      <td>11</td>
      <td>6</td>
      <td>15</td>
      <td>14</td>
      <td>12</td>
      <td>4</td>
      <td>30</td>
      <td>15</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
rating_test_results # for each model: training params, average rating of movies recommended for 16 test cases
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>vector_size</th>
      <th>corpus_count</th>
      <th>corpus_total_words</th>
      <th>window</th>
      <th>sg</th>
      <th>hs</th>
      <th>negative</th>
      <th>alpha</th>
      <th>min_alpha</th>
      <th>sample</th>
      <th>epochs</th>
      <th>ns_exponent</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mistakenot</td>
      <td>100</td>
      <td>17812</td>
      <td>904140</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0010</td>
      <td>10</td>
      <td>0.75</td>
      <td>7.011224</td>
      <td>7.272165</td>
      <td>7.114433</td>
      <td>7.092784</td>
      <td>6.901124</td>
      <td>7.245918</td>
      <td>7.402410</td>
      <td>6.954321</td>
      <td>7.267010</td>
      <td>7.111111</td>
      <td>7.225773</td>
      <td>7.240625</td>
      <td>6.906818</td>
      <td>7.255102</td>
      <td>7.397590</td>
      <td>7.011111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>limitingfactor_v1</td>
      <td>100</td>
      <td>17812</td>
      <td>817687</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>7.468041</td>
      <td>7.365306</td>
      <td>7.410417</td>
      <td>7.416667</td>
      <td>8.065574</td>
      <td>7.453571</td>
      <td>8.141667</td>
      <td>8.250000</td>
      <td>7.512903</td>
      <td>7.411458</td>
      <td>7.579570</td>
      <td>7.521739</td>
      <td>8.066129</td>
      <td>7.429213</td>
      <td>8.137288</td>
      <td>8.246296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>limitingfactor_v2</td>
      <td>100</td>
      <td>13641</td>
      <td>766970</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>7.443299</td>
      <td>7.403061</td>
      <td>7.365625</td>
      <td>7.361458</td>
      <td>7.987097</td>
      <td>7.512791</td>
      <td>8.106897</td>
      <td>8.175000</td>
      <td>7.427957</td>
      <td>7.445833</td>
      <td>7.507447</td>
      <td>7.452174</td>
      <td>7.936508</td>
      <td>7.468132</td>
      <td>8.098276</td>
      <td>8.174510</td>
    </tr>
    <tr>
      <th>3</th>
      <td>limitingfactor_v3</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.50</td>
      <td>7.435714</td>
      <td>7.303061</td>
      <td>7.395876</td>
      <td>7.383505</td>
      <td>8.227273</td>
      <td>7.562500</td>
      <td>8.268966</td>
      <td>8.386792</td>
      <td>7.554348</td>
      <td>7.306383</td>
      <td>7.509677</td>
      <td>7.595699</td>
      <td>8.189286</td>
      <td>7.551648</td>
      <td>8.255932</td>
      <td>8.339623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>limitingfactor_v3.36</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>1500</td>
      <td>0.30</td>
      <td>7.441237</td>
      <td>7.351546</td>
      <td>7.308333</td>
      <td>7.325000</td>
      <td>8.222414</td>
      <td>7.767500</td>
      <td>8.316364</td>
      <td>8.301852</td>
      <td>7.652128</td>
      <td>7.293617</td>
      <td>7.582796</td>
      <td>7.575532</td>
      <td>8.148276</td>
      <td>7.711628</td>
      <td>8.298214</td>
      <td>8.320000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>limitingfactor_v3.5</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>90</td>
      <td>0.35</td>
      <td>7.466327</td>
      <td>7.377778</td>
      <td>7.360825</td>
      <td>7.400000</td>
      <td>8.190164</td>
      <td>7.559036</td>
      <td>8.287931</td>
      <td>8.325926</td>
      <td>7.651064</td>
      <td>7.286458</td>
      <td>7.575269</td>
      <td>7.575269</td>
      <td>8.173333</td>
      <td>7.541111</td>
      <td>8.246552</td>
      <td>8.320755</td>
    </tr>
    <tr>
      <th>6</th>
      <td>limitingfactor_v3.51</td>
      <td>100</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>500</td>
      <td>0.35</td>
      <td>7.509184</td>
      <td>7.389691</td>
      <td>7.360825</td>
      <td>7.355208</td>
      <td>7.931148</td>
      <td>7.727160</td>
      <td>8.172414</td>
      <td>8.001724</td>
      <td>7.582796</td>
      <td>7.337895</td>
      <td>7.644086</td>
      <td>7.611957</td>
      <td>7.862903</td>
      <td>7.702326</td>
      <td>8.177586</td>
      <td>7.963793</td>
    </tr>
    <tr>
      <th>7</th>
      <td>limitingfactor_v3.6</td>
      <td>100</td>
      <td>44438</td>
      <td>986835</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>1500</td>
      <td>0.30</td>
      <td>7.101020</td>
      <td>7.320408</td>
      <td>7.312245</td>
      <td>7.293814</td>
      <td>8.351724</td>
      <td>8.074667</td>
      <td>8.295455</td>
      <td>8.327419</td>
      <td>7.592473</td>
      <td>7.532979</td>
      <td>7.747872</td>
      <td>7.770968</td>
      <td>8.351724</td>
      <td>8.041772</td>
      <td>8.279104</td>
      <td>8.329032</td>
    </tr>
    <tr>
      <th>8</th>
      <td>limitingfactor_v4</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>30</td>
      <td>0.50</td>
      <td>7.264646</td>
      <td>7.122222</td>
      <td>7.274490</td>
      <td>7.342857</td>
      <td>6.915464</td>
      <td>7.119792</td>
      <td>7.429348</td>
      <td>7.180000</td>
      <td>6.812121</td>
      <td>6.647312</td>
      <td>6.771277</td>
      <td>6.627083</td>
      <td>6.853608</td>
      <td>7.138144</td>
      <td>7.407778</td>
      <td>7.149474</td>
    </tr>
    <tr>
      <th>9</th>
      <td>limitingfactor_v4.1</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>100</td>
      <td>0.50</td>
      <td>7.292857</td>
      <td>7.184694</td>
      <td>7.282653</td>
      <td>7.303061</td>
      <td>6.713830</td>
      <td>7.515476</td>
      <td>7.019048</td>
      <td>6.835632</td>
      <td>6.558696</td>
      <td>7.260000</td>
      <td>6.797778</td>
      <td>6.534444</td>
      <td>6.694624</td>
      <td>7.516092</td>
      <td>6.939286</td>
      <td>6.860920</td>
    </tr>
    <tr>
      <th>10</th>
      <td>limitingfactor_v4.12</td>
      <td>300</td>
      <td>35371</td>
      <td>933359</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>10</td>
      <td>0.03</td>
      <td>0.0007</td>
      <td>0.0001</td>
      <td>500</td>
      <td>0.50</td>
      <td>7.208163</td>
      <td>7.274227</td>
      <td>7.267010</td>
      <td>7.209278</td>
      <td>6.677273</td>
      <td>7.671795</td>
      <td>6.970667</td>
      <td>6.781176</td>
      <td>7.009091</td>
      <td>7.313483</td>
      <td>7.265909</td>
      <td>7.193103</td>
      <td>6.715730</td>
      <td>7.687805</td>
      <td>6.975000</td>
      <td>6.732143</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_test_results # for each model: validation score and avg. rating for a user's inputs.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>eric</th>
      <th>wade</th>
      <th>aj</th>
      <th>kelly</th>
      <th>cooper</th>
      <th>thomas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mistakenot</td>
      <td>(1, 7.299999999999996)</td>
      <td>(4, 7.682474226804127)</td>
      <td>(6, 7.234375000000001)</td>
      <td>(8, 7.558064516129031)</td>
      <td>(22, 7.633333333333332)</td>
      <td>(11, 7.245918367346941)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>limitingfactor_v1</td>
      <td>(3, 7.618888888888891)</td>
      <td>(6, 7.460465116279073)</td>
      <td>(3, 7.043333333333335)</td>
      <td>(11, 7.730232558139535)</td>
      <td>(21, 7.665151515151515)</td>
      <td>(21, 7.45357142857143)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>limitingfactor_v2</td>
      <td>(1, 7.59186046511628)</td>
      <td>(10, 7.539325842696628)</td>
      <td>(6, 7.212765957446808)</td>
      <td>(12, 7.739772727272731)</td>
      <td>(21, 7.667187500000001)</td>
      <td>(21, 7.5127906976744185)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>limitingfactor_v3</td>
      <td>(4, 7.673493975903615)</td>
      <td>(11, 7.539759036144578)</td>
      <td>(5, 7.288043478260868)</td>
      <td>(16, 7.817857142857145)</td>
      <td>(22, 7.807272727272728)</td>
      <td>(26, 7.5625)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>limitingfactor_v3.36</td>
      <td>(4, 7.841772151898734)</td>
      <td>(17, 7.604225352112677)</td>
      <td>(6, 7.496666666666672)</td>
      <td>(20, 7.980000000000001)</td>
      <td>(28, 7.989795918367348)</td>
      <td>(35, 7.7675)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>limitingfactor_v3.5</td>
      <td>(4, 7.746428571428571)</td>
      <td>(10, 7.508045977011496)</td>
      <td>(6, 7.327472527472528)</td>
      <td>(14, 7.845000000000001)</td>
      <td>(22, 7.784210526315789)</td>
      <td>(27, 7.559036144578312)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>limitingfactor_v3.51</td>
      <td>(5, 7.801219512195122)</td>
      <td>(15, 7.611538461538464)</td>
      <td>(6, 7.559340659340662)</td>
      <td>(16, 7.940789473684211)</td>
      <td>(25, 7.8999999999999995)</td>
      <td>(32, 7.727160493827163)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>limitingfactor_v3.6</td>
      <td>(5, 8.104615384615384)</td>
      <td>(17, 7.642253521126764)</td>
      <td>(19, 8.032499999999997)</td>
      <td>(22, 8.120000000000003)</td>
      <td>(19, 8.171875)</td>
      <td>(47, 8.074666666666669)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>limitingfactor_v4</td>
      <td>(0, 6.976842105263159)</td>
      <td>(5, 7.422222222222221)</td>
      <td>(1, 6.941935483870969)</td>
      <td>(3, 7.352127659574468)</td>
      <td>(2, 7.329032258064515)</td>
      <td>(5, 7.119791666666667)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>limitingfactor_v4.1</td>
      <td>(0, 7.341025641025642)</td>
      <td>(8, 7.38875)</td>
      <td>(4, 6.950000000000001)</td>
      <td>(9, 7.545454545454544)</td>
      <td>(9, 7.366129032258064)</td>
      <td>(19, 7.5154761904761855)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>limitingfactor_v4.12</td>
      <td>(3, 7.556944444444446)</td>
      <td>(11, 7.363636363636366)</td>
      <td>(4, 7.296590909090906)</td>
      <td>(18, 7.788405797101449)</td>
      <td>(18, 7.604347826086957)</td>
      <td>(28, 7.67179487179487)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # Export test results
match_test_results.to_csv("w2v_match_test_results.csv", index=False)
rating_test_results.to_csv("w2v_rating_test_results.csv", index=False)
user_test_results.to_csv("w2v_user_test_results.csv", index=False)
```

# The End

In the original notebook, I show some examples of recommendations given from different parameters. Since the output is extremely long, I've left it out of this markdown version.
