---
permalink: /portfolio/
title: "Portfolio"
---
placeholder


### Gróa - an open source, NLP-based movie discovery system
<figure>
	<iframe width="560" height="315" src="https://www.youtube.com/embed/-XXOhunofT8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	<figcaption>Live Demo of the Gróa web prototype by the author and a co-conspirator.
	</figcaption>
</figure>

Commercial movie recommenders are closed source and coupled with IP silos, leading them to recommend movies within a relatively small subset of all movies. Moreoever, they are designed to maximize time spent on the streaming platform, rather than enjoyment and discovery. Gróa combines the totality of IMDb movie reviews with tried-and-true recommendation techniques to provide a user-driven movie discovery experience that imports user data from IMDb and Letterboxd.

As the project's machine learning engineer, I trained Word2Vec on positive user ratings histories to create a user-based collaborative filtering recommender. The algorithm embeds over 97,000 movie IDs into a 100-dimensional vector space according to their co-occurence in a user's positive ratings history. The ID for each movie is a key for its vector, which can be called from the model and compared with any other vector in that space for cosine-similarity. To provide recommendations given a new user's watch history, we simply find the vector average of the user's choice of "good movies" and find the top-n cosine-similar vectors from the model. We can improve the recommendations by subtracting a "bad movies" vector from the "good movies" vector before inferencing. Models trained in this way can be tested by treating a user's watchlist (unwatched movies saved for later) as a validation set.

The above model fulfills most requirements for a general-purpose movie recommender system. However, it is unable to make riskier recommendations for movies that a majority of reviewers do not enjoy (cult movies). To satisfy users who seek underrated movies, we also trained Doc2Vec on user review histories to create a review-based collaborative filtering model. This model does not recommend movies, but finds reviewers who write similarly to a new user. We then query the review database for positive reviews from these users, both in cases where the ratings count is 1k-10k (hidden gems), and where the reviewer rates a movie 3 stars more than the average.

The lightning-fast inferencing of the Word2Vec/Doc2vec algorithms allows us to incorporate user feedback into progressively updating recommendations. If the user elects to approve or disapprove of a movie, its corresponding vector is added to, or subtracted from, the user's overall taste vector. Weighting these feedback vectors by a factor like 1.2 increases the influence of that feedback on the user's taste vector, and this factor can be tweaked to change the effective "learning rate" of the re-recommendations process.


### COVID News Analysis - scoring news outlets on proactive and responsible COVID reporting

placeholder

### Camera Price Predictor - a tool for exploring the relationship between camera prices and features

placeholder