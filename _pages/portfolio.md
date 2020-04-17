---
permalink: /portfolio/
title: "Portfolio"
---


# My skills include:
Data science/software engineering - Python (Pandas, NumPy, Scikit-Learn, Keras, fast.ai, Gensim, Django, Flask, Dash), PostgreSQL, SQLite, MongoDB, HBase, Docker, AWS (EC2, ECR, RDS, Sagemaker), Excel, Tableau, Rust, Scheme, Git, Agile.

Other - Adobe Creative Suite, videography, technical communication.


### Gróa - an open source, NLP-based movie discovery system
<figure>
	<iframe width="560" height="315" src="https://www.youtube.com/embed/-XXOhunofT8" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	<figcaption>Live Demo of the Gróa web prototype by the author and a co-conspirator.
	</figcaption>
</figure>

Gróa is a movie recommender system that lets users upload their own data to interactively receive movie recommendations. I worked on this project as lead ML engineer, building the core functionality around Gensim's Word2Vec and Doc2Vec algorithms. I trained Word2Vec on positive user ratings histories to create a user-based collaborative filtering recommender, which I validated using user watchlists. Visit [here](https://projector.tensorflow.org/?config=https://raw.githubusercontent.com/coopwilliams/w2v_movie_projector/master/projector_config_top_10k.json) to play around with a visualization of the vector embeddings. I also helped with scraping IMDb's entire reviews database, conducted market research, and designed much of the front-end for the prototype.

This project is still in development by a team of data scientists and web developers. You can check it out at [groa.us](https://www.groa.us/). Be aware, though, that the latest iteration of the project may not have all the features shown in the video above.

### Thinkful Data Analytics Video Curriculum
<figure>
	<iframe width="560" height="315" src="https://www.youtube.com/embed/avXw2krbxgI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	<figcaption>Thinkful video lecture about responsible data storytelling.
	</figcaption>
</figure>

In Q4 2019, I wrote and produced a full suite of lecture and demo videos for Thinkful's online [Data Analytics program](https://www.thinkful.com/bootcamp/data-analytics/flexible/). My lessons covered Python, Excel, SQL, Tableau, statistics, data visualization, and BI techniques.

### Camera Price Predictor - a tool for exploring the relationship between camera prices and features

This web app predicts the price of a hypothetical (pre-2008) digital camera with a set of user-defined features. I built it using Dash and Scikit-Learn's random forest regressor. [View the code here.](https://github.com/coopwilliams/camera_prices)