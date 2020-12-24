# twitter-kMeans-Clustering
This is the third assignment I did when taking the course -- Big Data for Data Science. This assignment required me to familiarize myself with Machine Learning applications on big-data frameworks. This task was done in pyspark.  

### Problem Statement 

To group a large number of tweets into a small number of clusters according to their topics. 

#### Dataset 

Sentiment140 dataset with 1.6 million tweets. (https://www.kaggle.com/kazanova/sentiment140)

### Solution Design 

1. Filter out stopwords 
2. Build a word2vec vector for each tweet using spark's word2vec implementation. 
3. Use MLlib's k-means clustering algorithm to find clusters.  

### Learning outcomes 
1. Learnt to use MLlib's functions and algos  

