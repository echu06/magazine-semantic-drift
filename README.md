# magazine-semantic-drift


This repository is intended for those who want to compile magazine data and implement semantic drift analysis. More importantly, we take magazine data with article level observations observed at different times (time series data), implement word searches to extract context windows for groups or topics that one wants to see language change for, and create embeddings with parallel computing using BGE/BAAI-large-en-v1.5 model (or BGE/BAAI-small-en-v1.5 model for those who don't want to wait that long!)

Our files calculate three main measures after the data management steps described above:
- We calculate semantic drift by calculating the average centroid by time bins across different groups then after find measure of distance between centroids across time through cosine similarity.
- The next measure we calculate is cross divergence salience (Wallerstein Distance) by comparing distributions  $D_{M \rightarrow A}$ of how embeddings in one magazine differ from centroids in another magazine within a time bin $m_t$. Then after, we take a difference between distributions after doing this procedure twice (once for each set of centroids in the two magazines) to calculate the distance
- RoBERTa Sentiment, which is calculated through the RoBERTa model. This calculates, neutral, positive, and negative sentiment.


Main Pipeline:
1. The data is first collected in the **Text Data Magazines CSV** folder. We use these datasets and the word_search.py python file to generate the word search data. word_search.py contains a class "keyword_panel_builder" which takes a dictionary of lists and passes them through to the pipeline function to along with the text column to create a new panel dataset. The new dataset is stored in **Word Search Data** for the next step.
   
2. Secondly, we create embeddings in the semantic drift step. There are three things to note. In the **semantic_drift_pipeline.py** file, we call three helper packages created **time_parser.py**, **embeddings_package.py**, and **drift_model.py** which all need to be in the same location to run. The usage of **time_parser.py** is to create splits in our data based off of time bins (ex: split our data into monthly, yearly, or decade data). **embeddings_package** is a package made to call the large/small embeddings model and optimize embedding time by using parallel processing. **drift_model.py** is simply just a helper function which calculates the semantic drift manually through the model. If you simply run the semantic_drift_pipeline.py file, it will take the pre-existing made embeddings in the **Embeddings** folder and use it to calculate semantic drift and store it in **Semantic Drift Data**. See the full function for presets

3. Thirdly, we calculate roBERTa sentiment in **roberta_pipeline.py** using the **time_parser.py** package and other pre-sets. You only need to call the pipeline, add time bins, and text column names to calculate roberta sentiment. The roberta sentiment will be stored in the **Roberta Data** folder. 



