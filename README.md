# magazine-semantic-drift


This repository is intended for those who want to compile magazine data and implement semantic drift analysis. More importantly, we take magazine data with article level observations observed at different times (time series data), implement word searches to extract context windows for groups or topics that one wants to see language change for, and create embeddings with parallel computing using BGE/BAAI-large-en-v1.5 model (or BGE/BAAI-small-en-v1.5 model for those who don't want to wait that long!)

Our files calculate three main measures after the data management steps described above. 
- We calculate semantic drift by calculating the average centroid by time bins across different groups then after find measure of distance between centroids across time through cosine similarity.
- The next measure we calculate is cross divergence salience (Wallerstein Distance) by comparing distributions  $D_{M \rightarrow A}$ of how embeddings in one magazine differ from centroids in another magazine within a time bin $m_t$. Then after, we take a difference between distributions after doing this procedure twice (once for each set of centroids in the two magazines) and 

