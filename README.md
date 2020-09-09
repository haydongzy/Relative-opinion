# Relative opinion
Measuring relative opinion from location-based social media

The codes here are derived from the work in the article: Gong Z, Cai T, Thill J-C, Hale S, Graham M (2020) Measuring relative opinion from location-based social media: A case study of the 2016 U.S. presidential election. PLoS ONE 15(5): e0233660. https://doi.org/10.1371/journal.pone.0233660

## Abstract
Social media has become an emerging alternative to opinion polls for public opinion collection, while it is still posing many challenges as a passive data source, such as structurelessness, quantifiability, and representativeness. Social media data with geotags provide new opportunities to unveil the geographic locations of users expressing their opinions. This paper aims to answer two questions: 1) whether quantifiable measurement of public opinion can be obtained from social media and 2) whether it can produce better or complementary measures compared to opinion polls. This research proposes a novel approach to measure the relative opinion of Twitter users towards public issues in order to accommodate more complex opinion structures and take advantage of the geography pertaining to the public issues. To ensure that this new measure is technically feasible, a modeling framework is developed including building a training dataset by adopting a state-of-the-art approach and devising a new deep learning method called Opinion-Oriented Word Embedding. With a case study of tweets on the 2016 U.S. presidential election, we demonstrate the predictive superiority of our relative opinion approach and we show how it can aid visual analytics and support opinion predictions. Although the relative opinion measure is proved to be more robust than polling, our study also suggests that the former can advantageously complement the latter in opinion prediction.

## Introduction
### 1. Data preparation

Input: raw tweet json file

Script: USPE2016_get_tweets.py

Output: csv table with the following columns
[tw_id, user_id, country_code, place_full_name, place_name, place_type, place_bounding_box, tw_coor, quoted_status_user_id, in_reply_to_user_id, retweeted_user_id, mentioned_user_ids, text_message]

### 2. Generate training data from hashtag

Input: csv table from step 1 and identified hashtags for candidates

Script: create_training_data.py

Output: Two text file for candidates and one vocab csv file

### 3. Extract geolocation for users and assign users to states

Input: csv table from step 1

Script: geocode_st.py

Output: csv table with the state column

### 4. Opinion-Oriented Word Embedding training

Input: training files from step 2

Script: Java-based OOWE algorithm

Output: word embedding file

### 5. Embedding aggregation and opinion plot and prediction

Input: embedding file from step 4 and csv table from step 3

Script: plot_emb.py

Output: opinion plots and prediction results

