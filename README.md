# Classification via Clustering on Music-based dataset
**What this is**

This project explores music genre analysis using a combination of supervised classification and unsupervised clustering techniques. Utilizing two main datasets, GTZAN for lower-level (spectral centroid, rolloff, flatness, etc.) in-depth features derived from audio signals, and a Spotify-based dataset with more high-level, human-interpretable/readable features (Upbeat, happiness, sadness, etc.). The following is the analysis of three supervised classification models (K-Nearest Neighbors, Decision Tree, Random Forest), which are utilized to evaluate and understand/determine if these models can determine how well genres can be predicted from each feature representation, and to compare how model performance differs between low-level acoustic and high-level perceptual feature spaces.
With the support of these findings through unsupervised clustering algorithms (K-Means and Mean shift), we complement the classification analysis with further investigation to see if tracks are 'naturally' grouped into clear clusters with labels ignored. With a prior expectation of lower-level feature data sets to display a clearer distinction of apparent genres, we compare the following findings to find a clear picture of how well music genre is encoded within audio features to be classified/clustered, and if sub-genres that are obscure or even niche can be found.     

**Datasets utilized for this analysis**
We utilized two datasets that contrast each other in terms of feature-sets, with:

- **GTZAN**: Considered to be a widely used benchmark for genre classification, contains an in-depth music dataset with features that include lower-level features derived from audio signals. This dataset, containing approximately 1,000 30-second length tracks across 10 different genres, includes lower-level features such as spectral bandwidth and rolloff, RMS energy, and various other audio-derived signal features. Considered highly balanced, this controlled environment is how our chosen classification models distinguish genres from each other based on audio features.

Source: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

- **Spotify Dataset**: A spotify data set with approximately 114,000 data entries obtained through the Spotify Web API (Endpoint for this now deprecated), contains high level, human-interpretable/readable features derived from labels such as danceability, tempo, loudness, liveness, etc., where this dataset is considered highly imbalanced, through the characteristics of music instead of audio signal features.

Source: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/discussion/374642

Utilizing both of these Datasets, we compare the different feature properties from raw audio feature-sets to music, 'vibe', and genre attributes for classification and clustering. Both of these provide insights/perspectives on how genres may be classified with these various features.

**Supervised Classification Models and Unsupervised clustering algorithms used**
Utilizing a combination of supervised classification models and unsupervised clustering algorithms,  we employed supervised classification models to evaluate how well genres can be predicted when labels are available (Comparing Spotify and GTZAN), while unsupervised clustering provides both a supporting/exploratory analysis, through the approach of clustering without genre labels (Only GTZAN). The idea is to combine these approaches to offer a coupled insight into how genres may be seen with the different feature representations, and to find sub-genres that are obscure or niche within smaller/feature-specific groupings.

**Supervised Classification Models:**

- K-Nearest Neighbors: Considered to be a distance-based classifier and a 'lazy learner' due to it not having a training phase for either of our datasets, we utilize this classifier to group tracks to a genre based on a majority class found within a track's nearest neighbors within a feature space. With limitations affecting the performance of this model through redundant features found within the dataset, we mitigate the possibility of mislabeled tracks to a genre by dropping highly correlated features that are found to be redundant, to output well-seperated clusters with a simple baseline.

- Decision Tree Classifier: Considered as an 'eager learner', the Decision Tree classifier is utilized for its construction of a tree structure based on a recursive strategy of feature splits that is done during training and through the learnt decision rules. Although this model is partially susceptible to being affected by overlapping or redundant features, which are both partially found within the GTZAN dataset and mostly found within the Spotify dataset, the sensitivity is quite low in our case for GTZAN due to its balanced class distribution, coupled with its low-level audio signal features. With this classifier model, it may struggle more with the imbalanced Spotify dataset due to its unclear decision boundaries, as it has more high-level perceptual properties where the features can overlap between genres, but it is still a useful baseline for comparing both datasets, their features, and viewing the overlap between genres and less stable decision boundaries.

- Random Forest (RF) Classifier: An ensemble method of a singular decision tree classifier, the Random Forest model combines multiple 'weak' individual decision trees in an attempt to create a more accurate and stable model that is aggregated by prediction. With its higher performance and more accurate decision tree coupling (despite the decision trees possibly containing overlapping genre types and redundant feature spaces), there are still certain limitations with this approach, as classifications won't be as understandable, as it lacks the in-depth decision tree splits. Despite this, RF does reduce overfitting through the avoidance of high variance that you would see in a deeper, more complex singular decision tree. Effective for both the GTZAN and Spotify Datasets, we will see a strong balance between accuracy and readability, particularly for the data that comes up as complex or noisy.

**Unsupervised Clustering Algorithms:**

- K-Means Clustering: With the ability to partition tracks from either dataset into k-amount of clusters through reducing cluster variance, we utilize the elbow method to help select an appropriate number of clusters within a standardized feature space. Utilizing all standardized numeric features, dropping null or missing values throughout the dataset, and identifying compact clusters based on feature similarities is performed for a more feasible distinction/exploration of latent structure for unlabeled genres for tracks, and to see if meaningful genre groupings occur through the visualized output.

- Mean Shift Clustering: A density-based clustering algorithm that was utilized to identify cluster centers through modes in the dataset rather than defining a k-amount of clusters as done with the elbow method in the K-Means algorithm. By applying feature pairs, after cleaning the dataset from null/missing values, meanshift is applied with an estimated bandwidth through user-inputted quantile values alongside a number of samples, used to control the neighborhood for the cluster centers within dense areas. Mean shift, in our case, is utilized to further explore whether dense areas of similar tracks are found with relevant feature spaces to display potential genre or sub-genres without using labels during a training phase.





Presentation from our Classification/Clustering findings (This portion of the presentation mostly drills on clustering algorithms and the results/analysis of the findings)
[Music Genre Classification and Feature Comparison using ML (1).pptx](https://github.com/user-attachments/files/25328598/Music.Genre.Classification.and.Feature.Comparison.using.ML.1.pptx)



*This project was completed in part and with the help of Ryan Arevalo and Kyle Cushing, 2024*
*Credit to the research paper and inspiration for our implementation of these Classification and Clustering analysis:
Zhengxin Qi, Mohammed A. Jassim, Mohamed Rahouti, Nazli Siasi
https://www.researchgate.net/publication/361231964_Music_Genre_Classification_and_Feature_Comparison_using_ML* 


