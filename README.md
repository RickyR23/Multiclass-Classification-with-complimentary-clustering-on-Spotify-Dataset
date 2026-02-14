# Classification via Clustering on Music-based dataset
**What this is**
This archived repo contains a research/implementation of the classification analysis of three ML-based algorithms for music genre classification using multi-class classification analysis and compares how different types of audio features impact model performance between two music datasets (GTZAN & Spotify)
The main goal for this project is to find hidden relationships between genres and subgenres at an attempt to find similarities between audio features.

**Datasets utilized for this analysis:**
- **GTZAN**: A more balanced dataset that contains more lower level features.
- **Spotify** Dataset: High level interpreted features that categorize datasets for music that is more intelligble at a "human" level (ie. tempo vs upbeat, respectively)

Three ML-based algorithms were utilized for this research:

- K-nearest neighbor:
Based on pre-analysis of data, KNN was utilized due to its interpretable splits, simple baseline, and it's effective performance with clean and distinctive feature sets (such as GTZAN dataset)
Further classifier usage with KNN was due to its disuse of a 'training phase' as its a lazy learner.

- Decision-tree based algorithm:
Utilized due to its native multi-class classification for structured, 'tabular' type of data. Although innefficient for both datasets, was useful for the visualization of clustered datasets. This ideally works for feature thresholds for class seperations based (mostly based) on entropy.

- Random Forest:



