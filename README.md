# National-Basketball-Association
Data Analysis on NBA dataset from Kaggle as part of Project in course CSC522 at NCSU.

## Introduction
The National Basketball Association (NBA) is a professional basketball league in North America that is composed of 30 teams (29 in the United States and 1 in Canada). It is one of the four major professional sports leagues in the United States and Canada and is the premier men’s professional basketball league in the world[6]. This project analyzes the NBA dataset to discover patterns between various aspects of the player and their impact on the game. For an instance— career span of players on the basis of the positions they play or their biological aspects.

### The Challenge: Data
This project has been started with my no previous knowledge in the game of Basketball thus most of the Data mining techniques used, yielded no pattern.

## Data
Dataset has been taken from http://www.cs.cmu.edu/~awm/10701/project/databasebasketball2.0.zip

### Data Preparation
- Cleaning: A variety of techniques have been used to improve the representativeness of the data. The approaches applied are determined by the proportion of data missing against the sample, the type of missing values, and context-aware selection of other correlated variables.
- Draft Dataset: Attribute ilkid had 60% missing percentage. Used listwise deletion for nominal attributes.
- Players Regular Season dataset: Attribute turnover had a 1.5% missing percentage. Since it’s a continuous attribute, replacement done with mean and median of all available values, but a better estimation was obtained using a linear regression utilizing other performance attributes.
- Players dataset: Attributes hfeet, hinches, and weight had 0.25%, 0.25% and 0.5% respectively. These attributes are replaced with the mean and median of all available values since they are continuous. Linear regression was also considered for estimating hfeet against weight and vice versa, however, there are no records where only one from the two is missing. Attributes college and birthdates had 1.5% and 0.5% missing rates.
- Additionally, certain important attributes such as age, retirement age, career spans and debut age which were derived from the given attributes are also added to the original data.

## Basic Pipeline
- Data Visualizaton: K means clustering, silhouette method, elbow method and distortion.
- Data preprocessing: Estimation of null values using Mean, Medians and Linear Regressions. 
- Data mining techniques: Clustering (K-means, elbow method, Silhouette method), Binary Classification (Decision trees, Logistic Regression, KNN, SGD), Ensemble techniques (Random Forest), XGBoost.


## Data Visualizaton and Pattern recognition
For visualizing the raw data and understanding the hidden groups, K-means clustering has been applied for different values of K. Two extra columns for each coach has been made:
1. Win percentage in regular seasons
2. Win percentage in playoff seasons
These two derived attributes were plotted to understand groups of coaches who perform well in seasons or in playoffs or in both. After applying clustering for different values, it was noticed that there are no such distinguishable groups present in the coach data.

### Finding optimal K
- The elbow method in clustering is a heuristic used to determine the number of clusters in a data set.The elbow method here is not clearly indicating the optimal k value. Thus it can be said that, the data is not very clustered intrinsically.
- The silhouette score has been utilized to show how many clusters are ideal. Silhouette score explains the density of each cluster and the separation from cluster to cluster. Once the rate slows down significantly picking a higher number of ‘n’ clusters isn’t very useful[7]. It was found that 2 clusters are sufficient.

### Attribute Redundancy & Usefulness
Generally, we look for the uniqueness of nominal attributes, in order to decide whether the attribute will be useful or not in decision making. In the player data, it was found that, college attribute has 20% of uniqueness. So, it might not be a good factor in order to make any decisions about the success of a player. But, location (background information) and birthdates can be very good decision maker attributes as they have 60% and 80% of uniqueness respectively.

### Feature Selection
Covariance matrix is used in order to determine the features which contribute the most in the data. Principal Component Analysis (PCA) is used on player’s data (in
standardized data), which determined that 6 PCs, that covers approximately 93.28% of the total variance are sufficient and hence can significantly reduce the data size.

## Inference
> K - means clustering: To understand how a team’s coach affects the game plan, we have calculated the season success rate and the playoff success rate. The success rate was calculated as the number of wins upon the total number of games played by a team under a coach. We clustered this data to find a relationship between playoff success and season success under a coach. The Elbow method and the silhouette method used to find the optimal K suggested that K=2 is good for clustering. From the graphs generated we can conclude that our coaches’ data is not suited for clustering.

> Predictions on performance attributes of a player would be reliably performed on discretised data, reducing the precision required and enabling the use of classification techniques. Classification on performance attributes taken as an average across the total number of games played would perform better than classification on cumulative data. Readily available attributes such as position, weight and height can be used to predict performance attributes in the absence of game data. We introduced additional attributes for each performance attribute with values as the value of the respective original attribute divided by the number of games played. Attribute BMI was introduced as a ratio of weight and squared height. We used the silhouette method to determine the optimal number of bins for each attribute, followed by K-means clustering to obtain a label for each value and discretise all attributes. Then, we applied the random forest classifier and the Extreme Gradient Boosted (XGBoost) random forest classifier for each original attribute, using input as all other original attributes. The same was repeated for attributes averaged across games instead of original attributes. Finally, classification was performed using readily available attributes such as position and BMI.

> Decision Tree Algorithm for performance classification: Demographic background and biological history may affect the performance of a player. We tried to predict the player’s performance using his nominal attributes such as height, weight, age, college, state using Decision Tree. Decision Tree is generally really good for this kind of classification problem. We randomly split the dataset into a training and testing set using a 70-30 margin. After tuning the parameters of a decision tree model, we had 750 max leaf nodes as the upper bound of the complexity of a model. We carried out class wise accuracy using that model as the data was unevenly distributed among the class categories. However, we observed that by using those attributes we cannot accurately predict the class of a player

> All-star league prediction: Whether the player will be in the all-star league or not can be identified using his past history. We introduced a new attribute Isallstar in the player dataset. Matching with the all-star player data, we took binary values in that attribute. To handle this binary classification problem, we implemented various methods to fit the data in the model to accurately predict the player can be in the all star league or not using its attributes.

- Logistic Regression: We finetuned the model using over the number of iteration as parameter and then took the best number of iterations which gave the highest recall. Here, the number of all star players is very less than the number of non star players. Using a stratified sampling approach, we mainly focused on how well the predictor predicts the Class ‘1’ rather than the class ‘0’. We used Jacardian distance to measure that. (Figure 1) This model was able to achieve 94.05% overall accuracy on the unseen data and 72.00% recall. The model laid out some important attributes for predicting the output which were: Game played, rebounds, career span, regular season points and assists in the playoffs.

- Decision Tree: DT is a very suitable approach for simple classification problems. We tuned the algorithm for achieving an optimal complexity and achieved 91.64% overall accuracy and 64% Recall on class 1. The complexity of our model is 97 leaf nodes. We again tried to find the important factors which plays an important role for the decisions. Those features are: Free Throw made, Points earned both in regular season, age, steals in playoffs, career span and game played

- K nearest neighbour: We tuned the K value for the best KNN model. We found that the model got overfitted for low values of K but, for K = 9, it gave the minimum error with 7.85% misclassification error. We got to achieve 54% recall value for class 1.

- SGD: Since our data is linearly separable, we tried to implement stochastic gradient descent which is expected to work really well in this kind of problem. SGD was set into 90 max iteration and was able to achieve 87% recall value on class 1 which is really good. It gave 92.78% overall accuracy on unseen dataset.

- Model Comparison:Based on the 4 scores, all the models were evaluated. We were highly focused on the F1 Score in order to decide the better model as we cared about how precisely the model can identify class 1 and how many of them can be identified well. Logistic regression performs extremely well with all the competing models, having the highest F1 score of 75.39%, having 79.12% precision value and 94.05% overall accuracy

> Analysis of the career span of players on the basis of the positions has been performed. The three main positions taken into consideration are - C(Center), F(Forward) and G(Guard). The impact of these three affecting variables was observed on the variable ’careerspan’ which is calculated as the time difference of two given variables - ’firstseason’ and ’lastseason’. For analysis, the player data was divided into 3 sections according to their positions and descriptive analysis on each of the sections were done separately.

> Analysis of the relationship between height and weight of players, and their impact on the player position has been performed. We have used the K-means clustering method over weight and height. The positions of players are C, F and G.

## Results 
> We achieved only 47.47% overall accuracy using the Decision tree approach for classifying the player category. Weak players were generally more misclassified as average.
> All-star league prediction:

| Models | Accuracy | Recall | Precision | F1 Measure |
|------|------|------|------|------|
| Linear Regression  | 94.05% | 72% | 79.12% | 73.39% |
| Decision Tree  | 91.65% | 64% | 68.08% | 65.97% |
| KNN  | 92.15% | 54% | 77.14% | 63.53% |
| SGD  | 92.78% | 87%  | 66.41% | 75.32% |

> The following graphs show the accuracies for attributes that were predicted using all other attributes with at least 90% thrice repeated 3-fold cross validation accuracy.Note: Attributes with suffixes po and rg represent performance in playoffs and regular season respectively. Attributes without suffix denote performance in all star games.

- Random Forest Classifier (a:Average per game. b:Cumulative data)
- XGBoost random forest classifier (a:Average per game. b:Cumulative data)

> The following graphs show the accuracies for attributes averaged across the number of games played that were predicted using only BMI and position with at least 80% thrice repeated 3-fold cross validation accuracy. 

- Random Forest Classifier
- XGBoost random forest classifier 

> The results of analysis of career span of players showed that the mean, median and standard deviation of careerspan for players of position-C were - 5.38, 4.0, 5.02 respectively. Similarly, results for position-F and position-G were found to be - 4.15, 3.0, 4.34 and 4.03, 2.0, 4.22 respectively.

> The results of analysis showed that ’height’ and ’weight’ are highly correlated with correlation = 0.82 (Figure 13 (b)), which was intuitively correct since a player with more height tends to have more mass resulting in more weight. The further and main part of this analysis was 3-means clustering (k=3, since categories = C,F,G) over the weight and height. It was found that the height of players significantly impacts the position of their gameplay.

## Discussion and Conclusion
- Dataset has many missing values and outliers.
- The Coaches dataset is not suitable for clustering. If data is to be clustered then 2 clusters can be used which are derived from the elbow and Silhouette score.
- By using nominal attributes, we cannot predict the class category of a player effectively. Thus, in this data there cannot be any clear relationship identified between the player’s biological/ demographic data and the player’s performance.
- We can perform reliable classification using discretised performance attributes. Predictions on cumulative data tend to perform better than predictions on average values per game. Given a BMI class and position, classes for some average values per game may be obtained. Additionally, this approach may be extended to predict cumulative performance attributes, considering the number of games played as an input parameter along with BMI class and position.
- It can be clearly observed (using median) that players who used to play on position-C had a longer career span followed by players who played at position-F. At last, players who played at position-G had the smallest career span. Hence from the above analysis, we can say that the career span of players according to their positions is — C > F > G.
- Neither silhouette nor elbow method can clearly determine the optimal cluster in the coaches data. Thus the data cannot be clustered properly.
- The results of analysis of height and weight w.r.t the position of the player showed that ’height’ and ’weight’ are highly correlated with correlation = 0.82, which was intuitively correct since a player with more height tends to have more mass resulting in more weight. The further and main part of this analysis was 3-means clustering (k=3, since categories = C,F,G) over the weight and height. It was found that the height of players significantly impacts the position of their gameplay.
- The results clearly comprehend that the players with more height tend to play at position-C, followed by position-F and the shortest players play at position-G. Also, Tall players have a longer career span.

## References
- M. S. Oughali, M. Bahloul and S. A. El Rahman, "Analysis of NBA Players and Shot Prediction Using Random Forest and XGBoost Models," 2019 International Conference on Computer and Information Sciences (ICCIS), 2019, pp. 1-5, doi: 10.1109/ICCISci. 2019.8716412.

- A. Reed, J. Piorkowski and I. McCulloh, "Correlating NBA Team Network Centrality Measures with Game Performance," 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM), 2018, pp. 1291-1294, doi: 10.1109/ASONAM.2018.8508571.

- J. Hewko, R. Sullivan, S. Reige and M. El-Hajj, "Data Mining in The NBA: An Applied Approach," 2019 IEEE 10th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON), 2019, pp. 0426-0432, doi: 10.1109/UEMCON47517.2019.8993074.

- How to Determine the Optimal K for K-Means? | by Khyati Mahendru | Analytics Vidhya

- Evaluating goodness of clustering for unsupervised learning case

- Using K-Means Clustering Algorithm to Redefine NBA Positions and Explore Roster Construction

