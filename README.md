# metaflow-automatic-ml

![Drag Racing](https://i.ibb.co/r4MzCNV/Screen-Shot-2020-03-18-at-12-57-51-PM.png)


This repo contains a powerful, minimalistic, state-of-the-art, automated modeling pipeline. 

Its build using Metaflow, MITs Deep Feature Synthesis (Featuretools), and Stanfords Probabilistic gradient boosting (NGBoost). It is fast, scalable and  production ready. 


![Drag Racing](https://i.ibb.co/QftPB2Z/Screen-Shot-2020-03-23-at-1-33-16-PM.png=150px )



### Deep Feature Synthesis?

Deep Feature Synthesis (DFS) is an automated method for performing feature engineering on relational and temporal data.

### Natural Gradient Booosting? 

Gradient Boosting is a robust out of the box classifier that can learn complex non-linear decision boundaries, without having to worry about feature selection or colinearity.

It generally its been among the top performers in predictive accuracy over structured or tabular input data, typically yielding good accuracy, precision, and ROC area. However, because XGBoost/Catboost/LiteGBM do not solve the log-loss/corss-entropy objective directly, the outputs do not represent an accurate posterior probabilities. The results is poor squared error and cross-entropy.

NGBoost on is a state of the art solution (13 Feb 2020) using Probabilistic gradient boosting via Natural Gradients. Its output is a prediction of the posterior probability. This necessary for automated decision making, where you might want to use a cutt-off on the probability prediction.

