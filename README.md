# Hyperparameter optimization in Apache Spark
Code from my master's thesis, **"Hyperparameter optimization for large-scale machine learning"**

Hyperparameter optimization algorithms implemented:
* **Grid search**: In GridSurrogate.scala
* **Random search**: In RandomSurrogate.scala
* **SMAC**: In smac/RandomForestSurrogate.scala
* **Spearmint**: In spearmint/GaussianProcessSurrogate.scala
* **SH**, **RGH** and **BGH**: Using the adaptiveEvalTrial or parallelAdaptiveEvalTrial functions of SmboLoop.scala with different configurations

SmboModel.scala serves as a common interface to all the hyperparameter optimization algorithms.

Every algorithm gives values for the corresponding hyperparameters in the range [0,1]. This range needs to be then transformed to the actual values that are going to be used for the hyperparameters. TransformationUtils.scala offers functions to facilitate this transformation.

A helper class to run a hyperparameter optimization loop with the aforementioned implemented algorithms is given in SmboLoop.scala. Note that it is a basic interface which does not allow to include conditional hyperparameters nor classifier choices/pipeline items as hyperparameters (it would need to be done manually). Also, it currently assumes that the loss is given in form of classification error, for which it is necessary to evaluate the accuracy of the model. That error would need to be changed for different losses (e.g. RMSE).

ExampleExperiment.scala shows how to use the aforementioned implemented algorithms with SmboLoop.scala to perform a basic experiment optimizing three hyperparameters of Logistic Regression. This experiment configured is the same as the one in Section 6.3 of the thesis. This class takes a path to the dataset to be used (in libsvm format) as a first argument and the hyperparameter optimization algorithm to be run as a second argument (with 0 it runs every algorithm).

The rest of the classes (RandomForestSMAC.scala, DecisionTreeSMACMetadata.scala, RandomForestSMACRegressor.scala) are modifications of Spark's original classes (with the same name without "SMAC") in order to be able to run SMAC as currently implemented based on the original random forest implementation of MLlib.
