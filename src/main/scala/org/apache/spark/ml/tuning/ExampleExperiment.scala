package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.TransformationUtils.binaryTransformation
import org.apache.spark.ml.tuning.smac.RandomForestSurrogate
import org.apache.spark.ml.tuning.spearmint.GaussianProcessSurrogate
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

object ExampleExperiment {

  def main(args: Array[String]): Unit = {

    //Build session
    val sparkSession = SparkSession.builder
      .master("local") //TODO: Change for cluster
      .appName("Example_experiment")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")

    //Load data in libsvm format
    val data = sparkSession.sqlContext.read.format("libsvm").load(args(0))

    //Prepare training and validation data (caching since they are going to be used multiple times)
    val dataSplit = data.randomSplit(Array(0.8, 0.2), 41)
    val training = dataSplit(0).cache()
    val validation = dataSplit(1).cache()

    //Accuracy evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    //100 trials
    val nPoints = 100

    //Optimizing the hyperparameters of logistic regression
    val lr = new LogisticRegression()

    //Optimize three hyperparameters
    val params = Array(lr.regParam, lr.elasticNetParam, lr.fitIntercept).map(_.asInstanceOf[Param[Any]])

    //The third hyperparameter is a boolean
    val categories = Array(0, 0, 2)

    //Prepare grid as a sobol sequence for SMAC and Spearmint

    val finalCandidates = new ListBuffer[Array[Double]]
    val sobol = new SobolSequenceGenerator(params.length)
    for (i <- 0 until 1000) {
      val t = sobol.nextVector()
      finalCandidates += t
    }
    val finalCandidatesMatrix = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)

    //Define transformations
    //Searching for the regularization hyperparameter in a logarithmic range of (10^8,1)
    //Searching for the elastic net hyperparameter in a linear range of (0,1)
    //Transforming the fitIntercept hyperparameter to a boolean value

    def modelTransformations(input: DenseVector[Double]): Array[Any] = {
      val result = new Array[Any](input.length)
      result(0) = TransformationUtils.logarithmicTransformation(input(0), 1E-8, 1)
      result(1) = input(1)
      result(2) = binaryTransformation(input(2))
      return result
    }

    //Define feedback transformations
    //We don't actually transform anything as there are no conditional hyperparameters
    def modelFeedback(input: DenseVector[Double]): Array[Double] = {
      val result = new Array[Double](input.length)
      result(0) = input(0)
      result(1) = input(1)
      result(2) = input(2)
      return result
    }


    //Number of points to train and validate in each round of the adaptive evaluation procedure
    val arrPoints = Array(8, 4, 2, 1)

    //Dataset subsample proportions to use in each round of the adaptive evaluation procedure for BGH and SH
    val subsamplesSH = Array(1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 1.0)
    val subsamplesBGH = Array(1.0 / 64.0, 1.0 / 16.0, 1.0 / 4.0, 1.0 / 1.0)

    //Calculate these subsamples a priori and cache them since they are going to be used multiple times
    val trainingSH0 = training.sample(subsamplesSH(0)).cache()
    val trainingSH1 = training.sample(subsamplesSH(1)).cache()
    val trainingSH2 = training.sample(subsamplesSH(2)).cache()
    val trainingSH3 = training.sample(subsamplesSH(3)).cache()

    val trainingBGH0 = training.sample(subsamplesBGH(0)).cache()
    val trainingBGH1 = training.sample(subsamplesBGH(1)).cache()
    val trainingBGH2 = training.sample(subsamplesBGH(2)).cache()
    val trainingBGH3 = training.sample(subsamplesBGH(3)).cache()


    val datasetSamplesSH = Array(trainingSH0,trainingSH1,trainingSH2,trainingSH3)
    val datasetSamplesBGH = Array(trainingBGH0,trainingBGH1,trainingBGH2,trainingBGH3)


    //Run the actual hyperparameter optimization algorithms
    //0 runs all the algorithms, otherwise select the algorithm to run (1 to 7)

    if (args(1) == "0" || args(1) == "1") {
      println("STARTING RANDOM SEARCH ----------")
      val random0 = new RandomSurrogate()
      val loopR = new SmboLoop(random0, lr, evaluator, training, validation, params)

      loopR.setTransformations(modelTransformations)

      loopR.randomStart()

      val init0 = System.currentTimeMillis()
      for (n <- 0 until  nPoints) {
        println("Iteration " + n)
        loopR.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init0))
      }
    }

    if (args(1) == "0" || args(1) == "2") {
      println("STARTING SPEARMINT ----------")
      val gps = new GaussianProcessSurrogate()
      val loopS = new SmboLoop(gps, lr, evaluator, training, validation, params)

      loopS.setTransformations(modelTransformations)

      loopS.randomStart()

      val init1 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loopS.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init1))
      }
    }

    if (args(1) == "0" || args(1) == "3") {
      println("STARTING SMAC ----------")
      val smac0 = new RandomForestSurrogate(sparkSession.sqlContext, categories)
      val loopSMAC = new SmboLoop(smac0, lr, evaluator, training, validation, params)

      loopSMAC.setTransformations(modelTransformations, modelFeedback)
      loopSMAC.randomStart()

      val init2 = System.currentTimeMillis()
      for (n <- 0 until 2 * nPoints) {
        println("Iteration " + n)
        loopSMAC.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init2))
      }
    }


    if (args(1) == "0" || args(1) == "4") {
      println("STARTING RGH ----------")
      val random = new RandomSurrogate()
      val loopRGH = new SmboLoop(random, lr, evaluator, training, validation, params)

      loopRGH.setTransformations(modelTransformations)

      loopRGH.randomStart()

      val init3 = System.currentTimeMillis()
      for (n <- 0 until nPoints/2) {
        println("Iteration " + n)
        loopRGH.parallelAdaptiveEvalTrial(finalCandidatesMatrix, datasetSamplesBGH, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init3))
      }
    }

    if (args(1) == "0" || args(1) == "5") {
      println("STARTING BGH-SMAC ----------")
      val smac = new RandomForestSurrogate(sparkSession.sqlContext, categories)
      val loopBS = new SmboLoop(smac, lr, evaluator, training, validation, params)

      loopBS.setTransformations(modelTransformations, modelFeedback)
      loopBS.randomStart()

      val init4 = System.currentTimeMillis()
      for (n <- 0 until nPoints/2) {
        println("Iteration " + n)
        loopBS.parallelAdaptiveEvalTrial(finalCandidatesMatrix, datasetSamplesBGH, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init4))
      }
    }


    if (args(1) == "0" || args(1) == "6") {
      println("STARTING BGH-SPEARMINT ----------")
      val gp = new GaussianProcessSurrogate()
      val loopBSP = new SmboLoop(gp, lr, evaluator, training, validation, params)

      loopBSP.setTransformations(modelTransformations)

      loopBSP.randomStart()

      val init5 = System.currentTimeMillis()
      for (n <- 0 until nPoints/2) {
        println("Iteration " + n)
        loopBSP.parallelAdaptiveEvalTrial(finalCandidatesMatrix, datasetSamplesBGH, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init5))
      }
    }

    if (args(1) == "0" || args(1) == "7") {
      println("STARTING SUCCESSIVE HALVING ----------")
      val random = new RandomSurrogate()
      val loopSH = new SmboLoop(random, lr, evaluator, training, validation, params)

      loopSH.setTransformations(modelTransformations)

      loopSH.randomStart()

      val init3 = System.currentTimeMillis()
      for (n <- 0 until nPoints/2) {
        println("Iteration " + n)
        loopSH.parallelAdaptiveEvalTrial(finalCandidatesMatrix, datasetSamplesSH, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init3))
      }
    }


  }
}

