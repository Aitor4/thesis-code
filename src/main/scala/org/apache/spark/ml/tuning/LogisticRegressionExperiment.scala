package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator
import org.apache.spark.ml.classification.{LinearSVC, LogisticRegression}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.TransformationUtils.binaryTransformation
import org.apache.spark.ml.tuning.smac.RandomForestSurrogate
import org.apache.spark.ml.tuning.spearmint.GaussianProcess
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

object LogisticRegressionExperiment {

  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder
      //.master("local") //TODO
      .appName("Aitor_experiment")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")

    val data = sparkSession.sqlContext.read.format("libsvm").load(args(0))

    val dataSplit = data.randomSplit(Array(0.8, 0.2), 41)
    val training = dataSplit(0).cache()
    val validation = dataSplit(1).cache()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val nPoints = 25

/*
    val svm = new LogisticRegression()
    val params = Array(svm.regParam, svm.elasticNetParam, svm.fitIntercept).map(_.asInstanceOf[Param[Any]])
*/

    val lr = new LogisticRegression()

    val params = Array(lr.regParam, lr.elasticNetParam, lr.fitIntercept).map(_.asInstanceOf[Param[Any]])

    val categories = Array(0, 0, 2)

    val finalCandidates = new ListBuffer[Array[Double]]

    val sobol = new SobolSequenceGenerator(params.length)

    for (i <- 0 until 1000) {
      val t = sobol.nextVector()
      finalCandidates += t
    }

    val finalCandidatesMatrix = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)

    //Transformations

/*    def modelTransformations(input: DenseVector[Double]): Array[Any] = {
      val result = new Array[Any](input.length)
      result(0) = input(0)
      result(1) = input(1)
      result(2) = binaryTransformation(input(2))
      return result
    }*/

    def modelTransformations(input: DenseVector[Double]): Array[Any] = {
      val result = new Array[Any](input.length)
      result(0) = TransformationUtils.logarithmicTransformation(input(0), 1E-8, 1)
      result(1) = input(1)
      result(2) = binaryTransformation(input(2))
      return result
    }

    def modelFeedback(input: DenseVector[Double]): Array[Double] = {
      val result = new Array[Double](input.length)
      result(0) = input(0)
      result(1) = input(1)
      result(2) = math.round(input(2)).toDouble
      return result
    }


    val arrPoints = Array(8, 4, 2, 1)
    val subsamples = Array(1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 1.0)
    val training0 = training.sample(subsamples(0)).cache()
    val training1 = training.sample(subsamples(1)).cache()
    val training2 = training.sample(subsamples(2)).cache()
    val training3 = training.sample(subsamples(3)).cache()
    val datasetSamples = Array(training0,training1,training2,training3)

    if (args(1) == "0" || args(1) == "1") {
      println("STARTING Random search ----------")
      val random0 = new RandomSurrogate()
      val loopR0 = new SmboLoop(random0, lr, evaluator, training, validation, params)

      loopR0.setTransformations(modelTransformations)

      loopR0.randomStart()

      val init0 = System.currentTimeMillis()
      for (n <- 0 until 2 * nPoints) {
        println("Iteration " + n)
        loopR0.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init0))
      }
    }

    if (args(1) == "0" || args(1) == "2") {
      println("STARTING SPEARMINT ----------")
      val gps = new GaussianProcess()
      val loops = new SmboLoop(gps, lr, evaluator, training, validation, params)

      loops.setTransformations(modelTransformations)

      loops.randomStart()

      val init1 = System.currentTimeMillis()
      for (n <- 0 until 2 * nPoints) {
        println("Iteration " + n)
        loops.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init1))
      }
    }

    if (args(1) == "0" || args(1) == "3") {
      println("STARTING SMAC ----------")
      val smac0 = new RandomForestSurrogate(sparkSession.sqlContext, categories)
      val loop20 = new SmboLoop(smac0, lr, evaluator, training, validation, params)

      loop20.setTransformations(modelTransformations, modelFeedback)
      loop20.randomStart()

      val init2 = System.currentTimeMillis()
      for (n <- 0 until 2 * nPoints) {
        println("Iteration " + n)
        loop20.trial(finalCandidatesMatrix)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init2))
      }
    }


    if (args(1) == "0" || args(1) == "4") {
      println("STARTING SH ----------")
      val random = new RandomSurrogate()
      val loopR = new SmboLoop(random, lr, evaluator, training, validation, params)

      loopR.setTransformations(modelTransformations)

      loopR.randomStart()

      val init3 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loopR.shaTrial(finalCandidatesMatrix, datasetSamples, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init3))
      }
    }

    if (args(1) == "0" || args(1) == "5") {
      println("STARTING SMAC SH ----------")
      val smac = new RandomForestSurrogate(sparkSession.sqlContext, categories)
      val loop2 = new SmboLoop(smac, lr, evaluator, training, validation, params)

      loop2.setTransformations(modelTransformations, modelFeedback)
      loop2.randomStart()

      val init4 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loop2.shaTrial(finalCandidatesMatrix, datasetSamples, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init4))
      }
    }


    if (args(1) == "0" || args(1) == "6") {
      println("STARTING SPEARMINT SH----------")
      val gp = new GaussianProcess()
      val loop = new SmboLoop(gp, lr, evaluator, training, validation, params)

      loop.setTransformations(modelTransformations)

      loop.randomStart()

      val init5 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loop.shaTrial(finalCandidatesMatrix, datasetSamples, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init5))
      }
    }
  }
}

