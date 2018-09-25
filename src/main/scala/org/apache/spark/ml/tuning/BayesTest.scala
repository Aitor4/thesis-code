package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.TransformationUtils.binaryTransformation
import org.apache.spark.ml.tuning.smac.RandomForestSurrogate
import org.apache.spark.ml.tuning.spearmint.GaussianProcess
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.tuning.TransformationUtils._

import scala.collection.mutable.ListBuffer

object BayesTest {

  def main(args: Array[String]): Unit = {

    val sparkSession = SparkSession.builder
        .master("local") //TODO
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

    val nPoints = 50

    val nn = new MultilayerPerceptronClassifier().setSolver("gd")

    val params = Array(nn.stepSize, nn.layers, nn.layers, nn.layers).map(_.asInstanceOf[Param[Any]]) //Array(svm.regParam, svm.fitIntercept, svm.standardization).map(_.asInstanceOf[Param[Any]])

    val categories = Array(0,-3,0,0,0, 0)

    val finalCandidates = new ListBuffer[Array[Double]]

    val sobol = new SobolSequenceGenerator(params.length)

    for (i <- 0 until 1000) {
      val t = sobol.nextVector()
      finalCandidates += t
    }


    val finalCandidatesMatrix = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)


    //For Grid Search
    for (i <- 0 until 2*nPoints) {
      val t = sobol.nextVector()
      finalCandidates += t
    }

    val finalCandidatesMatrix2 = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)

    //Transformations

    def modelTransformations(input: DenseVector[Double]): Array[Any] = {
      val result = new Array[Any](input.length)
      result(0) = logarithmicTransformation(input(0),0.001,10.0)
      result(1) = integerTransformation(input(1),3,2)
      result(2) = integerTransformation(input(2),16,2)
      result(3) = integerTransformation(input(3),16,2)
      //result(4) = integerTransformation(input(4),100,10)
      //result(5) = integerTransformation(input(5),50,10)
      return result
    }

    def modelFeedback(input: DenseVector[Double]): Array[Double] = {
      val result = new Array[Double](input.length)
      result(0) = input(0)
      result(1) = input(1)
      result(2) = if(input(1)>(1.0/3.0)) result(2) else 0.5
      result(3) = if(input(1)>(2.0/3.0)) result(3) else 0.5
      //result(4) = if(input(1)>(3.0/5.0)) result(4) else 0.5
      //result(5) = if(input(1)>(4.0/5.0)) result(5) else 0.5

      return result
    }

    val arrPoints = Array(8, 4, 2, 1)
    val subsamples = Array(1.0 / 64.0, 1.0 / 16.0, 1.0 / 4.0, 1.0 / 1.0)
    val training0 = training.sample(subsamples(0)).cache()
    val training1 = training.sample(subsamples(1)).cache()
    val training2 = training.sample(subsamples(2)).cache()
    val training3 = training.sample(subsamples(3)).cache()
    val datasetSamples = Array(training0,training1,training2,training3)

/*    if (args(1) == "0" || args(1) == "-1") {
      println("STARTING Grid search ----------")
      val grid = new GridSurrogate(finalCandidatesMatrix2)
      val loopG = new SmboLoop(grid, rf, evaluator, training, validation, params)

      loopG.setTransformations(modelTransformations)
      loopG.randomStart()

      val init00 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loopG.trial(finalCandidatesMatrix)

        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init00))
      }
    }*/
    val nFeat = args(2).toInt
    val nClass = 2

      if (args(1) == "0" || args(1) == "1") {
        println("STARTING Random search ----------")
        val random0 = new RandomSurrogate()
        val loopR0 = new SmboLoop(random0, nn, evaluator, training, validation, params, null, nFeat, nClass)

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
      val loops = new SmboLoop(gps, nn, evaluator, training, validation, params, null, nFeat, nClass)

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
      val loop20 = new SmboLoop(smac0, nn, evaluator, training, validation, params, null, nFeat, nClass)

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
      println("STARTING Random search SH ----------")
      val random = new RandomSurrogate()
      val loopR = new SmboLoop(random, nn, evaluator, training, validation, params, null, nFeat, nClass)

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
      val loop2 = new SmboLoop(smac, nn, evaluator, training, validation, params, null, nFeat, nClass)

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
      val loop = new SmboLoop(gp, nn, evaluator, training, validation, params, null, nFeat, nClass)

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

