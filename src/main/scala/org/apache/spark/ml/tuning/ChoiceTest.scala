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

object ChoiceTest {

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

    val lr = new LogisticRegression()

    val svm = new LinearSVC()

    val params = Array(lr.standardization,lr.regParam, lr.elasticNetParam, lr.fitIntercept,
      svm.regParam,svm.standardization,svm.fitIntercept).map(_.asInstanceOf[Param[Any]]) //Array(svm.regParam, svm.fitIntercept, svm.standardization).map(_.asInstanceOf[Param[Any]])


    val categories = Array(2,0,0,2,0,2,2)

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
          result(0) = binaryTransformation(input(0))
           result(1) = TransformationUtils.logarithmicTransformation(input(1), 1E-8, 1)
            result(2) = input(2)
            result(3) = binaryTransformation(input(3))
      result(4) = TransformationUtils.logarithmicTransformation(input(4), 1E-8, 1)
      result(5) = binaryTransformation(input(5))
      result(6) = binaryTransformation(input(6))
      return result
    }

    def modelFeedback(input: DenseVector[Double]): Array[Double] = {
      val result = new Array[Double](input.length)
      result(0) = input(0)
      result(1) = if(input(0)<0.5) input(1) else 0.5
      result(2) = if(input(0)<0.5) input(2) else 0.5
      result(3) = if(input(0)<0.5) input(3) else 0.5
      result(4) = if(input(0)>=0.5) input(4) else 0.5
      result(5) = if(input(0)>=0.5) input(5) else 0.5
      result(6) = if(input(0)>=0.5) input(6) else 0.5
      return result
    }

    val arrPoints = Array(8, 4, 2, 1)
    //val subsamples = Array(1.0 / 64.0, 1.0 / 16.0, 1.0 / 4.0, 1.0 / 1.0)
    val subsamples = Array(1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0 / 1.0)//TODO
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

    if (args(1) == "0" || args(1) == "1") {
      println("STARTING Random search ----------")
      val random0 = new RandomSurrogate()
      val loopR0 = new SmboLoop(random0, svm, evaluator, training, validation, params, Array(lr,svm))

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
      val loops = new SmboLoop(gps, svm, evaluator, training, validation, params, Array(lr,svm))

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
      val loop20 = new SmboLoop(smac0, svm, evaluator, training, validation, params, Array(lr,svm))

      loop20.setTransformations(modelTransformations)
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
      println("STARTING SH ----------") //TODO
      val random = new RandomSurrogate()
      val loopR = new SmboLoop(random, svm, evaluator, training, validation, params, Array(lr,svm))

      loopR.setTransformations(modelTransformations)

      loopR.randomStart()

      val init3 = System.currentTimeMillis()
      for (n <- 0 until 25) { //TODO
        println("Iteration " + n)
        loopR.shaTrial(finalCandidatesMatrix, datasetSamples, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init3))
      }
    }

    if (args(1) == "0" || args(1) == "5") {
      println("STARTING SMAC SH ----------")
      val smac = new RandomForestSurrogate(sparkSession.sqlContext, categories)
      val loop2 = new SmboLoop(smac, svm, evaluator, training, validation, params, Array(lr,svm))

      loop2.setTransformations(modelTransformations)
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
      val loop = new SmboLoop(gp, svm, evaluator, training, validation, params,Array(lr,svm))

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

