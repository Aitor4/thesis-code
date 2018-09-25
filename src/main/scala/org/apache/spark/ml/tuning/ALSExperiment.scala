package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.commons.math3.random.SobolSequenceGenerator
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.TransformationUtils.binaryTransformation
import org.apache.spark.ml.tuning.smac.RandomForestSurrogate
import org.apache.spark.ml.tuning.spearmint.GaussianProcess
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.tuning.TransformationUtils._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

object ALSExperiment {

  def main(args: Array[String]): Unit = {

    case class Rating(userId: Int, movieId: Int, rating: Int, timestamp: Long)
    def parseRating(str: String): (Int, Int, Int, Long) = {
      val fields = str.split("::")//TODO: Change to spaces
      assert(fields.size == 4)
      (fields(0).toInt, fields(1).toInt, fields(2).toInt, fields(3).toLong)
    }

    val spark = SparkSession
      .builder.master("local") //TODO
      .appName("ALSExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    // $example on$
          var ratings = spark.read.textFile(args(0))
        .map(parseRating)
        .toDF("userId","movieId","rating","timeStamp")


    //ratings = ratings.withColumn("rating", ratings.col("ratingI").cast(DoubleType))

 /*   val ratings = spark.read.textFile("../data/mllib/als/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF().withColumnRenamed("_1", "userId")
      .withColumnRenamed("_2", "movieId").withColumnRenamed("_3", "rating").withColumnRenamed("_4", "timestamp")*/

    //ratings.withColumn("rating",ratings.column("ratingI").cast(DoubleType))

/*    val schema = StructType(List(
      StructField("userId", IntegerType, true),
      StructField("movieId", IntegerType, true),
      StructField("rating", FloatType, true),
      StructField("timeStamp", LongType, true)))

    val ratings = spark.sqlContext.read.schema(schema).option("header", "true").csv(args(0)).withColumnRenamed("_c0", "userId")
      .withColumnRenamed("_c1", "movieId").withColumnRenamed("_c2", "rating").withColumnRenamed("_c3", "timestamp")*/

    val dataSplit = ratings.randomSplit(Array(0.8, 0.2), 41)
    val training = dataSplit(0).cache()
    val validation = dataSplit(1).cache()

    val nPoints = 50

    val als = new ALS()
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
      .setColdStartStrategy("drop")
    //val params = Array(als.rank, als.regParam, als.nonnegative).map(_.asInstanceOf[Param[Any]]) //Array(svm.regParam, svm.fitIntercept, svm.standardization).map(_.asInstanceOf[Param[Any]])
    //    val categories = Array(0, 0, 2)

    val params = Array(als.regParam).map(_.asInstanceOf[Param[Any]])
    val categories = Array(0)


    val finalCandidates = new ListBuffer[Array[Double]]

    val sobol = new SobolSequenceGenerator(params.length)

    for (i <- 0 until 1000) {
      val t = sobol.nextVector()
      finalCandidates += t
    }



    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val finalCandidatesMatrix = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)

    //For Grid Search
    for (i <- 0 until 2 * nPoints) {
      val t = sobol.nextVector()
      finalCandidates += t
    }

    val finalCandidatesMatrix2 = new DenseMatrix(params.length, finalCandidates.length, finalCandidates.toArray.flatten)

    //Transformations

    def modelTransformations(input: DenseVector[Double]): Array[Any] = {
      val result = new Array[Any](input.length)
      result(0) = rangeTransformation(input(0), 1E-6, 1)
      //result(0) = integerTransformation(input(0), 49, 2)
      //result(1) = rangeTransformation(input(1), 1E-6, 1)
      //result(2) = binaryTransformation(input(2))
      return result
    }

    def modelFeedback(input: DenseVector[Double]): Array[Double] = {
      val result = new Array[Double](input.length)
      result(0) = input(0)
      //result(1) = input(1)
      //result(2) = math.round(input(2)).toDouble
      return result
    }

    val subsamples = Array(1.0 / 64.0, 1.0 / 16.0, 1.0 / 4.0, 1.0 / 1.0)
    val training0 = training.sample(subsamples(0)).cache()
    val training1 = training.sample(subsamples(1)).cache()
    val training2 = training.sample(subsamples(2)).cache()
    val training3 = training.sample(subsamples(3)).cache()
    val datasetSamples = Array(training0,training1,training2,training3)
    val arrPoints = Array(8, 4, 2, 1)

/*        if (args(1) == "0" || args(1) == "-1") {
      println("STARTING Grid search ----------")
      val grid = new GridSurrogate(finalCandidatesMatrix2)
      val loopG = new SmboLoop(grid, als, evaluator, training, validation, params)

      loopG.setTransformations(modelTransformations)
      loopG.randomStart()

      val init00 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loopG.trial(finalCandidatesMatrix)

        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init00))
      } }*/

    for (t <- 0 until 4) {
      if (args(1) == "0" || args(1) == "1") {
        println("STARTING Random search ----------")
        val random0 = new RandomSurrogate()
        val loopR0 = new SmboLoop(random0, als, evaluator, training, validation, params)

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
        val loops = new SmboLoop(gps, als, evaluator, training, validation, params)

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
        val smac0 = new RandomForestSurrogate(spark.sqlContext, categories)
        val loop20 = new SmboLoop(smac0, als, evaluator, training, validation, params)

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

    }

    /*if (args(1) == "0" || args(1) == "4") {
      println("STARTING Random search SH ----------")
      val random = new RandomSurrogate()
      val loopR = new SmboLoop(random, als, evaluator, training, validation, params)

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
      val smac = new RandomForestSurrogate(spark.sqlContext, categories)
      val loop2 = new SmboLoop(smac, als, evaluator, training, validation, params)

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
      val loop = new SmboLoop(gp, als, evaluator, training, validation, params)

      loop.setTransformations(modelTransformations)

      loop.randomStart()

      val init5 = System.currentTimeMillis()
      for (n <- 0 until nPoints) {
        println("Iteration " + n)
        loop.shaTrial(finalCandidatesMatrix, datasetSamples, arrPoints)
        val end = System.currentTimeMillis()
        println("Time at iteration: " + (end - init5))
      }
    }*/
  }
}

