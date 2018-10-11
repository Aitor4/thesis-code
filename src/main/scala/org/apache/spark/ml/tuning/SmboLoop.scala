package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector, argmin}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.{Dataset, Row}

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.util.Random
import scala.concurrent.ExecutionContext.Implicits.global

class SmboLoop (initialModel: SmboModel, estimatorS: Estimator[_], evaluator: Evaluator,
                initTraining: Dataset[_], initTesting: Dataset[_], params:Array[Param[Any]], estimators: Array[Estimator[_]] = null,
                nFeat : Int = 0, nClass: Int = 0) {

  var model = initialModel
  var training = initTraining
  var testing = initTesting
  var hyperHistory : ArrayBuffer[Array[Double]] = new ArrayBuffer[Array[Double]]
  var lossHistory : ArrayBuffer[Double] = new ArrayBuffer[Double]
  var modelTransformations : DenseVector[Double] => Array[Any] = x=>x.toArray.asInstanceOf[Array[Any]]
  var feedbackTransformations : DenseVector[Double] => Array[Double] = x=>x.toArray

  //Define the model to be trained and validated
  def setModel(model: SmboModel): this.type ={
    this.model = model
    this
  }

  //Set training data
  def setTraining(training: Dataset[_]): this.type ={
    this.training = training
    this
  }

  //Set validation data
  def setTesting(testing: Dataset[_]): this.type ={
    this.testing = testing
    this
  }

  //Form warm-starting a surrogate model with a history of (hyperparameter, loss)
  def setHistory(hypers: ArrayBuffer[Array[Double]], loss: ArrayBuffer[Double]): this.type ={
    this.hyperHistory = hypers
    this.lossHistory = loss
    this
  }

  //Set transformation of parameters from surrogate model (scale 0-1) to the actual values used in the algorithm
  //Also, optionally, set transformations to impute values of certain hyperparameters (e.g. inactive) to give as feedback
  //to the surrogate
  def setTransformations(model: DenseVector[Double] => Array[Any],feedback: DenseVector[Double] => Array[Double] = null): this.type ={
    this.modelTransformations = model
    if (feedback!=null) this.feedbackTransformations = feedback
    this
  }

  //To initialize the feedback history
  def randomStart() : Unit = {
    val points = new DenseVector[Double](params.length)
    for (i<- 0 until params.length){
      points(i) = Random.nextDouble()
    }
    println("Random point suggested: ")
    val init = System.currentTimeMillis()
    println(points.toString())

    val transformed = modelTransformations(points)
    val selected = transformed
    val selector = 0
    val estimator = estimatorS
    var i = 0
    for (param<- 0 until selected.length){
        estimator.set(params(selector+param), selected(i))
        i += 1
    }
    val feedback = feedbackTransformations(points)

    val mlModel = estimator.fit(training).asInstanceOf[Model[_]]
    val result = mlModel.transform(testing)

    //TODO: Depends on metric
    val error : Double = 1- evaluator.evaluate(result)

    val end = System.currentTimeMillis()
    println("Time for iteration: "+(end-init))

    println("Validation error achieved: " + error)

    hyperHistory += feedback
    lossHistory += error

  }

  //Perform a single trial with the usual evaluation procedure (single training and validation)
  def trial(grid: DenseMatrix[Double]) : Unit = {
    val y = new DenseVector[Double](lossHistory.toArray)
    val x = new DenseMatrix(hyperHistory(0).length, hyperHistory.length, hyperHistory.toArray.flatten)
    val initTop = System.currentTimeMillis()
    model.fit(x, y)
    val suggestedPoint = model.next(grid)
    val endTop = System.currentTimeMillis()
    println("Point suggested: ")
    println(suggestedPoint.toString())
    println("Time for suggestion: "+(endTop-initTop))
    val init = System.currentTimeMillis()
    var i = 0
    val transformed = modelTransformations(suggestedPoint)
    val selected = transformed
    val selector = 0
    val estimator = estimatorS
    //for (param<- params){
    for (param<- 0 until selected.length){
        estimator.set(params(selector+param), selected(i))
        i += 1
    }
    val feedback = feedbackTransformations(suggestedPoint)


    val mlModel = estimator.fit(training).asInstanceOf[Model[_]]
    val result = mlModel.transform(testing)

    //TODO: Depends on metric
    val error : Double = 1- evaluator.evaluate(result)

    val end = System.currentTimeMillis()
    println("Time for iteration: "+(end-init))

    println("Validation error achieved: " + error)


    hyperHistory += feedback
    lossHistory += error
  }

  //Perform a trial with an adaptive evaluation procedure (defined by subsamples and roundPoints)
  def adaptiveEvalTrial(grid: DenseMatrix[Double], subsamples: Array[Dataset[Row]], stagePoints: Array[Int]) : Unit = {
    val y = new DenseVector[Double](lossHistory.toArray)
    val x = new DenseMatrix(hyperHistory(0).length, hyperHistory.length, hyperHistory.toArray.flatten)
    val initTop = System.currentTimeMillis()
    model.fit(x, y)
    var previousPoints = model.topNext(stagePoints(0),grid) //Get suggested points
    val endTop = System.currentTimeMillis()
    println("Time for suggestion: "+(endTop-initTop))
    var finalError = 0.0
    val init = System.currentTimeMillis()
    for (i<- 0 until subsamples.length){ //Round i before halving
      val pointResult = new Array[(DenseVector[Double],Double)](stagePoints(i))
      val initIter = System.currentTimeMillis()
      val subsampledTraining = subsamples(i)

      for (j <- 0 until stagePoints(i)){ //Point j of this round
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoints(j))
        var i = 0
        for (param<- params){
          estimatorS.set(param,transformed(i))
          i+=1
        }
        val mlModel = estimatorS.fit(subsampledTraining).asInstanceOf[Model[_]]
        val result = mlModel.transform(testing)

        //TODO: Depends on metric
        val error = 1 -evaluator.evaluate(result)

        pointResult(j) = (previousPoints(j),error)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
      }
      val endIter = System.currentTimeMillis()
      println("Time for round: "+ (endIter-initIter))
      if(i+1<stagePoints.length) previousPoints = pointResult.sortBy(_._2).map(_._1).take(stagePoints(i+1))
      else { //Rounds are finished, taking the best point (from the last round)
        previousPoints=pointResult.sortBy(_._2).map(_._1).take(1)
        finalError=pointResult.sortBy(_._2).map(_._2).take(1)(0)
      }
      subsampledTraining.unpersist()
    }
    val suggestedPoint = previousPoints(0)
    println("Point suggested: ")
    println(suggestedPoint.toString())
    val end = System.currentTimeMillis()
    println("Time for iteration: "+(end-init))

    val feedback = feedbackTransformations(suggestedPoint)

    println("Validation error achieved: " + finalError)

    hyperHistory += feedback
    lossHistory += finalError
  }
  //Perform a trial with an adaptive evaluation procedure (defined by subsamples and roundPoints). Trains and validates
  //every configuration of the same round in parallel (except, obviously, the last one with only one configuration).
  //Can report benefits when the cluster resources are not fully utilized by one round
  def parallelAdaptiveEvalTrial(grid: DenseMatrix[Double], subsamples: Array[Dataset[Row]], roundPoints: Array[Int])
  : Unit = {
    val y = new DenseVector[Double](lossHistory.toArray)
    val x = new DenseMatrix(hyperHistory(0).length, hyperHistory.length, hyperHistory.toArray.flatten)
    val initTop = System.currentTimeMillis()
    model.fit(x, y)
    var previousPoints = model.topNext(roundPoints(0),grid) //Get suggested points
    val endTop = System.currentTimeMillis()
    println("Time for suggestion: "+(endTop-initTop))
    var finalError = 0.0
    val init = System.currentTimeMillis()
    for (i<- 0 until subsamples.length) { //Round i before halving
      var pointResult = new Array[(DenseVector[Double], Double)](roundPoints(i))
      val initIter = System.currentTimeMillis()
      val subsampledTraining = subsamples(i)

      def errorParallel(previousPoint: DenseVector[Double]) = Future {
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoint)
        var i = 0
        val pMap = new ParamMap()
        val selected = transformed
        val selector = 0
        val estimator = estimatorS
        for (param<- 0 until selected.length){
          pMap.put(params(selector+param), selected(i))
          i += 1
        }
        val itEstimator = estimator.copy(pMap)
        val mlModel = itEstimator.fit(subsampledTraining).asInstanceOf[Model[_]]
        val result = mlModel.transform(testing)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
        //TODO: Depends on metric
        (1 - evaluator.evaluate(result))
      }

      def errorSequential(previousPoint: DenseVector[Double]) : Double ={
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoint)
        var i = 0
        val pMap = new ParamMap()
       val selected = transformed
        val selector = 0
        val estimator = estimatorS
        for (param<- 0 until selected.length){
          pMap.put(params(selector+param), selected(i))
          i += 1
        }
        val itEstimator = estimator.copy(pMap)
        val mlModel = itEstimator.fit(subsampledTraining).asInstanceOf[Model[_]]
        val result = mlModel.transform(testing)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
        //TODO: Depends on metric
        (1 - evaluator.evaluate(result))
      }
      val rangeSubsamples = 0 until roundPoints(i) toList

      if(roundPoints(i)>1) {
        val futureResult = rangeSubsamples.map(x => errorParallel(previousPoints(x)))
        val futureErrors = Future.sequence(futureResult)
        val errors = Await.result(futureErrors, Duration.Inf).toArray
        for (t <- 0 until roundPoints(i)) {
          pointResult(t) = (previousPoints(t), errors(t))
        }
      }
      else{
        pointResult(0) = (previousPoints(0), errorSequential(previousPoints(0)))
      }
      val endIter = System.currentTimeMillis()
      println("Time for round: "+ (endIter-initIter))
      if(i+1<roundPoints.length) previousPoints = pointResult.sortBy(_._2).map(_._1).take(roundPoints(i+1))
      else { //Rounds are finished, taking the best point (from the last round)
        previousPoints=pointResult.sortBy(_._2).map(_._1).take(1)
        finalError=pointResult.sortBy(_._2).map(_._2).take(1)(0)
      }
    }
    val suggestedPoint = previousPoints(0)
    println("Point suggested: ")
    println(suggestedPoint.toString())
    val end = System.currentTimeMillis()
    println("Time for iteration: "+(end-init))

    val feedback = feedbackTransformations(suggestedPoint)

    println("Validation error achieved: " + finalError)

    hyperHistory += feedback
    lossHistory += finalError
  }

  //Returns best combination of hyperparameters found so far and their loss
  def getBest() : (Array[Double], Double) = {
    val best = argmin(lossHistory.toArray)
    return (hyperHistory(best),lossHistory(best))
  }
}
