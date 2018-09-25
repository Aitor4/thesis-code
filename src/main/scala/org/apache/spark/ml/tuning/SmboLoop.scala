package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector, argmin}
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.recommendation.ALSModel
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
  def setModel(model: SmboModel): this.type ={
    this.model = model
    this
  }

  def setTraining(training: Dataset[_]): this.type ={
    this.training = training
    this
  }

  def setTesting(testing: Dataset[_]): this.type ={
    this.testing = testing
    this
  }

  //Form warm-starting a surrogate model
  def setHistory(hypers: ArrayBuffer[Array[Double]], loss: ArrayBuffer[Double]): this.type ={
    this.hyperHistory = hypers
    this.lossHistory = loss
    this
  }

  def setTransformations(model: DenseVector[Double] => Array[Any],feedback: DenseVector[Double] => Array[Double] = null): this.type ={
    this.modelTransformations = model
    if (feedback!=null) this.feedbackTransformations = feedback
    this
  }

  //To initialize the feedback history when no warm-starting is present
  def randomStart() : Unit = {
    val points = new DenseVector[Double](params.length)
    for (i<- 0 until params.length){
      points(i) = Random.nextDouble()
    }
    println("Random point suggested: ")
    println(points.toString())

    var i = 0
    val transformed = modelTransformations(points)
//    println("Random points transformed: ")
//    transformed.foreach(x=> println(x.toString))
//TODO: Neural networks (784 or 780 first layer)
    /*var layers : Array[Int] = null
    println("Nlayers is: "+transformed(1))
    if(transformed(1)==6){
    layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
  transformed(4).asInstanceOf[Int], transformed(5).asInstanceOf[Int], nClass)
    }
    else if (transformed(1)==5){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
        transformed(4).asInstanceOf[Int], nClass)
    }
    else if (transformed(1)==4){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int], nClass)
    }
    else if(transformed(1)==3){
        layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], nClass)
    }
    else if(transformed(1)==2){
        layers = Array[Int](nFeat, nClass)
      }
    val selected = new Array[Any](3)
    var estimator : Estimator[_] = null
    var selector = 0
    if(transformed(0)==false){
      selected(0)=transformed(1)
      selected(1)=transformed(2)
      selected(2)=transformed(3)
      estimator = estimators(0)
      selector = 1

    }
    else {
      selected(0)=transformed(4)
      selected(1)=transformed(5)
      selected(2)=transformed(6)
      estimator = estimators(1)
      selector = 4
    }*/
    val selected = transformed
    val selector = 0
    val estimator = estimatorS
    //for (param<- params){
    for (param<- 0 until selected.length){
/*      //TODO: Neural networks
      if(param.name.equals("layers"))
      {

      }*/
      //else {
        //TODO: How to treat categorical etc.
        estimator.set(params(selector+param), selected(i))
        i += 1
      //}
    }
    //estimator.set(params(3),layers)
    val feedback = feedbackTransformations(points)
//    println("Random points feedback: ")
//    feedback.foreach(x=> println(x.toString))

    val mlModel = estimator.fit(training).asInstanceOf[Model[_]] //TODO
    //val mlModel = new OneVsRest().setClassifier(estimator.asInstanceOf[LinearSVC]).fit(training)
    val result = mlModel.transform(testing)

    //TODO: Depends on metric
    val error : Double = 1- evaluator.evaluate(result)

    println("Validation error achieved: " + error)

    hyperHistory += feedback
    lossHistory += error

  }

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
//    println("Points transformed: ")
//    transformed.foreach(x=> println(x.toString))
    //TODO: Neural networks (784 or 780 first layer)
/*var layers : Array[Int] = null
    if(transformed(1)==6){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
        transformed(4).asInstanceOf[Int], transformed(5).asInstanceOf[Int], nClass)
    }
    else if (transformed(1)==5){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
        transformed(4).asInstanceOf[Int], nClass)
    }
    else if (transformed(1)==4){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int], nClass)
    }
    else if(transformed(1)==3){
      layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], nClass)
    }
    else if(transformed(1)==2){
      layers = Array[Int](nFeat, nClass)
    }
val selected = new Array[Any](3)
    var estimator : Estimator[_] = null
    var selector = 0
    if(transformed(0)==false){
      selected(0)=transformed(1)
      selected(1)=transformed(2)
      selected(2)=transformed(3)
      estimator = estimators(0)
      selector = 1

    }
    else {
      selected(0)=transformed(4)
      selected(1)=transformed(5)
      selected(2)=transformed(6)
      estimator = estimators(1)
      selector = 4
    }*/
val selected = transformed
    val selector = 0
    val estimator = estimatorS
    //for (param<- params){
    for (param<- 0 until selected.length){
      //TODO: Neural networks
/*      if(param.name.equals("layers"))
      {

      }*/
      //else {
        //TODO: How to treat categorical etc.
        estimator.set(params(selector+param), selected(i))
        i += 1
      //}
    }
    //estimator.set(params(3),layers)
    val feedback = feedbackTransformations(suggestedPoint)
    //println("Points feedback: ")
    //feedback.foreach(x=> println(x.toString))


    val mlModel = estimator.fit(training).asInstanceOf[Model[_]] //TODO
    //val mlModel = new OneVsRest().setClassifier(estimator.asInstanceOf[LinearSVC]).fit(training)
    val result = mlModel.transform(testing)

    //TODO: Depends on metric
    val error : Double = 1- evaluator.evaluate(result)

    val end = System.currentTimeMillis()
    println("Time for iteration: "+(end-init))

    println("Validation error achieved: " + error)


    hyperHistory += feedback
    lossHistory += error
  }

  def shaTrial(grid: DenseMatrix[Double], subsamples: Array[Dataset[Row]], stagePoints: Array[Int]) : Unit = {
    val y = new DenseVector[Double](lossHistory.toArray)
    val x = new DenseMatrix(hyperHistory(0).length, hyperHistory.length, hyperHistory.toArray.flatten)
    val initTop = System.currentTimeMillis()
    model.fit(x, y)
    var previousPoints = model.topNext(stagePoints(0),grid) //Get suggested points
    val endTop = System.currentTimeMillis()
    println("Time for suggestion: "+(endTop-initTop))
    var finalError = 0.0
    val init = System.currentTimeMillis()
    for (i<- 0 until subsamples.length) { //Round i before halving
      //      val initB = System.currentTimeMillis()
      var pointResult = new Array[(DenseVector[Double], Double)](stagePoints(i))
      //TODO: Using same subsample for every point, maybe take different ones partitioning to parallelize??
      val initIter = System.currentTimeMillis()
      val subsampledTraining = subsamples(i)
      //      println("Subsample count is: "+subsampledTraining.count())
      //      println("Subsample partitions is: "+subsampledTraining.rdd.getNumPartitions)
      //      println ("iterations is: "+stagePoints(i))

      def error(previousPoint: DenseVector[Double]) = Future {
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoint)
        //        println("Points transformed: ")
        //        transformed.foreach(x=> println(x.toString))
        var i = 0
        val pMap = new ParamMap()
        /*//TODO: Neural networks (784 or 780 first layer)
        var layers : Array[Int] = null
        if(transformed(1)==6){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
            transformed(4).asInstanceOf[Int], transformed(5).asInstanceOf[Int], nClass)
        }
        else if (transformed(1)==5){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
            transformed(4).asInstanceOf[Int], nClass)
        }
        else if (transformed(1)==4){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int], nClass)
        }
        else if(transformed(1)==3){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], nClass)
        }
        else if(transformed(1)==2){
          layers = Array[Int](nFeat, nClass)

        }
        val selected = new Array[Any](3)
        var estimator : Estimator[_] = null
        var selector = 0
        if(transformed(0)==false){
          selected(0)=transformed(1)
          selected(1)=transformed(2)
          selected(2)=transformed(3)
          estimator = estimators(0)
          selector = 1

        }
        else {
          selected(0)=transformed(4)
          selected(1)=transformed(5)
          selected(2)=transformed(6)
          estimator = estimators(1)
          selector = 4
        }*/
        val selected = transformed
        val selector = 0
        val estimator = estimatorS
        //for (param<- params){
        for (param<- 0 until selected.length){
          //TODO: Neural networks
          /*      if(param.name.equals("layers"))
                {

                }*/
          //else {
          //TODO: How to treat categorical etc.
          pMap.put(params(selector+param), selected(i))
          i += 1
          //}
        }
        //pMap.put(params(3),layers)
        val itEstimator = estimator.copy(pMap)
        //        val trainInit = System.currentTimeMillis()
        //val mlModel = new OneVsRest().setClassifier(estimator.asInstanceOf[LinearSVC]).fit(training)
        val mlModel = itEstimator.fit(subsampledTraining).asInstanceOf[Model[_]] //TODO
        //        val trainEnd = System.currentTimeMillis()
        //        println("Training takes: "+(trainEnd-trainInit))
        //        val testInit = System.currentTimeMillis()
        val result = mlModel.transform(testing)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
        (1 - evaluator.evaluate(result))
      } //TODO

      def errorN(previousPoint: DenseVector[Double]) : Double ={
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoint)
        //        println("Points transformed: ")
        //        transformed.foreach(x=> println(x.toString))
        var i = 0
        val pMap = new ParamMap()
        //TODO: Neural networks (784 or 780 first layer)
       /* var layers : Array[Int] = null
        if(transformed(1)==6){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
            transformed(4).asInstanceOf[Int], transformed(5).asInstanceOf[Int], nClass)
        }
        else if (transformed(1)==5){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int],
            transformed(4).asInstanceOf[Int], nClass)
        }
        else if (transformed(1)==4){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], transformed(3).asInstanceOf[Int], nClass)
        }
        else if(transformed(1)==3){
          layers = Array[Int](nFeat, transformed(2).asInstanceOf[Int], nClass)
        }
        else if(transformed(1)==2){
          layers = Array[Int](nFeat, nClass)
        }
       val selected = new Array[Any](3)
        var estimator : Estimator[_] = null
        var selector = 0
        if(transformed(0)==false){
          selected(0)=transformed(1)
          selected(1)=transformed(2)
          selected(2)=transformed(3)
          estimator = estimators(0)
          selector = 1

        }
        else {
          selected(0)=transformed(4)
          selected(1)=transformed(5)
          selected(2)=transformed(6)
          estimator = estimators(1)
          selector = 4
        }*/
       val selected = transformed
        val selector = 0
        val estimator = estimatorS
        //for (param<- params){
        for (param<- 0 until selected.length){
          //TODO: Neural networks
          /*      if(param.name.equals("layers"))
                {

                }*/
          //else {
          //TODO: How to treat categorical etc.
          pMap.put(params(selector+param), selected(i))
          i += 1
          //}
        }
        //pMap.put(params(3),layers)
        val itEstimator = estimator.copy(pMap)
        //        val trainInit = System.currentTimeMillis()
        //val mlModel = new OneVsRest().setClassifier(estimator.asInstanceOf[LinearSVC]).fit(training)
        val mlModel = itEstimator.fit(subsampledTraining).asInstanceOf[Model[_]] //TODO
        //        val trainEnd = System.currentTimeMillis()
        //        println("Training takes: "+(trainEnd-trainInit))
        //        val testInit = System.currentTimeMillis()
        val result = mlModel.transform(testing)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
        (1 - evaluator.evaluate(result))
      } //TODO
//        val testEnd = System.currentTimeMillis()
//        println("Testing takes: "+(testEnd-testInit))
      val rangeSubsamples = 0 until stagePoints(i) toList

      if(stagePoints(i)>1) {
        val futureResult = rangeSubsamples.map(x => error(previousPoints(x)))
        val futureErrors = Future.sequence(futureResult)
        val errors = Await.result(futureErrors, Duration.Inf).toArray
        for (t <- 0 until stagePoints(i)) {
          pointResult(t) = (previousPoints(t), errors(t))
        }
      }
      else{
        pointResult(0) = (previousPoints(0), errorN(previousPoints(0)))
      }
      val endIter = System.currentTimeMillis()
      println("Time for rung: "+ (endIter-initIter))
      //      val endB = System.currentTimeMillis()
      //      println("Bracket time:")
      //      println(endB-initB)
      //If not finished, halve taking best points
      if(i+1<stagePoints.length) previousPoints = pointResult.sortBy(_._2).map(_._1).take(stagePoints(i+1))
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
//    println("Points feedback: ")
//    feedback.foreach(x=> println(x.toString))

    println("Validation error achieved: " + finalError)

    hyperHistory += feedback
    lossHistory += finalError
  }


  def oldShaTrial(grid: DenseMatrix[Double], subsamples: Array[Dataset[Row]], stagePoints: Array[Int]) : Unit = {
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
      //      val initB = System.currentTimeMillis()
      val pointResult = new Array[(DenseVector[Double],Double)](stagePoints(i))
      //TODO: Using same subsample for every point, maybe take different ones partitioning to parallelize??
      val initIter = System.currentTimeMillis()
      val subsampledTraining = subsamples(i)
      //      println("Subsample count is: "+subsampledTraining.count())
      //      println("Subsample partitions is: "+subsampledTraining.rdd.getNumPartitions)
      //      println ("iterations is: "+stagePoints(i))
      for (j <- 0 until stagePoints(i)){ //Point j of this round
        val initPoint = System.currentTimeMillis()
        val transformed = modelTransformations(previousPoints(j))
        //        println("Points transformed: ")
        //        transformed.foreach(x=> println(x.toString))
        var i = 0
        for (param<- params){
          estimatorS.set(param,transformed(i))
          i+=1
        }
        //        val trainInit = System.currentTimeMillis()
        //val mlModel = new OneVsRest().setClassifier(estimator.asInstanceOf[LinearSVC]).fit(training)
        val mlModel = estimatorS.fit(subsampledTraining).asInstanceOf[Model[_]] //TODO
        //        val trainEnd = System.currentTimeMillis()
        //        println("Training takes: "+(trainEnd-trainInit))
        //        val testInit = System.currentTimeMillis()
        val result = mlModel.transform(testing)
        val error = 1 -evaluator.evaluate(result) //TODO
        //        val testEnd = System.currentTimeMillis()
        //        println("Testing takes: "+(testEnd-testInit))
        pointResult(j) = (previousPoints(j),error)
        val endPoint = System.currentTimeMillis()
        println("Time for point: "+(endPoint-initPoint))
      }
      val endIter = System.currentTimeMillis()
      println("Time for rung: "+ (endIter-initIter))
      //      val endB = System.currentTimeMillis()
      //      println("Bracket time:")
      //      println(endB-initB)
      //If not finished, halve taking best points
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
    //    println("Points feedback: ")
    //    feedback.foreach(x=> println(x.toString))

    println("Validation error achieved: " + finalError)

    hyperHistory += feedback
    lossHistory += finalError
  }

  //Returns best combination of hyperparameters and their loss
  def getBest() : (Array[Double], Double) = {
    val best = argmin(lossHistory.toArray)
    println("Best is: ")
    println(DenseVector(hyperHistory(best)).toString)
    return (hyperHistory(best),lossHistory(best))
  }
}
