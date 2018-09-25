package org.apache.spark.ml.tuning.smac

import breeze.linalg.{DenseMatrix, DenseVector, argmax}
import breeze.numerics.{exp, log, sqrt}
import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.{VectorUDT, Vectors}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor, RandomForestSMACRegressionModel, RandomForestSMACRegressor}
import org.apache.spark.ml.tuning.{AcquisitionFunction, SmboModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext}
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Random

/**
  * Class containing a random forest surrogate to make suggestions of next hyperparameters to try based on past trials,
  * maxmizing the acquisition function over a the predictions of the random forest model (as in SMAC)
  * Note that in the matrix representation, columns correspond to the input vectors and rows correspond to each feature
  *
  *
  *
  * Feature indexer to check which variables are categorical
  Training them as categorical
  For that necessary that the feedback has less number of classes than set in feature indexer.
  Hence, for categorical variables return dummy feedback (e.g. 0.0,0.2,0.4,0.6,0.8,1.0) etc.
  and take care of that in model transformations (0.0 -> cat0, 0.2 -> cat1, etc.
  Moreover, necessary to do so as well in grid for searching and handling neighbours when refining search
  */
//Category Index for neighbour search: 0=> continous, positive => number of categories, negative => Number of integers
class RandomForestSurrogate (ctxt: SQLContext, categoryIndex: Array[Int] = null) extends SmboModel {

  //SMAC parameters (fixed so far)
  val numTrees = 10
  val splitRatio =   5.0/6.0
  val maxDepth = 20 //FROM SMAC DOC
  val minInstancePerNode = 3


  var model : RandomForestSMACRegressionModel = null
  //Best loss so far
  var min = Double.MinValue
  //Number of points to perform local search (best n)
  val nPointsLocalSearch = 10

  var assembler : VectorAssembler = null
  var schema: StructType = null
  var schemaNoLabel: StructType = null
  var categories = categoryIndex

  //Prepare learner with parameters
  val rf = new RandomForestSMACRegressor().setNumTrees(numTrees).setFeatureSubsetStrategy(splitRatio.toString)
    .setMaxDepth(maxDepth).setMinInstancesPerNode(minInstancePerNode)
  //Outputting random points in uneven iterations
  var iterations = 0

  /**
  Train the random forest model given a dataset, converting to appropriate spark format and using spark's random forest
    implementation
    **/
  def fit(xtrain: DenseMatrix[Double], ytrain: DenseVector[Double]): this.type ={
    min = breeze.linalg.min(ytrain.toDenseMatrix)

    if(categories==null) categories = Array.fill(xtrain.rows)(0)

    var featMap = Map.empty[Int, Int]
    for (i<-0 until categories.length){
      //Taking the minimum between number of categories and training examples to avoid errors
      if(categories(i)>0) featMap += (i->categories(i))//(i->math.min(categories(i),ytrain.length))
    }

    rf.setCategoricalFeatures(featMap)
    //Prepare assembler, feature names and schema to create dataset
    var featNames : Array[String] = null
    if(assembler == null){
      val featBuffer = new ArrayBuffer[String]()
      var nameF : String = ""
      val schemaSeq = new ArrayBuffer[StructField]()
      for (t<- 0 until xtrain.rows){
        nameF = "feat"++t.toString
        featBuffer += nameF
        schemaSeq += StructField(name = nameF, dataType = DoubleType, nullable = false)
      }
      schemaNoLabel = StructType(schemaSeq)
      schemaSeq += StructField(name = "label", dataType = DoubleType, nullable = false)
      featNames = featBuffer.toArray
      assembler = new VectorAssembler().setInputCols(featNames)
        .setOutputCol("features")
      schema =  StructType(schemaSeq)
    }

    //Concatenate hyperparameters and corresponding loss
    val concatMatrix = DenseMatrix.vertcat(xtrain,new DenseMatrix[Double](1,ytrain.length,ytrain.toArray))

    //Create a Row object for each column
    val rowArr = new ArrayBuffer[Row]
    for (t<- 0 until xtrain.cols) {
      val col = concatMatrix(::, t).toArray
      rowArr += Row.fromSeq(col)
    }
    //Create RDD from rows and dataset from RDD
    val rdd = ctxt.sparkContext.parallelize(rowArr,1)
    val ds = ctxt.createDataFrame(rdd,schema)

    //Train the model
    val trainDs= assembler.transform(ds)
    model = rf.fit(trainDs)
    return this
  }

  /**
  Suggest the best hyperparameters according to the model,
  converting to appropriate spark format and using spark's random forest implementation.
  It starts from a set of initial candidates (grid) which is later refined by local search
    **/
  def next(candidates: DenseMatrix[Double]): DenseVector[Double] = {

    //Create a Row object for each column
    if(iterations%2==0){//Return best predicted
    val rowArr = new ArrayBuffer[Row]
      for (t<- 0 until candidates.cols) {
        val col = candidates(::, t).toArray
        rowArr += Row.fromSeq(col)
      }
      //Create RDD from rows and dataset from RDD and prepare features
      val rdd = ctxt.sparkContext.parallelize(rowArr,1)
      val ds = ctxt.createDataFrame(rdd,schemaNoLabel)
      val testDs= assembler.transform(ds)
      //Take the best points and order according to EI

      //Including a column with the minimum value to be used in the UDF that calculates the EI in a parallel manner
      val result = model.transform(testDs).withColumn("min",lit(min))
      val resultEi= result.withColumn("EI",eiUDF(result.col("prediction"),result.col("empiricalVariance"),result.col("min")))
      val ordered = resultEi.orderBy(resultEi.col("EI").desc).limit(nPointsLocalSearch).cache()

      //Create array to store points and their EI
      val results = new ArrayBuffer[(DenseVector[Double],Double)](nPointsLocalSearch)
      for (t<- 0 until nPointsLocalSearch){
        //Prepare initial points for local search
        val vector = ordered.select(ordered.col(model.getFeaturesCol)).collect()(t).get(0).asInstanceOf[org.apache.spark.ml.linalg.Vector]
        val ei = ordered.select(ordered.col("EI")).collect()(t).getDouble(0)
        results += Tuple2(DenseVector(vector.toArray),ei)
      }
      //Do local search for each point found
      for (point<- 0 until nPointsLocalSearch){
        results(point)=localSearch(results(point)._1,results(point)._2,ctxt)
      }
      //Avoid choosing the same one always when a tie is present by shuffling
      val shuffledResults = Random.shuffle(results)
      //Take best result and return it
      val eis = shuffledResults.toArray.map(x=>x._2)
      val points = shuffledResults.toArray.map(x=>x._1)

      val best = argmax(eis)

      iterations += 1
      return points(best)}
    else{ //Return random suggestion
      val result = generateRandomPoint(candidates.rows)
      iterations+=1
      return result
    }
  }

  /**
  Generate a random combination of hyperparameters
    **/
  def generateRandomPoint(dims: Int) : DenseVector[Double] = {
    val result = new DenseVector[Double](dims)
    for (i <- 0 until dims){
      if(categories(i)<=0) result(i)=Random.nextDouble()
      else result(i)= Random.nextInt(categories(i))
    }
    return result
  }

  /**
  Search for the top proposals according to the random forest model. It is the same method as next but returning
    top N proposals instead of only the best one.
    Not using the variable iterations
    **/
  def topNext(n: Int, candidates: DenseMatrix[Double]): Array[DenseVector[Double]] = {

    //Create a Row object for each column
    val rowArr = new ArrayBuffer[Row]
    for (t<- 0 until candidates.cols) {
      val col = candidates(::, t).toArray
      rowArr += Row.fromSeq(col)
    }
    //Create RDD from rows and dataset from RDD and prepare features
    val rdd = ctxt.sparkContext.parallelize(rowArr,1)
    val ds = ctxt.createDataFrame(rdd,schemaNoLabel)
    val testDs= assembler.transform(ds)

    //Take the best points and order according to EI

    //Including a column with the minimum value to be used in the UDF that calculates the EI in a parallel manner
    val result = model.transform(testDs).withColumn("min",lit(min))
    val resultEi= result.withColumn("EI",eiUDF(result.col("prediction"),result.col("empiricalVariance"),result.col("min")))
    val ordered = resultEi.orderBy(resultEi.col("EI").desc).limit(nPointsLocalSearch).cache()

    //Create array to store points and their EI
    val results = new ArrayBuffer[(DenseVector[Double],Double)](nPointsLocalSearch)
    for (t<- 0 until nPointsLocalSearch){
      //Prepare initial points for local search
      val vector = ordered.select(ordered.col(model.getFeaturesCol)).collect()(t).get(0).asInstanceOf[org.apache.spark.ml.linalg.Vector]
      val ei = ordered.select(ordered.col("EI")).collect()(t).getDouble(0)
      results += Tuple2(DenseVector(vector.toArray),ei)
    }
    //Do local search for each point found
    for (point<- 0 until nPointsLocalSearch){
      results(point)=localSearch(results(point)._1,results(point)._2,ctxt)
    }
    //Avoid choosing the same one always when a tie is present by shuffling
    val shuffledResults = Random.shuffle(results)
    //Take best result and return it
    val eis = shuffledResults.toArray.map(x=>x._2)
    val points = shuffledResults.toArray.map(x=>x._1)

    val (_, indices) = eis.zipWithIndex.sorted.unzip
    val topPoints = new Array[DenseVector[Double]](n)
    for (indx <- 0 until n){
      if(indx%2==0)  topPoints(indx)=points(indices(indx/2))
      else topPoints(indx) = generateRandomPoint(candidates.rows) //Interleave random points in the list
    }
    return topPoints
  }


  /**
  Do a local search based on a starting point and EI.
    **/
  def localSearch(point:DenseVector[Double], ei: Double, ctxt:SQLContext): (DenseVector[Double], Double) = {

    var currentPoint = point
    var currentEi = 0.0
    var newEi = ei
    var it = 0
    var newPoint = point
    //We stop the search once none of the neighbours improves the previous EI
    while(currentEi<newEi){

      currentEi = newEi
      currentPoint = newPoint

      //Generate random neighbours to check their EI
      val v = randomNeighbours(currentPoint)

      //Create a Row object for each neighbour
      val rowArr = new ArrayBuffer[Row]
      for (t<- 0 until v.length) {
        val col = v(t).toArray
        rowArr += Row.fromSeq(col)
      }

      //Create RDD from rows and dataset from RDD and prepare features
      val rdd = ctxt.sparkContext.parallelize(rowArr,1)
      val ds = ctxt.createDataFrame(rdd,schemaNoLabel)
      val data = assembler.transform(ds)

      //Including a column with the minimum value to be used in the UDF that calculates the EI in a parallel manner
      val result = model.transform(data).withColumn("min",lit(min))

      val resultEi= result.withColumn("EI",eiUDF(result.col("prediction"),result.col("empiricalVariance"),result.col("min"))).cache()
      newEi = resultEi.select(resultEi.col("EI")).agg(org.apache.spark.sql.functions.max(resultEi.col("EI")))
        .collect()(0).getDouble(0)

      newPoint = DenseVector(resultEi.filter(resultEi.col("Ei").===(newEi)).select(resultEi.col(model.getFeaturesCol))
        .collect().toSeq(0).get(0).asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray)
      it+=1
    }
    return (currentPoint, currentEi)
  }

  /**
  Generate random neighbours given a point. To be used in local search.
    Category index has to be 0 for continuous and >=2 for categorical
    **/
  def randomNeighbours(v: DenseVector[Double]): Array[DenseVector[Double]] = {
    val vArr = v.toArray
    val neighbours = new ArrayBuffer[DenseVector[Double]]
    for (i<-0 until vArr.length) { //For each hyperparameter
      val newPoint = vArr.clone()
      if (categories(i) == 0) { //Numerical: Generate 4 neighbours
        val g = Gaussian(vArr(i), 0.2)
        for (r <- 0 until 4) { //For each of the 4 neighbours, randomly sample its value
          var sampled = g.sample()
          while (sampled < 0.0 || sampled > 1.0) sampled = g.sample()
          newPoint(i) = sampled
          neighbours += DenseVector(newPoint.clone())
        }
      }
      else if(categories(i)<=0) { // Integer if negative, sampling one more and one less
        val stepSize : Double = 1.0/((-categories(i))-1.0)
        newPoint(i) = vArr(i) + stepSize
        if(newPoint(i)<1.0) neighbours += DenseVector(newPoint.clone())
        newPoint(i) = vArr(i) - stepSize
        if(newPoint(i)>0.0) neighbours += DenseVector(newPoint.clone())
      }
      else { //Categorical sampling, one neighbour for each value of the categorical parameter
        val stepSize: Double = 1.0  /// (categories(i) - 1.0)
        for (t <- 0 until categories(i)) {
          newPoint(i) = stepSize * (t.toDouble)
          if (newPoint(i) != vArr(i)) {
            neighbours += DenseVector(newPoint.clone())
          }
        }
      }
    }
    return neighbours.toArray
  }

  //Defining UDF to calculate expected improvement on a dataset
  val eiUDF = udf((mean: Double, variance: Double, min: Double) => AcquisitionFunction.expectedImprovement(mean,variance,min))
  /*if(!SH){ eiUDF = udf((mean: Double, variance: Double, min: Double) => AcquisitionFunction.expectedImprovement(mean,variance,min))}
  else {eiUDF = udf((mean: Double, variance: Double, min: Double) => AcquisitionFunction.ucb(mean,variance))}
*/

}
