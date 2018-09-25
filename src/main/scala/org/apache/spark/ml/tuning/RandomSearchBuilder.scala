package org.apache.spark.ml.tuning

import scala.annotation.varargs
import scala.collection.mutable

import org.apache.spark.annotation.Since
import org.apache.spark.ml.param._

//nTrials: Number of random hyperparameter points to sample and try
class RandomSearchBuilder (nTrials: Int) {

  //Map of (parametper),(logarithmic scale,(init_range,end_range))
  private val rangeGrid = mutable.Map.empty[Param[_], ((_,_),Boolean)]
  private val categoricalGrid = mutable.Map.empty[Param[_], Array[_]]
  private val rand = scala.util.Random

  /**
   * Adds a param with multiple values (overwrites if the input param exists).
   */
  protected def addGrid[T](param: Param[T], ranges: (T,T), logScale:Boolean): this.type = {
    rangeGrid.put(param, (ranges,logScale))
    this
  }

  // specialized versions of addGrid for Java.

  /**
    * Adds a categorical param with several options
    */
  def addCategoricalGrid[T](param: Param[T], values: Array[T]): this.type = {
    categoricalGrid.put(param, values)
    this
  }

  /**
   * Adds a double param within a range.
   */
  def addGrid(param: DoubleParam, range: (Double,Double), logScale:  Boolean): this.type = {
    addGrid[Double](param, range, logScale)
  }

  /**
   * Adds an int param within a range.
   */
  def addGrid(param: IntParam, range: (Int, Int), logScale:  Boolean): this.type = {
    addGrid[Int](param, range, logScale)
  }

  /**
   * Adds a float param within a range.
   */
  def addGrid(param: FloatParam, range: (Float, Float),logScale:  Boolean): this.type = {
    addGrid[Float](param, range, logScale)
  }

  /**
   * Adds a long param within a range.
   */
  def addGrid(param: LongParam, range: (Long, Long), logScale:  Boolean): this.type = {
    addGrid[Long](param, range, logScale)
  }

  /**
   * Adds a boolean param with true and false.
   */
  def addGrid(param: BooleanParam): this.type = {
    addGrid[Boolean](param, (true, false), false)
  }


  /**
    * Generates a random point within a given range sampling from a uniform distribution for the corresponding type.
    */
  def generateRandomPoint(range:(Any,Any)) : Any =  {
    if(range._1.isInstanceOf[java.lang.Number] && range._2.isInstanceOf[java.lang.Number]) {
      if (range._1.asInstanceOf[java.lang.Number].doubleValue() > range._2.asInstanceOf[java.lang.Number].doubleValue()) {
        throw new Exception("The start of a range cannot be bigger than the end")
      }
    }

    range match {
    case (init:Double, end:Double) => init + (end-init)*rand.nextDouble()
    case (init:Int, end:Int)=> init + rand.nextInt((end - init))
    case (init:Float, end:Float) => init + (end-init)*rand.nextFloat()
    case (init:Long, end:Long) => init + (rand.nextLong() % (end - init))
    case (opt1:Boolean, opt2:Boolean) => if (rand.nextDouble()>=0.5) opt1 else opt2
    //If any other type, treat as boolean (e.g. strings as categorical variables)
    case (opt1:Any, opt2:Any) => if (rand.nextDouble()>=0.5) opt1 else opt2
    }
  }

  /**
    * Generates a random point sampling on a uniform distribution in the log scale within a given range for the corresponding type
    */
  def generateRandomPointLogScale(range:(Any,Any)) : Any =  {
    if(range._1.isInstanceOf[java.lang.Number] && range._2.isInstanceOf[java.lang.Number]) {
      if (range._1.asInstanceOf[java.lang.Number].doubleValue()>range._2.asInstanceOf[java.lang.Number].doubleValue()){
        throw new Exception("The start of a range cannot be bigger than the end")}
      if(range._1.asInstanceOf[java.lang.Number].doubleValue()<=0){
        throw new Exception("The start of a logarithmic range has to be bigger than 0")
      }
    }
    range match {
      case (init:Double, end:Double) => math.pow(10,(math.log10(end)-math.log10(init))*rand.nextDouble()+math.log10(init))
      case (init:Int, end:Int)=> math.round(math.pow(10,(math.log10(end)-math.log10(init))*rand.nextDouble()+math.log10(init))).toInt
      case (init:Float, end:Float) =>  math.pow(10,(math.log10(end)-math.log10(init))*rand.nextFloat()+math.log10(init))
      case (init:Long, end:Long) => math.round(math.pow(10,(math.log10(end)-math.log10(init))*rand.nextDouble()+math.log10(init)))
      //If any other type, treat as boolean (e.g. strings as categorical variables)
      case (opt1:Any, opt2:Any) => if (rand.nextDouble()>=0.5) opt1 else opt2
    }
  }

  def generateRandomCategoricalPoint(values:Array[_]) : Any = {
    val ranges = values.size
    val stepSize = 1.0/ranges
    val random = rand.nextDouble()
    val bucket = math.floor(random/stepSize).toInt
    println("ranges: "+ranges+" stepSize: "+stepSize+" rand: "+random+" bucket: "+bucket)
    return values(bucket)
  }

  /**
   * Builds and returns all smaples of parameters in the range specified by the random search grid.
    * Note that categorical grids will be overriden by range grids if the same parameter is defined through categorical
    * and through normal grids.
   */
  def build(): Array[ParamMap] = {
    val paramMaps : Array[ParamMap] = new Array[ParamMap](nTrials)
    for (t <- 0 until nTrials) {
      paramMaps(t) = new ParamMap()
      categoricalGrid.foreach((tuple)=> { //Tuple 1: Parameter Tuple 2: Array of possible values
        paramMaps(t).put(tuple._1.asInstanceOf[Param[Any]], generateRandomCategoricalPoint(tuple._2))
      })
      rangeGrid.foreach((tuple)=>{ //Tuple 1: Parameter Tuple 2: (range,logScale)
        if(!tuple._2._2) paramMaps(t).put(tuple._1.asInstanceOf[Param[Any]], generateRandomPoint(tuple._2._1))
        else paramMaps(t).put(tuple._1.asInstanceOf[Param[Any]], generateRandomPointLogScale(tuple._2._1))
      })
    }
    paramMaps
  }
}