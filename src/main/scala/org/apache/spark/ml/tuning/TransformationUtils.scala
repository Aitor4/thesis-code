package org.apache.spark.ml.tuning

import breeze.linalg.argmax

object TransformationUtils {

  //Transforming numbers in the 0-1 range resulting from the surrogate models to values used in the learning algorithm

  //Simple range transformation
  def rangeTransformation(input: Double, initRange: Double, endRange: Double) : Double = {
    if (endRange < initRange){
      throw new Exception("The start of a range cannot be bigger than the end")}
    return initRange + (endRange-initRange)*input
  }

  //Transform to logarithmic scale
  def logarithmicTransformation(input: Double, initRange: Double, endRange: Double) : Double = {
    if (endRange < initRange){
      throw new Exception("The start of a range cannot be bigger than the end")}
    if(initRange<=0){
      throw new Exception("The start of a logarithmic range has to be bigger than 0")
    }
  return math.pow(10,(math.log10(endRange)-math.log10(initRange))*input+math.log10(initRange))
  }

  //Transformation for boolean variables
  def binaryTransformation(input: Double) : Boolean = {
    if(input<0.5) return false else return true
  }

  //Transformation for integer variables
  def integerTransformation(input: Double, numberInt: Int, initInteger: Int) : Int = {
    val stepSize = (1.0/numberInt)+1E-8
    val bucket = math.floor(input/stepSize).toInt
    return initInteger+bucket
  }

  //Categorical transformation for Spearmint
  //TODO: Test, it has not been used so far
  def categoryTransformation (inputCat: Array[Double], categories: Array[Any]) : Any = {
    val result = Array[Double](inputCat.length)
    val best = argmax(inputCat)
    return Array.fill(inputCat.length)(categories(best))
  }


  //Transforming the numbers in the 0-1 range resulting from the surrogate models to what is going to be returned as
  //feedback. Useful for imputing conditional hyperparameters to avoid noise.

  def middlePointTransformation(input: Double) : Double = 0.5
}
