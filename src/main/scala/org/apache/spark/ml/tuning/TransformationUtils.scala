package org.apache.spark.ml.tuning

import breeze.linalg.argmax

object TransformationUtils {

  //Transforming numbers in the 0-1 range resulting from the surrogate models to what is going to be used in the
  //learning algorithm


  def rangeTransformation(input: Double, initRange: Double, endRange: Double) : Double = {
    if (endRange < initRange){
      throw new Exception("The start of a range cannot be bigger than the end")}
    return initRange + (endRange-initRange)*input
  }

  //CAUTION: The start of a logarithmic range has to be bigger than 0
  def logarithmicTransformation(input: Double, initRange: Double, endRange: Double) : Double = {
    if (endRange < initRange){
      throw new Exception("The start of a range cannot be bigger than the end")}
    if(initRange<=0){
      throw new Exception("The start of a logarithmic range has to be bigger than 0")
    }
  return math.pow(10,(math.log10(endRange)-math.log10(initRange))*input+math.log10(initRange))
  }

  def binaryTransformation(input: Double) : Boolean = {
    if(input<0.5) return false else return true
  }

  def integerTransformation(input: Double, numberInt: Int, initInteger: Int) : Int = {
    val stepSize = (1.0/numberInt)+1E-8
    val bucket = math.floor(input/stepSize).toInt
    return initInteger+bucket
  }

  //For Spearmint
  def categoryTransformation (inputCat: Array[Double], categories: Array[Any]) : Any = {
    val result = Array[Double](inputCat.length)
    val best = argmax(inputCat)
    return Array.fill(inputCat.length)(categories(best))
  }


  //Transforming the numbers in the 0-1 range resulting from the surrogate models to what is going to be used as
  //feedback. Useful for conditional hyperparameters to avoid noise, not recommendable to use for binary and
  // similar hyperparameters because the function can learn the discretization of the space anyway if they are used.

  def middlePointTransformation(input: Double) : Double = 0.5

  //NOT USE FROM HERE NOW!!

  def binaryFeedbackTransformation(input: Double) : Double = {
    if(input<0.5) return 0.0 else return 1.0
  }

  def integerFeedbackTransformation(input: Double, numberInt: Int) : Double = {
    val stepSize = (1.0/numberInt)+1E-8
    val bucket = math.floor(input/stepSize).toInt
    //Middle of the range
    return stepSize*bucket+stepSize/2.0
  }

  def categoryTransformation (inputCat: Array[Double]) : Array[Double] = {
    val result = Array.fill(inputCat.length)(0.0)
    val best = argmax(inputCat)
    result(best) = 1.0
    return result
  }
}
