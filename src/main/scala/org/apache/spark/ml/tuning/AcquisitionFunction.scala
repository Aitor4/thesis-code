package org.apache.spark.ml.tuning

import breeze.numerics.sqrt
import breeze.stats.distributions.Gaussian
import org.apache.spark.sql.functions.udf
import spire.random.Exponential

object AcquisitionFunction extends Serializable {

  /**
  Calculates expected improvement based on predicted mean and variance and current best loss (min)
    **/
  def expectedImprovement(mean: Double, variance: Double, min:Double): Double = {
    val g = Gaussian(0, 1)
    //Avoiding division by 0
    //TODO: Uncomment
    //val Z = (min - mean) / (sqrt(variance)+1e-6)
    val Z = (min - mean) / (sqrt(variance)+1e-6)
    val improvement = sqrt(variance)*(Z * g.cdf(Z) + g.pdf(Z))
    //    println("Min is: "+min)
    //    println("m is: "+mean)
    //    println("stdev is: "+sqrt(variance))
    //    println("EI is: "+improvement)
    //    println()

    return improvement
  }

  /**
  Calculates upper confidence bound with random parameter lambda
    **/
  def ucb(mean: Double, variance: Double): Double = {
    val expo = new breeze.stats.distributions.Exponential(1.0)
    val lambda = expo.sample()
    println("Using")
    return (-mean)+lambda*variance
  }


}
