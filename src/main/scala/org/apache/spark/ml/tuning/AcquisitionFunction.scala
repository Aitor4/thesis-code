package org.apache.spark.ml.tuning

import breeze.numerics.sqrt
import breeze.stats.distributions.Gaussian

object AcquisitionFunction extends Serializable {

  /**
  Calculates expected improvement based on predicted mean and variance and current best loss (min)
    **/
  def expectedImprovement(mean: Double, variance: Double, min:Double): Double = {
    val g = Gaussian(0, 1)
    //Avoiding division by 0
    val Z = (min - mean) / (sqrt(variance)+1e-6)
    val improvement = sqrt(variance)*(Z * g.cdf(Z) + g.pdf(Z))

    return improvement
  }

  /**
  Calculates upper confidence bound with random parameter lambda
    **/
  def ucb(mean: Double, variance: Double): Double = {
    val expo = new breeze.stats.distributions.Exponential(1.0)
    val lambda = expo.sample()
    return (-mean)+lambda*variance
  }


}
