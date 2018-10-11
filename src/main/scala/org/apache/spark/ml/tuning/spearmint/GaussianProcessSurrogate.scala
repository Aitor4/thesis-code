package org.apache.spark.ml.tuning.spearmint

import breeze.linalg.{*, Axis, DenseMatrix, DenseVector, argmin, argsort, cholesky, diag, inv, max, min, sum}
import breeze.stats.distributions.{Gaussian, Rand}
import breeze.numerics._
import breeze.optimize._
import breeze.util.LazyLogger
import org.apache.spark.ml.tuning.SmboModel

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import org.slf4j.helpers.NOPLogger

/**
  * Class containing a gaussian process surrogate to make suggestions of next hyperparameters to try based on past trials,
  * maxmizing the acquisition function over a gaussian process
  * Note that in the matrix representation, columns correspond to the input vectors (observations)
  * and rows correspond to each feature
  */
class GaussianProcessSurrogate extends SmboModel {

  //Training data
  var Xtrain = new DenseMatrix[Double](0, 0)
  var Ytrain = new DenseVector[Double](0)
  //Best loss so far
  var best = Double.MaxValue

  val SQRT_5 = sqrt(5.0)
  //Gaussian process hyperparameters
  var lastAmp2 : Double = 1.0
  var lastMean : Double = 0.0
  var lastNoise : Double = 0.001
  var lastLengthScale : DenseVector [Double] = null

  //Priors of hyperparameters of gaussian process
  val noiseScale = 0.1 //horseshoe prior
  val amp2Scale = 1.0 //zero-mean log normal prior
  var maxLs = 1.0 // top-hat prior on length scales

  //Iterations for marginilizing across samples of hyperparameters
  val mcmcIters=10

  //Number of points to keep further optimizing out of the grid
  val subsetCandidates=10

  //Sampled hyperparameters of last iteration
  val proposedHyper=new Array[(DenseVector[Double],DenseVector[Double])](mcmcIters)

  //Iterations to perform burnin and whether to perform it (only initially)
  var init = true
  var needsBurnin = true
  val burnin = 100

  /**
  Distance between two vectors
    **/
  def distance(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val square = (x - y).t * (x - y)
    sqrt(square)
  }

  /**
  Gradient of squared distance
    **/
  def gradDistance2(x : DenseVector[Double], y : DenseVector[Double]): DenseVector[Double] = {
    2.0*(x - y)
  }

  /**
  Squared exponential kernel (example for another kernel)
    **/
  def squaredExponential(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val pow = -0.5 * distance(x, y)
    exp(pow)
  }

  /**
  Matern 5/2 kernel commonly used in several bayesian optimization packages including spearmint
    **/
  def matern52(lengthScale:DenseVector[Double], x1: DenseVector[Double], x2:DenseVector[Double] = null): Double = {
    var nx1 : DenseVector[Double] = null
    var nx2 : DenseVector[Double] = null
    if (x2==null){
      //Find distance with self for x1.
      // Rescale.
      nx1 = x1 / lengthScale
      nx2 = nx1
    }
    else{
      //Rescale.
      nx1 = x1 / lengthScale
      nx2 = x2 / lengthScale
    }
    val r = distance(nx1, nx2)
    val r2 = r*r
    val cov = (1.0 + SQRT_5 * r + (5.0 / 3.0) * r2) * exp(-SQRT_5 * r)
    return cov
  }

  /**
  Gradient of the Matern 5/2 kernel
    **/
  def grad_Matern52(lengthScale:DenseVector[Double],x1:DenseVector[Double], x2:DenseVector[Double]=null): DenseVector[Double] ={
    var nx1 : DenseVector[Double] = null
    var nx2 : DenseVector[Double] = null
    if (x2==null){
      //Find distance with self for x1.
      // Rescale.
      nx1 = x1 / lengthScale
      nx2 = nx1
    }
    else{
      //Rescale.
      nx1 = x1 / lengthScale
      nx2 = x2 / lengthScale
    }
    val r       = distance(nx1, nx2)
    val grad_r2 = -(5.0/6.0)*exp(-SQRT_5*r)*(1 + SQRT_5*r)
    //We have to divide the gradient of the distance by the lengthscale for the derivation to be correct
    return grad_r2 * (gradDistance2(nx1, nx2) / lengthScale)
  }

  /**
  Covariance calculation of two matrices using the appropriate kernel
    **/
  def kernel(x1: DenseMatrix[Double], x2:DenseMatrix[Double], ls: DenseVector[Double]): DenseMatrix[Double] = {
    val K = DenseMatrix.zeros[Double](x1.cols, x2.cols)
    for(i <- 0 until x1.cols) {
      for(j <- 0 until x2.cols) {
        K(i, j) =  matern52(ls,x1(::, i), x2(::, j))
      }
    }
    return K
  }

  /**
  Covariance gradient calculation of two matrices using the appropriate kernel
    **/
  def kernelGrad(x1: DenseMatrix[Double], x2:DenseVector[Double], ls: DenseVector[Double]): DenseMatrix[Double] = {
    val K = DenseMatrix.zeros[Double](x1.cols, x2.length)
    for(i <- 0 until x1.cols) {
        K(i,::) :=  grad_Matern52(ls,x1(::,i), x2).t
    }
    return K
  }

  /**
  Set the training data and initial values for gaussian hyperparameters before sampling
    **/
  def fit(xtrain: DenseMatrix[Double], ytrain: DenseVector[Double]) : this.type = {
    Xtrain = xtrain
    Ytrain = ytrain
    best = breeze.linalg.min(ytrain)

    if(init){
      //Only setting default hyperparameters before sampling on initialization
      lastMean = breeze.stats.mean(Ytrain)
      lastAmp2 = breeze.stats.stddev(Ytrain) + 1e-4
      lastNoise = 1e-3
      maxLs = sqrt(Xtrain.rows) //Maximum distance between two poits
      lastLengthScale = 0.33*maxLs*DenseVector.ones[Double](Xtrain.rows) //Third of the maximum distance
      init = false
    }
    this
  }

  /**
  GP regression, given test and training data (call to predict mean and variance of a new point)
    **/
  def predict(x: DenseVector[Double],mean:Double, amp2:Double, noise:Double,
              lengthScale: DenseVector[Double]): (Double, Double) = {
    val K = amp2*kernel(Xtrain, Xtrain, lengthScale) + DenseMatrix.eye[Double](Xtrain.cols) * (noise + 1e-6)
    val L = cholesky(K)
    val Xtest = x.toDenseMatrix.t

    val K_s = amp2*kernel(Xtrain, Xtest, lengthScale)
    val Lk = L \ K_s


    val mu = Lk.t * (L \ (Ytrain-mean))+mean
    val cov = amp2*(1.0+1e-6) -  sum(pow(Lk,2),Axis._0)
    //Own kernel is 1 for the same vector, that why no computed

    //Returns only first element (prediction) because only predicting one element at a time
    (mu(0), sqrt(cov(0, 0)))

  }

  def predict(x:DenseVector[Double]) : (Double, Double) = predict(x,lastMean,lastAmp2,lastNoise,lastLengthScale)


  /**
  Calculate expected improvement of a point based on its mean and variance, and optionally the gradient of the EI
    **/
  def expectedImprovementWithGradient(x: DenseVector[Double], gradient:Boolean,
                                      mean:Double, amp2:Double, noise:Double, lengthScale: DenseVector[Double]):
  (Double,DenseVector[Double]) = {
    val K = amp2*kernel(Xtrain, Xtrain, lengthScale) + DenseMatrix.eye[Double](Xtrain.cols) * (noise + 1e-6)
    val L = cholesky(K)
    val Xtest = x.toDenseMatrix.t

    val K_s = amp2*kernel(Xtrain, Xtest, lengthScale)
    val Lk = L \ K_s


    val mut = Lk.t * (L \ (Ytrain-mean))+mean
    val mu = mut(0)
    val covt = amp2*(1.0+1e-6) -  sum(pow(Lk,2),Axis._0) //Own kernel is 1 for the same vector, that why no computed
    val sigma = sqrt(covt(0, 0))

    val g = Gaussian(0, 1)

    val Z = (best - mu) / sigma
    val improvement = sigma*(Z * g.cdf(Z) + g.pdf(Z))

    if(!gradient) return (improvement,null)

    else{
      //Gradients of ei w.r.t. mean and variance

      val g_ei_m = -g.cdf(Z)
      val g_ei_s2 = 0.5*g.pdf(Z) / sigma

      val grad_cross=kernelGrad(Xtrain,x,lengthScale)
      val grad_xp_m = (L \ (Ytrain-mean)).t * grad_cross
      val grad_xp_v = -2.0*(Lk.t * grad_cross)

      val grad_xp = 0.5*amp2*(grad_xp_m.t*g_ei_m + grad_xp_v(0,::).t * g_ei_s2)

      return (improvement, grad_xp)
    }
  }

  def expectedImprovement(x:DenseVector[Double], mean:Double, amp2:Double, noise:Double,
                          lengthScale: DenseVector[Double]) =
    expectedImprovementWithGradient(x,false, mean, amp2, noise, lengthScale)._1

  /**
  Function to be minimized for finding best point according to the acquisition function
    **/
  val objective = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      var cumulativeImprovement=0.0
      var cumulativeGradient=DenseVector.zeros[Double](x.length)
      //Marginilizing over hyperparameter samples
      var it = 0
      for (hs<- proposedHyper){ //proposed hyper: (DenseVector(mean,amp2,noise),LengthScale)
        val (improvement, gradient) = expectedImprovementWithGradient(x,  true, hs._1(0), hs._1(1), hs._1(2), hs._2)
        cumulativeImprovement += improvement
        cumulativeGradient = cumulativeGradient + gradient
        it+=1
      }
      (-cumulativeImprovement, cumulativeGradient)
    }

    def calculateWithoutGradient(x: DenseVector[Double]) = {
      var cumulativeImprovement=0.0
      //Marginilizing over hyperparameter samples
      for (hs<- proposedHyper){ //proposed hyper: (DenseVector(mean,amp2,noise),LengthScale)
        val improvement = expectedImprovement(x,hs._1(0), hs._1(1), hs._1(2), hs._2)
        cumulativeImprovement += improvement
      }
      (-cumulativeImprovement)
    }

  }

  /**
  Function to be minimized without sampling hyperparameters, that is, over fixed hyperparameters
    **/
  val simpleObjective = new DiffFunction[DenseVector[Double]] {
    def calculate(x: DenseVector[Double]) = {
      var cumulativeImprovement=0.0
      var cumulativeGradient=DenseVector.zeros[Double](x.length)
      val (improvement, gradient) = expectedImprovementWithGradient(x, true, lastMean, lastAmp2, lastNoise, lastLengthScale)
      cumulativeImprovement = improvement
      cumulativeGradient = gradient
      (-cumulativeImprovement, cumulativeGradient)
    }
  }

  def restrictRange(x: Double) : Double = {
    if(x<0.0) return 0.0
    else if (x>1.0) return 1.0
    else return x
  }


  /**
  Finds the best candidate point given the current data and returns it.
    It starts with a set of initial candidates (grid) which are then optimized,
  searching for the best value out of that grid
    **/

  def next(originalCandidates: DenseMatrix[Double]): DenseVector[Double] = {
    //Spray a set of candidates around the best value so far (like in spearmint code)
    val currentBest = breeze.linalg.argmin(Ytrain)

    var candidates = originalCandidates

    for (t <- 0 until 10){ //Spraying 10 candidates arround the best
          val surrounding = Rand.gaussian.samplesVector(Xtrain.rows)*0.001 + Xtrain(::,currentBest)
          val surroundingPoint = surrounding.map(x=>restrictRange(x))
          candidates = DenseMatrix.horzcat(candidates,new DenseMatrix[Double](Xtrain.rows,1,surroundingPoint.toArray))
        }
    val overallEi = DenseVector.zeros[Double](candidates.cols)

    //Sample hyperparameters

    if(needsBurnin){ //Burnin hyperparameters the first time before sampling to be used
      needsBurnin=false
      for (mcmcIter <- 0 until this.burnin) {
        sampleHypers()
      }
    }
    for (mcmcIter <- 0 until this.mcmcIters) { //Sample hyperparameters and store them
      proposedHyper(mcmcIter) = sampleHypers()
    }

    //Find best points from the grid
    def calculateEI (candidateN:Int) = Future{
      objective.calculateWithoutGradient(candidates(::,candidateN))
    }
    val rangeCandidates = 0 until candidates.cols toList

    val candidatesEI = rangeCandidates.map(x=>calculateEI(x))

    val futureEI = Future.sequence(candidatesEI)

    val overallEI = Await.result(futureEI,Duration.Inf).toArray

    val indxs = argsort(overallEi)

    //In case there are less than subset Candidates
    val subsetCandidates = min(candidates.cols,this.subsetCandidates)

    //For the best candidates of the grid, start a local search around them with lbfgs-b (in parallel)

    def optimize (candidateN:Int) = Future{
      try{
        {new LBFGSB(DenseVector
          .zeros[Double](candidates.rows),
          DenseVector.ones[Double](candidates.rows),maxIter = 100, m=5,tolerance = 1E-5){
          override val logger = new LazyLogger(NOPLogger.NOP_LOGGER)
        }
          .minimize(objective,candidates(::,indxs(candidateN)))}
      }
      catch{
        // With few data and ill-suited problem, the optimization throws an exception very rarely
        case e: AssertionError => {
          println("Assertion error optimizing!!")
          println("With point: "+candidates(::,indxs(candidateN)).toString)
          //Take the original point without optimizing in that case
          candidates(::,indxs(candidateN))

        }
      }
    }

    val rangeOpt = 0 until subsetCandidates toList

    val optimized = rangeOpt.map(x=>optimize(x))

    val optResult = Future.sequence(optimized)

    val optimum = Await.result(optResult,Duration.Inf).toArray


    val optimumPoints = DenseMatrix.zeros[Double](candidates.rows,subsetCandidates)
    val optimizedEi = DenseVector.zeros[Double](subsetCandidates)

    for (p <- 0 until subsetCandidates){
      optimumPoints(::,p) := optimum(p)
      optimizedEi(p) = objective.calculateWithoutGradient(optimum(p))
    }

    //Find best candidate and return it
    val bestCandidate = argmin(optimizedEi)
    return optimumPoints(::,bestCandidate)
  }

  /**
  Finds the top N best candidates. Uses the constant liar heuristic
    **/
  def topNext(n: Int, candidates: DenseMatrix[Double]): Array[DenseVector[Double]] = {

    //Value to return for the constant liar heuristic
    //Returning the max of seen losses to promote exploration
    val lie = breeze.stats.mean(Ytrain)

    val suggestedPoints = new Array[DenseVector[Double]](n)

    //Suggest point and train with new fantasize outcomes based on the constant lie
    for (point<- 0 until n){
      val suggestedPoint = next(candidates)
      suggestedPoints(point)=suggestedPoint
      val newX = DenseMatrix.horzcat(Xtrain, new DenseMatrix(Xtrain.rows,1,suggestedPoint.toArray))
      val newY = DenseVector.vertcat(Ytrain,DenseVector(Array(lie)))
      this.fit(newX, newY)
    }

    return suggestedPoints
  }

  /**
  Logprob to sample hyperparameters for the gaussian process
    **/
  def logProb(x:DenseVector[Double]): Double ={
    val mean = x(0)
    val amp2 = x(1)
    val noise = x(2)
    if (mean > max(Ytrain) || mean < min(Ytrain) || amp2<0 || noise<0) return Double.NegativeInfinity

    val cov   =  amp2 * kernel(Xtrain, Xtrain, lastLengthScale) + 1e-6*DenseMatrix.eye[Double](Xtrain.cols) + noise * DenseMatrix.eye[Double](Xtrain.cols) //Identity matrix times noise

    val chol = cholesky(cov)
    val logdet = 2 * sum(log(diag(chol)))

    val Rinv = inv(chol)
    val inverse=Rinv*Rinv.t

    var lp = -0.5*logdet-0.5*((Ytrain-mean).t * inverse * (Ytrain-mean))-0.5* Xtrain.cols.toDouble * log(2.0 * scala.math.Pi)

    //Roll in noise horseshoe prior.
    lp += log(log(1 + pow(noiseScale/noise,2)))
    // Roll in amplitude lognormal prior
    lp -= 0.5*pow((log(sqrt(amp2))/amp2Scale),2)

    return lp
  }

  /**
  Logprob to sample lengthscale hyperparameters for the gaussian process
    **/
  def logProbLs(ls:DenseVector[Double]): Double = {
    if (ls.exists(_ < 0) || ls.exists(_ > maxLs)) return Double.NegativeInfinity

    val cov   =  lastAmp2 * kernel(Xtrain, Xtrain, ls) + DenseMatrix.eye[Double](Xtrain.cols) * lastNoise + 1e-6*DenseMatrix.eye[Double](Xtrain.cols) //Identity matrix times noise
    val chol = cholesky(cov)
    val logdet = 2 * sum(log(diag(chol)))

    val Rinv = inv(chol)
    val inverse=Rinv*Rinv.t

    val lp = -0.5*logdet-0.5*((Ytrain-lastMean).t * inverse * (Ytrain-lastMean))-0.5* Xtrain.cols.toDouble * log(2.0 * scala.math.Pi)

    return lp
  }

  /**
  Sample hyperparameters, store them and return them
    **/
  def sampleHypers(): (DenseVector[Double],DenseVector[Double]) = {
    val parameters = sliceSample(DenseVector(lastMean, lastAmp2, lastNoise),logProb, false)
    lastMean = parameters(0)
    lastAmp2 = parameters(1)
    lastNoise = parameters(2)

    val ls = sliceSample(lastLengthScale,logProbLs, true)

    lastLengthScale = ls

    return(DenseVector[Double](parameters(0),parameters(1),parameters(2)),ls)
  }

  /**
  Performs MCMC slice sampling to sample hyperparameters given a direction
    **/
  def directionSlice(direction: DenseVector[Double], initX: DenseVector[Double], logProb: DenseVector[Double]=> Double):
  DenseVector[Double] = {
    val sigma = 1.0
    val max_steps_out=1000
    val r = scala.util.Random
    var upper = sigma*r.nextDouble()
    var lower = upper - sigma
    val llh_s = log(r.nextDouble()) + logProb(direction*0.0 + initX)
    var new_z : Double = 0.0
    var l_steps_out = 0
    var u_steps_out = 0
    while (logProb(direction*lower + initX) > llh_s && l_steps_out < max_steps_out){
     l_steps_out += 1
      lower -= sigma}
    while (logProb(direction*upper + initX) > llh_s && u_steps_out < max_steps_out){
    u_steps_out += 1
    upper += sigma
    }
    var steps_in = 0
    var finished = false
  while (!finished) {
    steps_in += 1
    new_z = (upper - lower) * r.nextDouble() + lower
    val new_llh = logProb(direction*new_z + initX)
    if (new_llh > llh_s) finished = true
    else if (new_z<0) lower=new_z
    else if (new_z>0) upper=new_z
  }
    return new_z*direction + initX
  }

  /**
  Performs MCMC slice sampling to sample hyperparameters
    **/
  def sliceSample(initX: DenseVector[Double], logProb: DenseVector[Double]=> Double,
                  compwise:Boolean = false) : DenseVector[Double] = {
    val dims = initX.length
    if(compwise){
      var ordering = (0 until dims).toArray
      ordering = breeze.linalg.shuffle(ordering)
      var curX = initX.copy
      for (d <- ordering){
        val direction = DenseVector.zeros[Double](dims)
        direction(d)=1.0
        curX = directionSlice(direction,curX, logProb)
      }
      return curX
    }
    else{
      var direction = DenseVector.rand(dims, Rand.gaussian)
      direction = direction / sqrt(sum(pow(direction,2)))
      return directionSlice(direction,initX,logProb)
    }
  }


}