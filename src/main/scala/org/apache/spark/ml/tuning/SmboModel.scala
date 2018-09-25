package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}

trait SmboModel {

  //Fit the surrogate model given a list of hyperparameters tried and corresponding loss achieved
  def fit(hypers:DenseMatrix[Double], loss: DenseVector[Double]): SmboModel

  //Returns a suggestion of hyperparameters based on the acquisition function from the surrogate model
  def next(grid: DenseMatrix[Double]): DenseVector[Double]

  //Returns a list of the top suggestions of the surrogate model
  def topNext(topN: Int, grid: DenseMatrix[Double]): Array[DenseVector[Double]]

}
