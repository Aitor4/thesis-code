package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.mutable.ArrayBuffer

class GridSurrogate(grid: DenseMatrix[Double]) extends SmboModel {

  var trials = -1
  override def next(grid: DenseMatrix[Double]): DenseVector[Double] = {
    trials +=1
    return grid(::,trials)
  }

  override def topNext(topN: Int, grid: DenseMatrix[Double]): Array[DenseVector[Double]] = {
    val top = new ArrayBuffer[DenseVector[Double]]()
    for (n <- 0 until topN) top += next(grid)
    return top.toArray
  }

  override def fit(hypers: DenseMatrix[Double], loss: DenseVector[Double]): SmboModel = this
}
