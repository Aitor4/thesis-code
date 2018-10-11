package org.apache.spark.ml.tuning

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class RandomSurrogate() extends SmboModel {

  override def next(grid: DenseMatrix[Double]): DenseVector[Double] = {
    val hypers = new Array[Double](grid.rows)
    for (h <- 0 until grid.rows){
      hypers(h)=Random.nextDouble()
    }
    return DenseVector(hypers)
  }

  override def topNext(topN: Int, grid: DenseMatrix[Double]): Array[DenseVector[Double]] = {
    val top = new ArrayBuffer[DenseVector[Double]]()
    for (n <- 0 until topN){
      top += next(grid)
    }
    return top.toArray
  }

  override def fit(hypers: DenseMatrix[Double], loss: DenseVector[Double]): SmboModel = this
}