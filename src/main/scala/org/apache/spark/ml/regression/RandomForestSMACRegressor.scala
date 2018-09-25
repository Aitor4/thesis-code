/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.{RandomForest, RandomForestSMAC}
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._


/**
  * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a>
  * learning algorithm for regression.
  * It supports both continuous and categorical features.
  */
@Since("1.4.0")
class RandomForestSMACRegressor @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends Predictor[Vector, RandomForestSMACRegressor, RandomForestSMACRegressionModel]
    with RandomForestRegressorParams with DefaultParamsWritable {

  //Modified: To define categorical features manually for smac
  var categoricalFeatures : Map[Int, Int] = null
  def setCategoricalFeatures(featMap: Map[Int,Int]): this.type = {
    this.categoricalFeatures = featMap
    this
  }

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("rfr"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeRegressorParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
    * Specifies how often to checkpoint the cached node IDs.
    * E.g. 10 means that the cache will get checkpointed every 10 iterations.
    * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
    * [[org.apache.spark.SparkContext]].
    * Must be at least 1.
    * (default = 10)
    * @group setParam
    */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = set(impurity, value)

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  @Since("1.4.0")
  override def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  override protected def train(dataset: Dataset[_]): RandomForestSMACRegressionModel = {
    //Modified: To handle categorircal features introduced manually
    val categoricalFeatures : Map[Int, Int] = if(this.categoricalFeatures==null)
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    else this.categoricalFeatures

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)

    val instr = Instrumentation.create(this, oldDataset)
    instr.logParams(labelCol, featuresCol, predictionCol, impurity, numTrees,
      featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, cacheNodeIds, checkpointInterval)

    val trees = RandomForestSMAC
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeRegressionModel])

    val numFeatures = oldDataset.first().features.size
    val m = new RandomForestSMACRegressionModel(uid, trees, numFeatures)
    instr.logSuccess(m)
    m
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): RandomForestSMACRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object RandomForestSMACRegressor extends DefaultParamsReadable[RandomForestSMACRegressor]{
  /** Accessor for supported impurity settings: variance */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeRegressorParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): RandomForestSMACRegressor = super.load(path)

}

/**
  * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> model for regression.
  * It supports both continuous and categorical features.
  *
  * @param _trees  Decision trees in the ensemble.
  * @param numFeatures  Number of features used by this model
  */
@Since("1.4.0")
class RandomForestSMACRegressionModel private[ml] (
                                                override val uid: String,
                                                private val _trees: Array[DecisionTreeRegressionModel],
                                                override val numFeatures: Int)
  extends PredictionModel[Vector, RandomForestSMACRegressionModel]
    with RandomForestRegressorParams with TreeEnsembleModel[DecisionTreeRegressionModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "RandomForestRegressionModel requires at least 1 tree.")

  /**
    * Construct a random forest regression model, with all trees weighted equally.
    *
    * @param trees  Component trees
    */
  private[ml] def this(trees: Array[DecisionTreeRegressionModel], numFeatures: Int) =
    this(Identifiable.randomUID("rfr"), trees, numFeatures)

  @Since("1.4.0")
  override def trees: Array[DecisionTreeRegressionModel] = _trees

  // Note: We may add support for weights (based on tree performance) later on.
  private lazy val _treeWeights: Array[Double] = Array.fill[Double](_trees.length)(1.0)

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  //Modified: To store the variance and mean in the first UDF calculation and avoid repeating it in the second one
  var variance = 0.0
  var mean = 0.0
  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)

    //Modified: Calculate mean and variance, return mean
    val predictUDF = udf { (features: Any) => {
      val result = bcastModel.value.predictWithVariance(features.asInstanceOf[Vector])
      mean = result._1
      variance = result._2
      mean
    }
    }
    //Modified: Return variance previously calculated variance (of the same row feature vector)
    val predictVarUDF = udf { (features: Any) =>
      variance
    }

    //MOdified: Column with variance
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
      .withColumn("empiricalVariance", predictVarUDF(col($(featuresCol))))
  }

  override protected def predict(features: Vector): Double = {
    // TODO: When we add a generic Bagging class, handle transform there.  SPARK-7128
    // Predict average of tree predictions.
    // Ignore the weights since all are 1.0 for now.
    _trees.map(_.rootNode.predictImpl(features).prediction).sum / getNumTrees
  }

  //Modified: To calculate variance as well
  def predictWithVariance(features: Vector): (Double, Double) = {
    // Predict average of tree predictions.
    // Ignore the weights since all are 1.0 for now.
    val predictions = _trees.map(_.rootNode.predictImpl(features).prediction)
    val mean=predictions.sum / getNumTrees
    val variance=predictions.map(x=>(x-mean)*(x-mean)).sum/getNumTrees

    return (mean, variance)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): RandomForestSMACRegressionModel = {
    copyValues(new RandomForestSMACRegressionModel(uid, _trees, numFeatures), extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"RandomForestRegressionModel (uid=$uid) with $getNumTrees trees"
  }

  /**
    * Estimate of the importance of each feature.
    *
    * Each feature's importance is the average of its importance across all trees in the ensemble
    * The importance vector is normalized to sum to 1. This method is suggested by Hastie et al.
    * (Hastie, Tibshirani, Friedman. "The Elements of Statistical Learning, 2nd Edition." 2001.)
    * and follows the implementation from scikit-learn.
    *
    * @see `DecisionTreeRegressionModel.featureImportances`
    */
  @Since("1.5.0")
  lazy val featureImportances: Vector = TreeEnsembleModel.featureImportances(trees, numFeatures)

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldRandomForestModel = {
    new OldRandomForestModel(OldAlgo.Regression, _trees.map(_.toOld))
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new RandomForestSMACRegressionModel.RandomForestSMACRegressionModelWriter(this)
}

@Since("2.0.0")
object RandomForestSMACRegressionModel extends MLReadable[RandomForestSMACRegressionModel] {

  @Since("2.0.0")
  override def read: MLReader[RandomForestSMACRegressionModel] = new RandomForestSMACRegressionModelReader

  @Since("2.0.0")
  override def load(path: String): RandomForestSMACRegressionModel = super.load(path)

  private[RandomForestSMACRegressionModel]
  class RandomForestSMACRegressionModelWriter(instance: RandomForestSMACRegressionModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numTrees" -> instance.getNumTrees)
      EnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
    }
  }

  private class RandomForestSMACRegressionModelReader extends MLReader[RandomForestSMACRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[RandomForestSMACRegressionModel].getName
    private val treeClassName = classOf[DecisionTreeRegressionModel].getName

    override def load(path: String): RandomForestSMACRegressionModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, Node)], treeWeights: Array[Double]) =
        EnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[DecisionTreeRegressionModel] = treesData.map { case (treeMetadata, root) =>
        val tree =
          new DecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
        DefaultParamsReader.getAndSetParams(tree, treeMetadata)
        tree
      }
      require(numTrees == trees.length, s"RandomForestRegressionModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new RandomForestSMACRegressionModel(metadata.uid, trees, numFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldRandomForestModel,
                           parent: RandomForestSMACRegressor,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1): RandomForestSMACRegressionModel = {
    require(oldModel.algo == OldAlgo.Regression, "Cannot convert RandomForestModel" +
      s" with algo=${oldModel.algo} (old API) to RandomForestRegressionModel (new API).")
    val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
      DecisionTreeRegressionModel.fromOld(tree, null, categoricalFeatures)
    }
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("rfr")
    new RandomForestSMACRegressionModel(uid, newTrees, numFeatures)
  }
}
