package org.apache.spark.mllib.tree

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl.DecisionTreeMetadata
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.model.{Bin, Split}
import org.apache.spark.rdd.RDD

/**
 * Created by luke on 17/11/14.
 */
object LukeUtil {
  def getSplitsAndBins(featuress:RDD[org.apache.spark.mllib.linalg.Vector], maxBins:Int):
  (Array[Array[Split]], Array[Array[Bin]]) = {
    println("getSplitsAndBins")
    println("partitions: " + featuress.partitions.length + ", features: " + featuress.count())
    val strategy = new Strategy(Classification, Gini, 0, 0, maxBins, Sort, Map[Int, Int]())
    val metadata = DecisionTreeMetadata.buildMetadataFromFeatures(featuress, strategy, 50, "sqrt")
    DecisionTree.findSplitsBins(featuress, metadata)
  }
}
