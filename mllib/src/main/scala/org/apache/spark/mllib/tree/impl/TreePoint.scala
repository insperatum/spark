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

package org.apache.spark.mllib.tree.impl

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.Bin
import org.apache.spark.rdd.RDD


/**
 * Internal representation of LabeledPoint for DecisionTree.
 * This bins feature values based on a subsampled of data as follows:
 *  (a) Continuous features are binned into ranges.
 *  (b) Unordered categorical features are binned based on subsets of feature values.
 *      "Unordered categorical features" are categorical features with low arity used in
 *      multiclass classification.
 *  (c) Ordered categorical features are binned based on feature values.
 *      "Ordered categorical features" are categorical features with high arity,
 *      or any categorical feature used in regression or binary classification.
 *
 * @param label  Label from LabeledPoint
 * @param binnedFeatures  Binned feature values.
 *                        Same length as LabeledPoint.features, but values are bin indices.
 */
class TreePoint(val label: Double, val arr:Array[Int], val data:BinnedFeatureData = null, val idx:Int = 0)
  extends Serializable {
  def binnedFeatures(f:Int) = {
    if(arr == null) data.getBin(idx, f) else arr(f)
  }
  def features(f:Int) = data.getValue(idx, f)
}


private[tree] object TreePoint {

  /**
   * Convert an input dataset into its TreePoint representation,
   * binning feature values in preparation for DecisionTree training.
   * @param input     Input dataset.
   * @param bins      Bins for features, of size (numFeatures, numBins).
   * @param metadata  Learning and dataset metadata
   * @return  TreePoint dataset representation
   */
  def convertToTreeRDD(
      input: RDD[LabeledPoint],
      bins: Array[Array[Bin]],
      metadata: DecisionTreeMetadata): RDD[TreePoint] = {
    // Construct arrays for featureArity and isUnordered for efficiency in the inner loop.
    val featureArity: Array[Int] = new Array[Int](metadata.numFeatures)
    val isUnordered: Array[Boolean] = new Array[Boolean](metadata.numFeatures)
    var featureIndex = 0
    while (featureIndex < metadata.numFeatures) {
      featureArity(featureIndex) = metadata.featureArity.getOrElse(featureIndex, 0)
      isUnordered(featureIndex) = metadata.isUnordered(featureIndex)
      featureIndex += 1
    }
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, bins, featureArity, isUnordered)
    }
  }

  /**
   * Convert one LabeledPoint into its TreePoint representation.
   * @param bins      Bins for features, of size (numFeatures, numBins).
   * @param featureArity  Array indexed by feature, with value 0 for continuous and numCategories
   *                      for categorical features.
   * @param isUnordered  Array index by feature, with value true for unordered categorical features.
   */
  private def labeledPointToTreePoint(
      labeledPoint: LabeledPoint,
      bins: Array[Array[Bin]],
      featureArity: Array[Int],
      isUnordered: Array[Boolean]): TreePoint = {
    val numFeatures = labeledPoint.features.size
    val arr = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      arr(featureIndex) = findBin(featureIndex, labeledPoint, featureArity(featureIndex),
        isUnordered(featureIndex), bins)
      featureIndex += 1
    }
    new TreePoint(labeledPoint.label, arr)
  }

  /**
   * Find bin for one (labeledPoint, feature).
   *
   * @param featureArity  0 for continuous features; number of categories for categorical features.
   * @param isUnorderedFeature  (only applies if feature is categorical)
   * @param bins   Bins for features, of size (numFeatures, numBins).
   */
  private[tree] def findBin(
      featureIndex: Int,
      input: LabeledPoint,
      featureArity: Int,
      isUnorderedFeature: Boolean,
      bins: Array[Array[Bin]]): Int = {
    val binForFeatures = bins(featureIndex)
    val feature = input.features(featureIndex)
    findBin(feature, featureArity, isUnorderedFeature, binForFeatures)
  }

  private[tree] def findBin(
      featureValue: Double,
      featureArity: Int,
      isUnorderedFeature: Boolean,
      binForFeatures: Array[Bin]): Int = {
    /**
     * Binary search helper method for continuous feature.
     */
    def binarySearchForBins(): Int = {
      var left = 0
      var right = binForFeatures.length - 1
      while (left <= right) {
        val mid = left + (right - left) / 2
        val bin = binForFeatures(mid)
        val lowThreshold = bin.lowSplit.threshold
        val highThreshold = bin.highSplit.threshold
        if ((lowThreshold < featureValue) && (highThreshold >= featureValue)) {
          return mid
        } else if (lowThreshold >= featureValue) {
          right = mid - 1
        } else {
          left = mid + 1
        }
      }
      -1
    }

    if (featureArity == 0) {
      // Perform binary search for finding bin for continuous features.
      val binIndex = binarySearchForBins()
      if (binIndex == -1) {
        throw new RuntimeException("No bin was found for continuous feature." +
          " This error can occur when given invalid data values (such as NaN)." +
          " Feature index: $featureIndex.  Feature value: $featureValue")
      }
      binIndex
    } else {
      // Categorical feature bins are indexed by feature values.
      if (featureValue < 0 || featureValue >= featureArity) {
        throw new IllegalArgumentException(
          s"DecisionTree given invalid data:" +
            " Feature $featureIndex is categorical with values in" +
            " {0,...,${featureArity - 1}," +
            " but a data point gives it value $featureValue.\n" +
            "  Bad data point: ${features.toString}")
      }
      featureValue.toInt
    }
  }
}
