import java.io.{RandomAccessFile, BufferedInputStream, File}
import java.nio.channels.{FileChannel, ByteChannel}
import java.nio.file.Path
import java.nio.{ByteBuffer, FloatBuffer, DoubleBuffer}
import java.util.RandomAccess

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors

import org.apache.spark.ml
import org.apache.spark.mllib.tree.model.Split
import org.apache.spark.mllib.tree.{LukeUtil, RandomForest}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl.{BinnedFeatureData, RawFeatureData, DecisionTreeMetadata}
import org.apache.spark.mllib.tree.impurity.Gini

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.model.Bin

object New extends App{
  val data_root = "/home/luke/spark/neuron-forest/data"
  val featureSubsetStrategy = "sqrt"
  val impurity = "entropy"
  val maxDepth = 4
  val maxBins = 100
  val nFeatures = 30
  val nTrees = 5


  println("Starting Spark!")
  val conf = new SparkConf()
    .setAppName("Hello")
    .setMaster("local")
    .set("spark.executor.memory", "4g")
  val sc = new SparkContext(conf)



  // -------------------  Train  --------------------------------

  val train = loadTrainingData(0.2, fromFront = true)
  println("Training on " + train.count() + " examples")
  val model = RandomForest.trainClassifier(train, 2, Map[Int, Int](), 50, featureSubsetStrategy, impurity, maxDepth, maxBins)
  //val model = RandomForest.trainRegressor(train, Map[Int, Int](), 50, "sqrt", "variance", 14, 100)

  println("trained.")

  // --------------------  Test  --------------------------------
//  val
//  val test = loadTrainingData(0.8, fromFront = false)
////  test.take(50).foreach(println)
//
//  val labelsAndPredictions = test.map { point =>
//    val prediction = model.predict(point.features)
//    (point.label, prediction)
//  }
//  val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow(v - p, 2)}.mean()
//  println("Test Mean Squared Error = " + testMSE)
//  println("Learned regression tree model:\n" + model)

  




  def loadTrainingData(p:Double, fromFront:Boolean, numFiles:Int = 1) = {
    val rawFeaturesData = sc.parallelize(1 to numFiles, numFiles).mapPartitionsWithIndex((i, _) => {
      val features_file = data_root + "/im" + (i + 1) + "/features.raw"
      Seq(new RawFeatureData(features_file, nFeatures)).toIterator
    }).cache()
    val featsRDD = rawFeaturesData.mapPartitions(_.next().toVectors).cache()
    println("getting splits and bins!")
    val (splits, bins) = LukeUtil.getSplitsAndBins(featsRDD, maxBins)
    featsRDD.unpersist()
    println("done!")
    bins.head.foreach(println)

    val trainingData = rawFeaturesData.mapPartitionsWithIndex((i, s) => {
      val binnedFeatureData = new BinnedFeatureData(s.next(), bins)

      val targets_file = data_root + "/im" + (i + 1) + "/targets.txt"
      val n_targets_total = Source.fromFile(targets_file).getLines().size //todo: store this at the top of the file
      val n_targets = (n_targets_total * p).toInt
      val target_index_offset = if (fromFront) 0 else n_targets_total - n_targets

      val allTargets = Source.fromFile(targets_file).getLines().map(_.split(" ").map(_.toDouble))
      val targets = if (fromFront)
        allTargets.take(n_targets)
      else
        allTargets.drop(target_index_offset)

      val dimensions_file = data_root + "/im" + (i + 1) + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val size = dimensions(0)
      val step = (size(1) * size(2), size(2), 1)
      val min_idx = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      println("Targets from " + min_idx + " to " + max_idx)

      val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
      val seg_step = (seg_size._2 * seg_size._3, seg_size._3, 1)
      targets.zipWithIndex.map { case (ts, i) =>
        val t = i + target_index_offset
        val y = ts(0)
        val example_idx =
          step._1 * (min_idx._1 + t / seg_step._1) +
            step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2) +
            (min_idx._3 + t % seg_step._2)
        LabeledPoint(y, Vectors.dense(binnedFeatureData.arr.slice(example_idx * nFeatures, (example_idx + 1) * nFeatures).map(_.toDouble)))
      }
    }).cache()

    trainingData
  }

}