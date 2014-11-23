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
import org.apache.spark.mllib.tree.model.{Bin, Split}
import org.apache.spark.mllib.tree.{LukeUtil, RandomForest}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impl._
import org.apache.spark.mllib.tree.impurity.{Impurity, Entropy, Gini}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import org.apache.spark.mllib.linalg.Vector

import com.esotericsoftware.kryo.Kryo
import org.apache.spark.serializer.KryoRegistrator

object New extends App{
  val s = getSettingsFromArgs(args)
  println("Settings:\n" + s)

  val offsets = for(x <- s.dimOffsets; y <- s.dimOffsets; z <- s.dimOffsets) yield (x, y, z)
  val nFeatures = s.nBaseFeatures * offsets.length

  println("Starting Spark!")
  val conf = new SparkConf()
    .setAppName("Hello")
  if(! s.master.isEmpty) conf.setMaster(s.master)
  conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  conf.set("spark.kryo.registrator", "MyRegistrator")
  val sc = new SparkContext(conf)



  // -------------------  Train  --------------------------------

  val (train, splits, bins) = loadData(0.2, fromFront = true)
  //train.persist(StorageLevel.MEMORY_ONLY_SER)
  val strategy = new Strategy(Classification, s.impurity, s.maxDepth, 2, s.maxBins, Sort, Map[Int, Int](), maxMemoryInMB = s.maxMemoryInMB)
  val model = RandomForest.trainClassifierFromTreePoints(
    train,
    strategy,
    s.nTrees,
    nFeatures,
    400000, // <------------------ TODO: NOT THIS <-------------------
    s.featureSubsetStrategy: String,
    1,
    splits,
    bins)

//  val model = RandomForest.trainClassifier(train, 2, Map[Int, Int](), 50, featureSubsetStrategy, "entropy", maxDepth, maxBins)
  //val model = RandomForest.trainRegressor(train, Map[Int, Int](), 50, "sqrt", "variance", 14, 100)

  println("trained.")

  // --------------------  Test  --------------------------------
  val (test, _, _) = loadData(0.8, fromFront = false)
  //val test = train

  val labelsAndPredictions = test.map { point =>
    val features = Array.tabulate[Double](nFeatures)(f => point.features(f))
    val prediction = model.predict(Vectors.dense(features))
    (point.label, prediction)
  }
  val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow(v - p, 2)}.mean()
  println("Test Mean Squared Error = " + testMSE)
  println("Learned regression tree model:\n" + model)

  




  def loadData(p:Double, fromFront:Boolean) = { //todo: use numFiles
    val rawFeaturesData = sc.parallelize(1 to s.subvolumes.size, s.subvolumes.size).mapPartitionsWithIndex((i, _) => {
      val features_file = s.data_root + "/" + s.subvolumes(i) + "/features.raw"
      Seq(new RawFeatureData(features_file, s.nBaseFeatures)).toIterator
    })
    rawFeaturesData.cache()

    val baseFeaturesRDD = rawFeaturesData.mapPartitions(_.next().toVectors)
    println("getting splits and bins")
    val (splits, bins) = LukeUtil.getSplitsAndBins(baseFeaturesRDD, s.maxBins, s.nBaseFeatures, offsets.length)
    println(" found bins!")

    val data = rawFeaturesData.mapPartitionsWithIndex((i, f) => {
      val startTime = System.currentTimeMillis()

      val targets_file = s.data_root + "/" + s.subvolumes(i) + "/targets.txt"
      val n_targets_total = Source.fromFile(targets_file).getLines().size //todo: store this at the top of the file (OR GET FROM DIMENSIONS!)
      val n_targets = (n_targets_total * p).toInt
      val target_index_offset = if (fromFront) 0 else n_targets_total - n_targets

      val allTargets = Source.fromFile(targets_file).getLines().map(_.split(" ").map(_.toDouble))
      val targets = if (fromFront)
        allTargets.take(n_targets)
      else
        allTargets.drop(target_index_offset)

      val dimensions_file = s.data_root + "/" + s.subvolumes(i) + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val size = dimensions(0)
      val step = (size(1) * size(2), size(2), 1)
      val min_idx = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      println("Targets from " + min_idx + " to " + max_idx)


      val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
      val seg_step = (seg_size._2 * seg_size._3, seg_size._3, 1)

      val binnedFeatureData = new BinnedFeatureData(f.next(), bins, seg_step, offsets)
      val d = targets.zipWithIndex.map { case (ts, i) =>
        val t = i + target_index_offset
        val y = ts(0)
        val example_idx =
          step._1 * (min_idx._1 + t / seg_step._1) +
            step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2) +
            (min_idx._3 + t % seg_step._2)
        //LabeledPoint(y, Vectors.dense(binnedFeatureData.arr.slice(example_idx * nFeatures, (example_idx + 1) * nFeatures).map(_.toDouble)))
        new TreePoint(y, null, binnedFeatureData, example_idx)
      }

      println("creating partition data took " + (System.currentTimeMillis() - startTime) + " ms")
      d
    })
    rawFeaturesData.unpersist()
    (data, splits, bins)
  }

  case class RunSettings(maxMemoryInMB:Int, data_root:String, subvolumes:Array[String], featureSubsetStrategy:String,
                         impurity:Impurity, maxDepth:Int, maxBins:Int, nBaseFeatures:Int, nTrees:Int,
                         dimOffsets:Array[Int], master:String)

  def getSettingsFromArgs(args:Array[String]):RunSettings = {
    val m = args.map(_.split("=")).map(arr => arr(0) -> arr(1)).toMap
    val impurityMap = Seq("entropy" -> Entropy, "gini" -> Gini).toMap
    RunSettings(
      maxMemoryInMB = m.getOrElse("maxMemoryInMB", "1000").toInt,
      data_root     = m.getOrElse("data_root",     "/home/luke/spark/neuron-forest/data/im1/split_2"),
      subvolumes    = m.getOrElse("subvolumes",    "000,001,010,011,100,101,110,111").split(",").toArray,
      featureSubsetStrategy = m.getOrElse("featureSubsetStrategy", "sqrt"),
      impurity      = impurityMap(m.getOrElse("impurity", "entropy")),
      maxDepth      = m.getOrElse("maxDepth",      "14").toInt,
      maxBins       = m.getOrElse("maxBins",       "100").toInt,
      nBaseFeatures = m.getOrElse("nBaseFeatures", "30").toInt,
      nTrees        = m.getOrElse("nTrees",        "50").toInt,
      dimOffsets    = m.getOrElse("dimOffsets",    "0").split(",").map(_.toInt).toArray,
      master        = m.getOrElse("master",        "local") // use empty string to not setdata_
    )
  }
}



class MyRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[BinnedFeatureData])
  }
}
