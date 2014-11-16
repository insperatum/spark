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
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

object Vanilla extends App{

  println("Starting Spark!")
  val conf = new SparkConf()
              .setAppName("Hello")
              .setMaster("local")
              .set("spark.executor.memory", "4g")
  val sc = new SparkContext(conf)

  val numFiles = 1
  val root = "/home/luke/neuron-forests/spark/data"
  val n = 1000
  println("Loading data...")

  val data = sc.parallelize(1 to numFiles, numFiles).mapPartitionsWithIndex( (i, _) => {
    val features_file = root + "/im" + (i+1) + "/features.raw"
    val features = loadFeatures(features_file, 30)

    val targets_file = root + "/im" + (i+1) + "/targets.txt"
    val targets = Source.fromFile(targets_file).getLines().take(n).map(_.split(" ").map(_.toDouble))

    val dimensions_file = root + "/im" + (i+1) + "/dimensions.txt"
    val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

    val size = dimensions(0)
    val step_size = (size(1) * size(0), size(1), 1)
    val min_idx = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
    val max_idx = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))
    val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)


    new Iterator[LabeledPoint] {
      var x = min_idx._1
      var y = min_idx._2
      var z = min_idx._3 - 1
      var i = min_idx._1 * step_size._1 + min_idx._2 * step_size._2 + min_idx._3 * step_size._3 - 1
      var j = -1

      override def hasNext: Boolean = targets.hasNext
      override def next(): LabeledPoint = {
        j += 1

        z += 1; i += 1

        if(z > max_idx._3) {
          z = min_idx._3; i -= (max_idx._3 - min_idx._3 + 1)
          y += 1; i += step_size._2
          if(y > max_idx._2) {
            y = min_idx._2; i -= (max_idx._2 - min_idx._2 + 1)
            x += 1 ; i += step_size._1
            if(x > max_idx._1) {
              throw new Exception("x = " + x + ", y = " + y + ", z = " + z + ", j = " + j)
            }
          }
        }
        LabeledPoint(targets.next()(0), Vectors.dense(features.slice(j*30, j*30+31).map(_.toDouble)))
      }
    }
    /*targets.zipWithIndex.map{ case (ts, i) =>
      val t = ts(0)
      /*val feature_idx = step_size._1 * (i / seg_size._1) +
        step_size._2 * ((i % seg_size._1) / seg_size._2) +
        step_size._3 * (i % seg_size._2)

      LabeledPoint(t, Vectors.dense(features(feature_idx)))*/
      LabeledPoint(t, Vectors.dense(features(0)))
    }*/
  }).cache()


  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "entropy"
  val maxDepth = 14
  val maxBins = 100

  val model = RandomForest.trainClassifier(data, 2, Map[Int, Int](), 50, "all", "entropy", 14, 100)

  // Evaluate model on training instances and compute training error
  val labelsAndPredictions = data.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }

  val trainMSE = labelsAndPredictions.map{ case(v, p) => math.pow(v - p, 2)}.mean()
  println("Training Mean Squared Error = " + trainMSE)
  println("Learned regression tree model:\n" + model)








  def loadFeatures(path:String, n_features:Int):Array[Float] = {
    val file = new RandomAccessFile(path, "r")
    val fileChannel = file.getChannel

    val byteBuffer = ByteBuffer.allocate(4 * 10000) //must be multiple of 4 for floats
    val outFloatBuffer = FloatBuffer.allocate(fileChannel.size.toInt/4)

    var bytesRead = fileChannel.read(byteBuffer)
    while(bytesRead > 0) {
      byteBuffer.flip()
      outFloatBuffer.put(byteBuffer.asFloatBuffer())
      byteBuffer.clear()
      bytesRead = fileChannel.read(byteBuffer)
    }

    outFloatBuffer.array()
  }

}