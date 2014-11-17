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

object Main extends App{

  println("Starting Spark!")
  val conf = new SparkConf()
    .setAppName("Hello")
    .setMaster("local")
    .set("spark.executor.memory", "4g")
  val sc = new SparkContext(conf)

  val train = loadData(0.2, fromFront = true)
  println("Training on " + train.count() + " examples")

  val categoricalFeaturesInfo = Map[Int, Int]()
  val impurity = "entropy"
  val maxDepth = 14
  val maxBins = 100


  val model = RandomForest.trainClassifier(train, 2, Map[Int, Int](), 50, "sqrt", "entropy", 14, 100)
  //val model = RandomForest.trainRegressor(train, Map[Int, Int](), 50, "sqrt", "variance", 14, 100)
  val test = loadData(0.8, fromFront = false)
//  test.take(50).foreach(println)

  val labelsAndPredictions = test.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow(v - p, 2)}.mean()
  println("Test Mean Squared Error = " + testMSE)
  println("Learned regression tree model:\n" + model)






  def loadData(p:Double, fromFront:Boolean, numFiles:Int = 1, nFeatures:Int = 30) =
    sc.parallelize(1 to numFiles, numFiles).mapPartitionsWithIndex( (i, _) => {
      val root = "/home/luke/spark/neuron-forest/data"

      val features_file = root + "/im" + (i+1) + "/features.raw"
      val features = loadFeatures(features_file, nFeatures)

      val targets_file = root + "/im" + (i+1) + "/targets.txt"
      val n_targets_total = Source.fromFile(targets_file).getLines().size //todo: store this at the top of the file
      val n_targets = (n_targets_total * p).toInt
      val target_index_offset = if(fromFront) 0 else n_targets_total - n_targets

      val allTargets = Source.fromFile(targets_file).getLines().map(_.split(" ").map(_.toDouble))
      val targets = if (fromFront)
        allTargets.take(n_targets)
      else
        allTargets.drop(target_index_offset)

      val dimensions_file = root + "/im" + (i+1) + "/dimensions.txt"
      val dimensions = Source.fromFile(dimensions_file).getLines().map(_.split(" ").map(_.toInt)).toArray

      val size = dimensions(0)
      val step = (size(1) * size(2), size(2), 1)
      val min_idx = (dimensions(1)(0), dimensions(1)(1), dimensions(1)(2))
      val max_idx = (dimensions(2)(0), dimensions(2)(1), dimensions(2)(2))

      println("Targets from " + min_idx + " to " + max_idx )

      val seg_size = (max_idx._1 - min_idx._1 + 1, max_idx._2 - min_idx._2 + 1, max_idx._3 - min_idx._3 + 1)
      val seg_step = (seg_size._2 * seg_size._3, seg_size._3, 1)
      /*println("size:" + size)
      println("step_size:" + step)
      println("min_idx:" + min_idx)
      println("max_idx:" + max_idx)
      println("seg_size:" + seg_size)*/

      /*new Iterator[LabeledPoint] {
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
      }*/
      targets.zipWithIndex.map{ case (ts, i) =>
        val t = i + target_index_offset
        val y = ts(0)
        val example_idx =
          step._1 * (min_idx._1 + t / seg_step._1) +
          step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2) +
                    (min_idx._3 + t % seg_step._2)
//        if(t < 129) {
//          println(t + " => " + example_idx)
//          println(step._1 * (min_idx._1 + t / seg_step._1))
//          println(step._2 * (min_idx._2 + (t % seg_step._1) / seg_step._2))
//          println((min_idx._3 + t % seg_step._2))
//        }
        /*if(22 <= i && i <= 25) {
          println("i = " + i)
          println("t = " + t)
          println("y = " + y)
          println("features_idx = " + example_idx)
          println("features = " + Vectors.dense(features.slice(example_idx*nFeatures, (example_idx+1)*nFeatures).map(_.toDouble)))
          println("x y z = ")
          println((min_idx._1 + t / seg_step._1))
          println((min_idx._2 + (t % seg_step._1) / seg_step._2))
          println((min_idx._3 + t % seg_step._2))
          println(" ")
        }*/
        LabeledPoint(y, Vectors.dense(features.slice(example_idx*nFeatures, (example_idx+1)*nFeatures).map(_.toDouble)))
      }
    }).cache()


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