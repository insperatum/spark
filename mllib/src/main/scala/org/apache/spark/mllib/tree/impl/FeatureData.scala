package org.apache.spark.mllib.tree.impl

import java.io.RandomAccessFile
import java.nio.{FloatBuffer, ByteBuffer}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.Bin

trait FeatureData {
  def getValue(i:Int, f:Int):Double
  def nFeatures:Int
  def nExamples:Int
}

class RawFeatureData(file:String, val nFeatures:Int) extends FeatureData with Serializable {
  val arr = loadFeatures(file)//todo: NO!
  val nExamples = arr.length / nFeatures
  def getValue(i:Int, f:Int) = arr(i*nFeatures + f)

  def loadFeatures(path:String):Array[Float] = {
    println("loading raw feature data: " + path)

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

    println("\tComplete")
    outFloatBuffer.array()
  }

  def toVectors = {
    println("to vectors...")
    val iter = (0 until nExamples).toIterator.map { i =>
      Vectors.dense(arr.view(i, i + nFeatures).map(_.toDouble).toArray) //todo: NO NO!
    }
    iter
  }
}

class BinnedFeatureData(featureData:RawFeatureData, bins:Array[Array[Bin]]) extends Serializable {
  val arr = featureData.arr
  val binnedFeatures = Array.ofDim[Int](arr.length)
  def nFeatures = featureData.nFeatures
  def nExamples = featureData.nExamples

  var i = 0
  while(i < nExamples) {
    var f = 0
    while(f < nFeatures) {
      val idx = i * nFeatures + f
      binnedFeatures(idx) = TreePoint.findBin(arr(idx), 0, false, bins(f))
      f += 1
    }
    i += 1
  }

  def getValue(i:Int, f:Int) = featureData.getValue(i, f)
  def getBin(i:Int, f:Int) = binnedFeatures(i * nFeatures + f)
}