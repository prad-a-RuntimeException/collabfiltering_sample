package com.collabsample

import java.net.URISyntaxException

import com.google.common.collect.Lists
import com.google.common.io.Resources
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.slf4j.LoggerFactory.getLogger


object Main {
  private val LOGGER = getLogger(Main.getClass)
  private val sparkSession = SparkSession.builder
    .appName("CollabFilteringSample").master("local")
    .config("spark.driver.memory", "1g")
    .config("spark.executor.memory", "2g")
    .config("SPARK_CONF_DIR", "./infrastructure/spark-config")
    .getOrCreate

  import sparkSession.implicits._

  def main(args: Array[String]): Unit = {
    loadDataset()
  }

  def loadDataset(): Unit = {
    LOGGER.info("Entering load")
    val context = sparkSession.sparkContext

    val sqlContext = new SQLContext(context)
    try {
      val userReviewFilePath = Resources.getResource("users_ratings.csv").toURI.getPath
      LOGGER.info("File path {} ", userReviewFilePath)
      val ratingDF = sqlContext.read.format("com.databricks.spark.csv")
        .option("header", "true") // Use first line of all files as header
        .option("inferSchema", "true")
        .load(userReviewFilePath)
        .map(r => {
          val userStr = r.getString(0)
          val userId = userStr.substring(1).toInt
          Rating(userId, r.getInt(1), r.getInt(2))
        })

      val dataSets: Array[Dataset[Rating]] = ratingDF.randomSplit(Array(0.6, 0.2, 0.2))

      assert(dataSets.length == 3)
      train(dataSets(0).rdd, dataSets(1).rdd)
    } catch {
      case e: URISyntaxException =>
        throw new RuntimeException("Cannot find file ", e)
    }
  }

  def train(training: RDD[Rating], validation: RDD[Rating]): Unit = {
    val ranks = Lists.newArrayList(8, 12)
    val lambdas = Lists.newArrayList(0.1, 10.0)
    val numIters = Lists.newArrayList(10, 20)
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    var bestModel: Option[MatrixFactorizationModel] = None
    Range.apply(0, ranks.size()).foreach((i: Int) => {
      def foo(i: Int) = {
        val rank = ranks.get(i)
        val lambda = lambdas.get(0)
        val numIter = numIters.get(0)
        val model = ALS.train(training, rank, numIter, lambda)
        val validationRmse = computeRmse(model, validation)
        LOGGER.info("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
          + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
        if (validationRmse < bestValidationRmse) {
          bestModel = Some(model)
          bestValidationRmse = validationRmse
          bestRank = rank
          bestLambda = lambda
          bestNumIter = numIter
        }
      }

      foo(i)
    })
  }

  private def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]) = {

    val predictions: DataFrame = model.predict(data.map(x => (x.user, x.product)))
      .toDF("user", "product", "rating")
    val validationData: DataFrame = data.toDF("user", "product", "rating")
    val predictionsAndRatings: DataFrame = predictions
      .join(validationData, Seq("user", "product"))

    val ratingValue: DataFrame = predictionsAndRatings.select(predictions.col("rating"), validationData.col("rating"))

    math.sqrt(ratingValue.map(x => (x.getDouble(0) - x.getDouble(1)) * (x.getDouble(0) - x.getDouble(1))).reduce(_ + _) /
      data.count())
  }
}



