package com.sparrowrecsys.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {
  /**
   * One-hot encoding example function
   *
   * @param samples movie samples dataframe
   */
  def oneHotEncoderExample(samples: DataFrame): Unit = {
    // samples样本集中的每一条数据代表一部电影的信息，其中movieId为电影id
    // 增加一列movieIdNumber
    val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))

    // 利用spark的机器学习库spark MLlib创建One-hot编码器
    // OneHotEncoderEstimator在PySpark3.0.0及以上版本更改为OneHotEncoder()
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      // 是否删除编码向量的最后一个类别（默认为true）
      .setDropLast(false)

    // 训练one-hot编码器，并完成从id特征到one-hot向量的转变
    val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    // 打印最终样本的数据结构
    oneHotEncoderSamples.printSchema()
    // 打印10条样本查看结果
    oneHotEncoderSamples.show(10)
  }

  // 生成vector的udf array2vec
  /**
   * def array2vec(genreIndexes, indexSize):
   * genreIndexes.sort()
   * fill_list = [1.0 for _ in range(len(genreIndexes))]
   * // 稀疏向量存储indexSize，有值的Indexes，对应Indexes上的填充值
   * return Vectors.sparse(indexSize, genreIndexes, fill_list)
   */
  val array2vec: UserDefinedFunction = udf { (a: Seq[Int], length: Int) => org.apache.spark.ml.linalg.Vectors.sparse(length, a.sortWith(_ < _).toArray, Array.fill[Double](a.length)(1.0)) }

  /**
   * Multi-hot encoding example function
   * @param samples movie samples dataframe
   */
  def multiHotEncoderExample(samples: DataFrame): Unit = {
    // 将genres切割为genre，多行
    val samplesWithGenre = samples.select(col("movieId"), col("title"), explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
    // 转换为index；StringIndexer()将一组字符串标签编码成一组标签索引，索引范围从0到标签数量；
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")

    val stringIndexerModel: StringIndexerModel = genreIndexer.fit(samplesWithGenre)

    // 转换数据类型为Int
    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt", col("genreIndex").cast(sql.types.IntegerType))

    // 计算编码向量大小
    val indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0) + 1

    // 根据movieId聚合genreIndexInt，增加indexSize列
    val processedSamples = genreIndexSamples
      .groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize))

    // array2Vec生成稀疏向量
    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))

    finalSample.printSchema()
    finalSample.show(10)
  }

  // create a dense vector from a double array
  val double2vec: UserDefinedFunction = udf { (value: Double) => org.apache.spark.ml.linalg.Vectors.dense(value) }

  /**
   * Process rating samples
   *
   * @param samples rating samples
   */
  def ratingFeatures(samples: DataFrame): Unit = {
    samples.printSchema()
    samples.show(10)

    // 计算打分表ratings计算电影的平均分、被打分次数等数值特征
    val movieFeatures = samples.groupBy(col("movieId"))
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar")) // 方差
      .withColumn("avgRatingVec", double2vec(col("avgRating")))

    movieFeatures.show(10)

    //bucketing, 分桶处理，将打分次数这一特征分到100个桶中
    val ratingCountDiscretizer = new QuantileDiscretizer()
      .setInputCol("ratingCount")
      .setOutputCol("ratingCountBucket")
      .setNumBuckets(100)

    //Normalization，归一化处理，将平均得分进行归一化
    val ratingScaler = new MinMaxScaler()
      .setInputCol("avgRatingVec")
      .setOutputCol("scaleAvgRating")

    // 创建一个pipeline，依次进行两个特征处理过程
    val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
    val featurePipeline = new Pipeline().setStages(pipelineStage)

    val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
    movieProcessedFeatures.show(10)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("Raw Movie Samples:")
    movieSamples.printSchema()
    movieSamples.show(10)

    println("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)

    println("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)

    println("Numerical features Example:")
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingFeatures(ratingSamples)

  }
}