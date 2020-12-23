import breeze.linalg.split
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession, functions}
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.functions.{col, concat_ws, when}
import org.apache.spark.sql.{functions => F}
import org.apache.spark.ml.feature.StopWordsRemover


object Main {

  def filterTweets(tweets: DataFrame, stopWords: Array[String]): DataFrame = {
    // remove elements from an array column in spark : https://stackoverflow.com/questions/56180887/how-to-remove-elements-from-an-array-column-in-spark
    // val filteredTweets = tweets.withColumn("tweets", F.array_except(tweets("tweets"), F.lit(stopWords)))
    // the above method does not do inbuilt lowercase conversion. The method below does it

    val remover = new StopWordsRemover().setStopWords(stopWords).setInputCol("tweets").setOutputCol("filteredTweets")
    val filteredTweets = remover.transform(tweets)
    filteredTweets
  }

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .appName("Simple Application")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    // write your code here

    val csvFilePath = args(0)
    val txtFilePath = args(1)

    val unfilteredTweetsDf = spark.read.csv(csvFilePath)
    val tweetsStringDf = unfilteredTweetsDf.drop("_c0", "_c1", "_c2", "_c3", "_c4")
    // convert string column to array column : https://sparkbyexamples.com/spark/convert-delimiter-separated-string-to-array-column-in-spark/#:~:text=Spark%20SQL%20provides%20split(),e.t.c%2C%20and%20converting%20into%20ArrayType.
    val tweetsDf = tweetsStringDf.select(functions.split(col("_c5"), " ").as("tweets")).drop("_c5")


    val stopWordsDf = spark.read.textFile(txtFilePath)
    val stopWords = stopWordsDf.collect()
    val filteredTweetsDf = filterTweets(tweetsDf, stopWords)

    // filteredTweetsDf.show()
    val filteredTweetsOnlyDf = filteredTweetsDf.drop("tweets")
    // filteredTweetsOnlyDf.show()
    // filteredTweetsOnlyDf.printSchema()
//    val theDf = filteredTweetsOnlyDf.map(f => {
//      val tweet = f.getList(0).toArray().mkString(" ")
//      tweet
//    })
//    theDf.printSchema()
//    theDf.show()

    // Create word2vector
    /**
     * Parameters for Word2Vec
     * setVectorSize (default 100)
     * setNumIterations (default 1)
     * setMinCount min no of times a token must appear to be included in word2vec model's vocab (default 5)
     * setMaxSentenceLength
     */
    val vectorSize = 10
    val word2Vec = new Word2Vec().setInputCol("filteredTweets").setOutputCol("features").setVectorSize(vectorSize)
    val model = word2Vec.fit(filteredTweetsOnlyDf)
    val vectorsDf = model.transform(filteredTweetsOnlyDf)
    // vectorsDf.show()

    // kmean clustering

    val numClusters = 5
    val numIterations = 20
    val kmeans = new KMeans().setK(numClusters).setMaxIter(numIterations)
    val kmeansModel = kmeans.fit(vectorsDf)
    val predictions = kmeansModel.transform(vectorsDf)

    predictions.show()

    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println("Size of each vector: " + vectorSize)
    println("Number of Clusters: " + numClusters)
    println("Number of Iterations: " + numIterations)
    println(s"Silhouette with squared euclidean distance = $silhouette")
    kmeansModel.clusterCenters.foreach(println)

    spark.stop()
  }
























}