from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler


spark = SparkSession.builder.appName('randomForest').getOrCreate()

df_train = spark.read.options(header="True", inferSchema="True").csv("file:///D:/train-feature.csv")
# df_test =sc.read.options(header="True", inferSchema="True").csv("test-feature.csv")

data = VectorAssembler(inputCols=['age_range', 'gender', 'categories', "one_clicks", "shopping_carts", "purchase_times", "favourite_times"], outputCol="features").transform(df_train)

dataNum = data.select("label").count()
#print("nums:".format(dataNum))

#assign higher weights to the positives(cancelled == 1).
balancingRatio = float(data.select("label").where("label==0").count() / data. count())
calculateWeights = udf(lambda x: balancingRatio if x == 1 else (1.0-balancingRatio), FloatType())
dataSet = data.withColumn("classWeightCol", calculateWeights('label'))

train, test = dataSet.randomSplit([0.75, 0.25], seed=10)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", weightCol="classWeightCol", numTrees=100)

pipline = Pipeline(stages=[dataSet, rf])

model = pipline.fit(train)

predictions = model.transform(test)
rf_prob = udf(lambda x: float(x[1]), FloatType())
predictions = predictions.withColumn("prob", rf_prob("probability"))

result = predictions.select("user_id", "merchant_id", "prob")
result.toPandas().to_csv("file:///D:/result.csv", index=False)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
ACC = evaluator.evaluate(predictions)

print("acc = %g" % ACC)
rfModel = model.stages[2]
print("rfModelSummary:", rfModel)





