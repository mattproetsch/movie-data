from pyspark import SparkContext, SparkConf
from pyspark.sql.types import FloatType
from pyspark.sql import HiveContext
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.linalg import DenseVector


def main(sc, sqlContext):
    train = sqlContext.sql("""SELECT * 
                           FROM hospitalhcahpsscores hs
                           INNER JOIN hospital h
                           ON hs.providerid = h.providerid""")
    train = train.withColumn('scoreFloat', train['rating'].cast('double'))
    train_df = train.rdd.map(lambda x: [DenseVector(x[1:12]), x[-1]]).toDF(['features', 'label'])
    lr = LinearRegression(featuresCol='features', labelCol='label')
    lr_model = lr.fit(train_df)
    score_pred = lr_model.transform(train_df)
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='label')
    train_df.limit(10).show()
    print 'r2 = ' + str(evaluator.evaluate(score_pred))
    
    score_pred.limit(10).show()

    #feats = train.select('communicationwithnurses',
    #                     'communicationwithdoctors',
    #                     'responsivenessofhospitalstaff',
    #                     'painmanagement',
    #                     'communicationaboutmedicines',
    #                     'cleanlinessandquietness',
    #                     'dischargeinformation',
    #                     'overallratingofhospital',
    #                     'basescore',
    #                     'consistencyscore')
    #targets = train.select('rating')
    #feats.limit(10).show()
    #targets.limit(10).show()

if __name__=='__main__':
    conf = SparkConf().setAppName('test')
    conf.setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sqlContext = HiveContext(sc)
    main(sc, sqlContext)

