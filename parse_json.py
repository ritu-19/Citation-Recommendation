import pyspark
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
from pyspark.sql import SQLContext


# In[4]:


sqlContext = SQLContext (sc)
sc


# In[8]:
for i in range(100):

 name = str(i)
 if i<10:
  name = '0' + name

 json_file_path = 'data/x'+ name
 json_to_df = sqlContext.read.json(json_file_path).limit(50)


# In[9]:


 json_to_df.schema


# In[10]:


 from pyspark.sql.functions import explode, avg, col

 df1 = json_to_df.select("id", "paperAbstract", explode("outCitations").alias("ref"))
#df1.show()

 df1.printSchema()

# In[11]:


 df2 = df1
#df1.show()
#df2.show()
#spark.conf.set( "spark.sql.crossJoin.enabled" , "true" )

 df1Join = df1.alias("a").crossJoin(df1.alias("b")).where('a.ref != b.id').select(col("a.id").alias('id1'), col('a.ref').alias('id2'),"a.paperAbstract",
                                                                                col('b.paperAbstract').alias('paperAbstract2')).dropDuplicates()\




# In[12]:


 from pyspark.sql.functions import lit
 df1Join = df1Join.withColumn("label", lit(0))
#df1Join.show()


# In[14]:


 output_file = 'data.csv'
 df1Join.repartition(1).write.format('com.databricks.spark.csv').save(output_file, mode="append",header = 'true')


# In[ ]:





# In[ ]:





