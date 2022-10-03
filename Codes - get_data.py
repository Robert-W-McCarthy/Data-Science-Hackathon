#!/usr/bin/env python
# coding: utf-8

# In[1]:


query = '''
SELECT
  DISTINCT(postid_data.id) AS id,
  approved_tagid_data.tagid AS tagId,
--   postid_data.features,
  postid_data.url,
  ocr_raw_text.fullText
FROM ((
  SELECT
    id,
    url
  FROM
   maximal-furnace-783.ds_feature_storage.content_intelligence_features
  WHERE
    version = "resnet_101_openimage_vggish_select"
    AND DATE(time) >= "2020-06-15"
    AND DATE(time) <= "2020-08-10"
    AND featureType = "video"
    AND lang = "Hindi") postid_data
INNER JOIN (
  SELECT
    tagid,
    LANGUAGE,
    postid
  FROM
    maximal-furnace-783.sc_analytics.tagFilterTool_Actions
  WHERE
    DATE(time) >= "2020-06-15"
    AND DATE(time) <= "2020-08-10"
    AND LANGUAGE IN ("Hindi")
    AND action LIKE ("approved")
    AND CAST(tagid AS INT64) IN (
    SELECT
      tagId
    FROM
      maximal-furnace-783.sanjit.sc_tags_top50_hindi_video_ugc_100days_approved) ) approved_tagid_data
ON
  CAST(postid_data.id AS string) = approved_tagid_data.postid
INNER JOIN (
  SELECT
    postId,
    fullText
  FROM
    maximal-furnace-783.sc_analytics.postOcrData
  WHERE
    DATE(time) >= "2020-06-15"
    AND DATE(time) <= "2020-08-10"
   ) ocr_raw_text
ON
  CAST(ocr_raw_text.postId AS string) = approved_tagid_data.postid)
'''


# In[2]:


from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1
import time

bqclient = bigquery.Client(
)
bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
)


# In[3]:


start_time = time.time()
dataframe = (
    bqclient.query(query)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)
print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:


dataframe.head()


# In[5]:


dataframe.to_pickle("multimodal_video_data.pkl")


# In[7]:


dataframe.shape


# In[ ]:




