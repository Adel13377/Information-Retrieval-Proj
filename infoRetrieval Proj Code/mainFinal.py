import pandas as pd
from nltk.stem import PorterStemmer
from collections import defaultdict
import math
import numpy as np
from natsort import natsorted
ps = PorterStemmer()

#spread method
def spread(arg):
  ret = []
  for i in arg:
    ret.extend(i) if isinstance(i, list) else ret.append(i)
  return ret
 
##creating dictionary of words final form : {'docnum' : [list of stemmed terms]}     ex: {1: ['antoni', 'brutu', 'caeser', 'cleopatra', 'merci', 'worser']}
notStemmedDict={} #initial Dictionary
#reading files
for s in range(1,11): #loop 1:10
    f = open(f"../DocumentCollection/{s}.txt", "r")
    fileContentStr = f.read()
    notStemmedDict[s]=fileContentStr.split()


#stemming terms
stemmedDict = {}
for key, words_list in notStemmedDict.items():
    stemmed_words = [ps.stem(word) for word in words_list]
    stemmedDict[key] = stemmed_words

# print(stemmedDict)
# print(spread(list(stemmedDict.values())))

    
##create inverted index from stemmedDict final form : {'term' : [list which elements are ['docnum', 'term position']] }   ex: {'antoni': [[1, 0], [2, 0], [6, 0]]}
invertedIndex={}

def create_inverted_index(documents):
    for doc_id, terms_list in documents.items():
        for index, term in enumerate(terms_list):
            if term not in invertedIndex:
                invertedIndex[term] = [[doc_id, index]]
            else:
                invertedIndex[term].append([doc_id, index])


create_inverted_index(stemmedDict)
# print(invertedIndex)

# Printing the inverted index
for term, posting_list in invertedIndex.items():
    print(f"{term}, ({len(posting_list)}): {spread(posting_list)[::2]}")




#printing the Positional index

for term, doc_info_list in invertedIndex.items():
    spreaded_doc_info_list=spread(doc_info_list)[::2]
    num_docs=[]
    for i in spreaded_doc_info_list:
      if i not in num_docs:
        num_docs.append(i)
    num_docs = len(num_docs)
    print(f"{term}, ({num_docs} documents containing the term)")

    for doc_info in doc_info_list:
        doc_id, positions = doc_info
        print(f"  doc{doc_id}: {positions+1} ;")

    print()    


#invertedIndex -->  {'antoni': [[1, 0], [1, 6], [2, 0], [6, 0]]
#Creating a dict to store term frequencies
term_frequencies = {}

def get_term_frequencies(invertedIndex):

    for term, positions in invertedIndex.items():
        term_frequencies[term] = {}

        for doc_position in positions:
            doc_id, position = doc_position
            doc_key = 'd' + str(doc_id)

            if doc_key not in term_frequencies[term]:
                term_frequencies[term][doc_key] = 1
            else:
                term_frequencies[term][doc_key] += 1  
get_term_frequencies(invertedIndex)

# print(term_frequencies)


#####################################| Term Frequency DataFrame |####################################
# Convert the dictionary to a pandas DataFrame and transpose it
tf = pd.DataFrame(term_frequencies).fillna(0).T.astype(int)
tf = tf[natsorted(tf.columns.tolist())] #sorting the docs order


# Display the DataFrame
print(tf)

########################| Weighted Term Frequency DataFrame  (1 + log(TF)) |#########################
getWTF = lambda x: 1+math.log10(x) if x > 0 else 0
wtf = tf.map(getWTF)
# Display weighted TF DataFrame
print(wtf)



############################| Document frequency / Inverse DF DataFrame |############################
# Calculate document frequency for each term
document_frequency = {term: sum(1 for freq in doc_frequencies.values() if freq > 0) for term, doc_frequencies in term_frequencies.items()}
# print(document_frequency)

# Display the document frequency DataFrame
df = pd.DataFrame(list(document_frequency.items()), columns=['Term', 'DF']).set_index('Term')
N = len(stemmedDict) # Number of documents
df['IDF'] = np.log10(N / df['DF'])
# Display the DataFrame
print(df)

###########################################| TF*IDF DataFrame |######################################
# Multiply term frequencies by IDF
tf_idf = wtf.mul(df['IDF'], axis=0)
# Display the DataFrame
print(tf_idf)

###########################################| Document Length |#######################################
document_length = pd.DataFrame()

def get_doc_length(col):
  return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())


for column in tf_idf.columns:
  document_length.loc[0, column+" length"] = get_doc_length(column)
  
print(document_length.T)


##########################################| Normalized tf.idf |###################################
normalized_tf_idf = tf_idf.div(document_length.iloc[0].values, axis=1).fillna(0)

print(normalized_tf_idf)


###############################################| Query |##########################################
q = input('Enter your Query : ')    #'antony brutus'
q = " ".join([ps.stem(word) for word in q.split()])
def queryProccessing(query):
  query= pd.DataFrame(index = normalized_tf_idf.index)
  query['tf'] = [1 if x in q.split() else 0 for x in list(normalized_tf_idf.index)]
  query['w_tf'] = query['tf'].apply(lambda x : getWTF(x))
  query['idf'] = df['IDF'] * query['w_tf']
  query['tf_idf'] = query['w_tf'] * query['idf']
  query['normalized'] = query['tf_idf'] / math.sqrt(sum(query['tf_idf'].values**2))

  print(query[query['tf'] != 0])

  product = normalized_tf_idf.multiply(query['w_tf'], axis=0)
  product2 = product.multiply(query['normalized'], axis=0)



  scores = {}
  for col in product2.columns:
      if 0 not in product2[col].loc[q.split()].values:
          scores[col] = product2[col].sum()
      
    ##query length
  qLength=f"Query length is : {math.sqrt(sum([x**2 for x in query['idf'].loc[q.split()]]))}"
  print(qLength)

  ###########################################################################
    ##prod dataframe
  prod_res = product2.loc[q.split(), list(scores)] #product
  prod_res.loc['Total'] = prod_res.sum(numeric_only=True) #total row
  print(prod_res)
  
  ##display similarity
  total_similarity = prod_res.loc['Total']

  for column_name, similarity in total_similarity.items():
    print(f"Similarity(q, {column_name}) = {similarity}")
    
    ##returned docs sorted
  final_score = sorted(scores.items(), key = lambda x: x[1], reverse=True)
  print(*[doc[0] for doc in final_score], sep=', ')
  
queryProccessing(q)



