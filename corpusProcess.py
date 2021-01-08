
import nltk
import re
import bs4 as bs 
from nltk.tokenize import word_tokenize
import string  
import pandas as pd
import hashedindex
import json
import sys

class createCorpus():
  def __init__(self):
    
    if(len(sys.argv) > 1):
      filename = []
      for i in range(1, len(sys.argv)):
        filename.append(sys.argv[1])
      print("the corpus given : "+filename)
    else:
      filename = ["wiki_56"]
      print("Default corpus :")
      print(filename)
        
    self.process_index(filename)

  def make_posting(self, docs):
    #inverted index creation
    
    indexlist  =  hashedindex.HashedIndex()
    for (i,j) in docs.items():
      for tokens in nltk.word_tokenize(j):
        if tokens not in string.punctuation:
          indexlist.add_term_occurrence(tokens,i)
    return indexlist


  def make_bigram_Improvement(self, docs):
    #improvement made using bigram index
    
    indices  =  hashedindex.HashedIndex()
    for (i,j) in docs.items():
      tokens = nltk.word_tokenize(j)
      for bigrams in nltk.ngrams(tokens,2):
        indices.add_term_occurrence(bigrams,i)
    return indices


  def store_Data(self, filename, data):
    with open(filename, "w+") as f:
      temp  = { str(k):str(json.dumps(data[k])) for k in data.items()}
      j  = json.dumps(temp)
      f.write(j)
      f.close()


  def store_docids(self, filename, docDATA):
    f = open(filename,"w+")
    f.write(str(list(docDATA.keys()))) 
    f.close()

  def clean_docID(self, filenames):
    #extracting individual documents in corpus
    docs_clean = {}
    for filename in filenames :
      with  open(filename, "r", encoding="utf-8") as file:
        data  = file.read()
        docs = { doc.group(1).strip():doc.group(2).strip() for doc in re.finditer(r'<doc id="([0-9]*)".*?>(.*?)</doc>',data,flags = re.M|re.S)}
        docs_clean.update({ k: bs.BeautifulSoup(v,features="lxml").get_text() for k,v in docs.items()})
    
    return docs_clean


  def process_index(self, filenames):
    #storing indices in .txt 
    docsID = self.clean_docID(filenames)
    indices = self.make_posting(docsID)
    bigram_indices = self.make_bigram_Improvement(docsID)
    
    self.store_Data("indices.txt", indices)
    self.store_Data("bigram_improved_indices.txt", bigram_indices)
    
    print("No of documents processed in corpus: "+ str(len(docsID))) 
    self.store_docids("docIDs.txt", docsID)



if __name__ == "__main__":
  createCorpus()
