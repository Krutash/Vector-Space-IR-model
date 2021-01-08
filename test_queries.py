import os
import bs4 as bs 
import string  
import numpy as np
import nltk
import re
import sys
import time
import json
import ast
import math
import pandas as pd
from scipy import spatial

class testQ():
    def __init__(self):
        if(len(sys.argv) > 4):
            qFile = sys.argv[1]
            IFile = sys.argv[2]
            bigramIFile = sys.argv[3]
            DFile  =  sys.argv[4]
            print("Command line atgument Recieved !")

        else:
            qFile = "query.txt"
            IFile = "indices.txt"
            bigramIFile = "bigram_improved_indices.txt"
            DFile  =  "docIDs.txt"
            print("Insufficent or no command line arguments \nPerforming IR in Default Mode....")
        
        
        print("\n\nReading Index File.....")
        postDictionary = self.ReadPostL(IFile)
        print("Successfull !")
        print("\n\nReading Document IDs...")
        D_IDList = self.DOC_ID_INPUT(DFile)
        print("Successfull !")
        print("\n\nCreating Posting List....")
        T_S_Matrix = self.Calculate_TFID(D_IDList, postDictionary)
        print("Successfull !")
        Query_Input = self.Query_Input_input(qFile)
        print("\nQuery_Input Recieved : " + Query_Input + "\n")
        print("\nLoading Data Structure in Memory.....")
        Query_Inputidf  = self.Calculate_Q_vector(Query_Input.split(' '),postDictionary,len(postDictionary))
        print("Successfull !")
        start_TIME = time.process_time()
        Result_OUT = self.TOP_K_LIST(D_IDList, T_S_Matrix,Query_Inputidf,10,Query_Input,postDictionary)
        end_TIME  = time.process_time()
        print("\nUsing vector space model:\n")
        for docs in Result_OUT:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\nTime taken = " + str(end_TIME-start_TIME) + "\n")


        start_TIME =  time.process_time()
        Result_OUT = self.Champ_List(Query_Input.split(' '), postDictionary, T_S_Matrix, Query_Inputidf,10)
        end_TIME  =  time.process_time()
        print("\nUsing Champion list: \n")
        for docs in Result_OUT:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\nTime taken = " + str(end_TIME-start_TIME) + "\n")

        B_POST_LIST = self.ReadPostL(bigramIFile)
        start_TIME = time.process_time()
        Result_OUT = self.phraseQuery_Inputparser(10,Query_Input,B_POST_LIST,postDictionary,T_S_Matrix,D_IDList,Query_Inputidf)
        end_TIME  = time.process_time()
        print("\nUsing phrasal Query_Input (bigram): \n")
        for docs in Result_OUT:
            print("Doc id: " + docs[0] + " Score: " + str(docs[1]))
        print("\nTime taken = " + str(end_TIME-start_TIME) + "\n")


    def DOC_ID_INPUT(self, filename):
        # getting doc ids
        D_FILE = open(filename, 'r')
        D_IDList = [n.strip() for n in ast.literal_eval(D_FILE.read())]
        return D_IDList

    def Calculate_TFID(self, docids , postDictionary):
        # calculating tf
        totaldocs = len(docids)
        rowname = docids
        colname = postDictionary.keys()
        T_S_Matrix = pd.DataFrame(0, index = rowname, columns = colname)
        for word in postDictionary:
            for docno in postDictionary[word]:
                temp = postDictionary[word][docno]
                T_S_Matrix[word][docno] = 1 + np.log10(temp)
        return T_S_Matrix

    def ReadPostL(self, filename):
        # reading the posting list
        fp = open(filename, 'r')
        ipStr = fp.read()
        postDictionary = json.loads(ipStr)
        postDictionary = {k:json.loads(postDictionary[k]) for k in postDictionary}
        fp.close()
        return postDictionary

    def Calculate_Q_vector(self, Query_Inputdoc,postDictionary,totaldocs):
        #calculating  the Doc freq in Query_Input
        Query_Inputidf =  pd.DataFrame(columns = postDictionary.keys())
        Query_Inputidf.loc[0] = np.zeros(len(postDictionary))
        for word in Query_Inputdoc:
            if(word not in postDictionary.keys()):
                continue
            Query_Inputidf[word] = np.log10((totaldocs/len(postDictionary[word])))
        return Query_Inputidf

    def TOP_K_LIST(self, docids, T_S_Matrix,Query_Inputidf,k,Query_Input,postDictionary):
        # getting the top k docs
        COS_SIMILARITY = {}

        for docs in docids:
            COS_SIMILARITY[docs] = 1 - spatial.distance.cosine(T_S_Matrix.loc[docs], Query_Inputidf.loc[0])
        SORTED_COS = sorted(COS_SIMILARITY.items(),  key= lambda x: x[1])[::-1]
        return SORTED_COS[:k]


    def phraseQuery_Input(self, rawQuery_Input,BIGRAM_POSTING_LIST):
        # getting the relevant SET_DOC according to bigram index
        Query_Inputtokens =  nltk.word_tokenize(rawQuery_Input)
        Query_InputbGR = nltk.ngrams(Query_Inputtokens,2)
        
        temp = set()
        SET_DOC = set()
        
        for bGR in Query_InputbGR:
            if(str(bGR) not in BIGRAM_POSTING_LIST.keys()):
                continue
            temp = set((BIGRAM_POSTING_LIST[str(bGR)]).keys())
            if(len(SET_DOC) == 0):
                SET_DOC = temp
            else:
                SET_DOC = SET_DOC.intersection(temp)
        return SET_DOC

    def phraseQuery_Inputparser(self, k,rawQuery_Input,BIGRAM_POSTING_LIST, postDictionary, T_S_Matrix,docid,Query_Inputidf):
        # getting the ranked doc list acc to phrasal queries
        SET_DOC = self.phraseQuery_Input(rawQuery_Input,BIGRAM_POSTING_LIST)
        if(len(SET_DOC) >= k):
            return [(x,1) for x in list(SET_DOC)[0:k]]
        else:
            RANKING_DOC_LIST = [(x,1) for x in list(SET_DOC)[0:k]]
            DOC_LIST  = [x[0] for x in RANKING_DOC_LIST]
            SET_NEXT = self.TOP_K_LIST(docid, T_S_Matrix,Query_Inputidf,2*k ,rawQuery_Input.split(' '),postDictionary)
            index  = 0
            while(len(RANKING_DOC_LIST) <k):
                if(SET_NEXT[index][0] not in DOC_LIST):
                    RANKING_DOC_LIST.append(SET_NEXT[index])
                index =  index + 1
        return RANKING_DOC_LIST
        

    def Champ_List(self, Query_Inputdoc, postDictionary, T_S_Matrix,Query_Inputidf,topk):
        # getting the top k docs using the champion list
        
        Query_InputChamp_List = {}
        Query_Inputdoc  = [ word for word in Query_Inputdoc if word in postDictionary.keys()]
        for word in Query_Inputdoc:
            Query_InputChamp_List[word] = sorted(postDictionary[word].items(), key=lambda x:x[1])[::-1]
        Pruned_L = {k:Query_InputChamp_List[k][:15] for k in Query_Inputdoc}

        DOC_ID_CHAMP_L = set()

        for word in Query_Inputdoc:
            for k in Pruned_L[word]:
                DOC_ID_CHAMP_L.add(k[0])

        Cos_similarity = {}

        for docs in DOC_ID_CHAMP_L:
            Cos_similarity[docs] = 1 - spatial.distance.cosine(T_S_Matrix.loc[docs], Query_Inputidf)

        Sorted_Cos = sorted(Cos_similarity.items(), key=lambda x:x[1])[::-1]

        return Sorted_Cos[:topk]


            
    def Query_Input_input(self, filename):
    # main function calling the above functions and printing their Result_OUTults

        Query_Input = []
        try:
            f = open(filename, 'r')
        except IOError:
            print("Error: File does not appear to exist. Try Running corpusProcess.py first")
            exit()
        Query_Input  = f.read()
        return Query_Input 

if __name__ == "__main__":
    testQ()





