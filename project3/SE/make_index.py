import os.path

import pandas as pd
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from math import log

schema = Schema(docID=NUMERIC(stored=True), contents=TEXT(stored=True))     #text 저장을 통해 keyword 접근
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)

writer = ix.writer()
df = pd.DataFrame()
with open('doc/document.txt', 'r') as f:
    text = f.read()
    docs = text.split('   /\n')[:-1]
    StopWords = stopwords.words('english')
    nomark = str.maketrans('\n?!,.()-/', '         ')
    lm = WordNetLemmatizer()

    for doc in docs:
        br = doc.find('\n')
        docID = int(doc[:br])
        doc_text = doc[br+1:]

        new_w = ''
        doc_text = doc_text.translate(nomark)

        for word in doc_text.split(' '):

            if (word.lower() not in StopWords) and (word.isdigit() is False) and (len(word) != 1):
                words = lm.lemmatize(word.lower())

                if words not in '':
                    new_w += words + ' '


        new_string = {'index' : docID, 'doc' : new_w}
        df = df.append(new_string, ignore_index=True)
        writer.add_document(docID=docID, contents=new_w)




writer.commit()