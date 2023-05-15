import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

def getSearchEngineResult(query_dict):
    result_dict = {}
    ix = index.open_dir("index")


    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # Define your own query parser
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.7) )
        stopWords = set(stopwords.words('english'))
        lm = WordNetLemmatizer()
        nomark = str.maketrans('\n?!,.()-' , '        ')

        for qid, q in query_dict.items():
            new_q = ''
            q = q.translate(nomark)
            for word in q.split(' '):
                if (word.lower() not in stopWords) and (word.isdigit() is False) and (len(word) != 1):

                    word_lm = lm.lemmatize(word)
                    if word_lm not in '':
                        new_q += word_lm + ' '

            #pos tag
            new_q_postag = pos_tag(new_q.split())
            new2_q = ''
            for word, tag in new_q_postag:
                if tag in ['NN', 'NNS']:
                    new2_q += word + '^1 '
                elif tag in ['NP','NPS']:
                    new2_q += word + '^1 '
                elif tag in ['JJ', 'JJR', 'JJS']:
                    new2_q += word + '^1.1 '
                elif tag in ['VBG', 'VBN']:
                    new2_q += word + '^1 '
                elif tag in ['VB', 'VBD', 'VBP']:
                    new2_q += word + '^1.1 '
                elif tag in ['CC', 'IN']:
                    new2_q += word + '^0.75 '
                elif tag in ['RB', 'RBR', 'RBS']:
                    new2_q += word + '^0.8 '
                else:
                    new2_q += word + '^0.9 '

            new_q = new2_q
            query = parser.parse(new_q.lower())
            results = searcher.search(query, limit=None)

            #keywords를 강조
            keywords = [keyword for keyword, score in results.key_terms("contents", docs=7, numterms=4)]
            for keyword in keywords:
                new_q += keyword + '^1.25'
            query = parser.parse(new_q.lower())
            results = searcher.search(query, limit=None, terms=True)

            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict