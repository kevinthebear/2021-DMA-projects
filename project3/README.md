# Project #3 Info
## Document search engine & Classification and Clustering
This project aims to implement a search engine module, as well as classification and clustering models for text data. The project is divided into two main parts.  

**PART I** - *Document search engine*  
**PART II** - *Document classification and clustering*  

### <ins>PART I. Document search engine</ins>
PART I aims to implement a search engine module that lists documents in order of relevance to the given query using the Python library called "whoosh."

> Data used: Cranfield Dataset
> - document.txt: 1,400 document files
> - query.txt: 225 query files
> - relevance.txt: Answer file specifying the actual related documents for each query

> Modules to be created:
> - [Optional] make_index.py: A function to store the ID and contents of the documents in the index. This module should be written if you're not using the provided index.
> -  QueryResult.py: Takes a text-based query and converts it into a whoosh query object, returning the search results.
> -  CustomScoring.py: A scoring function used to list the documents in order of relevance to the query. Available basic information includes:
>>   - Term frequency (TF) within the document
>>   - Inverse document frequency (IDF)
>>   - Term frequency within the entire dataset
>>   - Number of documents
>>   - Document length (number of words)
>>   - Total number of terms in the entire dataset
>>   - Average number of words per document
>   
> If you can extract additional information beyond the provided information, you may use it as well.

> Evaluation Method:
> - Evaluate the search performance for a randomly selected subset of 37 test queries out of the 225 queries.
> - Use BPREF as the evaluation metric, which will be automatically calculated in the evaluate.py script.

> Notice:
> - Matching query-document pairs based on the project's specific context using the provided queries is prohibited.
> => Both the score for performance improvement methods and the performance evaluation score will be 0.
> - Document analysis is allowed, but query analysis is prohibited.

### <ins>PART II. Document classification and clustering</ins>
PART II requires implementing a model to classify and cluster English newspaper articles from The New York Times using the sklearn library in Python.

> **_(R2-1)_** English newspaper article classification
> - Eight categories in total: opinion, business, world, us, arts, sports, books, movies
> - Approximately 300 article data provided for each category
> - Train/test folders are separated in the text folder, and there is a text file for each category in each folder named after the category.
> - The test folder contains only about 5 of the most recent articles for each category.
> - Additional data obtained through crawling, modifying files, or using only part of the provided dataset is not allowed.
> - The trained models for each category should be saved as a pickle file and submitted along with the code.
>> - 2-1-1. Naïve Bayes Classifier
15 points: the number of correctly classified articles out of 30 * 0.5
>> - 2-1-2. SVM
15 points: the number of correctly classified articles out of 30 * 0.5

> **_(R2-2)_** English newspaper article clustering
> - K-means Clustering
> - Same data used for classification. Train/test separation is not necessary, so the text_all folder is used.
> - V-measure is used as the evaluation metric (harmonic mean of homogeneity and completeness)
