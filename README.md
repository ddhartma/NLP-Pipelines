[image1]: text_processing1.png "text_processing1"
[image2]: text_processing2.png "text_processing2"
[image3]: text_processing3.png "text_processing3"
[image4]: text_processing4.png "text_processing4"
[image5]: text_processing5.png "text_processing5"
[image6]: text_processing6.png "text_processing6"
[image7]: text_processing7.png "text_processing7"
[image8]: text_processing8.png "text_processing8"
[image9]: text_processing9.png "text_processing9"
[image10]: text_processing10.png "text_processing10"
[image11]: text_processing11.png "text_processing11"
[image12]: text_processing12.png "text_processing12"
[image13]: text_processing13.png "text_processing13"
[image14]: text_processing14.png "text_processing14"
[image15]: text_processing15.png "text_processing15"

# NLP Pipelines
Natural Language Processing is one of the fastest growing fields in the world. NLP is making its way into a number of products and services which we use every day.

NLP Pipelines contains of three stages:
1. ***Text Processing***:
  - Cleaning
  - Normalization
  - Tokenization
  - Stop Word Removal
  - Part of Speech Tagging
  - Named Entity Recognition
  - Stemming and Lemmatization
2. ***Feature Extraction***:
  - Bag of Words
  - TF-IDF
  - Word Embeddings
3. ***Modeling***

Each stage transforms text in some way and produces a result that the next stage needs.
The goal of text processing is to take raw input text, clean it, normalize it and convert it into a form that is suitable for feature extraction.
Similarly, the next stage needs to extract and produce feature representations that are appropriate for the type of model one is planning to use and the NLP task one is trying to accomplish. This process isn't always linear and may require additional steps.

## Text Processing

- ***Extracting plain text***: Textual data can come from a wide variety of sources: the web, PDFs, word documents, speech recognition systems, book scans, etc. Your goal is to extract plain text that is free of any source specific markup or constructs that are not relevant to your task.

![image1]

- ***Reducing complexity***: Some features of our language like capitalization, punctuation, and common words such as a, of, and the, often help provide structure, but don't add much meaning. Sometimes it's best to remove them if that helps reduce the complexity of the procedures you want to apply later.

![image2]

In this section we will prepare text data from different sources with the following text processing steps:
  1. Cleaning to remove irrelevant items, such as HTML tags
  2. Normalizing by converting to all lowercase and removing punctuation
  3. Splitting text into words or tokens
  4. Removing words that are too common, also known as stop words
  5. Identifying different parts of speech and named entities
  6. Converting words into their dictionary forms, using stemming and lemmatization

After performing these steps, the text will capture the essence of what was being conveyed in a form that is easier to work with.

### Cleaning
Open notebook ***./text_processing/cleaning.ipynb*** to handle text cleaning
- Let's walk through an example of cleaning text data from a popular source - the web. There are helpful tools in working with this data, including the
 - [requests library](https://2.python-requests.org/en/master/user/quickstart/#make-a-request)
 - [regular expressions](https://docs.python.org/3/library/re.html)
 - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

- Request a web page
  ```
  import requests
  # fetch web page
  r = requests.get('https://www.udacity.com/courses/all')
  ```

  Downloaded successfully (status = 200)
  ```
  r.status_code
  ```

- Parsing a Page with Beautifulsoup
  ```
  from bs4 import BeautifulSoup
  soup = BeautifulSoup(r.text, "lxml")
  soup.text
  ```

  Print with readable indent
  ```
  print(soup.prettify())
  ```

  List all tags that are nested
  ```
  list(soup.children)
  ```

  List children of children
  ```
  html = list(soup.children)[2]
  body = list(html.children)[3]
  p = list(body.children)[1]
  ```

  Get text from children
  ```
  p.get_text()
  ```
- Finding all instances of a tag at once
  ```
  soup.find_all('p')
  ```

  Use list indexing, it to extract text:
  ```
  soup.find_all('p')[0].get_text()
  ```

  If only first p instance is needed
  ```
  soup.find('p')
  ```

- Searching for tags by class and id
  Search for any ```p``` tag that has the ```class='outer-text'```
  ```
  soup.find_all('p', class_='outer-text')
  ```

  Search for any tag that has the ```class='outer-text'```
  ```
  soup.find_all(class_="outer-text")
  ```

  Find id
  ```
  soup.find_all(id="first")
  ```

  CSS selectors to find all the p tags in a page that are inside of a div
  ```
  soup.select("div p")
  ```

### Normalization
Open notebook ***./text_processing/normalization.ipynb*** to handle text normalization
- ***Case Normalization***: In Machine Learning it does not make sense to differentiate between 'car', 'Car' and 'CAR'. These all three words have the same meaning. Therefore: Normalize all words to lower case
  ```
  text = text.lower()
  ```
- ***Punctual Removal***: Dependenfing on the NLP task, one wants to remove special characters like periods, question marks, exclamation points and only keep letters of the alphabet and maybe numbers (especially usefull for document classification and clustering where low level details do not matter a lot)
  ```
  import re
  # Remove punctuation from text and
  # only keep letters of the alphabet and maybe numbers
  # everything else is replaced by a space

  text = re.sub(r"[^a-zA-Z0-9]", " ", text)
  ```

### Tokenization
Open notebook ***./text_processing/tokenization.ipynb*** to handle text tokenization
- Tokenization is simply splitting each sentence into a sequence of words.
- Simple method: ```split()```
  ```
  # Split text into words
  words = text.split()
  print(words)
  ```
- [NLTK](www.nltk.org/api/nltk.tokenize.html) library - NATURAL LANGUAGE TOOLKIT

  - It is smarter e.g. in terms of punctuation ['Dr.', 'Smith', 'graduated', ... , '.']
  - Split text into words
    ```
    from nltk.tokenize import word_tokenize
    # split text into words using NLTK
    words = word_tokenize(text)
    ```
  - Split text into sentences (e.g. for translation)
    ```
    from nltk.tokenize import sent_tokenize
    # split text into sent4ences using NLTK
    sentences = sent_tokenize(text)
    ```

  - NLTK has several other options e.g.
    - a regular expression base tokenizer to ***remove punctuation*** and ***perform tokenization*** in a single step.
    - a tweet tokenizer that is aware of twitter handles, hash tags and emoticons


### Stop Words Removal
Open notebook ***./text_processing/stop_words.ipynb*** to handle stop word removal
- Stop words are uninformative words like ***is. our, the, in, at, ...*** that do not add a lot of meaning to a sentence.
- Remove them to reduce the vocabulary size (complexity of later procedures)
- [NLTK](www.nltk.org/api/nltk.tokenize.html) library  can identify stop words
  ```
  # List stop words
  from nltk.corpus import stopwords
  print(stopwords.words("english"))
  ```
- Remove stop words with a Python list comprehension with a filtering condition
  ```
  # Remove stop words
  words = [w for w in words if w not in stopwords.words("english")]
  ```

### Part-of-Speech Tagging
Open notebook ***./text_processing/pos_ner.ipynb*** to handle Part-of-Speech Tagging
- To know the parts of speech (like nouns, verbs, pronouns) can help to understand the meaning of a sentence better
- It can point out relationships betwee nwords and recognize cross references
- [NLTK](www.nltk.org/api/nltk.tokenize.html) library  can identify parts-of-speech
  ```
  from nltk import pos_tag
  # Tag parts of speech (PoS)
  sentence = word_tokenize("I always lie down to tell a lie")
  pos_tag(sentence)
  ```

  ![image3]

- Sentence Parsing
  ```
  # Define a custom grammar
  my_grammar = nltk.CFG.fromstring("""
  S -> NP VP
  PP -> P NP
  NP -> Det N | Det N PP | 'I'
  VP -> V NP | VP PP
  Det -> 'an' | 'my'
  N -> 'elephant' | 'pajamas'
  V -> 'shot'
  P -> 'in'
  """)
  parser = nltk.ChartParser(my_grammar)
  ```
  ```
  # Parse a sentence
  sentence = word_tokenize("I shot an elephant in my pajamas")
  for tree in parser.parse(sentence):
      print(tree)
  ```

### Named Entity Recognition
Open notebook ***./text_processing/pos_ner.ipynb*** to handle Named Entity Recognition
- Named entities are typically noun phrases that refer to some specific object, person, or place
- Named entity recognition is often used to index and search for news articles on companies of interest
- [NLTK](www.nltk.org/api/nltk.tokenize.html) provides the ne_chunk function to label named entities in text
- Before using ne_chunk one has to
  - first tokenize and then
  - tag parts of speech

  ```
  from nltk import pos_tag, ne_chunk
  from nltk.tokenize import word_tokenize

  # Recognize named entities in  a tagged sentence
  ne_chunk(pos_tag(word_tokenize("Antonio joined Udacity Inc. in California.")))
  ```

  Entity types are recognized
  ![image4]


### Stemming and Lemmatization
Open notebook ***./text_processing/stem_lem.ipynb*** to handle Stemming and Lemmatization
- ***Stemming***: In order to further simplify text data, stemming is the process of reducing a word to its stem or root form.
- For instance, branching, branched, branches et cetera, can all be reduced to branch.
- the suffixes 'ing' and 'ed' can be dropped off, 'ies' can be replaced by 'y' et cetera.
- Stemming is meant to be a fast operation
- NLTK has a few different stemmers for you to choose from
  - PorterStemmer
  - SnowballStemmer
  - other language-specific stemmers

- PorterStemmer (remove stop words beforehand)
  ```
  from nltk.stem.porter import PorterStemmer

  # Reduce words to their stem
  stemmed = [PorterStemmer().stem(w) for w in words]
  print(stemmed)
  ```
- ***Lemmatization***: This is another technique to reduce words to a normalize form
- In this case the transformation uses a ***dictionary*** to map different variants of a word back to its root.
- With this approach, we are able to reduce non-trivial inflections such as 'is', 'was', 'were', back to the root 'be'.
- [NLTK](www.nltk.org/api/nltk.tokenize.html) uses the default lemmatizer Wordnet database.
  ```
  from ntlk.stem.wordnet import WordNetLemmatizer

  # Reduce words to their root form
  lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
  print(lemmed)
  ```

-  A lemmatizer needs to know about the part of speech for each word it's trying to transform. In this case, WordNetLemmatizer defaults to nouns, but one can override that by specifying the PoS parameter. Let's pass in 'v' for verbs.

  ```
  Lemmatize verbs by specifying pos
  lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
  print(lemmed)
  ```

- Stemming sometimes results in stems that are not complete words in English. Lemmatization is similar to stemming with one difference, the final form is also a meaningful word. Stemming does not need a dictionary like lemmatization does. Stemming maybe a less memory intensive option.

  ![image5]


### Summary of Text Processing
 1. Normalize
 2. Tokenize
 3. Remove Stop Words
 4. Stem / Lemmatize

 ![image6]


## Feature extraction
- Now text is clean and normalized. Can we feed this into a statistical or a machine learning model? Text data is represented using an encoding such as ASCII or Unicode that maps every character to a number. Computer store and transmit these values as binaries. These numbers also have an implicit ordering, 65 < 66 < 67. But does that mean A is less than B. No. This would mislead the natural language processing algorithms. Moreover, individual characters don't carry much meaning at all. It is words that one should be concerned with,

- So the question is, how do we come up with a representation for text data that we can use as features for modeling?
The answer again depends on what kind of model you're using and what task you're trying to accomplish.

- If you want to use a graph based model to extract insights, you may want to represent your words as symbolic nodes with relationships between them like WordNet.

  ![image7]

- For statistical models some sort of numerical representation would fit better.

  ![image8]


- For document level task, such as spam detection or sentiment analysis bag-of-words or doc2vec may fit better.

  ![image9]


- For working with individual words and phrases such as for text generation or machine translation, you'll need a word level representation such as word2vec or glove.

  ![image10]

### Bag of Words
Open notebook ***./feature_extraction/bag_of_words_and_TF_IDF.ipynb*** to handle Bag of Words
- The Bag of Words model treats each document as an un-ordered collection or bag of words.
- To obtain a bag of words from a piece of raw text, apply text processing steps:
  - cleaning
  - normalizing
  - splitting into words
  - stemming, lemmatization


- Turn each document into a vector of numbers, representing how many times each word occurs in a document. A set of documents is known as a corpus, and this gives the context for the vectors to be calculated.

  1. Collect all the unique words present in the corpus to form your vocabulary.
  2. Each row is a document
  3. Ech column is a word representation
  4. Count the number of occurrences of each word in each document and enter the value in the respective column.

  ![image12]


- What you can do with this representation?

  - Check the similarity of documents by computing the dot product between the two row vectors
  - dot product meaning: how many words they have in common or how similar their term frequencies are?
  - dot product = the sum of the products of corresponding elements.
  - The greater the dot product, the more similar the two vectors are.

  ![image13]

- The dot product has one drawback: it only captures the portions of overlap. So, pairs that are very different can end up with the same product as ones that are identical.

  - A better measure is cosine similarity: the product is divided by their magnitudes or Euclidean norms.
  - The angle theta is the angle between the vectors of the dot product
  - Identical vectors have cosine equals 1
  - Orthogonal vectors have cosine equal 0
  - Antiparallel vectorshave cosine equal  -1

### TF-IDF
Open notebook ***./feature_extraction/bag_of_words_and_TF_IDF.ipynb*** to handle TF-IDF

- One limitation of bag of words approach: It treats every word as being equally important. To solve this problem:  
  - Count the number of documents in which each word occurs and divide the term frequency by the document frequency of that terms
  - This provides a metric that is proportional to the frequency of occurrence of a term in a document
  - and inversely proportional to the number of documents it appears in
  - It highlights words that are unique for that document and thus this is better for characterizing those words

  ![image14]

- ***TF-IDF transformation***: It's simply the product of two weights
  - The product consists of a term frequency and an inverse document frequency
  - ***term frequency***: the raw count of a term 't' in document 'd' / total number of terms in 'd'
  - ***inverse document frquency***: log of the total number of documents in collection 'D' / number of documents where t is present

  ![image15]


- This approach can be used to implement a Spam Detection application based on a supervised learning model.

## Setup Instructions
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a good Command Line Interface (CLI) under Windowsa you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/NLP-Pipelines.git
```

- Change Directory
```
$ cd 6_NLP_Pipelines
```

- Create a new Python environment, e.g. ds_nlp. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_nlp
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
```

- Check the environment installation via
```
$ conda env list
```

- Activate the installed environment via
```
$ conda activate ds_nlp
```

## Acknowledgments
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.
