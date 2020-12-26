[image1]: text_processing1.png "text_processing1"
[image2]: text_processing2.png "text_processing2"
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
