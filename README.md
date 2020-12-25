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
Let's walk through an example of cleaning text data from a popular source - the web. There are helpful tools in working with this data, including the
 - [requests library](docs.python-requests.org/en/master/user/quickstart/#make-a-request)
 - [regular expressions](https://docs.python.org/3/library/re.html)
 - [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
 
