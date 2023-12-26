import glob
import numpy as np
from collections import defaultdict
import sys
from io import StringIO
import matplotlib.pyplot as plt
import topic

class DataSet:
    def __init__(self,dirname="data",
                 length_limit=3,
                 count_limit=20):
        # lower bound on length of word
        self.length_limit = length_limit
        # lower limit on min occurences in dataset
        self.count_limit = count_limit

        self.titles = []
        self.pages = []
        self._load_data(dirname)
        self._load_stopwords()
        self._make_word_list()

        self.page_count = len(self.pages)
        self.word_count = len(self.words)

        self._pages_to_vectors()

    def _load_data(self,dirname):
        """Read all txt files in the dirname directory."""
        for title, id, text, link in dirname:
            text = [ word.lower() for word in text.split() ]
            self.pages.append(text)
            self.titles.append(title)

    def _load_stopwords(self,filename="stopwords"):
        """Read list of stopwords."""
        with open(filename,'r') as f:
            lines = f.readlines()
        self.stopwords = [ line.strip() for line in lines ]

    def _make_word_list(self):
        """Find all unique words in the corpus."""

        # count the occurences of every word
        word_counts = defaultdict(lambda: 0)
        for page in self.pages:
            for word in page:
                word_counts[word] += 1

        for word in list(word_counts):
            if len(word) < self.length_limit:
                # skip short words
                del word_counts[word]
            elif word in self.stopwords:
                # skip common words (stopwords)
                del word_counts[word]
            elif word_counts[word] < self.count_limit:
                # skip words that appear only a few times
                del word_counts[word]

        # list of words
        self.words = sorted(list(word_counts))
        # map from word to index in sorted list
        self.word_index = { w:i for i,w in enumerate(self.words) }
        # set of words (fast lookup)
        self.word_set = set(self.words)

    def _page_to_vector(self,page):
        """convert a single page to a vector of word counts"""
        page = [ word for word in page if word in self.word_set ]
        indices = [ self.word_index[word] for word in page ]
        vector = np.bincount(indices)
        #vector.resize(self.word_count)
        # AC: you need to set refcheck=False if you want to debug
        vector.resize(self.word_count,refcheck=False)
        return vector

    def _pages_to_vectors(self):
        """convert all pages to a vector of word counts"""
        vectors = [ self._page_to_vector(page) for page in self.pages ]
        self.vectors = np.array(vectors)

    def print_chart(top_x, word_pr):
        for x in top_x:
            plt.barh(x, word_pr[0], color="red")
            plt.barh(x, word_pr[1], color="purple")
            plt.barh(x, word_pr[2], color="blue")
        plt.show()

    def print_word_probability_table(self,pr,header,length=20, f=sys.stdout):
        """print out a word probability table.
        print out the top most probable words, based on length"""
        print("========================================" +"<br>\n", file = f)
        word_pr = [ (w,p) for w,p in enumerate(pr) ]
        word_pr.sort(key=lambda x: x[1],reverse=True)
        top_three = ""
        
        b = []
        for w, pr in word_pr[0:3]:
            word = self.words[w]
            b.append(word)
        topic.topics_list.append(b)
        
        for w, pr in word_pr[0:1]:
            word = self.words[w]
            a = []
            a.append(word)
            a.append(pr)
            (topic.wp1).append(a)
            top_three += word + " | "
        
        for w, pr in word_pr[1:2]:
            word = self.words[w]
            a = []
            a.append(word)
            a.append(pr)
            (topic.wp2).append(a)
            top_three += word + " | "
        
        for w, pr in word_pr[2:3]:
            word = self.words[w]
            a = []
            a.append(word)
            a.append(pr)
            (topic.wp3).append(a)
            top_three += word + " | "
        
        print(header + " { " + "%20s " % (top_three) + "}" + "<br>\n", file = f)
        #print("==", header +"<br>\n", file = f)
        print("========================================" +"<br>\n", file = f)
        for w,pr in word_pr[:length]:
            word = self.words[w]
            print("%20s | %.4f%%" % (word,100*pr) +"<br>\n", file = f)
        #print_chart(top_x, word_pr)
        return top_three

    
    def print_topic_probability_table(self,pr,header,length=20, f=sys.stdout):
        """print out a topic probability table.
        print out the topics given a documenth"""
        print("========================================", file = f)
        print("==", header +"<br>\n", file = f)
        print("========================================", file = f)
        topic_pr = [ (w,p) for w,p in enumerate(pr) ]
        topic_pr.sort(key=lambda x: x[1],reverse=True)
        for t,pr in topic_pr[:length]:
            print("%20d | %.4f%%" % (t,100*pr) +"<br>\n", file = f)
         

    def print_common_words(self, f = sys.stdout):
        """print list of most frequently appearing words"""
        counts = self.vectors.sum(axis=0)
        total = counts.sum()
        pr = [ c/total for c in counts ]

        header = "common words (out of %d total)" % total
        self.print_word_probability_table(pr,header, f = f)
        prd = self.vectors.sum(axis=1) / self.vectors.sum()
        return prd