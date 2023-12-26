import numpy as np
from dataset import DataSet
from io import StringIO
import sys
import sqlite3 as sql
import download
import app
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('agg')
from matplotlib import colors, colormaps

count_limit = 20     # minimum times a word has to appear in the corpus
topic_count = 10     # number of topics
max_iterations = 100 # maximum number of EM iterations

class TopicModel:
    def __init__(self,data, seed=None, 
                 topic_count=10):
        self.page_count = data.page_count
        self.topic_count = topic_count
        self.word_count = data.word_count
        self.data = data

        if seed is None:
            seed = np.random.randint(10000)
        print("random number seed: %d" % seed)
        np.random.seed(seed)
        self.seed = seed
        
        # sample a random topic-given-document distribution
        alpha = [2]*self.topic_count
        self.pr_td = np.random.dirichlet(alpha,self.page_count)
        # sample a random word-given-topic distribution
        alpha = [2]*self.word_count
        self.pr_wt = np.random.dirichlet(alpha,self.topic_count)

    def _learn(self,topics):
        """learn parameters from completed data"""

        # learn topic-given-document distributions
        self.pr_td = np.ones([self.page_count,self.topic_count])
        for d,(topic,vector) in enumerate(zip(topics,self.data.vectors)):
            vector = vector[:,np.newaxis]
            self.pr_td[d] += (topic*vector).sum(axis=0)
        self.pr_td /= self.pr_td.sum(axis=1,keepdims=True)

        # learn word-given-topic distributions
        self.pr_wt = np.ones([self.topic_count,self.word_count])
        topics = np.swapaxes(topics,0,1)
        for w,(topic,vector) in enumerate(zip(topics,self.data.vectors.T)):
            vector = vector[:,np.newaxis]
            self.pr_wt.T[w] += (topic*vector).sum(axis=0)
        self.pr_wt /= self.pr_wt.sum(axis=1,keepdims=True)       
        
    #for given word in a given doc, probability of each topic
    def twd(self, word, doc):
        topic_d = self.pr_td[doc]
        topic_w = self.pr_wt[:, word]
        pr_twd = np.multiply(topic_d, topic_w)
        pr_twd = pr_twd/sum(pr_twd)
        return pr_twd

    def prtd(self):
        global prd_v
        prd_vt = prd_v.reshape(-1,1)
        pr_dt = np.multiply(prd_vt, self.pr_td)
        pr_dt = pr_dt/sum(pr_dt)
        print(pr_dt)
        return pr_dt
    
    def _predict(self):
        """predict (fractional) topics given data"""
        topics = []
        for pr_t,page in zip(self.pr_td,self.data.pages):
            doc_pr = (self.pr_wt * pr_t[:,np.newaxis]).T
            doc_sum = doc_pr.sum(axis=1)[:,np.newaxis]
            doc_pr /= doc_sum
            topics.append(doc_pr)
        return np.array(topics)

    def _log_likelihood(self):
        """compute the log likelihood of the data
        (we skip the Pr(doc) factor, which is a constant)"""
        N = self.data.vectors.sum()  # total # of words in dataset
        ll = 0.0                # log likelihood
        for pr_t,vector in zip(self.pr_td,self.data.vectors):
            doc_pr = self.pr_wt * pr_t[:,np.newaxis]
            doc_ll = np.log(doc_pr.sum(axis=0))
            ll += doc_ll@vector
        return ll/N
# create function to generate top 3 words & pass into print word prob table

    def print_topics(self, f=sys.stdout):
        global top3
        top3 = []
        for t,pr_w in enumerate(self.pr_wt):
            header = "topic %d" % t
            top3.append(self.data.print_word_probability_table(pr_w,header, f=f))
    
            #create new function taking in topic index, and return top 3 words
            
    def print_topic_probability_table(self,pr,header,length=20, f=sys.stdout):
        """print out a word probability table.
        print out the top most probable words, based on length"""
        print("========================================", file = f)
        print("==", header +"<br>\n", file = f)
        print("========================================", file = f)
        topic_pr = [ (w,p) for w,p in enumerate(pr) ]
        topic_pr.sort(key=lambda x: x[1],reverse=True)
        
        maindocs.append(topic_pr)
        
        for w, pr in topic_pr[0:1]:
            #word = self.words[w]
            a = []
            a.append(w)
            a.append(pr)
            (tp1).append(a)
        
        for w, pr in topic_pr[1:2]:
            #word = self.words[w]
            a = []
            a.append(w)
            a.append(pr)
            (tp2).append(a)
        
        for w, pr in topic_pr[2:3]:
            #word = self.words[w]
            a = []
            a.append(w)
            a.append(pr)
            (tp3).append(a)
            
        for t,pr in topic_pr[:length]:
            print("{%20s} | %.4f%%" % (top3[t],100*pr) +"<br>\n", file = f)
         
    def print_documents(self, f=sys.stdout):
        global doc_names 
        doc_names= []
        for d,pr_t in enumerate(self.pr_td):
            header = "%s" % self.data.titles[d]
            doc_names.append(header)
            self.print_topic_probability_table(pr_t,header, f=f)

    def em(self):
        """run expectation-maximization"""
        for it in range(max_iterations):
            topics = self._predict()
            self._learn(topics)
            ll = self._log_likelihood()
            print("iteration %d/%d: ll = %.4f" % (it+1,max_iterations,ll))
        return ll

# load dataset
def main():
    global wp1
    global wp2
    global wp3
    wp1 = []
    wp2 = []
    wp3 = []
    
    global tp1
    global tp2
    global tp3
    tp1 = []
    tp2 = []
    tp3 = []
    
    global topics_list
    topics_list = []
    
    global maindocs
    maindocs = []
    
    conn = sql.connect('article_db.db', check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT article_name, article_id, article_text, article_link FROM " + download.tablename)
    articles = c.fetchall()
    
    data = DataSet(count_limit=count_limit, dirname = articles)
    topic_link = StringIO()

    # print stats
    print("====================<br>" , file = topic_link)
    print("word length limit: %d " % data.length_limit +"\n", file = topic_link)
    print("word count limit: %d" % data.count_limit +"<br>\n", file = topic_link)
    print("====================" +"<br>\n", file = topic_link)
    print("page count: %d" % data.page_count +"<br>\n", file = topic_link)
    print("word count: %d" % data.word_count +"<br>\n", file = topic_link)
    print("====================" +"<br>\n", file = topic_link)
    global prd_v
    prd_v = data.print_common_words(f=topic_link)

    # topic model, run EM and print the learned topics
    global tm
    tm = TopicModel(data, topic_count=topic_count)
    ll = tm.em()
    tm.print_topics(f=topic_link)
    tm.print_documents(f=topic_link)
    print("final log likelihood = %.8f" % ll +"<br>\n", file = topic_link)
    
    td = tm.prtd()
    
    for i in list(range(10)):
        t1 = enumerate(td[:,i])
        sort_t1 = sorted(t1,key=lambda x: x[1],reverse=True)
        print("========================================", file = topic_link)
        print("== Topic: %d | {%20s}" % (i,top3[i]) +"<br>\n", file = topic_link)
        print("========================================" +"<br>\n", file = topic_link)
        for d,pr in sort_t1[:10]:
            print("%20s | %.4f%%" % (doc_names[d],100*pr) +"<br>\n", file = topic_link)
    
    topic_link.seek(0)
    print_topic = topic_link.read()
    topic_link.close()
    return print_topic
    
def get_text(doc_index):
    global tm
    a_text = " ".join(tm.data.pages[doc_index])
    return a_text
        
def highlight_word(doc_index, article_text, t_index):
    global tm
    cmap = colormaps.get_cmap("YlGnBu")
    new_text = []
    for w in article_text.split():
        if w not in tm.data.word_index:
            continue
        word_index = tm.data.word_index[w]
        twd = tm.twd(word_index, doc_index)
        twd_t = twd[t_index]
        color = colors.rgb2hex(cmap(int(twd_t * 255)))
        new_word = '<mark style="background: %s "> %s </mark>' % (color + "BF", w)
        new_text.append(new_word)
    return " ".join(new_text) 
    
def topic_chart():
        plt.clf()
        plt.figure(figsize=(14,12))
        b1 = []
        w1 = []
        for inner in wp1:
            b1.append(inner[1] * 100)
            w1.append(inner[0])
        
        b2 = []
        w2 = []
        for inner in wp2:
            b2.append(inner[1] * 100)
            w2.append(inner[0])
            
        b3 = []
        w3 = []
        for inner in wp3:
            b3.append(inner[1] * 100)
            w3.append(inner[0])
            
        topic = list(range(topic_count))
        left_b3 = np.add(b1[1:], b2[1:]).tolist()

        bar1 = plt.barh(topic, b1[1:], color = '#8FD28B')
        plt.bar_label(bar1, labels=w1[1:], label_type='center', fontsize = 18)
 
        bar2 = plt.barh(topic, b2[1:], left=b1[1:], tick_label = ["topic %d" % i for i in range(10)], color = '#D8A6E5')
        plt.bar_label(bar2, labels=w2[1:], label_type='center', fontsize = 18)
        
        bar3 = plt.barh(topic, b3[1:], left=left_b3, color = '#AEE2F0')
        plt.bar_label(bar3, labels=w3[1:], label_type='center', fontsize = 18)
        
        #plt.title("Topics Represented by Words")
        plt.gca().invert_yaxis()
        plt.yticks(fontsize = 18)
        plt.gca().set_xticks([])
class DocPlot():
    def __init__(self, doc_list, doc_num):
        #lists of top topics per document
        self.topt1 = [doc_list[i][0][0] for i in list(range(len(doc_list)))]
        self.topt2 = [doc_list[i][1][0] for i in list(range(len(doc_list)))]
        self.topt3 = [doc_list[i][2][0] for i in list(range(len(doc_list)))]
        self.topt4 = [doc_list[i][3][0] for i in list(range(len(doc_list)))]
        self.topt5 = [doc_list[i][4][0] for i in list(range(len(doc_list)))]
        self.topt6 = [doc_list[i][5][0] for i in list(range(len(doc_list)))]
        self.topt7 = [doc_list[i][6][0] for i in list(range(len(doc_list)))]
        self.topt8 = [doc_list[i][7][0] for i in list(range(len(doc_list)))]
        self.topt9 = [doc_list[i][8][0] for i in list(range(len(doc_list)))]
        self.topt10 = [doc_list[i][9][0] for i in list(range(len(doc_list)))]
        
        #lists of probabliities for each top topic in documents
        self.top_pr1 = [doc_list[i][0][1] for i in list(range(len(doc_list)))]
        self.top_pr2 = [doc_list[i][1][1] for i in list(range(len(doc_list)))]
        self.top_pr3 = [doc_list[i][2][1] for i in list(range(len(doc_list)))]
        self.top_pr4 = [doc_list[i][3][1] for i in list(range(len(doc_list)))]
        self.top_pr5 = [doc_list[i][4][1] for i in list(range(len(doc_list)))]
        self.top_pr6 = [doc_list[i][5][1] for i in list(range(len(doc_list)))]
        self.top_pr7 = [doc_list[i][6][1] for i in list(range(len(doc_list)))]
        self.top_pr8 = [doc_list[i][7][1] for i in list(range(len(doc_list)))]
        self.top_pr9 = [doc_list[i][8][1] for i in list(range(len(doc_list)))]
        self.top_pr10 = [doc_list[i][9][1] for i in list(range(len(doc_list)))]
        
        self.topic1 = doc_list[doc_num][0][0]
        self.pr1 = doc_list[doc_num][0][1]
        self.topic2 = doc_list[doc_num][1][0]
        self.pr2 = doc_list[doc_num][1][1]
        self.topic3 = doc_list[doc_num][2][0]
        self.pr3 = doc_list[doc_num][2][1]
        self.topic4 = doc_list[doc_num][3][0]
        self.pr4 = doc_list[doc_num][3][1]
        self.topic5 = doc_list[doc_num][4][0]
        self.pr5 = doc_list[doc_num][4][1]
        self.topic6 = doc_list[doc_num][5][0]
        self.pr6 = doc_list[doc_num][5][1]
        self.topic7 = doc_list[doc_num][6][0]
        self.pr7 = doc_list[doc_num][6][1]
        self.topic8 = doc_list[doc_num][7][0]
        self.pr8 = doc_list[doc_num][7][1]
        self.topic9 = doc_list[doc_num][8][0]
        self.pr9 = doc_list[doc_num][8][1]
        self.topic10 = doc_list[doc_num][9][0]
        self.pr10 = doc_list[doc_num][9][1]
        
        name = doc_names[doc_num]  

def doc_chart():
    #clears previous topic bar chart
    plt.clf()
    #list of numbers the length of the amount of documents present in dataset
    documents = list(range(len(doc_names)))
    if len(doc_names) < 650:
        plt.figure(figsize=(14,len(doc_names) - 10))
    else:
        plt.figure(figsize=(14,655))
    
    d = DocPlot(maindocs, 0)
    t1list = d.topt1
    t2list = d.topt2
    t3list = d.topt3
    pr1list = d.top_pr1
    pr2list = d.top_pr2
    pr3list = d.top_pr3

    t4list = d.topt4
    pr4list = d.top_pr4
    t5list = d.topt5
    pr5list = d.top_pr5
    t6list = d.topt6
    pr6list = d.top_pr6
    t7list = d.topt7
    pr7list = d.top_pr7
    t8list = d.topt8
    pr8list = d.top_pr8
    t9list = d.topt9
    pr9list = d.top_pr9
    t10list = d.topt10
    pr10list = d.top_pr10
    
    #will be list of top topics for each doc in order
    doc_t1 = []
    document_list = []
    
    t_enum = list(enumerate(zip(t1list, pr1list, t2list, t3list)))
    e1 = sorted(t_enum,key=lambda x: x[1])
    
    sorted_t1 = [t1list[i] for i, x in e1]
    sorted_t3 = [t3list[i] for i, x in e1]
    sorted_p3 = [pr3list[i] for i,x in e1]
    sorted_p1 = [pr1list[i] for i,x in e1]
    sorted_t2 = [t2list[i] for i, x in e1]
    sorted_p2 = [pr2list[i] for i, x  in e1]
    
    sorted_t4 = [t4list[i] for i, x in e1]
    sorted_t5 = [t5list[i] for i, x in e1]
    sorted_t6 = [t6list[i] for i, x in e1]
    sorted_t7 = [t7list[i] for i, x in e1]
    sorted_t8 = [t8list[i] for i, x in e1]
    sorted_t9 = [t9list[i] for i, x in e1]
    sorted_t10 = [t10list[i] for i, x in e1]
    sorted_p4 = [pr4list[i] for i,x in e1]
    sorted_p5 = [pr5list[i] for i,x in e1]
    sorted_p6 = [pr6list[i] for i,x in e1]
    sorted_p7 = [pr7list[i] for i,x in e1]
    sorted_p8 = [pr8list[i] for i,x in e1]
    sorted_p9 = [pr9list[i] for i,x in e1]
    sorted_p10 = [pr10list[i] for i,x in e1]

    sorted_doc = [i for i,x in e1]

    for i, (w, x, y, z, ) in e1:
        #lables for top topics for each doc
        doc_t1.append(top3[w])
        #makes y-axis, list of documents in order
        document_list.append(doc_names[i] + " (" + str(i) + ")")
     
    left_c3 = np.add(sorted_p1, sorted_p2).tolist()
    left_c4 = np.add(left_c3, sorted_p3).tolist()
    left_c5 = np.add(left_c4, sorted_p4).tolist()
    left_c6 = np.add(left_c5, sorted_p5).tolist()
    left_c7 = np.add(left_c6, sorted_p6).tolist()
    left_c8 = np.add(left_c7, sorted_p7).tolist()
    left_c9 = np.add(left_c8, sorted_p8).tolist()
    left_c10 = np.add(left_c9, sorted_p9).tolist()

  # another function prepares data to be plotted, sorting list
    
    #creating colormap
    cmap = plt.colormaps['Paired']

    #creating colors for each topic doc list
    color1 = [cmap.colors[topic] for topic in sorted_t1]
    color2 = [cmap.colors[topic] for topic in sorted_t2]
    color3 = [cmap.colors[topic] for topic in sorted_t3]
    color4 = [cmap.colors[topic] for topic in sorted_t4]
    color5 = [cmap.colors[topic] for topic in sorted_t5]
    color6 = [cmap.colors[topic] for topic in sorted_t6]
    color7 = [cmap.colors[topic] for topic in sorted_t7]
    color8 = [cmap.colors[topic] for topic in sorted_t8]
    color9 = [cmap.colors[topic] for topic in sorted_t9]
    color10 = [cmap.colors[topic] for topic in sorted_t10]

    #plotting each bar chart
    bar1 = plt.barh(documents, sorted_p1, tick_label = ["%s" % i for i in document_list], color = color1)
    plt.bar_label(bar1, labels=doc_t1, label_type='center')
    bar2 = plt.barh(documents, sorted_p2, left=sorted_p1, color = color2)
    bar3 = plt.barh(documents, sorted_p3, left=left_c3, color = color3)
    
    bar4 = plt.barh(documents, sorted_p4, left=left_c4, color = color4)
    bar5 = plt.barh(documents, sorted_p5, left=left_c5, color = color5)
    bar6 = plt.barh(documents, sorted_p6, left=left_c6, color = color6)
    bar7 = plt.barh(documents, sorted_p7, left=left_c7, color = color7)
    bar8 = plt.barh(documents, sorted_p8, left=left_c8, color = color8)
    bar9 = plt.barh(documents, sorted_p9, left=left_c9, color = color9)
    bar10 = plt.barh(documents, sorted_p10, left=left_c10, color = color10)

    #plotting invisible bar chart to create color legend
    barcolors = plt.barh(list(range(topic_count)), [0]*topic_count, color = cmap.colors[:topic_count])
    
    #creating topic legend
    plt.legend(barcolors, top3, loc="upper right")
    
    #labelling bar chart
    plt.title("Documents Represented by Topics")
    plt.xlabel("Probabilities of Topics Present in Documents")
    plt.ylabel("Documents")
    plt.gca().invert_yaxis()