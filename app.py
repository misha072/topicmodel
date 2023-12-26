from flask import Flask,render_template, request, url_for, redirect
from markupsafe import Markup
import os, re
import wikipedia as wp
import topic
import download
import sqlite3 as sql
import matplotlib.pyplot as plt
plt.rcParams['figure.autolayout'] = True
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def plot_to_img():
        topic.topic_chart()
        plt.savefig("static/topicchart.png")
        plt.savefig("static/topicchart.pdf")
        topic.doc_chart()
        plt.savefig("static/docchart.png")

app = Flask(__name__,template_folder="templates", static_folder="static")

@app.route("/")
def hello():
    import os
    print(os.getcwd())
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    a_name = request.form.get('data')
    articles = download.checktable(a_name)
    return render_template("articles.html", rows = articles, article_length = len(articles), table_len = (len(articles))/3)

@app.route('/topic')
def topics():
    global topics
    topics = topic.main()
    topics = Markup(topics)
    # Convert plot to image
    plot_to_img()

    topchart = Markup(' <img src = "static/topicchart.png"> ')
    docchart = Markup(' <img src = "static/docchart.png"> ')
    return render_template("topic.html", t=topics, top_img = topchart, doc_img = docchart)

@app.route('/text', methods=['POST'])
def text():
    global topics
    topics = Markup(topics)
    d_index = int(request.form.get('d_id'))
    topic_index = int(request.form.get('t_id'))
    article_text = topic.get_text(d_index)
    highlight_text = topic.highlight_word(d_index, article_text, topic_index)
    highlight_text = Markup(highlight_text)
    return render_template("text.html", t = topics, a_text = highlight_text)

if __name__ == '__main__':
    app.run(debug=True)