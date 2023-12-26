#provides functions for modifying directory
import os
#provides provides regular expression support
import re
#wikipedia module that contains all the articles data
import wikipedia as wp
#shutil is a module that allows us to remove the data file
import shutil
#creates an in-memory file-like object
from io import StringIO
#imports sqlite module for database
import sqlite3 as sql
import pandas as pd

conn = sql.connect('article_db.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS articles (article_id integer, article_name text, article_text text, article_link text)')
conn.commit()


def checktable(a_name):
    main = wp.page(a_name,auto_suggest=False)
    main_page_id = main.pageid
    global tablename
    tablename = 'article_' + main_page_id
    #c.execute("DROP TABLE IF EXISTS " + tablename)
    listOfTables = c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=? ", (tablename, )).fetchall()
    if listOfTables != [] :
        print('Table exists')
        c.execute("SELECT article_name, article_id, article_text, article_link FROM " + tablename)
        return c.fetchall()
    else:
        c.execute('CREATE TABLE IF NOT EXISTS ' + tablename + ' (article_id integer, article_name text, article_text text, article_link text)')
        conn.commit()
        return download_articles(a_name)
        
def add_data(title, id, mainid, text, link):
    tablename = 'article_' + mainid   
    c.execute('INSERT INTO ' + tablename + ' (article_id, article_name, article_text, article_link) VALUES (?,?,?,?)', (id, title, text, link))
    conn.commit()

def download_articles(article_name):
    f_links = StringIO()
    # process the data using Python code
    #import pdb; pdb.set_trace()
    # this is the name of the Wikipedia page
    main_page = article_name
    #main_page = "Turing_Award"
    #main_page = "The_Matrix"
    
    # fetch the webpage
    main = wp.page(main_page,auto_suggest=False)
    main_page_id = main.pageid
    artdir = "data/" + main_page_id
    os.makedirs(artdir, exist_ok=True)
    
    
    # get page id from mainpage here
    print("== Downloading %s: %d links" % (main_page,len(main.links)))
    print("Downloading to 'data' folder")
    print("Remove or rename 'data' folder to download new dataset")
    
    #deletes the previous data folder
    #shutil.rmtree("/Users/misha/Downloads/homework04/data")
    
    #deletes the previous articles text
    #os.remove("/Users/misha/Downloads/homework04/articles.txt")
    
    # make a data/ directory, if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # make a articles txt file, if it doesn't exist
    os.makedirs("articles", exist_ok=True)
    
    # add page, and all links on page to the list of pages to download
    links = [main_page] + main.links

    for link in links:
        # skip pages that are "List of" or "Category" pages
        if link.startswith("List"): continue
        
        #with open("articles.txt", "a") as a_links:
        #    print(link +"\n", file = a_links)
        
        # try to download the page
        try:
            page = wp.page(link,auto_suggest=False,preload=False)
        except wp.exceptions.PageError:
            print("    page not found, skipping")
            continue
        except wp.exceptions.DisambiguationError:
            print("    ambiguous name, skipping")
            continue
        except:
            print("    unexpected error, skipping")
            continue
        
        # check if we already downloaded the page, and skip if it exists
        pageid = page.pageid
        filename = "%s/%s.txt" % (artdir, pageid)
        #if os.path.exists(filename):
        #    print("    page previously saved, skipping")
        #   continue

        # get the title and text from the page
        title = page.title
        text = page.content
        # remove non-alphabetic characters from text
        clean_text = re.sub('[^A-Za-z]+', ' ', text)
        a_link = "https://en.wikipedia.org/w/index.php?curid=" + pageid
        add_data(link, pageid, main_page_id, clean_text, a_link)
        # save article as text file
        with open(filename,'w') as f:
            f.write("title: %s\n" % title)
            f.write("id: %s\n" % pageid)
            f.write(clean_text)
    
    c.execute("SELECT article_name, article_id, article_text, article_link FROM article_" + main_page_id)
    return c.fetchall()