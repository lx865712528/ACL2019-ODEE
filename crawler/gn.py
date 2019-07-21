import base64
import json
import os
import pickle
import time
import urllib
from datetime import datetime

from bs4 import BeautifulSoup
from newsplease import NewsPlease


class MySimpleCrawler():

    @staticmethod
    def fetch_url(url, timeout=None):
        """
        Crawls the html content of the parameter url and returns the html
        :param url:
        :param timeout: in seconds, if None, the urllib default is used
        :return:
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,ru;q=0.6',
            # 'Accept-Encoding': 'gzip, deflate, br'
        }
        req = urllib.request.Request(url, None, headers)
        html = urllib.request.urlopen(req, data=None, timeout=timeout).read()

        return html


class CrawlArticleItem:
    def __init__(self, url, base64_id, title):
        self.article = NewsPlease.from_url(url)
        self.base64_id = base64_id
        self.true_title = title

    def __str__(self):
        ss = ""
        ss += str(self.article.date_download) + "\n"
        ss += str(self.article.date_publish) + "\n"
        ss += self.article.description + "\n"
        ss += self.article.language + "\n"
        ss += self.article.title + "\n"
        ss += self.article.source_domain + "\n"
        ss += self.article.text + "\n"
        ss += self.article.url + "\n"
        ss += self.true_title + "\n"
        ss += self.base64_id.decode("utf-8")
        return ss

    def to_dict(self):
        return {
            "date_download": str(self.article.date_download),
            "date_publish": str(self.article.date_publish),
            "description": self.article.description,
            "language": self.article.language,
            "title": self.article.title,
            "source_domain": self.article.source_domain,
            "text": self.article.text,
            "url": self.article.url,
            "gntitle": self.true_title,
            "base64_id": self.base64_id.decode("utf-8")
        }


class GNArticle:
    base_url = "https://news.google.com"

    def __init__(self, partial_url):
        self.url = GNArticle.base_url + partial_url[1:]
        time.sleep(1)
        html = MySimpleCrawler.fetch_url(self.url)
        html = BeautifulSoup(html, "html.parser")
        nodes = html.find_all("a", rel="nofollow")
        self.url = nodes[0].get("href")


class GNFlipper:
    def __init__(self, base_url):
        self.base_url = base_url
        self.cache = set()
        self.today = str(datetime.today().date()) + "-" + str(datetime.today().hour)
        print("This is %s" % self.today)

    def init_cache(self, cache_dir):
        if os.path.exists(cache_dir):
            with open(cache_dir, "rb") as f:
                self.cache = pickle.load(f)

    def log_page(self, page_id, base64_id):
        print("Meeting " + page_id + " " + base64_id.decode("utf-8"))
        self.cache.add(base64_id)

    def save_cache(self, cache_dir):
        with open(cache_dir, "wb") as f:
            pickle.dump(self.cache, f)

    def save_items(self, store_dir, no, items):
        root = os.path.join(store_dir, str(no + 1))
        os.makedirs(root)

        with open(os.path.join(root, "total.json"), "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

    def let_it_go(self, store_root, cache_dir):
        store_dir = os.path.join(store_root, self.today)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        print("Saving crawled news to %s" % store_dir)

        # self.init_cache(cache_dir)

        html = MySimpleCrawler.fetch_url(self.base_url)
        node_root = BeautifulSoup(html, 'html.parser')
        nodes = node_root.find_all("div", class_="xrnccd F6Welf R7GTQ keNKEd j7vNaf")

        d_count = 0
        g_count = 0

        for no, node in enumerate(nodes):
            items = []
            article_nodes = node.find_all("a", class_="ipQwMb Q7tWef")
            for article_node in article_nodes:
                href = article_node.get("href")
                title = article_node.span.string
                page_id = "%s %s" % (href, title)
                base64_id = base64.b64encode(title.encode())

                # if base64_id in self.cache:
                #     continue

                try:
                    article = GNArticle(href)
                    item = CrawlArticleItem(article.url, base64_id, title)
                    items.append(item.to_dict())
                    self.log_page(page_id, base64_id)
                except:
                    print("Error on %s" % page_id)

            if len(items) > 0:
                self.save_items(store_dir, no, items)
                d_count += len(items)
                g_count += 1
                time.sleep(5)
                print("Group saved!")
            else:
                print("Group skipped!")
        # self.save_cache(cache_dir)
        print("Done for today!")
        print("Crawled %d documents in %d groups." % (d_count, g_count))


if __name__ == "__main__":
    base_url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
    store_root = "/home/liuxiao/projects/schema/crawler/data/gnbusiness"
    cache_dir = "/home/liuxiao/projects/schema/crawler/data/gnbusiness.pickle"

    crawler = GNFlipper(base_url)
    crawler.let_it_go(store_root, cache_dir)
