# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from joblib import Parallel, delayed


def get_article_urls(year: int, month: int, catid: int = 0):
    base_pink_bike_url = "https://www.pinkbike.com/news/archive/?"

    url = f"{base_pink_bike_url}catid={catid}&year={year}&month={month}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    article_url_list = []
    for element in soup.find_all(class_="news-style1"):
        a_tag = element.find("a")
        if a_tag and "href" in a_tag.attrs:
            href = a_tag["href"]
        for ele in element.find_all("a"):
            href = ele.get("href")
            if href:
                article_url_list.append(href)

    # Regular expression pattern to match URLs
    # starting with "https://www.pinkbike.com/news/"
    # ending with "html", otherwise target pages don't have same
    # structure
    pattern = r"^https://www\.pinkbike\.com/news/.*\.html$"

    # Use list comprehension to filter URLs that match the pattern
    filtered_urls = [url for url in article_url_list if re.match(pattern, url)]

    return filtered_urls


# %%
def get_article_details(url):
    try:
        response = requests.get(url)
    except requests.exceptions.ConnectTimeout:
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    if soup.title:
        title = soup.title.string
    else:
        title = "N/A"

    # Find the <meta> element with name="description"
    meta_description = soup.find("meta", attrs={"name": "description"})

    # Extract the content attribute from the <meta> element
    if meta_description:
        description = meta_description.get("content")
    else:
        description = "NA"

    # Find an element and extract its text using .get_text()
    blog_section_inside = soup.find(class_="blog-section-inside")
    if blog_section_inside is None:
        text = "N/A"
    else:
        for child in blog_section_inside.find_all(class_="media-media-width"):
            child.decompose()
        text = blog_section_inside.get_text()

    # Find the comments
    try:
        comment_div = soup.find("div", id="comment_wrap")
        top_comment_div = comment_div.find(class_="ppcont")
        top_comment_text_div = top_comment_div.find(class_="comtext")
        top_comment = top_comment_text_div.get_text(strip=True)
    except:
        top_comment = "N/A"

    dict = {
        "URL": url,
        "Title": str(title),
        "Description": description,
        "Article Text": text,
        "Top Comment Text": top_comment,
    }
    return dict


if __name__ == "__main__":
    # Settings
    years = list(range(2023, 1999, -1))
    months = list(range(12, 0, -1))
    sleep_length = 0.1

    # Article Links
    article_urls = []
    for year in tqdm(years, desc="Years"):
        for month in months:
            article_urls.append(get_article_urls(0, year, month))
            time.sleep(sleep_length)

    pd.DataFrame(article_urls).to_csv("article_urls.csv")
    article_urls = pd.read_csv("article_urls.csv", index_col=None, names=["url"], header=0).url.to_list()

    # Article Details
    articles = []
    n_jobs = 8
    url_chunks = np.array_split(article_urls, n_jobs)

    def get_article_details_parallel(url_chunk, job_num, sleep_length=1):
        details = []
        for url in tqdm(url_chunk, desc=f"Job {job_num}", position=job_num, leave=False):
            details.append(get_article_details(url))
            time.sleep(sleep_length)
        return details

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_article_details_parallel)(url_chunk, job_num) for job_num, url_chunk in enumerate(url_chunks)
    )
    results = [result for job_result in results for result in job_result]
    df_details = pd.DataFrame(results)
    df_details.columns = df_details.columns.str.lower().str.replace(" ", "_")
    df_details = (
        df_details.replace("N/A", np.nan)
        .dropna()
        .assign(title_article_text=lambda _df: (_df.title + ": " + _df.article_text))
    )
    df_details.to_json("article_details.json", orient="records")
