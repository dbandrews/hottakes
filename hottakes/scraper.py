# %%
import re
import time
import json

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed
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


# %%
def get_comment_votes(url):
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all div elements with a class name starting with "cmcont"
        divs = soup.find_zall("div", class_="cmcont")

        # Iterate through the found div elements
        results = []
        for div in divs:
            result = {}
            result["comment_id"] = div.get("id")
            result["url"] = url
            result["comment_text"] = div.find(class_="comtext").get_text(strip=True)
            # Find the child div with class "pcp" within the current div element
            pcp_div = div.find("div", class_="pcp")

            # Find the span with class "pcu" within the pcp_div
            pcu_span = pcp_div.find("span", class_="pcu") if pcp_div else None

            # Get the text content of the pcu_span, which is the value of "pcu"
            pcu_value = pcu_span.get_text(strip=True) if pcu_span else None

            # Find the span with class "pcd" within the pcp_div
            pcd_span = pcp_div.find("span", class_="pcd") if pcp_div else None

            # Get the text content of the pcd_span, which is the value of "pcd"
            pcd_value = pcd_span.get_text(strip=True) if pcd_span else None

            # Print the extracted values
            result["pcu_value"] = pcu_value
            result["pcd_value"] = pcd_value
            results.append(result)
        return results


if __name__ == "__main__":
    # Settings
    years = list(range(2023, 1999, -1))
    months = list(range(12, 0, -1))
    sleep_length = 0.1

    # _______________Article Links _______________
    article_urls = []
    for year in tqdm(years, desc="Years"):
        for month in months:
            article_urls.append(get_article_urls(0, year, month))
            time.sleep(sleep_length)

    pd.DataFrame(article_urls).to_csv("article_urls.csv")
    article_urls = pd.read_csv("article_urls.csv", index_col=None, names=["url"], header=0).url.to_list()

    # # _______________Article Details_______________
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
    df_details.to_csv("article_details.csv", index=False)

    # _______________Article Comments_______________
    comments = []
    n_jobs = 8
    article_urls = pd.read_csv("article_urls.csv", index_col=None, names=["url"], header=0).url.to_list()
    url_chunks = np.array_split(article_urls, n_jobs)

    def get_comment_votes_parallel(url_chunk, job_num, sleep_length=1):
        details = []
        for url in tqdm(url_chunk, desc=f"Job {job_num}", position=job_num, leave=False):
            try:
                details.append(get_comment_votes(url))
            except Exception as e:
                print(e)
            time.sleep(sleep_length)
        return details

    results = Parallel(n_jobs=n_jobs)(
        delayed(get_comment_votes_parallel)(url_chunk, job_num) for job_num, url_chunk in enumerate(url_chunks)
    )
    # Remove empty lists from chunks
    results = [result for job_result in results for result in job_result if result]
    # Flatten list of lists
    results = [result for job_result in results for result in job_result]

    df_comments = pd.DataFrame(results)
    df_comments.columns = df_comments.columns.str.lower().str.replace(" ", "_")
    df_comments.to_csv("article_comments.csv", index=False)
