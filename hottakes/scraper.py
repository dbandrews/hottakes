# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from joblib import Parallel, delayed

# %%
years = list(range(2023, 1999, -1))
# years = list(range(2023, 2021, -1))
years

months = list(range(12, 0, -1))
# months = list(range(12, 4, -1))
months

sleep_length = 0.1


# %%
def get_article_urls(catid, year, month):
    base_pink_bike_url = "https://www.pinkbike.com/news/archive/?"

    url = f"{base_pink_bike_url}catid={catid}&year={year}&month={month}"
    # print(url)

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    article_url_list = []
    for element in soup.find_all(class_="news-style1"):
        a_tag = element.find("a")
        if a_tag and "href" in a_tag.attrs:
            href = a_tag["href"]
            # href_list.append(href)
        for ele in element.find_all("a"):
            href = ele.get("href")
            if href:
                article_url_list.append(href)

    # Regular expression pattern to match URLs
    # starting with "https://www.pinkbike.com/news/"
    # ending with "html"
    pattern = r"^https://www\.pinkbike\.com/news/.*\.html$"

    # Use list comprehension to filter URLs that match the pattern
    filtered_urls = [url for url in article_url_list if re.match(pattern, url)]

    return filtered_urls


# %%
filtered_urls = get_article_urls(0, 2023, 10)
print(len(filtered_urls))
print(filtered_urls)
# %%


def get_article_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string

    # Find the <meta> element with name="description"
    meta_description = soup.find("meta", attrs={"name": "description"})

    # Extract the content attribute from the <meta> element
    if meta_description:
        description = meta_description.get("content")

    # Find an element and extract its text using .get_text()
    blog_section_inside = soup.find(class_="blog-section-inside")
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
article_urls = []
for year in tqdm(years, desc="Years"):
    for month in months:
        article_urls.append(get_article_urls(0, year, month))
        time.sleep(sleep_length)
        # print(year)
        # print(month)

len(article_urls)
# %%
# what does this do?
article_urls = [item for sublist in article_urls for item in sublist]

# %%
# remove duplicates by converting to a set and back to a list
article_urls = list(set(article_urls))
article_urls
# %%
df = pd.DataFrame(article_urls)
# df.to_csv("article_urls.csv")
# %%
# Define the function you want to parallelize
# def process_item(item):
#     # Your processing logic for each item goes here
#     result = item * 2
#     return result

# # Create a list of items to process
# items = [1, 2, 3, 4, 5]

# # Use joblib to parallelize the processing of items
# %%
results = Parallel(n_jobs=-1)(delayed(get_article_details)(item) for item in article_urls[0:100])

# # Print the results
# print(results)

# %%
article_urls = pd.read_csv("article_urls.csv", index_col=None, names=["url"], header=0).url.to_list()
article_urls

# %%
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

df = pd.DataFrame(articles)

# %%
df.to_csv("article_details.csv")
print(df)
# %%
for k, v in get_article_details(article_urls[0]).items():
    print(k, v)
    print(type(k))
    print(type(v))
