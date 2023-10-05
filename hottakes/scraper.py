# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
from tqdm import tqdm

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
    comment_div = soup.find("div", id="comment_wrap")
    top_comment_div = comment_div.find(class_="ppcont")
    top_comment_text_div = top_comment_div.find(class_="comtext")
    top_comment = top_comment_text_div.get_text(strip=True)

    dict = {
        "URL": url,
        "Title": title,
        "Description": description,
        "Article Text": text,
        "Top Comment Text": top_comment,
    }
    return dict


# %%
articles = []
for url in filtered_urls[0:3]:
    details = get_article_details(url)
    articles.append(details)

df = pd.DataFrame(articles)
print(df)
# %%
articles
# %%
years = list(range(2023, 1999, -1))
# years = list(range(2023, 2021, -1))
years

months = list(range(12, 0, -1))
# months = list(range(12, 4, -1))
months
# %%
article_urls = []
for year in tqdm(years, desc="Years"):
    for month in months:
        article_urls.append(get_article_urls(0, year, month))
        time.sleep(1)
        # print(year)
        # print(month)

len(article_urls)
# %%
article_urls = [item for sublist in article_urls for item in sublist]

# %%
article_urls = list(set(article_urls))
article_urls
# %%
