# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

base_pink_bike_url = "https://www.pinkbike.com/news/archive/?"
catid = 0  # catid 0 is for all categories
year = 2023
month = 10
test_url = f"{base_pink_bike_url}catid={catid}&year={year}&month={month}"

response = requests.get(base_pink_bike_url)

soup = BeautifulSoup(response.text, "html.parser")

# %%
href_list = []
for element in soup.find_all(class_="news-style1"):
    a_tag = element.find("a")
    if a_tag and "href" in a_tag.attrs:
        href = a_tag["href"]
        # href_list.append(href)
    for ele in element.find_all("a"):
        href = ele.get("href")
        if href:
            href_list.append(href)
# Print the list of href attributes
print(href_list)

# %%
# Regular expression pattern to match URLs starting with "https://www.pinkbike.com/news/"
pattern = r"^https://www\.pinkbike\.com/news/.*\.html$"

# Use list comprehension to filter URLs that match the pattern
filtered_urls = [url for url in href_list if re.match(pattern, url)]

# Print the filtered list of URLs
print(filtered_urls)
# %%
art = requests.get(filtered_urls[1])
s = BeautifulSoup(art.text, "html.parser")
s
title = s.title.string
title
# %%
# Find the <meta> element with name="description"
meta_description = s.find("meta", attrs={"name": "description"})

# Extract the content attribute from the <meta> element
if meta_description:
    description = meta_description.get("content")
    print(description)
else:
    print("No meta description found.")
# %%
text_list = []
# Find an element and extract its text using .get_text()
blog_section_inside = s.find(class_="blog-section-inside")
for child in blog_section_inside.find_all(class_="media-media-width"):
    child.decompose()
text = blog_section_inside.get_text()
if text:
    text_list.append(text)
# Print the list of href attributes
print(text)

# %%
print(title)
print(description)
print(text)

# %%
