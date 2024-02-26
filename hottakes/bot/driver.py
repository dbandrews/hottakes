import os
from datetime import datetime
import time

import pandas as pd
import numpy as np
from playwright.sync_api import Playwright, sync_playwright, expect
from dotenv import load_dotenv

from hottakes.scraper import get_article_details, get_comment_votes, get_article_urls
from hottakes.bot.inference import generate_comments

USERAGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81"


class CommentBot:
    "Comment on articles, that haven't already been commented on"

    def __init__(self, username: str, password: str, headless: bool = False):
        self.username = username
        self.password = password
        self.headless = headless
        playwright = sync_playwright().start()
        self.browser = playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(user_agent=USERAGENT)

    def login(self):
        page = self.context.new_page()
        page.goto("https://www.pinkbike.com/")
        page.get_by_role("link", name="Log in").click()
        # Random delay+mvmt to avoid bot detection
        time.sleep(4)
        page.mouse.move(0, 100)
        page.mouse.move(100, 100)
        page.mouse.move(100, 0)
        page.mouse.move(0, 0)

        page.locator('input[name="username-login-loginlen"]').click()
        page.locator('input[name="username-login-loginlen"]').fill(self.username)
        page.locator('input[name="password-password-lt200"]').click()
        page.locator('input[name="password-password-lt200"]').fill(self.password)
        page.locator("#loginButton").click()
        self.page = page

    def comment(self, url: str, comment: str):
        self.page.goto(url)
        self.page.get_by_label("Post a Comment").click()
        self.page.get_by_label("Post a Comment").fill(comment)
        # page.get_by_role("button", name="Submit").click()
        print(f"Commented on {url}: ")
        print(f"{comment}")

    def get_article_candidates(self, num_candidates: int = 3) -> list[str]:
        """
        Get articles in current month, that haven't been commented on yet
        """
        articles = get_article_urls(year=datetime.now().year, month=datetime.now().month, catid=0)
        articles = np.random.choice(articles, size=num_candidates)
        article_comments = [get_comment_votes(article) for article in articles]
        article_comments = pd.DataFrame([comment for comments in article_comments for comment in comments])
        articles_commented_already = article_comments.query("username==@self.username")
        print(f"Commented articles found: {articles_commented_already.shape[0]}")
        print(f"Total articles found: {len(articles)}")
        non_commented_articles = [
            article for article in articles if article not in articles_commented_already.url.to_list()
        ]
        return self.filter_article_candidates(non_commented_articles)

    def filter_article_candidates(self, candidates: list[str]):
        """
        Filter candidates based on some criteria
        """
        return [candidate for candidate in candidates if "video" not in candidate and "photo" not in candidate]

    def get_candidate_article_texts(self, candidates: list[str]) -> list[str]:
        """
        Get article texts for candidates
        """
        article_details = [get_article_details(candidate) for candidate in candidates]
        article_title_and_text = [f"{article['title']} {article['article_text']}" for article in article_details]
        return article_title_and_text

    def run(self, num_comments_desired: int = 3) -> None:
        candidate_articles = list(np.random.choice(self.get_article_candidates(), size=3))
        candidate_article_texts = self.get_candidate_article_texts(candidate_articles)
        comments = generate_comments(candidate_article_texts)
        self.login()
        for article, comment in zip(candidate_articles, comments):
            self.comment(article, comment)


# def run(playwright: Playwright) -> None:
#     browser = playwright.chromium.launch(headless=False)
#     context = browser.new_context()
#     page = context.new_page()
#     page.goto("https://www.pinkbike.com/")
#     page.get_by_role("link", name="Log in").click()
#     page.locator('input[name="username-login-loginlen"]').click()
#     page.locator('input[name="username-login-loginlen"]').fill("joeyjoejoe")
#     page.locator('input[name="username-login-loginlen"]').press("Tab")
#     page.locator('input[name="password-password-lt200"]').click()
#     page.locator('input[name="password-password-lt200"]').fill("Eligible-Tribune-Destitute8")
#     page.locator("#loginButton").click()
#     page.get_by_role("link", name="News").click()
#     page.get_by_role("link", name="2023").click()
#     page.get_by_role("link", name="2022", exact=True).click()
#     page.get_by_role("link", name="2024", exact=True).click()
#     page.get_by_role(
#         "link", name="Finals Photo Epic: Red Bull Hardline Tasmania 2024 Finals Photo Epic: Red Bull Hardline Tasmania"
#     ).click()
#     page.get_by_label("Post a Comment").click()
#     page.get_by_label("Post a Comment").click()
#     page.get_by_label("Post a Comment").fill("Unreal riding!")
#     page.get_by_role("button", name="Submit").click()
#     page.goto("https://www.pinkbike.com/news/archive/?catid=0&year=2024&month=2")
#     page.get_by_role(
#         "link", name="Finals Photo Epic: Red Bull Hardline Tasmania 2024 Finals Photo Epic: Red Bull Hardline Tasmania"
#     ).click()
#     page.goto("https://www.pinkbike.com/news/archive/?catid=0&year=2024&month=2")

#     # ---------------------
#     context.close()
#     browser.close()


# with sync_playwright() as playwright:
#     run(playwright)

if __name__ == "__main__":
    load_dotenv(".env")
    comment_bot = CommentBot(username=os.getenv("PINKBIKE_USER"), password=os.getenv("PINKBIKE_PASS"))
    # comment_bot.login()
    # comment_bot.comment(
    #     url="https://www.pinkbike.com/news/slack-randoms-snaix-neurobike-swampfest-carnage-cannonball-whip-off-and-more.html",
    #     comment="Swampfest.....carnage!",
    # )
    comment_bot.run()
