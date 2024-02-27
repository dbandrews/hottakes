import os
from datetime import datetime
import time

import pandas as pd
import numpy as np
from playwright.sync_api import Playwright, sync_playwright, expect
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from hottakes.scraper import get_article_details, get_comment_votes, get_article_urls
from hottakes.bot.inference import generate_comments
import json

USERAGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.81"


class CommentBot:
    """
    Comment on articles, that haven't already been commented on on Pinkbike.

    Log comments to Azure blob storage.
    """

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

    def comment(self, url: str, comment: str) -> str:
        """
        Post a comment on an article - emit the comment id for logging
        """
        self.page.goto(url)
        self.page.get_by_label("Post a Comment").click()
        self.page.get_by_label("Post a Comment").fill(comment)
        self.page.get_by_role("button", name="Submit").click()
        time.sleep(2)
        article_comments = get_comment_votes(url)
        comment_id = pd.DataFrame(article_comments).query("username==@self.username").comment_id.iloc[0]
        print("**************************************")
        print(f"Commented on {url}: ")
        print(f"{comment}")
        print("**************************************")
        return comment_id

    def get_article_candidates(self) -> list[str]:
        """
        Get articles in current month, that haven't been commented on yet
        """
        articles = get_article_urls(year=datetime.now().year, month=datetime.now().month, catid=0)
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
        Filter candidates to remove video and photo articles
        """
        return [candidate for candidate in candidates if "video" not in candidate and "photo" not in candidate]

    def get_candidate_article_texts(self, candidates: list[str]) -> list[str]:
        """
        Get article texts for candidates.
        """
        article_details = [get_article_details(candidate) for candidate in candidates]
        article_title_and_text = [f"{article['title']} {article['article_text']}" for article in article_details]
        return article_title_and_text

    def run(self, num_comments_desired: int = 3) -> None:
        # Commenton most recent articles
        candidate_articles = self.get_article_candidates()[:num_comments_desired]
        candidate_article_texts = self.get_candidate_article_texts(candidate_articles)
        comments = generate_comments(candidate_article_texts)
        self.login()
        for article, comment in zip(candidate_articles, comments):
            if self.problem_comment_check(comment):
                print("Problematic comment detected, skipping")
                print(f"Comment: \n{comment}")
                continue
            comment_id = self.comment(article, comment)
            self.log_comment(comment_id, url=article, comment=comment)

    @staticmethod
    def problem_comment_check(comment: str) -> bool:
        """
        Check if a comment is problematic, and has generation artifacts
        """
        # Check if comment contains ###
        # Check if comment says "funniest"
        # Check if comment contains ``` or code
        if "###" in comment:
            return True
        if "funniest" in comment:
            return True
        if "```" in comment:
            return True

        return False

    def log_comment(
        self,
        comment_id: str,
        url: str,
        comment: str,
    ) -> None:
        """
        Log the comment to Azure blob storage
        """
        comment_log = {
            "comment_id": comment_id,
            "url": url,
            "comment": comment,
            "datetime": datetime.now().isoformat(),
        }

        blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONN_STR"))
        container_name = "hottakes"
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(f"{comment_id}.json")
        blob_client.upload_blob(json.dumps(comment_log))


if __name__ == "__main__":
    load_dotenv(".env")
    comment_bot = CommentBot(username=os.getenv("PINKBIKE_USER"), password=os.getenv("PINKBIKE_PASS"))
    comment_bot.run()
