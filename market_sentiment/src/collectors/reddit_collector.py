"""
Reddit Collector Module

Collects posts and comments from Reddit using the PRAW library.
Monitors finance-related subreddits for ticker mentions.

Features:
- Search for ticker mentions across multiple subreddits
- Collect post metadata (title, body, score, comments)
- Deduplication by post_id
- Caching to JSON files
- Rate limit handling
- Top comments collection per post

DISCLAIMER: Reddit data is public and for educational purposes only.
May not represent all investor sentiment (selection bias).
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import praw
from loguru import logger
from praw.models import Submission


class RedditCollector:
    """
    Collects Reddit posts and comments for sentiment analysis.
    
    Uses PRAW (Python Reddit API Wrapper) in read-only mode.
    No login required for public subreddit access.
    
    Attributes:
        client_id: Reddit API client ID
        client_secret: Reddit API client secret
        user_agent: User agent string for API requests
        cache_dir: Directory for caching raw data
        reddit: PRAW Reddit instance
    
    Example:
        >>> collector = RedditCollector(client_id, secret, agent, cache)
        >>> posts = collector.search_ticker_mentions("AAPL", ["wallstreetbets"])
        >>> print(len(posts))
        150
    """
    
    # Maximum number of posts to fetch per subreddit per ticker
    MAX_POSTS_PER_SUBREDDIT = 100
    
    # Maximum comments to fetch per post
    MAX_COMMENTS_PER_POST = 5
    
    # Rate limit delay between API calls (seconds)
    RATE_LIMIT_DELAY = 1.0
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        cache_dir: str,
        rate_limit_delay: float = 1.0
    ) -> None:
        """
        Initialize the Reddit collector.
        
        Args:
            client_id: Reddit API client ID from https://www.reddit.com/prefs/apps
            client_secret: Reddit API client secret
            user_agent: User agent string (format: app_name/version by username)
            cache_dir: Directory path for caching JSON files
            rate_limit_delay: Seconds to wait between API calls
        
        Note:
            Reddit API requires creating a "script" type application.
            The application is free but requires registration.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir) / "reddit"
        self.rate_limit_delay = rate_limit_delay
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PRAW in read-only mode
        # No login required for public subreddits
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                check_for_async=False  # Disable async mode for simplicity
            )
            
            # Verify read-only access
            logger.info(f"Reddit read-only mode: {self.reddit.read_only}")
            logger.info("RedditCollector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def _get_cache_path(self, ticker: str, date: Optional[str] = None) -> Path:
        """Get the cache file path for a ticker."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        return self.cache_dir / f"{ticker.upper()}_{date}.json"
    
    def _is_cache_valid(self, cache_path: Path, max_age_hours: int = 24) -> bool:
        """
        Check if cached file exists and is recent enough.
        
        Args:
            cache_path: Path to the cached file
            max_age_hours: Maximum age in hours
            
        Returns:
            True if cache exists and is recent
        """
        if not cache_path.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mtime
        
        return age < timedelta(hours=max_age_hours)
    
    def _load_from_cache(self, ticker: str) -> Optional[list[dict]]:
        """
        Load posts from cache if available.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of post dictionaries, or None if cache invalid
        """
        cache_path = self._get_cache_path(ticker)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.debug(f"Loaded {ticker} posts from cache ({len(data)} posts)")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
        
        return None
    
    def _save_to_cache(self, ticker: str, posts: list[dict]) -> None:
        """
        Save posts to cache.
        
        Args:
            ticker: Stock ticker symbol
            posts: List of post dictionaries
        """
        cache_path = self._get_cache_path(ticker)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved {ticker} posts to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {ticker}: {e}")
    
    def _extract_post_data(self, submission: Submission, ticker: str) -> dict:
        """
        Extract relevant data from a Reddit submission.
        
        Args:
            submission: PRAW Submission object
            ticker: Stock ticker symbol being searched
            
        Returns:
            Dictionary with post data
        """
        # Combine title and selftext for sentiment analysis
        # This gives full context for the post
        combined_text = f"{submission.title}"
        if submission.selftext:
            combined_text += f" {submission.selftext}"
        
        # Clean up the text
        combined_text = combined_text.strip()
        
        # Convert UTC timestamp to datetime
        created_at = datetime.fromtimestamp(submission.created_utc)
        
        post_data = {
            "post_id": submission.id,
            "ticker": ticker.upper(),
            "title": submission.title,
            "selftext": submission.selftext or "",
            "combined_text": combined_text,
            "score": submission.score,  # Upvotes - downvotes
            "upvote_ratio": submission.upvote_ratio,
            "num_comments": submission.num_comments,
            "subreddit": str(submission.subreddit),
            "url": f"https://reddit.com{submission.permalink}",
            "created_at": created_at.isoformat(),
            "fetched_at": datetime.now().isoformat(),
        }
        
        return post_data
    
    def _extract_comments(self, submission: Submission, limit: int = 5) -> list[dict]:
        """
        Extract top comments from a submission.
        
        Comments provide additional sentiment context.
        We get the top comments by score (upvotes).
        
        Args:
            submission: PRAW Submission object
            limit: Maximum number of comments to extract
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        
        try:
            # Replace "more comments" placeholders with actual comments
            # This may take time for posts with many comments
            submission.comments.replace_more(limit=0)
            
            # Get top comments sorted by score
            for i, comment in enumerate(submission.comments[:limit]):
                if comment.body and comment.body != "[deleted]":
                    comments.append({
                        "comment_id": comment.id,
                        "body": comment.body,
                        "score": comment.score,
                        "created_at": datetime.fromtimestamp(
                            comment.created_utc
                        ).isoformat(),
                    })
                    
        except Exception as e:
            logger.debug(f"Failed to fetch comments: {e}")
        
        return comments
    
    def search_ticker_mentions(
        self,
        ticker: str,
        subreddits: list[str],
        days_back: int = 30,
        max_posts: int = 100,
        use_cache: bool = True
    ) -> list[dict]:
        """
        Search for ticker mentions across specified subreddits.
        
        Search query: ticker OR $ticker
        Example: "AAPL OR $AAPL"
        
        This captures posts that mention the ticker either with or without
        the dollar sign prefix (common in finance discussions).
        
        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            subreddits: List of subreddit names to search
            days_back: Number of days to look back
            max_posts: Maximum posts to fetch per subreddit
            use_cache: Whether to use cached data
            
        Returns:
            List of dictionaries, each containing:
            - post_id: Unique Reddit post ID
            - ticker: Stock symbol
            - title: Post title
            - selftext: Post body text
            - combined_text: Title + body (for sentiment analysis)
            - score: Net upvotes
            - num_comments: Comment count
            - subreddit: Subreddit name
            - url: Reddit URL
            - created_at: Post timestamp
            - comments: List of top comments
            
        Note:
            Reddit API rate limit is 60 requests per minute.
            We add delays between requests to stay within limits.
        """
        ticker = ticker.upper()
        
        # Try to load from cache first
        if use_cache:
            cached_posts = self._load_from_cache(ticker)
            if cached_posts:
                logger.info(
                    f"Loaded {ticker} posts from cache ({len(cached_posts)} posts)"
                )
                return cached_posts
        
        logger.info(
            f"Searching for {ticker} mentions in {len(subreddits)} subreddits "
            f"(last {days_back} days)"
        )
        
        all_posts = []
        seen_ids = set()  # Track seen post_ids for deduplication
        
        # Calculate time filter
        time_filter = "month" if days_back <= 30 else "year"
        
        # Build search query
        # Search for ticker mentions with or without $ prefix
        query = f'"{ticker}" OR "${ticker}"'
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                logger.debug(f"Searching r/{subreddit_name} for {ticker}...")
                
                # Search for posts
                search_results = subreddit.search(
                    query,
                    sort="relevance",  # Get most relevant posts first
                    time_filter=time_filter,
                    limit=max_posts
                )
                
                posts_found = 0
                
                for submission in search_results:
                    # Deduplicate by post_id
                    if submission.id in seen_ids:
                        continue
                    
                    # Check if post is within the time window
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    if post_time < datetime.now() - timedelta(days=days_back):
                        continue
                    
                    seen_ids.add(submission.id)
                    
                    # Extract post data
                    post_data = self._extract_post_data(submission, ticker)
                    
                    # Extract top comments
                    post_data["comments"] = self._extract_comments(
                        submission, limit=self.MAX_COMMENTS_PER_POST
                    )
                    
                    all_posts.append(post_data)
                    posts_found += 1
                
                logger.info(
                    f"Found {posts_found} posts in r/{subreddit_name} for {ticker}"
                )
                
                # Rate limit delay between subreddits
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error searching r/{subreddit_name}: {e}")
                continue
        
        # Sort by score (upvotes) descending
        all_posts.sort(key=lambda x: x["score"], reverse=True)
        
        # Save to cache
        if all_posts:
            self._save_to_cache(ticker, all_posts)
        
        logger.info(
            f"Collected {len(all_posts)} total posts for {ticker} "
            f"(deduplicated from {len(seen_ids)} unique posts)"
        )
        
        return all_posts
    
    def get_daily_mention_count(
        self,
        posts: list[dict]
    ) -> pd.DataFrame:
        """
        Aggregate posts into daily mention counts.
        
        This measures "buzz volume" - how much a ticker is being discussed.
        Independent of sentiment, this shows attention/interest level.
        
        Args:
            posts: List of post dictionaries from search_ticker_mentions
            
        Returns:
            DataFrame with columns:
            - date: Trading date
            - ticker: Stock symbol
            - post_count: Number of posts that day
            - avg_score: Average upvotes per post
            - total_comments: Total comments across all posts
            - total_upvotes: Sum of all post scores
        """
        if not posts:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Parse dates
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        # Group by date and aggregate
        daily = df.groupby(["date", "ticker"]).agg(
            post_count=("post_id", "count"),
            avg_score=("score", "mean"),
            total_comments=("num_comments", "sum"),
            total_upvotes=("score", "sum"),
        ).reset_index()
        
        # Round avg_score
        daily["avg_score"] = daily["avg_score"].round(2)
        
        logger.info(
            f"Generated daily mention counts: {len(daily)} days, "
            f"{daily['post_count'].sum()} total posts"
        )
        
        return daily
    
    def get_recent_posts(
        self,
        ticker: str,
        subreddits: list[str],
        hours: int = 24,
        max_posts: int = 50
    ) -> list[dict]:
        """
        Get very recent posts for real-time monitoring.
        
        Useful for dashboard displays showing latest activity.
        
        Args:
            ticker: Stock ticker symbol
            subreddits: List of subreddit names
            hours: Hours to look back (default: 24)
            max_posts: Maximum posts to return
            
        Returns:
            List of recent post dictionaries
        """
        posts = self.search_ticker_mentions(
            ticker,
            subreddits,
            days_back=min(hours // 24 + 1, 7),  # Max 7 days
            max_posts=max_posts * 2,  # Get more, then filter
            use_cache=False  # Always fresh data for recent
        )
        
        # Filter to recent posts
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_posts = []
        for post in posts:
            post_time = datetime.fromisoformat(post["created_at"])
            if post_time >= cutoff:
                recent_posts.append(post)
        
        # Sort by creation time, most recent first
        recent_posts.sort(
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        return recent_posts[:max_posts]
    
    def collect_all_tickers(
        self,
        tickers: list[str],
        subreddits: list[str],
        days_back: int = 30,
        use_cache: bool = True
    ) -> dict[str, list[dict]]:
        """
        Collect posts for all tickers.
        
        Iterates through all tickers and collects posts.
        Adds delays between requests to avoid rate limiting.
        
        Args:
            tickers: List of ticker symbols
            subreddits: List of subreddit names
            days_back: Days to look back
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping ticker to list of posts:
            {"AAPL": [post1, post2, ...], "TSLA": [...]}
        """
        results = {}
        
        logger.info(
            f"Starting Reddit collection for {len(tickers)} tickers"
        )
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
            
            try:
                posts = self.search_ticker_mentions(
                    ticker,
                    subreddits,
                    days_back=days_back,
                    use_cache=use_cache
                )
                results[ticker] = posts
                
                # Log progress
                logger.info(
                    f"[{i}/{len(tickers)}] {ticker}: {len(posts)} posts collected"
                )
                
                # Rate limit delay between tickers
                if i < len(tickers):
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Failed to collect posts for {ticker}: {e}")
                results[ticker] = []
        
        total_posts = sum(len(p) for p in results.values())
        logger.info(
            f"Reddit collection complete: {total_posts} posts for {len(results)} tickers"
        )
        
        return results
    
    def get_trending_tickers(
        self,
        subreddits: list[str],
        min_mentions: int = 10,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Find trending tickers by mention frequency.
        
        Searches for $ prefixed stock mentions and counts occurrences.
        Useful for discovering new tickers to track.
        
        Args:
            subreddits: List of subreddit names
            min_mentions: Minimum mentions to be considered trending
            days_back: Days to look back
            
        Returns:
            DataFrame with ticker and mention count
        """
        import re
        from collections import Counter
        
        # Pattern to match $TICKER mentions
        ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        
        all_mentions = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get recent posts
                for submission in subreddit.new(limit=500):
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    if post_time < datetime.now() - timedelta(days=days_back):
                        break
                    
                    # Find ticker mentions in title and body
                    text = f"{submission.title} {submission.selftext or ''}"
                    mentions = ticker_pattern.findall(text)
                    all_mentions.extend(mentions)
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error scanning r/{subreddit_name}: {e}")
        
        # Count mentions
        mention_counts = Counter(all_mentions)
        
        # Filter to valid tickers with minimum mentions
        trending = [
            {"ticker": ticker, "mentions": count}
            for ticker, count in mention_counts.items()
            if count >= min_mentions
        ]
        
        df = pd.DataFrame(trending).sort_values("mentions", ascending=False)
        
        logger.info(f"Found {len(df)} trending tickers")
        
        return df.reset_index(drop=True)


if __name__ == "__main__":
    # Test the collector (requires valid credentials)
    import os
    
    print("Reddit Collector Test")
    print("=" * 50)
    
    # Check for credentials
    if not os.getenv("REDDIT_CLIENT_ID"):
        print("Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        print("to test the Reddit collector.")
        print("\nGet credentials at: https://www.reddit.com/prefs/apps")
    else:
        from src.config import settings
        
        collector = RedditCollector(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
            cache_dir=settings.cache_dir
        )
        
        # Test single ticker search
        print("\nTesting ticker search for AAPL...")
        posts = collector.search_ticker_mentions(
            "AAPL",
            ["wallstreetbets", "investing"],
            days_back=7,
            max_posts=10
        )
        
        print(f"\nFound {len(posts)} posts:")
        for post in posts[:5]:
            print(f"\n[{post['subreddit']}] {post['title'][:60]}...")
            print(f"  Score: {post['score']}, Comments: {post['num_comments']}")
