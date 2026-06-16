"""Tests for linkedin.py — ScrapeCreators LinkedIn organic-post search.

Mirrors test_instagram_sc.py / the SC-source test style. Covers parsing of the
/v1/linkedin/search/posts response, the date_posted=last-month request param,
the no-token short-circuit, env availability, the opt-in pipeline gate, and the
microblog normalization path.
"""

import unittest
from unittest.mock import patch

from lib import env, linkedin, normalize, pipeline


def _sc_payload():
    """A realistic /v1/linkedin/search/posts response (two posts)."""
    return {
        "success": True,
        "credits_remaining": 98,
        "query": "tail spend procurement",
        "posts": [
            {
                "url": "https://www.linkedin.com/posts/simfoni_procurement-tailspend-activity-7462427548824629248-Jj7c",
                "datePublished": "2026-05-19T09:02:52.809Z",
                "description": "Tail spend is often treated as procurement noise, yet it represents thousands of fragmented suppliers.",
                "author": {"name": "Simfoni", "url": "https://www.linkedin.com/company/simfoni", "followers": 43608},
                "likeCount": 9,
                "commentCount": 2,
                "comments": [],
            },
            {
                "url": "https://www.linkedin.com/posts/janedoe_procurement-activity-7462000000000000001-AbCd",
                "datePublished": "2026-05-25T12:00:00.000Z",
                "description": "Maverick spend thrives when the compliant path is slower than the shortcut.",
                "author": {"name": "Jane Doe", "url": "https://www.linkedin.com/in/janedoe"},
                "likeCount": 120,
                "commentCount": 14,
            },
        ],
        "cursor": None,
    }


class TestLinkedInDepthConfig(unittest.TestCase):
    def test_all_depths_exist(self):
        for depth in ("quick", "default", "deep"):
            self.assertIn(depth, linkedin.DEPTH_CONFIG)


class TestLinkedInSearch(unittest.TestCase):
    def test_no_token_short_circuits(self):
        from lib import http as http_module
        with patch.object(http_module, "get") as mock_get:
            result = linkedin.search_linkedin(
                "tail spend procurement", "2026-05-01", "2026-05-31",
                depth="default", token=None,
            )
            mock_get.assert_not_called()
            self.assertIn("error", result)
            self.assertIn("SCRAPECREATORS_API_KEY", result["error"])

    def test_sends_last_month_filter_and_query(self):
        from lib import http as http_module
        with patch.object(http_module, "get", return_value=_sc_payload()) as mock_get:
            linkedin.search_linkedin(
                "tail spend procurement", "2026-05-01", "2026-05-31",
                depth="quick", token="fake-token",
            )
            params = mock_get.call_args_list[0].kwargs["params"]
            self.assertEqual(params["date_posted"], "last-month")
            self.assertIn("query", params)
            url = mock_get.call_args_list[0].args[0]
            self.assertTrue(url.endswith("/linkedin/search/posts"))

    def test_parses_and_sorts_by_likes(self):
        from lib import http as http_module
        with patch.object(http_module, "get", return_value=_sc_payload()):
            result = linkedin.search_linkedin(
                "tail spend procurement", "2026-05-01", "2026-05-31",
                depth="quick", token="fake-token",
            )
        items = result["items"]
        self.assertEqual(len(items), 2)
        # Sorted by likes desc: Jane Doe (120) before Simfoni (9)
        self.assertEqual(items[0]["display_name"], "Jane Doe")
        self.assertEqual(items[0]["engagement"]["likes"], 120)
        self.assertEqual(items[0]["engagement"]["comments"], 14)
        # Stable id derived from the activity number in the URL
        self.assertEqual(items[1]["id"], "LI7462427548824629248")
        self.assertEqual(items[1]["text"][:9], "Tail spen")
        self.assertEqual(items[1]["date"], "2026-05-19")

    def test_filters_out_of_range_dates(self):
        from lib import http as http_module
        payload = _sc_payload()
        payload["posts"][0]["datePublished"] = "2026-01-01T00:00:00.000Z"  # out of window
        with patch.object(http_module, "get", return_value=payload):
            result = linkedin.search_linkedin(
                "tail spend procurement", "2026-05-01", "2026-05-31",
                depth="quick", token="fake-token",
            )
        # Only the in-range Jane Doe post survives
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["display_name"], "Jane Doe")


class TestLinkedInEnvAvailability(unittest.TestCase):
    def test_available_with_sc_key(self):
        self.assertTrue(env.is_linkedin_available({"SCRAPECREATORS_API_KEY": "k"}))

    def test_unavailable_without_sc_key(self):
        self.assertFalse(env.is_linkedin_available({}))

    def test_token_is_sc_key(self):
        self.assertEqual(env.get_linkedin_token({"SCRAPECREATORS_API_KEY": "k"}), "k")


class TestLinkedInPipelineGate(unittest.TestCase):
    """LinkedIn is OPT-IN: present only when 'linkedin' is in INCLUDE_SOURCES
    or the per-run requested sources, even with an SC key configured."""

    def test_off_by_default_with_sc_key(self):
        avail = pipeline.available_sources({"SCRAPECREATORS_API_KEY": "k"})
        self.assertNotIn("linkedin", avail)

    def test_on_via_include_sources(self):
        avail = pipeline.available_sources(
            {"SCRAPECREATORS_API_KEY": "k", "INCLUDE_SOURCES": "linkedin"}
        )
        self.assertIn("linkedin", avail)

    def test_on_via_requested_sources(self):
        avail = pipeline.available_sources(
            {"SCRAPECREATORS_API_KEY": "k"}, requested_sources=["linkedin"]
        )
        self.assertIn("linkedin", avail)

    def test_off_without_key_even_if_requested(self):
        avail = pipeline.available_sources({}, requested_sources=["linkedin"])
        self.assertNotIn("linkedin", avail)


class TestLinkedInNormalize(unittest.TestCase):
    def test_normalizes_via_microblog(self):
        raw = [{
            "id": "LI123",
            "handle": "Simfoni",
            "display_name": "Simfoni",
            "text": "Tail spend management for the enterprise.",
            "url": "https://www.linkedin.com/posts/simfoni-activity-123",
            "date": "2026-05-19",
            "engagement": {"likes": 9, "comments": 2},
            "relevance": 0.8,
            "why_relevant": "LinkedIn: Simfoni",
        }]
        items = normalize.normalize_source_items("linkedin", raw, "2026-05-01", "2026-05-31")
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.source, "linkedin")
        # Author is a plain name, not an @handle
        self.assertEqual(item.author, "Simfoni")
        self.assertEqual(item.engagement.get("likes"), 9)


if __name__ == "__main__":
    unittest.main()
