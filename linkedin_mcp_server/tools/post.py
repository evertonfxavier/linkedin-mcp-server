"""
LinkedIn post/content search tools.

Enables searching for job opportunities and other content shared as posts
using boolean query operators.
"""

import logging
from typing import Any

from fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from linkedin_mcp_server.drivers.browser import (
    ensure_authenticated,
    get_or_create_browser,
)
from linkedin_mcp_server.error_handler import handle_tool_error
from linkedin_mcp_server.scraping import LinkedInExtractor

logger = logging.getLogger(__name__)


def register_post_tools(mcp: FastMCP) -> None:
    """Register all post-related tools with the MCP server."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search Posts",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def search_posts(
        keywords: str,
        ctx: Context,
        date_posted: str | None = None,
        sort_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for posts/content on LinkedIn.

        This is useful for finding job opportunities shared as posts, company
        announcements, and other content that matches specific search queries.
        Many recruiters and companies post job openings directly in their feed
        rather than using LinkedIn Jobs.

        Args:
            keywords: Search query with boolean operators support.
                Use AND, OR, and parentheses for complex queries.
                Examples:
                - "React" AND ("Pleno" OR "Mid" OR "PL")
                - ("Front" OR "Frontend") AND ("Pleno" OR "Mid")
                - ("Vue" OR "Vue.js") AND ("Junior" OR "JR" OR "Entry")
                - "Python" AND "Remote" AND ("Senior" OR "SR")
            ctx: FastMCP context for progress reporting
            date_posted: Filter by date posted. Options:
                - "past-24h" (last 24 hours)
                - "past-week" (last week)
                - "past-month" (last month)
                Default: None (no date filter)
            sort_by: Sort order. Options:
                - "date_posted" (most recent first)
                - "relevance" (default LinkedIn sorting)
                Default: None (LinkedIn default)

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text to extract individual posts and job opportunities.
        """
        try:
            await ensure_authenticated()

            logger.info(
                "Searching posts: keywords='%s', date_posted='%s', sort_by='%s'",
                keywords,
                date_posted,
                sort_by,
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting post search"
            )

            result = await extractor.search_posts(keywords, date_posted, sort_by)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "search_posts")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create Post",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def create_post(
        content: str,
        ctx: Context,
        visibility: str = "anyone",
    ) -> dict[str, Any]:
        """
        Create a new LinkedIn post.

        Use this to share content, updates, thoughts, or job-related information
        with your network.

        Args:
            content: The text content of your post. Can include:
                - Plain text
                - Hashtags (#tech #hiring #python)
                - Mentions (use @name but LinkedIn will autocomplete)
                - Emojis
                Maximum recommended: 3000 characters for engagement.
            ctx: FastMCP context for progress reporting
            visibility: Who can see your post. Options:
                - "anyone" (public - visible to everyone on LinkedIn)
                - "connections" (only your 1st-degree connections)
                Default: "anyone"

        Returns:
            Dict with status, message, and content_preview.
            Status can be: "success", "partial", or "error"
        """
        try:
            await ensure_authenticated()

            logger.info(
                "Creating post: visibility='%s', content_length=%d",
                visibility,
                len(content),
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Creating post"
            )

            result = await extractor.create_post(content, visibility)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "create_post")
