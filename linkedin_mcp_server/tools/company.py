"""
LinkedIn company profile scraping tools.

Uses innerText extraction for resilient company data capture
with configurable section selection.
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
from linkedin_mcp_server.scraping import LinkedInExtractor, parse_company_sections

logger = logging.getLogger(__name__)


def register_company_tools(mcp: FastMCP) -> None:
    """Register all company-related tools with the MCP server."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Company Profile",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_company_profile(
        company_name: str,
        ctx: Context,
        sections: str | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific company's LinkedIn profile.

        Args:
            company_name: LinkedIn company name (e.g., "docker", "anthropic", "microsoft")
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to scrape.
                The about page is always included.
                Available sections: posts, jobs
                Examples: "posts", "posts,jobs"
                Default (None) scrapes only the about page.

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text in each section.
        """
        try:
            await ensure_authenticated()

            fields, unknown = parse_company_sections(sections)

            logger.info(
                "Scraping company: %s (sections=%s)",
                company_name,
                sections,
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting company profile scrape"
            )

            result = await extractor.scrape_company(company_name, fields)

            if unknown:
                result["unknown_sections"] = unknown

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_company_profile")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Company Posts",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_company_posts(
        company_name: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Get recent posts from a company's LinkedIn feed.

        Args:
            company_name: LinkedIn company name (e.g., "docker", "anthropic", "microsoft")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text to extract individual posts.
        """
        try:
            await ensure_authenticated()

            logger.info("Scraping company posts: %s", company_name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting company posts scrape"
            )

            url = f"https://www.linkedin.com/company/{company_name}/posts/"
            text = await extractor.extract_page(url)

            sections: dict[str, str] = {}
            if text:
                sections["posts"] = text

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return {
                "url": url,
                "sections": sections,
                "pages_visited": [url],
                "sections_requested": ["posts"],
            }

        except Exception as e:
            return handle_tool_error(e, "get_company_posts")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Follow Company",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def follow_company(
        company_name: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Follow or unfollow a company on LinkedIn.

        Args:
            company_name: LinkedIn company identifier (e.g., "google", "microsoft", "anthropic")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, message, company_name, company_url, and action (followed/unfollowed).
        """
        try:
            await ensure_authenticated()

            logger.info("Following company: %s", company_name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Following company"
            )

            result = await extractor.follow_company(company_name)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "follow_company")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Company Jobs",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_company_jobs(
        company_name: str,
        ctx: Context,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get job openings from a specific company on LinkedIn.

        Args:
            company_name: LinkedIn company identifier (e.g., "google", "microsoft", "anthropic")
            ctx: FastMCP context for progress reporting
            limit: Maximum number of jobs to retrieve (default: 50)

        Returns:
            Dict with company jobs including total_jobs count, and job details
            (job_id, title, location, posted_time, job_url, easy_apply).
        """
        try:
            await ensure_authenticated()

            logger.info("Getting jobs from company: %s (limit=%d)", company_name, limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Fetching company jobs"
            )

            result = await extractor.get_company_jobs(company_name, limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_company_jobs")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Company Employees",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_company_employees(
        company_name: str,
        ctx: Context,
        role_filter: str | None = None,
        location_filter: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get employees from a specific company on LinkedIn.

        Args:
            company_name: LinkedIn company identifier (e.g., "google", "microsoft", "anthropic")
            ctx: FastMCP context for progress reporting
            role_filter: Optional filter for job role (e.g., "Engineer", "Manager", "Recruiter")
            location_filter: Optional filter for location (e.g., "San Francisco", "New York")
            limit: Maximum number of employees to retrieve (default: 50)

        Returns:
            Dict with company employees including total_employees count, and employee details
            (name, username, title, location, profile_url, connection_degree, mutual_connections).
        """
        try:
            await ensure_authenticated()

            logger.info(
                "Getting employees from company: %s (role=%s, location=%s, limit=%d)",
                company_name, role_filter, location_filter, limit
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Fetching company employees"
            )

            result = await extractor.get_company_employees(
                company_name, role_filter, location_filter, limit
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_company_employees")
