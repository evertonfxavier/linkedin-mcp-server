"""
LinkedIn job scraping tools with search and detail extraction.

Uses innerText extraction for resilient job data capture.
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


def register_job_tools(mcp: FastMCP) -> None:
    """Register all job-related tools with the MCP server."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Job Details",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_job_details(job_id: str, ctx: Context) -> dict[str, Any]:
        """
        Get job details for a specific job posting on LinkedIn.

        Args:
            job_id: LinkedIn job ID (e.g., "4252026496", "3856789012")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text to extract job details.
        """
        try:
            await ensure_authenticated()

            logger.info("Scraping job: %s", job_id)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting job scrape"
            )

            result = await extractor.scrape_job(job_id)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_job_details")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search Jobs",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def search_jobs(
        keywords: str,
        ctx: Context,
        location: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for jobs on LinkedIn.

        Args:
            keywords: Search keywords (e.g., "software engineer", "data scientist")
            ctx: FastMCP context for progress reporting
            location: Optional location filter (e.g., "San Francisco", "Remote")

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text to extract job listings.
        """
        try:
            await ensure_authenticated()

            logger.info(
                "Searching jobs: keywords='%s', location='%s'",
                keywords,
                location,
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting job search"
            )

            result = await extractor.search_jobs(keywords, location)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "search_jobs")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Save Job",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def save_job(job_id: str, ctx: Context) -> dict[str, Any]:
        """
        Save a job for later on LinkedIn.

        Args:
            job_id: LinkedIn job ID (e.g., "4252026496", "3856789012")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, message, job_id, job_url, and action (saved/unsaved).
        """
        try:
            await ensure_authenticated()

            logger.info("Saving job: %s", job_id)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Saving job"
            )

            result = await extractor.save_job(job_id)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "save_job")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Apply to Job",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def apply_to_job(job_id: str, ctx: Context) -> dict[str, Any]:
        """
        Apply to a job using Easy Apply on LinkedIn.

        This only works for jobs with Easy Apply enabled.
        For jobs requiring external applications, returns the application URL.

        Args:
            job_id: LinkedIn job ID (e.g., "4252026496", "3856789012")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status and relevant info:
            - modal_opened: Easy Apply modal opened (additional steps needed)
            - clicked: Button clicked but modal not detected
            - external: Job requires external application (includes external_url)
            - not_available: No apply button found
            - error: Something went wrong
        """
        try:
            await ensure_authenticated()

            logger.info("Applying to job: %s", job_id)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting job application"
            )

            result = await extractor.apply_to_job(job_id)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "apply_to_job")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Saved Jobs",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_saved_jobs(
        ctx: Context,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get jobs saved by the user on LinkedIn.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum number of saved jobs to retrieve (default: 50)

        Returns:
            Dict with saved jobs list including job_id, title, company, location,
            job_url, saved_date, and is_active status.
        """
        try:
            await ensure_authenticated()

            logger.info("Getting saved jobs (limit=%d)", limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Fetching saved jobs"
            )

            result = await extractor.get_saved_jobs(limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_saved_jobs")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Applied Jobs",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_applied_jobs(
        ctx: Context,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get jobs the user has applied to on LinkedIn.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum number of applied jobs to retrieve (default: 50)

        Returns:
            Dict with applied jobs list including job_id, title, company, location,
            job_url, applied_date, and status (applied, viewed, in_progress, rejected, offered).
        """
        try:
            await ensure_authenticated()

            logger.info("Getting applied jobs (limit=%d)", limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Fetching applied jobs"
            )

            result = await extractor.get_applied_jobs(limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_applied_jobs")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Job Alerts",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_job_alerts(ctx: Context) -> dict[str, Any]:
        """
        Get all job alerts configured by the user on LinkedIn.

        Args:
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with job alerts list including keywords and location for each alert.
        """
        try:
            await ensure_authenticated()

            logger.info("Getting job alerts")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Fetching job alerts"
            )

            result = await extractor.get_job_alerts()

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_job_alerts")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Create Job Alert",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def create_job_alert(
        keywords: str,
        ctx: Context,
        location: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new job alert on LinkedIn.

        Args:
            keywords: Search keywords for the alert (e.g., "software engineer", "data scientist")
            ctx: FastMCP context for progress reporting
            location: Optional location filter (e.g., "Remote", "San Francisco", "Portugal")

        Returns:
            Dict with creation status and alert details.
        """
        try:
            await ensure_authenticated()

            logger.info("Creating job alert: keywords='%s', location='%s'", keywords, location)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Creating job alert"
            )

            result = await extractor.create_job_alert(keywords, location)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "create_job_alert")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Easy Apply Complete",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def easy_apply_complete(
        job_id: str,
        ctx: Context,
        phone_number: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Complete the Easy Apply flow for a job on LinkedIn.

        This handles the entire Easy Apply process including contact info,
        resume selection, additional questions, and submission.

        Args:
            job_id: LinkedIn job ID (e.g., "4252026496")
            ctx: FastMCP context for progress reporting
            phone_number: Optional phone number with country code (e.g., "+1 555-123-4567")
            answers: Optional dict mapping question keywords to answers.
                    Example: {"years experience": "5", "work authorization": "yes", "salary": "100000"}

        Returns:
            Dict with application result:
            - success: Application submitted
            - external: Requires external application
            - incomplete: Flow didn't complete (may need manual intervention)
            - error: Something went wrong
        """
        try:
            await ensure_authenticated()

            logger.info("Starting Easy Apply for job: %s", job_id)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting Easy Apply"
            )

            result = await extractor.easy_apply_complete(job_id, phone_number, answers)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "easy_apply_complete")
