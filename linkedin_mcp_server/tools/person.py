"""
LinkedIn person profile scraping tools.

Uses innerText extraction for resilient profile data capture
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
from linkedin_mcp_server.scraping import LinkedInExtractor, parse_person_sections

logger = logging.getLogger(__name__)


def register_person_tools(mcp: FastMCP) -> None:
    """Register all person-related tools with the MCP server."""

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Person Profile",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def get_person_profile(
        linkedin_username: str,
        ctx: Context,
        sections: str | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific person's LinkedIn profile.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates")
            ctx: FastMCP context for progress reporting
            sections: Comma-separated list of extra sections to scrape.
                The main profile page is always included.
                Available sections: experience, education, interests, honors, languages, contact_info
                Examples: "experience,education", "contact_info", "honors,languages"
                Default (None) scrapes only the main profile page.

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            Sections may be absent if extraction yielded no content for that page.
            The LLM should parse the raw text in each section.
        """
        try:
            await ensure_authenticated()

            fields, unknown = parse_person_sections(sections)

            logger.info(
                "Scraping profile: %s (sections=%s)",
                linkedin_username,
                sections,
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting person profile scrape"
            )

            result = await extractor.scrape_person(linkedin_username, fields)

            if unknown:
                result["unknown_sections"] = unknown

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_person_profile")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Search People",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def search_people(
        keywords: str,
        ctx: Context,
        network: str | None = None,
        location: str | None = None,
        current_company: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for people on LinkedIn.

        This is useful for finding recruiters, potential connections, or
        professionals in specific roles or companies.

        Args:
            keywords: Search query (e.g., "tech recruiter", "software engineer react").
            ctx: FastMCP context for progress reporting
            network: Filter by connection degree. Comma-separated values:
                - "F" (1st degree connections)
                - "S" (2nd degree connections)
                - "O" (3rd+ / Out of network)
                Example: "S,O" for 2nd and 3rd+ connections
                Default: None (all connections)
            location: Location filter as geoUrn code.
                Common codes:
                - "106057199" = Brazil
                - "101174742" = São Paulo Area
                - "103644278" = United States
                - "101165590" = United Kingdom
                - "102095887" = California, USA
                Default: None (worldwide)
            current_company: Filter by current company name.
                Default: None (all companies)

        Returns:
            Dict with url, sections (name -> raw text), pages_visited, and sections_requested.
            The LLM should parse the raw text to extract person information.
        """
        try:
            await ensure_authenticated()

            # Parse network filter
            network_list = None
            if network:
                network_list = [n.strip().upper() for n in network.split(",")]

            logger.info(
                "Searching people: keywords='%s', network=%s, location='%s', company='%s'",
                keywords,
                network_list,
                location,
                current_company,
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Starting people search"
            )

            result = await extractor.search_people(
                keywords, network_list, location, current_company
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "search_people")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Connect with Person",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def connect_with_person(
        linkedin_username: str,
        ctx: Context,
        note: str | None = None,
    ) -> dict[str, Any]:
        """
        Send a connection request to a LinkedIn user.

        Use this to expand your network by connecting with recruiters,
        professionals, or people you've interacted with.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates").
                This is the part after linkedin.com/in/ in their profile URL.
            ctx: FastMCP context for progress reporting
            note: Optional personalized note to include with the connection request.
                Maximum 200 characters (LinkedIn free tier limit). Good notes mention
                how you found them or why you want to connect.
                Default: None (sends without a note)

        Returns:
            Dict with status, message, and profile_url.
            Status can be: "success", "error", "already_connected", or "pending"
        """
        try:
            await ensure_authenticated()

            logger.info(
                "Connecting with: %s (note=%s)",
                linkedin_username,
                "yes" if note else "no",
            )

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Sending connection request"
            )

            result = await extractor.connect_with_person(linkedin_username, note)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "connect_with_person")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Send Message",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def send_message(
        linkedin_username: str,
        message: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Send a direct message to a LinkedIn user.

        Use this to reach out to recruiters, professionals, or connections.
        Note: Some users may have messaging disabled for non-connections.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates").
                This is the part after linkedin.com/in/ in their profile URL.
            message: The message text to send. Keep it professional and concise.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, message, and profile_url.
            Status can be: "success" or "error"
        """
        try:
            await ensure_authenticated()

            logger.info("Sending message to: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Sending message"
            )

            result = await extractor.send_message(linkedin_username, message)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "send_message")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Follow Person",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=True,
        )
    )
    async def follow_person(
        linkedin_username: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Follow a LinkedIn user.

        Use this to follow people whose content you want to see,
        especially when you can't connect (they have Follow as primary action).

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates").
                This is the part after linkedin.com/in/ in their profile URL.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, message, and profile_url.
            Status can be: "success", "already_following", or "error"
        """
        try:
            await ensure_authenticated()

            logger.info("Following: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Following user"
            )

            result = await extractor.follow_person(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "follow_person")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Withdraw Connection Request",
            readOnlyHint=False,
            destructiveHint=True,
            openWorldHint=True,
        )
    )
    async def withdraw_connection(
        linkedin_username: str,
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Withdraw a pending connection request.

        Use this to cancel a connection request you've sent but hasn't been accepted yet.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates").
                This is the part after linkedin.com/in/ in their profile URL.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, message, and profile_url.
            Status can be: "success", "no_pending", or "error"
        """
        try:
            await ensure_authenticated()

            logger.info("Withdrawing connection request from: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Withdrawing connection request"
            )

            result = await extractor.withdraw_connection(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "withdraw_connection")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get My Network",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_my_network(
        ctx: Context,
    ) -> dict[str, Any]:
        """
        Get your LinkedIn network: pending invitations and connection suggestions.

        Use this to see who wants to connect with you and discover new people to connect with.

        Args:
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with status, pending_invitations (list), suggestions (list), and message.
            Each invitation/suggestion has: name, headline, profile_url
        """
        try:
            await ensure_authenticated()

            logger.info("Getting my network")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Loading network"
            )

            result = await extractor.get_my_network()

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_my_network")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Messages",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_messages(
        ctx: Context,
        linkedin_username: str | None = None,
    ) -> dict[str, Any]:
        """
        Get your LinkedIn messages/inbox.

        Use this to read messages you've received or view conversation history with someone.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: Optional - LinkedIn username to view conversation with.
                If not provided, returns list of recent conversations in your inbox.
                Example: "margaridajcosta" to see messages with that person.

        Returns:
            If no username: Dict with conversations list (name, last_message, time, unread)
            If username provided: Dict with messages list (sender, content, time)
        """
        try:
            await ensure_authenticated()

            if linkedin_username:
                logger.info("Getting messages with: %s", linkedin_username)
            else:
                logger.info("Getting inbox conversations")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Loading messages"
            )

            result = await extractor.get_messages(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_messages")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get My Profile",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_my_profile(
        ctx: Context,
        language: str = "en",
    ) -> dict[str, Any]:
        """
        Get your own LinkedIn profile data.

        Use this to read your current profile information including name, headline,
        about, experiences, education, skills, languages, and more.

        Args:
            ctx: FastMCP context for progress reporting
            language: Profile language to view - "en" for English, "pt" for Portuguese.
                Default is "en".

        Returns:
            Dict with profile sections (name, headline, about, experiences, education,
            skills, languages, certifications, projects, etc.)
        """
        try:
            await ensure_authenticated()

            logger.info("Getting own profile in language: %s", language)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Loading profile"
            )

            result = await extractor.get_my_profile(language)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_my_profile")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Update Profile Intro",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def update_profile_intro(
        ctx: Context,
        first_name: str | None = None,
        last_name: str | None = None,
        headline: str | None = None,
        industry: str | None = None,
        city: str | None = None,
        country: str | None = None,
        pronouns: str | None = None,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """
        Update your profile intro section (name, headline, location, industry).

        Use this to modify your professional headline, name, location, or industry.
        You can update in either English or Portuguese profile.

        Args:
            ctx: FastMCP context for progress reporting
            first_name: New first name (optional)
            last_name: New last name (optional)
            headline: New professional headline, max 220 chars (optional)
            industry: New industry (optional)
            city: New city location (optional)
            country: New country/region (optional)
            pronouns: Pronouns - "He/Him", "She/Her", "They/Them" (optional)
            profile_language: Which profile to edit - "en" or "pt". Default "en".

        Returns:
            Status dictionary with what was updated.
        """
        try:
            await ensure_authenticated()

            logger.info("Updating profile intro")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Updating profile"
            )

            result = await extractor.update_profile_intro(
                first_name=first_name,
                last_name=last_name,
                headline=headline,
                industry=industry,
                city=city,
                country=country,
                pronouns=pronouns,
                profile_language=profile_language,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "update_profile_intro")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Update Profile About",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def update_profile_about(
        ctx: Context,
        about_text: str,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """
        Update your profile About/Summary section.

        Use this to modify your about/summary text that appears on your profile.

        Args:
            ctx: FastMCP context for progress reporting
            about_text: New about/summary text (max 2600 characters)
            profile_language: Which profile to edit - "en" or "pt". Default "en".

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Updating profile about section")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Updating about section"
            )

            result = await extractor.update_profile_about(about_text, profile_language)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "update_profile_about")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Experience",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_experience(
        ctx: Context,
        title: str,
        company: str,
        location: str | None = None,
        start_month: str | None = None,
        start_year: str | None = None,
        end_month: str | None = None,
        end_year: str | None = None,
        is_current: bool = False,
        description: str | None = None,
        employment_type: str | None = None,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """
        Add a new work experience to your profile.

        Use this to add a job position to your experience section.

        Args:
            ctx: FastMCP context for progress reporting
            title: Job title (required)
            company: Company name (required)
            location: Job location (optional)
            start_month: Start month name, e.g. "January" (optional)
            start_year: Start year, e.g. "2024" (optional)
            end_month: End month name (optional, ignored if is_current=True)
            end_year: End year (optional, ignored if is_current=True)
            is_current: Set True if this is your current position
            description: Job description/responsibilities (optional)
            employment_type: "Full-time", "Part-time", "Contract", etc. (optional)
            profile_language: Which profile to edit - "en" or "pt". Default "en".

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding experience: %s at %s", title, company)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding experience"
            )

            result = await extractor.add_experience(
                title=title,
                company=company,
                location=location,
                start_month=start_month,
                start_year=start_year,
                end_month=end_month,
                end_year=end_year,
                is_current=is_current,
                description=description,
                employment_type=employment_type,
                profile_language=profile_language,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_experience")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Education",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_education(
        ctx: Context,
        school: str,
        degree: str | None = None,
        field_of_study: str | None = None,
        start_year: str | None = None,
        end_year: str | None = None,
        description: str | None = None,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """
        Add a new education entry to your profile.

        Use this to add a school, university, or course to your education section.

        Args:
            ctx: FastMCP context for progress reporting
            school: School/University name (required)
            degree: Degree type, e.g. "Bachelor's degree" (optional)
            field_of_study: Field of study, e.g. "Computer Science" (optional)
            start_year: Start year (optional)
            end_year: End year or expected graduation (optional)
            description: Description or activities (optional)
            profile_language: Which profile to edit - "en" or "pt". Default "en".

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding education: %s", school)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding education"
            )

            result = await extractor.add_education(
                school=school,
                degree=degree,
                field_of_study=field_of_study,
                start_year=start_year,
                end_year=end_year,
                description=description,
                profile_language=profile_language,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_education")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Skill",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_skill(
        ctx: Context,
        skill_name: str,
    ) -> dict[str, Any]:
        """
        Add a new skill to your profile.

        Use this to add a skill to your skills section.

        Args:
            ctx: FastMCP context for progress reporting
            skill_name: Name of the skill to add (e.g. "Python", "React", "Project Management")

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding skill: %s", skill_name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding skill"
            )

            result = await extractor.add_skill(skill_name)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_skill")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Switch Profile Language",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def switch_profile_language(
        ctx: Context,
        language: str,
    ) -> dict[str, Any]:
        """
        Switch your profile view to a different language.

        LinkedIn allows having your profile in multiple languages.
        Use this to switch between your available profile languages.

        Args:
            ctx: FastMCP context for progress reporting
            language: Target language - "en" for English, "pt" for Portuguese

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Switching profile language to: %s", language)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Switching language"
            )

            result = await extractor.switch_profile_language(language)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "switch_profile_language")

    # ===== NETWORK MANAGEMENT TOOLS =====

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Pending Invitations",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_pending_invitations(ctx: Context) -> dict[str, Any]:
        """
        Get list of pending connection invitations.

        Use this to see who has sent you connection requests.

        Args:
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with count and list of invitations (name, headline, profile_url).
        """
        try:
            await ensure_authenticated()

            logger.info("Getting pending invitations")

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Getting invitations"
            )

            result = await extractor.get_pending_invitations()

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_pending_invitations")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Accept Connection",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def accept_connection(
        ctx: Context,
        person_name: str,
    ) -> dict[str, Any]:
        """
        Accept a pending connection invitation.

        Args:
            ctx: FastMCP context for progress reporting
            person_name: Name of the person to accept (partial match supported)

        Returns:
            Status dictionary indicating success or if not found.
        """
        try:
            await ensure_authenticated()

            logger.info("Accepting connection from: %s", person_name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Accepting connection"
            )

            result = await extractor.accept_connection(person_name)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "accept_connection")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Reject Connection",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def reject_connection(
        ctx: Context,
        person_name: str,
    ) -> dict[str, Any]:
        """
        Ignore/reject a pending connection invitation.

        Args:
            ctx: FastMCP context for progress reporting
            person_name: Name of the person to reject (partial match supported)

        Returns:
            Status dictionary indicating success or if not found.
        """
        try:
            await ensure_authenticated()

            logger.info("Rejecting connection from: %s", person_name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Rejecting connection"
            )

            result = await extractor.reject_connection(person_name)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "reject_connection")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Connections",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_connections(
        ctx: Context,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get list of your current connections.

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum number of connections to retrieve (default 50)

        Returns:
            Dict with total_connections, retrieved count, and connections list.
        """
        try:
            await ensure_authenticated()

            logger.info("Getting connections (limit=%d)", limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Getting connections"
            )

            result = await extractor.get_connections(limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_connections")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Remove Connection",
            readOnlyHint=False,
            destructiveHint=True,
            openWorldHint=False,
        )
    )
    async def remove_connection(
        ctx: Context,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """
        Remove an existing connection from your network.

        WARNING: This action cannot be undone. You will need to send a new
        connection request to reconnect with this person.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the connection to remove

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Removing connection: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Removing connection"
            )

            result = await extractor.remove_connection(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "remove_connection")

    # ===== NOTIFICATIONS =====

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Notifications",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_notifications(
        ctx: Context,
        filter_type: str = "all",
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Get your LinkedIn notifications.

        Args:
            ctx: FastMCP context for progress reporting
            filter_type: Filter by type - "all", "jobs", "my_posts", "mentions"
            limit: Maximum number of notifications to retrieve (default 20)

        Returns:
            Dict with filter, count, and notifications list.
            Each notification has: text, type, time, url, is_unread.
            Types include: job_alert, post, reaction, comment, mention,
                          hiring, job_change, birthday, endorsement.
        """
        try:
            await ensure_authenticated()

            logger.info("Getting notifications (filter=%s, limit=%d)", filter_type, limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Getting notifications"
            )

            result = await extractor.get_notifications(filter_type, limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_notifications")

    # ===== PROFILE ADVANCED =====

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Certification",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_certification(
        ctx: Context,
        name: str,
        issuing_organization: str,
        issue_month: str | None = None,
        issue_year: str | None = None,
        expiration_month: str | None = None,
        expiration_year: str | None = None,
        has_expiration: bool = True,
        credential_id: str | None = None,
        credential_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a new certification or license to your profile.

        Use this to add certifications like AWS, Google Cloud, PMP, etc.

        Args:
            ctx: FastMCP context for progress reporting
            name: Certification name (required)
            issuing_organization: Organization that issued the certification (required)
            issue_month: Month issued, e.g. "January" (optional)
            issue_year: Year issued, e.g. "2024" (optional)
            expiration_month: Expiration month (optional)
            expiration_year: Expiration year (optional)
            has_expiration: Set False if credential doesn't expire (default True)
            credential_id: Credential ID (max 80 chars, optional)
            credential_url: URL to verify credential (optional)

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding certification: %s from %s", name, issuing_organization)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding certification"
            )

            result = await extractor.add_certification(
                name=name,
                issuing_organization=issuing_organization,
                issue_month=issue_month,
                issue_year=issue_year,
                expiration_month=expiration_month,
                expiration_year=expiration_year,
                has_expiration=has_expiration,
                credential_id=credential_id,
                credential_url=credential_url,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_certification")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Project",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_project(
        ctx: Context,
        name: str,
        description: str | None = None,
        start_month: str | None = None,
        start_year: str | None = None,
        end_month: str | None = None,
        end_year: str | None = None,
        is_current: bool = False,
        project_url: str | None = None,
    ) -> dict[str, Any]:
        """
        Add a new project to your profile.

        Use this to showcase personal or work projects.

        Args:
            ctx: FastMCP context for progress reporting
            name: Project name (required)
            description: Project description (max 2000 chars, optional)
            start_month: Start month, e.g. "January" (optional)
            start_year: Start year, e.g. "2024" (optional)
            end_month: End month (optional, ignored if is_current=True)
            end_year: End year (optional, ignored if is_current=True)
            is_current: Set True if currently working on this project
            project_url: URL to the project (optional)

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding project: %s", name)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding project"
            )

            result = await extractor.add_project(
                name=name,
                description=description,
                start_month=start_month,
                start_year=start_year,
                end_month=end_month,
                end_year=end_year,
                is_current=is_current,
                project_url=project_url,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_project")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Add Language",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def add_language(
        ctx: Context,
        language: str,
        proficiency: str = "Professional working",
    ) -> dict[str, Any]:
        """
        Add a new language to your profile.

        Use this to add languages you speak and your proficiency level.

        Args:
            ctx: FastMCP context for progress reporting
            language: Language name (e.g. "Spanish", "French", "Mandarin")
            proficiency: Proficiency level. Options:
                - "Elementary" - Elementary proficiency
                - "Limited working" - Limited working proficiency
                - "Professional working" - Professional working proficiency (default)
                - "Full professional" - Full professional proficiency
                - "Native or bilingual" - Native or bilingual proficiency

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Adding language: %s (%s)", language, proficiency)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Adding language"
            )

            result = await extractor.add_language(
                language=language,
                proficiency=proficiency,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "add_language")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Profile Views",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_profile_views(
        ctx: Context,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Get the list of people who viewed your profile.

        Shows who has viewed your profile recently. Some viewers may be
        anonymous (LinkedIn Premium shows all viewers).

        Args:
            ctx: FastMCP context for progress reporting
            limit: Maximum number of viewers to retrieve (default 20)

        Returns:
            Dict with total_views and viewers list.
            Each viewer has: name, headline, time, profile_url, is_anonymous.
        """
        try:
            await ensure_authenticated()

            logger.info("Getting profile views (limit=%d)", limit)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Getting profile views"
            )

            result = await extractor.get_profile_views(limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_profile_views")

    # ===== MESSAGES ADVANCED =====

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Get Conversation",
            readOnlyHint=True,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def get_conversation(
        ctx: Context,
        linkedin_username: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Get the full conversation/messages with a specific person.

        Retrieves the message history with a LinkedIn connection.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the person
            limit: Maximum number of messages to retrieve (default 50)

        Returns:
            Dict with conversation details and messages list.
            Each message has: sender, text, time, is_you (boolean).
        """
        try:
            await ensure_authenticated()

            logger.info("Getting conversation with: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Getting conversation"
            )

            result = await extractor.get_conversation(linkedin_username, limit)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "get_conversation")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Archive Conversation",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def archive_conversation(
        ctx: Context,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """
        Archive a conversation with a specific person.

        Moves a conversation to the archive folder.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the person

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Archiving conversation with: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Archiving conversation"
            )

            result = await extractor.archive_conversation(linkedin_username)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "archive_conversation")

    # ===== ENDORSEMENTS & RECOMMENDATIONS =====

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Endorse Skill",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def endorse_skill(
        ctx: Context,
        linkedin_username: str,
        skill_name: str,
    ) -> dict[str, Any]:
        """
        Endorse a skill on someone's profile.

        Adds your endorsement to a connection's skill.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the person to endorse
            skill_name: Name of the skill to endorse (e.g. "Python", "Leadership")

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Endorsing skill %s for %s", skill_name, linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Endorsing skill"
            )

            result = await extractor.endorse_skill(linkedin_username, skill_name)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "endorse_skill")

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Request Recommendation",
            readOnlyHint=False,
            destructiveHint=False,
            openWorldHint=False,
        )
    )
    async def request_recommendation(
        ctx: Context,
        linkedin_username: str,
        message: str | None = None,
        relationship: str | None = None,
        position_at_time: str | None = None,
    ) -> dict[str, Any]:
        """
        Request a recommendation from a connection.

        Sends a recommendation request to someone you've worked with.

        Args:
            ctx: FastMCP context for progress reporting
            linkedin_username: LinkedIn username of the person to request from
            message: Custom message for the request (optional)
            relationship: How you know them, e.g. "colleague", "manager" (optional)
            position_at_time: Your position when you worked with them (optional)

        Returns:
            Status dictionary indicating success or error.
        """
        try:
            await ensure_authenticated()

            logger.info("Requesting recommendation from: %s", linkedin_username)

            browser = await get_or_create_browser()
            extractor = LinkedInExtractor(browser.page)

            await ctx.report_progress(
                progress=0, total=100, message="Requesting recommendation"
            )

            result = await extractor.request_recommendation(
                linkedin_username=linkedin_username,
                message=message,
                relationship=relationship,
                position_at_time=position_at_time,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except Exception as e:
            return handle_tool_error(e, "request_recommendation")
