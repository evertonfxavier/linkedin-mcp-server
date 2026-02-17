"""Core extraction engine using innerText instead of DOM selectors."""

import asyncio
import logging
import re
from typing import Any
from urllib.parse import quote_plus

from patchright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.core.utils import (
    detect_rate_limit,
    handle_modal_close,
    scroll_to_bottom,
)

from .fields import (
    COMPANY_SECTION_MAP,
    PERSON_SECTION_MAP,
    CompanyScrapingFields,
    PersonScrapingFields,
)

logger = logging.getLogger(__name__)

# Delay between page navigations to avoid rate limiting
_NAV_DELAY = 2.0

# Backoff before retrying a rate-limited page
_RATE_LIMIT_RETRY_DELAY = 5.0

# Returned as section text when LinkedIn rate-limits the page
_RATE_LIMITED_MSG = "[Rate limited] LinkedIn blocked this section. Try again later or request fewer sections."

# Patterns that mark the start of LinkedIn page chrome (sidebar/footer).
# Everything from the earliest match onwards is stripped.
_NOISE_MARKERS: list[re.Pattern[str]] = [
    # Footer nav links: "About" immediately followed by "Accessibility" or "Talent Solutions"
    re.compile(r"^About\n+(?:Accessibility|Talent Solutions)", re.MULTILINE),
    # Sidebar profile recommendations
    re.compile(r"^More profiles for you$", re.MULTILINE),
    # Sidebar premium upsell
    re.compile(r"^Explore premium profiles$", re.MULTILINE),
    # InMail upsell in contact info overlay
    re.compile(r"^Get up to .+ replies when you message with InMail$", re.MULTILINE),
]


def strip_linkedin_noise(text: str) -> str:
    """Remove LinkedIn page chrome (footer, sidebar recommendations) from innerText.

    Finds the earliest occurrence of any known noise marker and truncates there.
    """
    earliest = len(text)
    for pattern in _NOISE_MARKERS:
        match = pattern.search(text)
        if match and match.start() < earliest:
            earliest = match.start()

    return text[:earliest].strip()


class LinkedInExtractor:
    """Extracts LinkedIn page content via navigate-scroll-innerText pattern."""

    def __init__(self, page: Page):
        self._page = page

    async def extract_page(self, url: str) -> str:
        """Navigate to a URL, scroll to load lazy content, and extract innerText.

        Retries once after a backoff when the page returns only LinkedIn chrome
        (sidebar/footer noise with no actual content), which indicates a soft
        rate limit.

        Raises LinkedInScraperException subclasses (rate limit, auth, etc.).
        Returns _RATE_LIMITED_MSG sentinel when soft-rate-limited after retry.
        Returns empty string for unexpected non-domain failures (error isolation).
        """
        try:
            result = await self._extract_page_once(url)
            if result != _RATE_LIMITED_MSG:
                return result

            # Retry once after backoff
            logger.info("Retrying %s after %.0fs backoff", url, _RATE_LIMIT_RETRY_DELAY)
            await asyncio.sleep(_RATE_LIMIT_RETRY_DELAY)
            return await self._extract_page_once(url)

        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract page %s: %s", url, e)
            return ""

    async def _extract_page_once(self, url: str) -> str:
        """Single attempt to navigate, scroll, and extract innerText."""
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await detect_rate_limit(self._page)

        # Wait for main content to render
        try:
            await self._page.wait_for_selector("main", timeout=5000)
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", url)

        # Dismiss any modals blocking content
        await handle_modal_close(self._page)

        # Scroll to trigger lazy loading
        await scroll_to_bottom(self._page, pause_time=0.5, max_scrolls=5)

        # Extract text from main content area
        raw = await self._page.evaluate(
            """() => {
                const main = document.querySelector('main');
                return main ? main.innerText : document.body.innerText;
            }"""
        )

        if not raw:
            return ""
        cleaned = strip_linkedin_noise(raw)
        if not cleaned and raw.strip():
            logger.warning(
                "Page %s returned only LinkedIn chrome (likely rate-limited)", url
            )
            return _RATE_LIMITED_MSG
        return cleaned

    async def _extract_overlay(self, url: str) -> str:
        """Extract content from an overlay/modal page (e.g. contact info).

        LinkedIn renders contact info as a native <dialog> element.
        Falls back to `<main>` if no dialog is found.

        Retries once after a backoff when the overlay returns only LinkedIn
        chrome (noise), mirroring `extract_page` behavior.
        """
        try:
            result = await self._extract_overlay_once(url)
            if result != _RATE_LIMITED_MSG:
                return result

            logger.info(
                "Retrying overlay %s after %.0fs backoff",
                url,
                _RATE_LIMIT_RETRY_DELAY,
            )
            await asyncio.sleep(_RATE_LIMIT_RETRY_DELAY)
            return await self._extract_overlay_once(url)

        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract overlay %s: %s", url, e)
            return ""

    async def _extract_overlay_once(self, url: str) -> str:
        """Single attempt to extract content from an overlay/modal page."""
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await detect_rate_limit(self._page)

        # Wait for the dialog/modal to render (LinkedIn uses native <dialog>)
        try:
            await self._page.wait_for_selector(
                "dialog[open], .artdeco-modal__content", timeout=5000
            )
        except PlaywrightTimeoutError:
            logger.debug("No modal overlay found on %s, falling back to main", url)

        # NOTE: Do NOT call handle_modal_close() here — the contact-info
        # overlay *is* a dialog/modal. Dismissing it would destroy the
        # content before the JS evaluation below can read it.

        raw = await self._page.evaluate(
            """() => {
                const dialog = document.querySelector('dialog[open]');
                if (dialog) return dialog.innerText.trim();
                const modal = document.querySelector('.artdeco-modal__content');
                if (modal) return modal.innerText.trim();
                const main = document.querySelector('main');
                return main ? main.innerText.trim() : document.body.innerText.trim();
            }"""
        )

        if not raw:
            return ""
        cleaned = strip_linkedin_noise(raw)
        if not cleaned and raw.strip():
            logger.warning(
                "Overlay %s returned only LinkedIn chrome (likely rate-limited)",
                url,
            )
            return _RATE_LIMITED_MSG
        return cleaned

    async def scrape_person(
        self, username: str, fields: PersonScrapingFields
    ) -> dict[str, Any]:
        """Scrape a person profile with configurable sections.

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
        """
        fields |= PersonScrapingFields.BASIC_INFO
        base_url = f"https://www.linkedin.com/in/{username}"
        sections: dict[str, str] = {}
        pages_visited: list[str] = []

        # Map flags to (section_name, url_suffix, is_overlay)
        page_map: list[tuple[PersonScrapingFields, str, str, bool]] = [
            (PersonScrapingFields.BASIC_INFO, "main_profile", "/", False),
            (
                PersonScrapingFields.EXPERIENCE,
                "experience",
                "/details/experience/",
                False,
            ),
            (
                PersonScrapingFields.EDUCATION,
                "education",
                "/details/education/",
                False,
            ),
            (
                PersonScrapingFields.INTERESTS,
                "interests",
                "/details/interests/",
                False,
            ),
            (
                PersonScrapingFields.HONORS,
                "honors",
                "/details/honors/",
                False,
            ),
            (
                PersonScrapingFields.LANGUAGES,
                "languages",
                "/details/languages/",
                False,
            ),
            (
                PersonScrapingFields.CONTACT_INFO,
                "contact_info",
                "/overlay/contact-info/",
                True,
            ),
        ]

        for flag, section_name, suffix, is_overlay in page_map:
            if not (flag & fields):
                continue

            url = base_url + suffix
            try:
                if is_overlay:
                    text = await self._extract_overlay(url)
                else:
                    text = await self.extract_page(url)

                if text:
                    sections[section_name] = text
                pages_visited.append(url)
            except LinkedInScraperException:
                raise
            except Exception as e:
                logger.warning("Error scraping section %s: %s", section_name, e)
                pages_visited.append(url)

            # Delay between navigations
            await asyncio.sleep(_NAV_DELAY)

        # Build sections_requested from flags
        requested = ["main_profile"]
        reverse_map = {v: k for k, v in PERSON_SECTION_MAP.items()}
        for flag in PersonScrapingFields:
            if flag in fields and flag in reverse_map:
                requested.append(reverse_map[flag])

        return {
            "url": f"{base_url}/",
            "sections": sections,
            "pages_visited": pages_visited,
            "sections_requested": requested,
        }

    async def scrape_company(
        self, company_name: str, fields: CompanyScrapingFields
    ) -> dict[str, Any]:
        """Scrape a company profile with configurable sections.

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
        """
        fields |= CompanyScrapingFields.ABOUT
        base_url = f"https://www.linkedin.com/company/{company_name}"
        sections: dict[str, str] = {}
        pages_visited: list[str] = []

        page_map: list[tuple[CompanyScrapingFields, str, str]] = [
            (CompanyScrapingFields.ABOUT, "about", "/about/"),
            (CompanyScrapingFields.POSTS, "posts", "/posts/"),
            (CompanyScrapingFields.JOBS, "jobs", "/jobs/"),
        ]

        for flag, section_name, suffix in page_map:
            if not (flag & fields):
                continue

            url = base_url + suffix
            try:
                text = await self.extract_page(url)
                if text:
                    sections[section_name] = text
                pages_visited.append(url)
            except LinkedInScraperException:
                raise
            except Exception as e:
                logger.warning("Error scraping section %s: %s", section_name, e)
                pages_visited.append(url)

            await asyncio.sleep(_NAV_DELAY)

        # Build sections_requested from flags
        requested = ["about"]
        reverse_map = {v: k for k, v in COMPANY_SECTION_MAP.items()}
        for flag in CompanyScrapingFields:
            if flag in fields and flag in reverse_map:
                requested.append(reverse_map[flag])

        return {
            "url": f"{base_url}/",
            "sections": sections,
            "pages_visited": pages_visited,
            "sections_requested": requested,
        }

    async def scrape_job(self, job_id: str) -> dict[str, Any]:
        """Scrape a single job posting.

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
        """
        url = f"https://www.linkedin.com/jobs/view/{job_id}/"
        text = await self.extract_page(url)

        sections: dict[str, str] = {}
        if text:
            sections["job_posting"] = text

        return {
            "url": url,
            "sections": sections,
            "pages_visited": [url],
            "sections_requested": ["job_posting"],
        }

    async def search_jobs(
        self, keywords: str, location: str | None = None
    ) -> dict[str, Any]:
        """Search for jobs and extract the results page.

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
        """
        params = f"keywords={quote_plus(keywords)}"
        if location:
            params += f"&location={quote_plus(location)}"

        url = f"https://www.linkedin.com/jobs/search/?{params}"
        text = await self.extract_page(url)

        sections: dict[str, str] = {}
        if text:
            sections["search_results"] = text

        return {
            "url": url,
            "sections": sections,
            "pages_visited": [url],
            "sections_requested": ["search_results"],
        }

    async def search_posts(
        self,
        keywords: str,
        date_posted: str | None = None,
        sort_by: str | None = None,
    ) -> dict[str, Any]:
        """Search for posts/content on LinkedIn.

        This is useful for finding job postings shared as posts, announcements,
        and other content that matches specific search queries.

        Args:
            keywords: Search query with boolean operators support.
                Examples:
                - "React" AND ("Pleno" OR "Mid" OR "PL")
                - ("Front" OR "Frontend") AND ("Pleno" OR "Mid")
                - ("Vue" OR "Vue.js") AND ("Junior" OR "JR" OR "Entry")
            date_posted: Filter by date. Options:
                - "past-24h" (last 24 hours)
                - "past-week" (last week)
                - "past-month" (last month)
                Default: None (no date filter)
            sort_by: Sort order. Options:
                - "date_posted" (most recent first)
                - "relevance" (default LinkedIn sorting)
                Default: None (LinkedIn default)

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
        """
        params = f"keywords={quote_plus(keywords)}"
        params += "&origin=FACETED_SEARCH"

        if date_posted:
            # LinkedIn expects quoted values for datePosted
            quoted_date = '"' + date_posted + '"'
            params += f"&datePosted={quote_plus(quoted_date)}"

        if sort_by:
            # LinkedIn expects quoted values for sortBy
            quoted_sort = '"' + sort_by + '"'
            params += f"&sortBy={quote_plus(quoted_sort)}"

        url = f"https://www.linkedin.com/search/results/content/?{params}"
        text = await self._extract_posts_page(url)

        sections: dict[str, str] = {}
        if text:
            sections["posts"] = text

        return {
            "url": url,
            "sections": sections,
            "pages_visited": [url],
            "sections_requested": ["posts"],
        }

    async def _extract_posts_page(self, url: str) -> str:
        """Extract content from LinkedIn search results/posts page.

        This page has a different structure and requires waiting for
        the search results container to load.
        """
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await detect_rate_limit(self._page)

        # Wait for search results container
        try:
            await self._page.wait_for_selector(
                ".search-results-container, .scaffold-finite-scroll__content",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("No search results container found on %s", url)

        # Dismiss any modals blocking content
        await handle_modal_close(self._page)

        # Scroll more to load lazy content (posts need more scrolling)
        await scroll_to_bottom(self._page, pause_time=0.8, max_scrolls=8)

        # Extract text from search results area
        raw = await self._page.evaluate(
            """() => {
                // Try specific search results container first
                const container = document.querySelector('.search-results-container') ||
                                  document.querySelector('.scaffold-finite-scroll__content') ||
                                  document.querySelector('main');
                return container ? container.innerText : document.body.innerText;
            }"""
        )

        if not raw:
            return ""
        cleaned = strip_linkedin_noise(raw)
        if not cleaned and raw.strip():
            logger.warning(
                "Posts page %s returned only LinkedIn chrome (likely rate-limited)", url
            )
            return _RATE_LIMITED_MSG
        return cleaned

    async def search_people(
        self,
        keywords: str,
        network: list[str] | None = None,
        location: str | None = None,
        current_company: str | None = None,
    ) -> dict[str, Any]:
        """Search for people on LinkedIn.

        Args:
            keywords: Search query (e.g., "tech recruiter", "software engineer").
            network: Filter by connection degree. Options:
                - "F" (1st degree connections)
                - "S" (2nd degree connections)
                - "O" (3rd+ / Out of network)
                Example: ["S", "O"] for 2nd and 3rd+ connections
                Default: None (all connections)
            location: Location filter as geoUrn code.
                Common codes:
                - "106057199" = Brazil
                - "101174742" = São Paulo Area
                - "103644278" = United States
                - "101165590" = United Kingdom
                Default: None (worldwide)
            current_company: Filter by current company name.
                Default: None (all companies)

        Returns:
            {url, sections: {name: text}, pages_visited, sections_requested}
            The text will contain person cards with name, title, location, and profile URL.
        """
        params = f"keywords={quote_plus(keywords)}"
        params += "&origin=FACETED_SEARCH"

        if network:
            # LinkedIn expects JSON array format for network filter
            network_json = "[" + ",".join(f'"{n}"' for n in network) + "]"
            params += f"&network={quote_plus(network_json)}"

        if location:
            # LinkedIn expects JSON array format for geoUrn
            geo_json = '["' + location + '"]'
            params += f"&geoUrn={quote_plus(geo_json)}"

        if current_company:
            params += f"&currentCompany={quote_plus(current_company)}"

        url = f"https://www.linkedin.com/search/results/people/?{params}"
        text = await self._extract_people_search_page(url)

        sections: dict[str, str] = {}
        if text:
            sections["people"] = text

        return {
            "url": url,
            "sections": sections,
            "pages_visited": [url],
            "sections_requested": ["people"],
        }

    async def _extract_people_search_page(self, url: str) -> str:
        """Extract content from LinkedIn people search results page."""
        await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await detect_rate_limit(self._page)

        # Wait for search results container
        try:
            await self._page.wait_for_selector(
                ".search-results-container, .reusable-search__entity-result-list",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("No people search results container found on %s", url)

        # Dismiss any modals blocking content
        await handle_modal_close(self._page)

        # Scroll to load more results
        await scroll_to_bottom(self._page, pause_time=0.8, max_scrolls=5)

        # Extract profile links and text from search results
        raw = await self._page.evaluate(
            """() => {
                const results = [];
                // Find all person result cards
                const cards = document.querySelectorAll('.reusable-search__result-container, [data-view-name="search-entity-result-universal-template"]');
                
                for (const card of cards) {
                    // Get profile link
                    const link = card.querySelector('a[href*="/in/"]');
                    const profileUrl = link ? link.href.split('?')[0] : '';
                    
                    // Get text content
                    const text = card.innerText;
                    
                    if (profileUrl) {
                        results.push('Profile: ' + profileUrl + '\\n' + text);
                    } else {
                        results.push(text);
                    }
                }
                
                if (results.length > 0) {
                    return results.join('\\n---\\n');
                }
                
                // Fallback to container text
                const container = document.querySelector('.search-results-container') ||
                                  document.querySelector('.reusable-search__entity-result-list') ||
                                  document.querySelector('main');
                return container ? container.innerText : document.body.innerText;
            }"""
        )

        if not raw:
            return ""
        cleaned = strip_linkedin_noise(raw)
        if not cleaned and raw.strip():
            logger.warning(
                "People search page %s returned only LinkedIn chrome (likely rate-limited)",
                url,
            )
            return _RATE_LIMITED_MSG
        return cleaned

    async def connect_with_person(
        self,
        linkedin_username: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Send a connection request to a LinkedIn user.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel", "williamhgates")
            note: Optional personalized note to include with the connection request.
                Maximum 200 characters. If not provided, sends without a note.

        Returns:
            {
                "status": "success" | "error" | "already_connected" | "pending",
                "message": str,
                "profile_url": str
            }
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"
        invite_url = f"https://www.linkedin.com/preload/custom-invite/?vanityName={linkedin_username}"

        try:
            # Navigate directly to the custom invite URL
            # This is more reliable than clicking the Connect button on the profile
            await self._page.goto(invite_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)

            # Wait for the invite modal dialog to appear
            await asyncio.sleep(2)

            # Check what dialog appeared
            dialog_state = await self._page.evaluate("""() => {
                const dialogs = document.querySelectorAll('[role="dialog"]');
                for (const dialog of dialogs) {
                    const text = dialog.innerText.toLowerCase();
                    const buttons = Array.from(dialog.querySelectorAll('button')).map(b => b.innerText.trim());
                    
                    // Check for "Add a note" dialog
                    if (text.includes('add a note') && text.includes('invitation')) {
                        return { 
                            type: 'invite_modal', 
                            buttons,
                            hasAddNote: buttons.some(b => b.toLowerCase().includes('add a note')),
                            hasSendWithout: buttons.some(b => b.toLowerCase().includes('send without'))
                        };
                    }
                    
                    // Check for note input dialog (after clicking "Add a note")
                    if (text.includes('personalized invitations') || 
                        (dialog.querySelector('textarea') && text.includes('personal note'))) {
                        return {
                            type: 'note_input_modal',
                            buttons,
                            hasTextarea: true
                        };
                    }
                    
                    // Check for already pending
                    if (text.includes('pending') || text.includes('withdraw')) {
                        return { type: 'pending', buttons };
                    }
                    
                    // Check for already connected
                    if (text.includes('already connected') || text.includes('1st degree')) {
                        return { type: 'already_connected', buttons };
                    }
                }
                
                // No relevant dialog found - check page content
                const pageText = document.body.innerText.toLowerCase();
                if (pageText.includes('pending')) {
                    return { type: 'pending_on_page' };
                }
                
                return { 
                    type: 'unknown', 
                    dialogCount: dialogs.length,
                    dialogTexts: Array.from(dialogs).map(d => d.innerText.substring(0, 100))
                };
            }""")

            logger.info("Dialog state for %s: %s", linkedin_username, dialog_state)

            # Handle different dialog states
            if dialog_state.get("type") == "already_connected":
                return {
                    "status": "already_connected",
                    "message": f"Already connected with {linkedin_username}",
                    "profile_url": profile_url,
                }

            if dialog_state.get("type") in ("pending", "pending_on_page"):
                return {
                    "status": "pending",
                    "message": f"Connection request already pending for {linkedin_username}",
                    "profile_url": profile_url,
                }

            if dialog_state.get("type") == "invite_modal":
                if note:
                    # Click "Add a note" button
                    add_note_clicked = await self._page.evaluate("""() => {
                        const dialog = document.querySelector('[role="dialog"]');
                        if (!dialog) return false;
                        const buttons = dialog.querySelectorAll('button');
                        for (const btn of buttons) {
                            if (btn.innerText.toLowerCase().includes('add a note')) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }""")

                    if add_note_clicked:
                        logger.info("Clicked 'Add a note' for %s", linkedin_username)
                        await asyncio.sleep(2)

                        # Wait for and fill the textarea
                        textarea = self._page.locator('[role="dialog"] textarea')
                        try:
                            await textarea.first.wait_for(timeout=5000)
                            # LinkedIn limits to 200 chars in free tier
                            await textarea.first.fill(note[:200])
                            logger.info("Filled note for %s", linkedin_username)
                            await asyncio.sleep(0.5)

                            # Click "Send invitation" button
                            send_clicked = await self._page.evaluate("""() => {
                                const dialog = document.querySelector('[role="dialog"]');
                                if (!dialog) return false;
                                const buttons = dialog.querySelectorAll('button');
                                for (const btn of buttons) {
                                    const text = btn.innerText.toLowerCase();
                                    if (text.includes('send invitation') || text === 'send') {
                                        btn.click();
                                        return true;
                                    }
                                }
                                return false;
                            }""")
                            logger.info("Send button clicked: %s for %s", send_clicked, linkedin_username)

                        except Exception as e:
                            logger.error("Error filling note for %s: %s", linkedin_username, e)
                            return {
                                "status": "error",
                                "message": f"Error filling note: {str(e)}",
                                "profile_url": profile_url,
                            }
                    else:
                        # Fallback: send without note
                        logger.warning("Could not find 'Add a note' button for %s, sending without", linkedin_username)
                        await self._page.evaluate("""() => {
                            const dialog = document.querySelector('[role="dialog"]');
                            if (!dialog) return false;
                            const buttons = dialog.querySelectorAll('button');
                            for (const btn of buttons) {
                                if (btn.innerText.toLowerCase().includes('send without')) {
                                    btn.click();
                                    return true;
                                }
                            }
                            return false;
                        }""")
                else:
                    # No note - click "Send without a note"
                    send_clicked = await self._page.evaluate("""() => {
                        const dialog = document.querySelector('[role="dialog"]');
                        if (!dialog) return false;
                        const buttons = dialog.querySelectorAll('button');
                        for (const btn of buttons) {
                            if (btn.innerText.toLowerCase().includes('send without')) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    logger.info("Send without note clicked: %s for %s", send_clicked, linkedin_username)

                # Wait and verify
                await asyncio.sleep(2)
                
                # Check if modal closed (success indicator)
                modal_closed = await self._page.evaluate("""() => {
                    const dialog = document.querySelector('[role="dialog"]');
                    if (!dialog) return true;
                    // Check if it's still the invite modal
                    const text = dialog.innerText.toLowerCase();
                    return !(text.includes('add a note') || text.includes('invitation') || text.includes('personal note'));
                }""")

                if modal_closed:
                    return {
                        "status": "success",
                        "message": f"Connection request sent to {linkedin_username}" +
                                   (" with note" if note else ""),
                        "profile_url": profile_url,
                    }
                else:
                    return {
                        "status": "partial",
                        "message": f"Modal may still be visible for {linkedin_username}",
                        "profile_url": profile_url,
                    }

            # Unknown state - try to provide helpful debug info
            return {
                "status": "error",
                "message": f"Could not determine invite modal state for {linkedin_username}. Debug: {dialog_state}",
                "profile_url": profile_url,
            }

        except Exception as e:
            logger.error("Error connecting with %s: %s", linkedin_username, e)
            return {
                "status": "error",
                "message": f"Error connecting with {linkedin_username}: {str(e)}",
                "profile_url": profile_url,
            }

    async def send_message(
        self,
        linkedin_username: str,
        message: str,
    ) -> dict[str, Any]:
        """Send a direct message to a LinkedIn user.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel")
            message: The message text to send

        Returns:
            {
                "status": "success" | "error",
                "message": str,
                "profile_url": str
            }
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"
        messaging_url = f"https://www.linkedin.com/messaging/compose/?recipients={linkedin_username}"

        try:
            await self._page.goto(messaging_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Wait for the message compose area
            try:
                msg_box = self._page.locator('[role="textbox"], .msg-form__contenteditable, textarea[name="message"]')
                await msg_box.first.wait_for(timeout=10000)
            except PlaywrightTimeoutError:
                return {
                    "status": "error",
                    "message": f"Message compose area not found for {linkedin_username}. They may have messaging disabled.",
                    "profile_url": profile_url,
                }

            # Fill the message
            await msg_box.first.click()
            await asyncio.sleep(0.3)
            await msg_box.first.fill(message)
            logger.info("Filled message for %s", linkedin_username)
            await asyncio.sleep(0.5)

            # Click send button
            send_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.innerText.toLowerCase().trim();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    if (text === 'send' || ariaLabel.includes('send message') || ariaLabel === 'send') {
                        btn.click();
                        return true;
                    }
                }
                // Try form submit button
                const submitBtn = document.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.click();
                    return true;
                }
                return false;
            }""")

            if send_clicked:
                await asyncio.sleep(2)
                return {
                    "status": "success",
                    "message": f"Message sent to {linkedin_username}",
                    "profile_url": profile_url,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Could not find send button for {linkedin_username}",
                    "profile_url": profile_url,
                }

        except Exception as e:
            logger.error("Error sending message to %s: %s", linkedin_username, e)
            return {
                "status": "error",
                "message": f"Error sending message: {str(e)}",
                "profile_url": profile_url,
            }

    async def follow_person(
        self,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """Follow a LinkedIn user.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel")

        Returns:
            {
                "status": "success" | "error" | "already_following",
                "message": str,
                "profile_url": str
            }
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)

            try:
                await self._page.wait_for_selector("main", timeout=10000)
            except PlaywrightTimeoutError:
                return {
                    "status": "error",
                    "message": f"Profile page did not load for {linkedin_username}",
                    "profile_url": profile_url,
                }

            await asyncio.sleep(1.5)
            await handle_modal_close(self._page)

            # Check current follow state and click Follow button
            result = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                
                for (const btn of buttons) {
                    const text = btn.innerText.trim().toLowerCase();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    
                    // Check if already following
                    if (text === 'following' || ariaLabel.includes('unfollow')) {
                        return { status: 'already_following' };
                    }
                    
                    // Find and click Follow button
                    if (text === 'follow' || ariaLabel.includes('follow') && !ariaLabel.includes('following')) {
                        btn.click();
                        return { status: 'clicked' };
                    }
                }
                
                return { status: 'not_found' };
            }""")

            if result.get("status") == "already_following":
                return {
                    "status": "already_following",
                    "message": f"Already following {linkedin_username}",
                    "profile_url": profile_url,
                }

            if result.get("status") == "clicked":
                await asyncio.sleep(1)
                return {
                    "status": "success",
                    "message": f"Now following {linkedin_username}",
                    "profile_url": profile_url,
                }

            return {
                "status": "error",
                "message": f"Follow button not found for {linkedin_username}",
                "profile_url": profile_url,
            }

        except Exception as e:
            logger.error("Error following %s: %s", linkedin_username, e)
            return {
                "status": "error",
                "message": f"Error following: {str(e)}",
                "profile_url": profile_url,
            }

    async def withdraw_connection(
        self,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """Withdraw a pending connection request.

        Args:
            linkedin_username: LinkedIn username (e.g., "stickerdaniel")

        Returns:
            {
                "status": "success" | "error" | "no_pending",
                "message": str,
                "profile_url": str
            }
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)

            try:
                await self._page.wait_for_selector("main", timeout=10000)
            except PlaywrightTimeoutError:
                return {
                    "status": "error",
                    "message": f"Profile page did not load for {linkedin_username}",
                    "profile_url": profile_url,
                }

            await asyncio.sleep(1.5)
            await handle_modal_close(self._page)

            # Find and click Pending button
            pending_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                
                for (const btn of buttons) {
                    const text = btn.innerText.trim().toLowerCase();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    
                    if (text === 'pending' || ariaLabel.includes('pending') || ariaLabel.includes('withdraw')) {
                        btn.click();
                        return { clicked: true };
                    }
                }
                
                return { clicked: false };
            }""")

            if not pending_clicked.get("clicked"):
                return {
                    "status": "no_pending",
                    "message": f"No pending connection request for {linkedin_username}",
                    "profile_url": profile_url,
                }

            # Wait for confirmation dialog and click withdraw
            await asyncio.sleep(1.5)
            
            withdraw_clicked = await self._page.evaluate("""() => {
                const dialogs = document.querySelectorAll('[role="dialog"], .artdeco-modal');
                for (const dialog of dialogs) {
                    const buttons = dialog.querySelectorAll('button');
                    for (const btn of buttons) {
                        const text = btn.innerText.toLowerCase().trim();
                        if (text.includes('withdraw') || text.includes('confirm')) {
                            btn.click();
                            return true;
                        }
                    }
                }
                return false;
            }""")

            if withdraw_clicked:
                await asyncio.sleep(1)
                return {
                    "status": "success",
                    "message": f"Withdrawn connection request from {linkedin_username}",
                    "profile_url": profile_url,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Could not confirm withdrawal for {linkedin_username}",
                    "profile_url": profile_url,
                }

        except Exception as e:
            logger.error("Error withdrawing connection from %s: %s", linkedin_username, e)
            return {
                "status": "error",
                "message": f"Error withdrawing: {str(e)}",
                "profile_url": profile_url,
            }

    async def get_my_network(self) -> dict[str, Any]:
        """Get your LinkedIn network: pending invitations and connection suggestions.

        Returns:
            {
                "status": "success" | "error",
                "pending_invitations": list of dicts with name, headline, profile_url,
                "suggestions": list of dicts with name, headline, profile_url,
                "message": str
            }
        """
        network_url = "https://www.linkedin.com/mynetwork/"

        try:
            await self._page.goto(network_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)

            try:
                await self._page.wait_for_selector("main", timeout=10000)
            except PlaywrightTimeoutError:
                return {
                    "status": "error",
                    "message": "Network page did not load",
                }

            await asyncio.sleep(2)

            # Extract network data
            network_data = await self._page.evaluate("""() => {
                const result = {
                    pending_invitations: [],
                    suggestions: []
                };
                
                // Get pending invitations section
                const inviteSection = document.querySelector('[data-view-name="profile-pending-invite-card"]') ||
                                      document.querySelector('.invitation-card');
                if (inviteSection) {
                    const inviteCards = document.querySelectorAll('.invitation-card, [data-view-name="profile-pending-invite-card"]');
                    inviteCards.forEach(card => {
                        const nameEl = card.querySelector('.invitation-card__name, a[href*="/in/"]');
                        const headlineEl = card.querySelector('.invitation-card__subtitle, .invitation-card__occupation');
                        const linkEl = card.querySelector('a[href*="/in/"]');
                        
                        if (nameEl) {
                            result.pending_invitations.push({
                                name: nameEl.innerText.trim(),
                                headline: headlineEl ? headlineEl.innerText.trim() : '',
                                profile_url: linkEl ? linkEl.href : ''
                            });
                        }
                    });
                }
                
                // Get suggestions ("People you may know")
                const suggestionCards = document.querySelectorAll('.discover-entity-card, .mn-pymk-list__card');
                suggestionCards.forEach(card => {
                    const nameEl = card.querySelector('.discover-person-card__name, .mn-connection-card__name');
                    const headlineEl = card.querySelector('.discover-person-card__occupation, .mn-connection-card__occupation');
                    const linkEl = card.querySelector('a[href*="/in/"]');
                    
                    if (nameEl && linkEl) {
                        const profileUrl = linkEl.href.split('?')[0];
                        const username = profileUrl.split('/in/')[1]?.replace('/', '') || '';
                        
                        result.suggestions.push({
                            name: nameEl.innerText.trim(),
                            headline: headlineEl ? headlineEl.innerText.trim() : '',
                            profile_url: profileUrl,
                            username: username
                        });
                    }
                });
                
                return result;
            }""")

            return {
                "status": "success",
                "pending_invitations": network_data.get("pending_invitations", []),
                "suggestions": network_data.get("suggestions", [])[:10],  # Limit to 10
                "message": f"Found {len(network_data.get('pending_invitations', []))} pending invitations and {len(network_data.get('suggestions', []))} suggestions",
            }

        except Exception as e:
            logger.error("Error getting network: %s", e)
            return {
                "status": "error",
                "message": f"Error getting network: {str(e)}",
            }

    async def get_messages(
        self,
        linkedin_username: str | None = None,
    ) -> dict[str, Any]:
        """Get LinkedIn messages/inbox.

        Args:
            linkedin_username: Optional - if provided, gets conversation with this user.
                If None, gets list of recent conversations.

        Returns:
            {
                "status": "success" | "error",
                "conversations": list of conversations (when no username),
                "messages": list of messages (when username provided),
                "message": str
            }
        """
        try:
            if linkedin_username:
                # Get specific conversation
                # LinkedIn messaging URL uses profile URN, but we can search for the conversation
                messaging_url = f"https://www.linkedin.com/messaging/thread/new/?recipients={linkedin_username}"
                await self._page.goto(messaging_url, wait_until="domcontentloaded", timeout=30000)
            else:
                # Get inbox
                messaging_url = "https://www.linkedin.com/messaging/"
                await self._page.goto(messaging_url, wait_until="domcontentloaded", timeout=30000)

            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            if linkedin_username:
                # Extract messages from conversation
                messages = await self._page.evaluate("""() => {
                    const messages = [];
                    const msgElements = document.querySelectorAll('.msg-s-message-list__event, .msg-s-event-listitem');
                    
                    msgElements.forEach(el => {
                        const senderEl = el.querySelector('.msg-s-message-group__name, .msg-s-event-listitem__sender');
                        const contentEl = el.querySelector('.msg-s-event-listitem__body, .msg-s-message-group__content');
                        const timeEl = el.querySelector('.msg-s-message-list__time-heading, time');
                        
                        if (contentEl) {
                            messages.push({
                                sender: senderEl ? senderEl.innerText.trim() : 'Unknown',
                                content: contentEl.innerText.trim(),
                                time: timeEl ? timeEl.innerText.trim() : ''
                            });
                        }
                    });
                    
                    return messages;
                }""")

                return {
                    "status": "success",
                    "messages": messages[-20:],  # Last 20 messages
                    "conversation_with": linkedin_username,
                    "message": f"Found {len(messages)} messages with {linkedin_username}",
                }
            else:
                # Extract conversation list
                conversations = await self._page.evaluate("""() => {
                    const conversations = [];
                    const convElements = document.querySelectorAll('.msg-conversation-listitem, .msg-conversations-container__convo-item');
                    
                    convElements.forEach(el => {
                        const nameEl = el.querySelector('.msg-conversation-listitem__participant-names, .msg-conversation-card__participant-names');
                        const previewEl = el.querySelector('.msg-conversation-listitem__message-snippet, .msg-conversation-card__message-snippet');
                        const timeEl = el.querySelector('.msg-conversation-listitem__time-stamp, .msg-conversation-card__time-stamp');
                        const linkEl = el.querySelector('a[href*="/messaging/"]');
                        const unreadEl = el.querySelector('.msg-conversation-listitem__unread-count, .notification-badge');
                        
                        if (nameEl) {
                            // Try to extract username from link
                            let username = '';
                            if (linkEl) {
                                const href = linkEl.href;
                                const match = href.match(/thread\\/([^\\/\\?]+)/);
                                if (match) username = match[1];
                            }
                            
                            conversations.push({
                                name: nameEl.innerText.trim(),
                                last_message: previewEl ? previewEl.innerText.trim() : '',
                                time: timeEl ? timeEl.innerText.trim() : '',
                                unread: unreadEl ? parseInt(unreadEl.innerText.trim()) || 1 : 0,
                                thread_id: username
                            });
                        }
                    });
                    
                    return conversations;
                }""")

                return {
                    "status": "success",
                    "conversations": conversations[:15],  # Limit to 15
                    "message": f"Found {len(conversations)} conversations",
                }

        except Exception as e:
            logger.error("Error getting messages: %s", e)
            return {
                "status": "error",
                "message": f"Error getting messages: {str(e)}",
            }

    async def create_post(
        self,
        content: str,
        visibility: str = "anyone",
    ) -> dict[str, Any]:
        """Create a LinkedIn post.

        Args:
            content: The text content of the post
            visibility: Who can see the post - "anyone" (public), "connections" (connections only)

        Returns:
            {
                "status": "success" | "error",
                "message": str,
                "post_url": str (if available)
            }
        """
        feed_url = "https://www.linkedin.com/feed/"

        try:
            await self._page.goto(feed_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click "Start a post" button to open the post modal
            start_post_clicked = await self._page.evaluate("""() => {
                // Look for "Start a post" button or the share box
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.innerText.toLowerCase();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    if (text.includes('start a post') || ariaLabel.includes('start a post') ||
                        text.includes('share') || ariaLabel.includes('create a post')) {
                        btn.click();
                        return true;
                    }
                }
                
                // Try the share box placeholder
                const shareBox = document.querySelector('.share-box-feed-entry__trigger, .share-box__open');
                if (shareBox) {
                    shareBox.click();
                    return true;
                }
                
                return false;
            }""")

            if not start_post_clicked:
                return {
                    "status": "error",
                    "message": "Could not find 'Start a post' button",
                }

            # Wait for the post modal to open
            await asyncio.sleep(2)

            # Find and fill the text editor
            editor = self._page.locator('[role="textbox"], .ql-editor, .editor-content, [contenteditable="true"]')
            try:
                await editor.first.wait_for(timeout=5000)
                await editor.first.click()
                await asyncio.sleep(0.3)
                
                # Use keyboard to type (more reliable than fill for contenteditable)
                await self._page.keyboard.type(content, delay=10)
                logger.info("Filled post content")
            except PlaywrightTimeoutError:
                return {
                    "status": "error",
                    "message": "Post editor not found",
                }

            await asyncio.sleep(1)

            # Set visibility if not default
            if visibility == "connections":
                # Click the visibility dropdown
                visibility_clicked = await self._page.evaluate("""() => {
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                        const text = btn.innerText.toLowerCase();
                        if (ariaLabel.includes('visibility') || ariaLabel.includes('who can see') ||
                            text.includes('anyone') || text.includes('connections only')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }""")

                if visibility_clicked:
                    await asyncio.sleep(1)
                    # Select "Connections only"
                    await self._page.evaluate("""() => {
                        const options = document.querySelectorAll('[role="radio"], [role="option"], .share-visibility-list__item');
                        for (const opt of options) {
                            const text = opt.innerText.toLowerCase();
                            if (text.includes('connections only') || text.includes('conexões')) {
                                opt.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    await asyncio.sleep(0.5)

            # Click Post button
            post_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.innerText.trim().toLowerCase();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    // Look for Post button (not "Start a post")
                    if ((text === 'post' || text === 'publicar' || ariaLabel === 'post') &&
                        !text.includes('start') && btn.offsetParent !== null) {
                        // Check if button is enabled
                        if (!btn.disabled) {
                            btn.click();
                            return { clicked: true };
                        } else {
                            return { clicked: false, reason: 'button_disabled' };
                        }
                    }
                }
                return { clicked: false, reason: 'not_found' };
            }""")

            if not post_clicked.get("clicked"):
                return {
                    "status": "error",
                    "message": f"Could not click Post button: {post_clicked.get('reason', 'unknown')}",
                }

            # Wait for post to be published
            await asyncio.sleep(3)

            # Check if modal closed (success indicator)
            modal_closed = await self._page.evaluate("""() => {
                const modal = document.querySelector('[role="dialog"]');
                if (!modal) return true;
                const text = modal.innerText.toLowerCase();
                // Check if it's still the post creation modal
                return !(text.includes('create a post') || text.includes('start a post'));
            }""")

            if modal_closed:
                return {
                    "status": "success",
                    "message": f"Post published successfully! Visibility: {visibility}",
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                }
            else:
                return {
                    "status": "partial",
                    "message": "Post may not have been published - modal still visible",
                }

        except Exception as e:
            logger.error("Error creating post: %s", e)
            return {
                "status": "error",
                "message": f"Error creating post: {str(e)}",
            }

    async def get_my_profile(self, language: str = "en") -> dict[str, Any]:
        """Get the current user's profile data.

        Args:
            language: Profile language - "en" for English, "pt" for Portuguese

        Returns:
            A dictionary containing profile sections data.
        """
        profile_url = "https://www.linkedin.com/in/me/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Switch profile language if needed
            if language == "pt":
                lang_switched = await self._page.evaluate("""() => {
                    const languageSection = document.querySelector('[id*="profile-language"]');
                    if (languageSection) {
                        const ptRadio = languageSection.querySelector('input[value*="pt"], label:has-text("Português")');
                        if (ptRadio) {
                            ptRadio.click();
                            return true;
                        }
                    }
                    return false;
                }""")
                if lang_switched:
                    await asyncio.sleep(2)

            # Extract profile data using JavaScript
            profile_data = await self._page.evaluate("""() => {
                const data = {
                    name: '',
                    headline: '',
                    location: '',
                    about: '',
                    current_position: '',
                    company: '',
                    profile_url: window.location.href,
                    profile_languages: [],
                    experiences: [],
                    education: [],
                    skills: [],
                    languages: [],
                    certifications: [],
                    projects: [],
                    connection_count: '',
                    follower_count: ''
                };

                // Name
                const nameEl = document.querySelector('h1');
                if (nameEl) data.name = nameEl.innerText.trim();

                // Headline
                const headlineEl = document.querySelector('.text-body-medium.break-words');
                if (headlineEl) data.headline = headlineEl.innerText.trim();

                // Location
                const locationEl = document.querySelector('.text-body-small.inline.t-black--light.break-words');
                if (locationEl) data.location = locationEl.innerText.trim();

                // About section
                const aboutSection = document.querySelector('#about');
                if (aboutSection) {
                    const aboutContent = aboutSection.closest('section')?.querySelector('.inline-show-more-text span[aria-hidden="true"], .inline-show-more-text');
                    if (aboutContent) data.about = aboutContent.innerText.trim();
                }

                // Profile languages available
                const langSection = document.querySelector('section:has-text("Profile language")');
                if (langSection) {
                    const langRadios = langSection.querySelectorAll('input[type="radio"]');
                    langRadios.forEach(radio => {
                        const label = radio.closest('label') || radio.parentElement;
                        if (label) data.profile_languages.push(label.innerText.trim());
                    });
                }

                // Connection/Follower count
                const connEl = document.querySelector('a[href*="connections"] span, li.text-body-small');
                if (connEl) {
                    const text = connEl.innerText;
                    if (text.includes('connection')) data.connection_count = text;
                    if (text.includes('follower')) data.follower_count = text;
                }

                // Experience section
                const expSection = document.getElementById('experience');
                if (expSection) {
                    const expItems = expSection.closest('section')?.querySelectorAll('[data-view-name="profile-component-entity"]') || [];
                    expItems.forEach(item => {
                        const title = item.querySelector('.t-bold span')?.innerText.trim() || '';
                        const company = item.querySelector('.t-normal span')?.innerText.trim() || '';
                        const duration = item.querySelector('.t-black--light span')?.innerText.trim() || '';
                        if (title) {
                            data.experiences.push({ title, company, duration });
                        }
                    });
                }

                // Education section
                const eduSection = document.getElementById('education');
                if (eduSection) {
                    const eduItems = eduSection.closest('section')?.querySelectorAll('[data-view-name="profile-component-entity"]') || [];
                    eduItems.forEach(item => {
                        const school = item.querySelector('.t-bold span')?.innerText.trim() || '';
                        const degree = item.querySelector('.t-normal span')?.innerText.trim() || '';
                        const years = item.querySelector('.t-black--light span')?.innerText.trim() || '';
                        if (school) {
                            data.education.push({ school, degree, years });
                        }
                    });
                }

                // Skills section
                const skillsSection = document.getElementById('skills');
                if (skillsSection) {
                    const skillItems = skillsSection.closest('section')?.querySelectorAll('.t-bold span') || [];
                    skillItems.forEach(item => {
                        const skill = item.innerText.trim();
                        if (skill && !skill.includes('Show all')) {
                            data.skills.push(skill);
                        }
                    });
                }

                // Languages section
                const langSectionContent = document.querySelector('section:has(h2:has-text("Languages"))');
                if (langSectionContent) {
                    const langItems = langSectionContent.querySelectorAll('.t-bold');
                    langItems.forEach(item => {
                        const lang = item.innerText.trim();
                        const proficiency = item.nextElementSibling?.innerText.trim() || '';
                        if (lang && lang !== 'Languages') {
                            data.languages.push({ language: lang, proficiency });
                        }
                    });
                }

                return data;
            }""")

            return {
                "status": "success",
                "profile": profile_data,
                "language": language,
            }

        except Exception as e:
            logger.error("Error getting profile: %s", e)
            return {
                "status": "error",
                "message": f"Error getting profile: {str(e)}",
            }

    async def update_profile_intro(
        self,
        first_name: str | None = None,
        last_name: str | None = None,
        headline: str | None = None,
        industry: str | None = None,
        city: str | None = None,
        country: str | None = None,
        pronouns: str | None = None,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """Update profile intro section (name, headline, location, etc).

        Args:
            first_name: First name (optional)
            last_name: Last name (optional)
            headline: Professional headline (optional)
            industry: Industry (optional)
            city: City location (optional)
            country: Country/Region (optional)
            pronouns: Pronouns - "He/Him", "She/Her", "They/Them", "Custom" (optional)
            profile_language: Which profile language to edit - "en" or "pt"

        Returns:
            Status dictionary with success/error info.
        """
        edit_url = "https://www.linkedin.com/in/me/edit/intro/"

        try:
            await self._page.goto(edit_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Switch to the correct profile language tab if needed
            if profile_language == "pt":
                pt_button = self._page.locator('button:has-text("Portuguese"), button:has-text("Português")')
                if await pt_button.count() > 0:
                    await pt_button.first.click()
                    await asyncio.sleep(1)

            updates_made = []

            # Update first name
            if first_name:
                first_name_input = self._page.locator('input[id*="firstName"], input[name*="firstName"]')
                if await first_name_input.count() > 0:
                    await first_name_input.first.fill(first_name)
                    updates_made.append(f"first_name: {first_name}")

            # Update last name
            if last_name:
                last_name_input = self._page.locator('input[id*="lastName"], input[name*="lastName"]')
                if await last_name_input.count() > 0:
                    await last_name_input.first.fill(last_name)
                    updates_made.append(f"last_name: {last_name}")

            # Update headline
            if headline:
                headline_input = self._page.locator('[role="textbox"], textarea').filter(has_text=lambda el: el)
                # Try to find headline specifically
                headline_found = await self._page.evaluate("""(newHeadline) => {
                    const labels = document.querySelectorAll('label, span');
                    for (const label of labels) {
                        if (label.innerText.toLowerCase().includes('headline')) {
                            // Find associated input
                            const parent = label.closest('.artdeco-text-input--container, div');
                            const input = parent?.querySelector('[role="textbox"], textarea, .ql-editor');
                            if (input) {
                                if (input.contentEditable === 'true') {
                                    input.innerText = newHeadline;
                                } else {
                                    input.value = newHeadline;
                                }
                                input.dispatchEvent(new Event('input', { bubbles: true }));
                                return true;
                            }
                        }
                    }
                    return false;
                }""", headline)
                if headline_found:
                    updates_made.append(f"headline: {headline[:50]}...")

            # Update industry
            if industry:
                industry_input = self._page.locator('input[id*="industry"], input[aria-label*="Industry"]')
                if await industry_input.count() > 0:
                    await industry_input.first.fill(industry)
                    await asyncio.sleep(0.5)
                    # Select from autocomplete
                    await self._page.keyboard.press("ArrowDown")
                    await self._page.keyboard.press("Enter")
                    updates_made.append(f"industry: {industry}")

            # Update city
            if city:
                city_input = self._page.locator('input[id*="city"], input[aria-label*="City"]')
                if await city_input.count() > 0:
                    await city_input.first.fill(city)
                    await asyncio.sleep(0.5)
                    await self._page.keyboard.press("ArrowDown")
                    await self._page.keyboard.press("Enter")
                    updates_made.append(f"city: {city}")

            # Update country
            if country:
                country_input = self._page.locator('input[id*="country"], input[aria-label*="Country"]')
                if await country_input.count() > 0:
                    await country_input.first.fill(country)
                    await asyncio.sleep(0.5)
                    await self._page.keyboard.press("ArrowDown")
                    await self._page.keyboard.press("Enter")
                    updates_made.append(f"country: {country}")

            # Update pronouns
            if pronouns:
                pronouns_updated = await self._page.evaluate("""(newPronouns) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const options = select.querySelectorAll('option');
                        for (const opt of options) {
                            if (opt.innerText.toLowerCase().includes(newPronouns.toLowerCase())) {
                                select.value = opt.value;
                                select.dispatchEvent(new Event('change', { bubbles: true }));
                                return true;
                            }
                        }
                    }
                    return false;
                }""", pronouns)
                if pronouns_updated:
                    updates_made.append(f"pronouns: {pronouns}")

            if not updates_made:
                return {
                    "status": "warning",
                    "message": "No fields were provided to update",
                }

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Profile intro updated successfully",
                "updates": updates_made,
                "profile_language": profile_language,
            }

        except Exception as e:
            logger.error("Error updating profile intro: %s", e)
            return {
                "status": "error",
                "message": f"Error updating profile intro: {str(e)}",
            }

    async def update_profile_about(
        self,
        about_text: str,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """Update profile About section.

        Args:
            about_text: New about/summary text (max 2600 characters)
            profile_language: Which profile language to edit - "en" or "pt"

        Returns:
            Status dictionary with success/error info.
        """
        profile_url = "https://www.linkedin.com/in/me/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Edit About button
            edit_about_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    if (ariaLabel.includes('edit about')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }""")

            if not edit_about_clicked:
                return {
                    "status": "error",
                    "message": "Could not find Edit About button",
                }

            await asyncio.sleep(2)

            # Switch to the correct profile language tab if needed
            if profile_language == "pt":
                pt_button = self._page.locator('button:has-text("Portuguese"), button:has-text("Português")')
                if await pt_button.count() > 0:
                    await pt_button.first.click()
                    await asyncio.sleep(1)

            # Find and fill the about text area
            about_editor = self._page.locator('[role="textbox"], textarea, .ql-editor')
            if await about_editor.count() > 0:
                await about_editor.first.click()
                await asyncio.sleep(0.3)
                
                # Clear existing content
                await self._page.keyboard.press("Meta+a")
                await asyncio.sleep(0.1)
                
                # Type new content
                await self._page.keyboard.type(about_text, delay=5)
            else:
                return {
                    "status": "error",
                    "message": "About editor not found",
                }

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": "About section updated successfully",
                "about_preview": about_text[:200] + "..." if len(about_text) > 200 else about_text,
                "profile_language": profile_language,
            }

        except Exception as e:
            logger.error("Error updating about section: %s", e)
            return {
                "status": "error",
                "message": f"Error updating about section: {str(e)}",
            }

    async def add_experience(
        self,
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
        """Add a new experience entry to the profile.

        Args:
            title: Job title
            company: Company name
            location: Job location (optional)
            start_month: Start month name, e.g. "January" (optional)
            start_year: Start year, e.g. "2024" (optional)
            end_month: End month name (optional, ignored if is_current=True)
            end_year: End year (optional, ignored if is_current=True)
            is_current: Whether this is current position
            description: Job description (optional)
            employment_type: Type like "Full-time", "Part-time", "Contract" etc (optional)
            profile_language: Which profile language to edit - "en" or "pt"

        Returns:
            Status dictionary with success/error info.
        """
        add_exp_url = "https://www.linkedin.com/in/me/details/experience/"

        try:
            await self._page.goto(add_exp_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Add experience button
            add_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    if (ariaLabel.includes('add experience') || text.includes('add experience') ||
                        ariaLabel.includes('add position')) {
                        btn.click();
                        return true;
                    }
                }
                // Try the floating + button
                const addBtn = document.querySelector('[data-view-name="profile-add-position"] button');
                if (addBtn) {
                    addBtn.click();
                    return true;
                }
                return false;
            }""")

            if not add_clicked:
                # Try navigating directly to add form
                await self._page.goto("https://www.linkedin.com/in/me/overlay/add-experience/", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

            await asyncio.sleep(2)

            # Switch to the correct profile language tab if needed
            if profile_language == "pt":
                pt_button = self._page.locator('button:has-text("Portuguese"), button:has-text("Português")')
                if await pt_button.count() > 0:
                    await pt_button.first.click()
                    await asyncio.sleep(1)

            # Fill title
            title_input = self._page.locator('input[id*="title"], input[aria-label*="Title"]')
            if await title_input.count() > 0:
                await title_input.first.fill(title)
                await asyncio.sleep(0.3)

            # Fill company
            company_input = self._page.locator('input[id*="company"], input[aria-label*="Company"]')
            if await company_input.count() > 0:
                await company_input.first.fill(company)
                await asyncio.sleep(0.5)
                # Select from autocomplete
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")

            # Fill location if provided
            if location:
                location_input = self._page.locator('input[id*="location"], input[aria-label*="Location"]')
                if await location_input.count() > 0:
                    await location_input.first.fill(location)
                    await asyncio.sleep(0.5)
                    await self._page.keyboard.press("ArrowDown")
                    await self._page.keyboard.press("Enter")

            # Set employment type if provided
            if employment_type:
                await self._page.evaluate("""(empType) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`);
                        if (label?.innerText.toLowerCase().includes('employment') || 
                            label?.innerText.toLowerCase().includes('type')) {
                            const options = select.querySelectorAll('option');
                            for (const opt of options) {
                                if (opt.innerText.toLowerCase().includes(empType.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    return true;
                                }
                            }
                        }
                    }
                }""", employment_type)

            # Set "I currently work here" checkbox if current position
            if is_current:
                current_checkbox = self._page.locator('input[type="checkbox"]').filter(has_text=lambda el: el)
                await self._page.evaluate("""() => {
                    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                    for (const cb of checkboxes) {
                        const label = cb.closest('label') || document.querySelector(`label[for="${cb.id}"]`);
                        if (label?.innerText.toLowerCase().includes('current')) {
                            if (!cb.checked) cb.click();
                            return true;
                        }
                    }
                }""")

            # Set start date if provided
            if start_year:
                await self._page.evaluate("""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`) || select.previousElementSibling;
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        // Find start year select
                        if (labelText.includes('start') && (labelText.includes('year') || select.querySelector('option[value="2024"]'))) {
                            for (const opt of select.options) {
                                if (opt.value === year || opt.innerText === year) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                        
                        // Find start month select
                        if (month && labelText.includes('start') && labelText.includes('month')) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", start_year, start_month)

            # Set end date if provided and not current
            if end_year and not is_current:
                await self._page.evaluate("""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`) || select.previousElementSibling;
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (labelText.includes('end') && (labelText.includes('year') || select.querySelector('option[value="2024"]'))) {
                            for (const opt of select.options) {
                                if (opt.value === year || opt.innerText === year) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                        
                        if (month && labelText.includes('end') && labelText.includes('month')) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", end_year, end_month)

            # Fill description if provided
            if description:
                desc_editor = self._page.locator('[role="textbox"], textarea').filter(has_text=lambda el: el).last
                if await desc_editor.count() > 0:
                    await desc_editor.click()
                    await self._page.keyboard.type(description, delay=5)

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Experience added successfully: {title} at {company}",
                "profile_language": profile_language,
            }

        except Exception as e:
            logger.error("Error adding experience: %s", e)
            return {
                "status": "error",
                "message": f"Error adding experience: {str(e)}",
            }

    async def add_education(
        self,
        school: str,
        degree: str | None = None,
        field_of_study: str | None = None,
        start_year: str | None = None,
        end_year: str | None = None,
        description: str | None = None,
        profile_language: str = "en",
    ) -> dict[str, Any]:
        """Add a new education entry to the profile.

        Args:
            school: School/University name
            degree: Degree type, e.g. "Bachelor's degree" (optional)
            field_of_study: Field of study, e.g. "Computer Science" (optional)
            start_year: Start year (optional)
            end_year: End year (optional)
            description: Description/activities (optional)
            profile_language: Which profile language to edit - "en" or "pt"

        Returns:
            Status dictionary with success/error info.
        """
        add_edu_url = "https://www.linkedin.com/in/me/overlay/add-education/"

        try:
            await self._page.goto(add_edu_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Switch to the correct profile language tab if needed
            if profile_language == "pt":
                pt_button = self._page.locator('button:has-text("Portuguese"), button:has-text("Português")')
                if await pt_button.count() > 0:
                    await pt_button.first.click()
                    await asyncio.sleep(1)

            # Fill school
            school_input = self._page.locator('input[id*="school"], input[aria-label*="School"]')
            if await school_input.count() > 0:
                await school_input.first.fill(school)
                await asyncio.sleep(0.5)
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")

            # Fill degree
            if degree:
                degree_input = self._page.locator('input[id*="degree"], input[aria-label*="Degree"]')
                if await degree_input.count() > 0:
                    await degree_input.first.fill(degree)
                    await asyncio.sleep(0.3)

            # Fill field of study
            if field_of_study:
                field_input = self._page.locator('input[id*="field"], input[aria-label*="Field"]')
                if await field_input.count() > 0:
                    await field_input.first.fill(field_of_study)
                    await asyncio.sleep(0.3)

            # Set years
            if start_year or end_year:
                await self._page.evaluate("""(startYear, endYear) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`);
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (startYear && labelText.includes('start')) {
                            for (const opt of select.options) {
                                if (opt.value === startYear || opt.innerText === startYear) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                        
                        if (endYear && labelText.includes('end')) {
                            for (const opt of select.options) {
                                if (opt.value === endYear || opt.innerText === endYear) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", start_year, end_year)

            # Fill description
            if description:
                desc_editor = self._page.locator('[role="textbox"], textarea')
                if await desc_editor.count() > 0:
                    await desc_editor.last.click()
                    await self._page.keyboard.type(description, delay=5)

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Education added successfully: {school}",
                "profile_language": profile_language,
            }

        except Exception as e:
            logger.error("Error adding education: %s", e)
            return {
                "status": "error",
                "message": f"Error adding education: {str(e)}",
            }

    async def add_skill(
        self,
        skill_name: str,
    ) -> dict[str, Any]:
        """Add a new skill to the profile.

        Args:
            skill_name: Name of the skill to add

        Returns:
            Status dictionary with success/error info.
        """
        add_skill_url = "https://www.linkedin.com/in/me/details/skills/"

        try:
            await self._page.goto(add_skill_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Add skill button
            add_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    if (ariaLabel.includes('add skill') || text.includes('add skill') ||
                        ariaLabel.includes('add a skill')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }""")

            if not add_clicked:
                await self._page.goto("https://www.linkedin.com/in/me/overlay/add-skill/", wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

            await asyncio.sleep(2)

            # Fill skill name
            skill_input = self._page.locator('input[id*="skill"], input[aria-label*="Skill"]')
            if await skill_input.count() > 0:
                await skill_input.first.fill(skill_name)
                await asyncio.sleep(0.5)
                # Select from autocomplete
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")

            await asyncio.sleep(1)

            # Click Save/Add button
            save_button = self._page.locator('button:has-text("Save"), button:has-text("Add")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Skill added successfully: {skill_name}",
            }

        except Exception as e:
            logger.error("Error adding skill: %s", e)
            return {
                "status": "error",
                "message": f"Error adding skill: {str(e)}",
            }

    async def switch_profile_language(
        self,
        language: str,
    ) -> dict[str, Any]:
        """Switch the profile view between available languages.

        Args:
            language: Target language - "en" for English, "pt" for Portuguese

        Returns:
            Status dictionary with success/error info.
        """
        profile_url = "https://www.linkedin.com/in/me/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Find and click the language radio button
            lang_label = "English" if language == "en" else "Português"
            
            switched = await self._page.evaluate("""(langLabel) => {
                // Find the Profile language section
                const sections = document.querySelectorAll('section');
                for (const section of sections) {
                    const heading = section.querySelector('h2');
                    if (heading?.innerText.toLowerCase().includes('profile language')) {
                        const radios = section.querySelectorAll('input[type="radio"]');
                        for (const radio of radios) {
                            const label = radio.closest('label') || document.querySelector(`label[for="${radio.id}"]`);
                            if (label?.innerText.includes(langLabel)) {
                                radio.click();
                                return { success: true, language: langLabel };
                            }
                        }
                    }
                }
                return { success: false };
            }""", lang_label)

            if switched.get("success"):
                await asyncio.sleep(2)
                return {
                    "status": "success",
                    "message": f"Profile language switched to {lang_label}",
                    "language": language,
                }
            else:
                return {
                    "status": "error",
                    "message": f"Could not find {lang_label} profile language option. Make sure you have a profile in that language.",
                }

        except Exception as e:
            logger.error("Error switching profile language: %s", e)
            return {
                "status": "error",
                "message": f"Error switching profile language: {str(e)}",
            }

    # ===== JOB MANAGEMENT METHODS =====

    async def save_job(self, job_id: str) -> dict[str, Any]:
        """Save or unsave a job for later.

        Args:
            job_id: The LinkedIn job ID (from the job URL).

        Returns:
            Status dictionary with save status.
        """
        job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"

        try:
            await self._page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Save button
            save_result = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    if (ariaLabel.includes('save') || text.includes('save')) {
                        const wasSaved = ariaLabel.includes('unsave') || text.includes('unsave');
                        btn.click();
                        return { clicked: true, wasSaved };
                    }
                }
                return { clicked: false };
            }""")

            if save_result.get("clicked"):
                await asyncio.sleep(1)
                action = "unsaved" if save_result.get("wasSaved") else "saved"
                return {
                    "status": "success",
                    "message": f"Job {action} successfully",
                    "job_id": job_id,
                    "job_url": job_url,
                    "action": action,
                }
            else:
                return {
                    "status": "error",
                    "message": "Could not find Save button",
                    "job_id": job_id,
                }

        except Exception as e:
            logger.error("Error saving job %s: %s", job_id, e)
            return {
                "status": "error",
                "message": f"Error saving job: {str(e)}",
                "job_id": job_id,
            }

    async def apply_to_job(self, job_id: str) -> dict[str, Any]:
        """Apply to a job using Easy Apply.

        This only works for jobs with Easy Apply enabled.
        For jobs requiring external applications, returns the application URL.

        Args:
            job_id: The LinkedIn job ID (from the job URL).

        Returns:
            Status dictionary with application result.
        """
        job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"

        try:
            await self._page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Check for Easy Apply button
            apply_result = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    
                    // Check for Easy Apply
                    if (text.includes('easy apply') || ariaLabel.includes('easy apply')) {
                        btn.click();
                        return { type: 'easy_apply', clicked: true };
                    }
                    
                    // Check for external Apply button
                    if ((text === 'apply' || ariaLabel.includes('apply')) && 
                        !text.includes('easy')) {
                        return { type: 'external', clicked: false };
                    }
                }
                return { type: 'not_found' };
            }""")

            if apply_result.get("type") == "easy_apply" and apply_result.get("clicked"):
                await asyncio.sleep(2)

                # Check if Easy Apply modal opened
                modal_state = await self._page.evaluate("""() => {
                    const dialogs = document.querySelectorAll('[role="dialog"]');
                    for (const dialog of dialogs) {
                        const text = dialog.innerText.toLowerCase();
                        if (text.includes('apply') || text.includes('resume') || 
                            text.includes('contact info')) {
                            const buttons = Array.from(dialog.querySelectorAll('button'))
                                .map(b => b.innerText.trim());
                            return { 
                                opened: true, 
                                buttons,
                                hasNext: buttons.some(b => b.toLowerCase().includes('next')),
                                hasSubmit: buttons.some(b => b.toLowerCase().includes('submit'))
                            };
                        }
                    }
                    return { opened: false };
                }""")

                if modal_state.get("opened"):
                    return {
                        "status": "modal_opened",
                        "message": "Easy Apply modal opened. Additional steps required (resume, questions, etc.)",
                        "job_id": job_id,
                        "job_url": job_url,
                        "modal_buttons": modal_state.get("buttons", []),
                    }
                else:
                    return {
                        "status": "clicked",
                        "message": "Easy Apply button clicked but modal not detected",
                        "job_id": job_id,
                        "job_url": job_url,
                    }

            elif apply_result.get("type") == "external":
                # Get external application URL
                external_url = await self._page.evaluate("""() => {
                    const links = document.querySelectorAll('a');
                    for (const link of links) {
                        const text = link.innerText.toLowerCase();
                        if (text.includes('apply')) {
                            return link.href;
                        }
                    }
                    return null;
                }""")
                return {
                    "status": "external",
                    "message": "This job requires external application",
                    "job_id": job_id,
                    "job_url": job_url,
                    "external_url": external_url,
                }
            else:
                return {
                    "status": "not_available",
                    "message": "No Apply button found. Job may be closed or not available.",
                    "job_id": job_id,
                    "job_url": job_url,
                }

        except Exception as e:
            logger.error("Error applying to job %s: %s", job_id, e)
            return {
                "status": "error",
                "message": f"Error applying to job: {str(e)}",
                "job_id": job_id,
            }

    # ===== COMPANY METHODS =====

    async def follow_company(self, company_name: str) -> dict[str, Any]:
        """Follow or unfollow a company on LinkedIn.

        Args:
            company_name: LinkedIn company identifier (from URL, e.g., "google" for linkedin.com/company/google)

        Returns:
            Status dictionary with follow status.
        """
        company_url = f"https://www.linkedin.com/company/{company_name}/"

        try:
            await self._page.goto(company_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Follow/Following button
            follow_result = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    
                    if (text === 'follow' || ariaLabel.includes('follow')) {
                        const wasFollowing = text === 'following' || ariaLabel.includes('unfollow');
                        btn.click();
                        return { clicked: true, wasFollowing };
                    }
                    if (text === 'following') {
                        btn.click();
                        return { clicked: true, wasFollowing: true };
                    }
                }
                return { clicked: false };
            }""")

            if follow_result.get("clicked"):
                await asyncio.sleep(1)
                action = "unfollowed" if follow_result.get("wasFollowing") else "followed"
                return {
                    "status": "success",
                    "message": f"Company {action} successfully",
                    "company_name": company_name,
                    "company_url": company_url,
                    "action": action,
                }
            else:
                return {
                    "status": "error",
                    "message": "Could not find Follow button",
                    "company_name": company_name,
                }

        except Exception as e:
            logger.error("Error following company %s: %s", company_name, e)
            return {
                "status": "error",
                "message": f"Error following company: {str(e)}",
                "company_name": company_name,
            }

    # ===== NETWORK MANAGEMENT METHODS =====

    async def get_pending_invitations(self) -> dict[str, Any]:
        """Get list of pending connection invitations.

        Returns:
            Dictionary with pending invitations list.
        """
        network_url = "https://www.linkedin.com/mynetwork/invitation-manager/"

        try:
            await self._page.goto(network_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Extract invitation data
            invitations = await self._page.evaluate("""() => {
                const items = [];
                const inviteCards = document.querySelectorAll('[class*="invitation-card"], li[class*="invitation"]');
                
                for (const card of inviteCards) {
                    const nameEl = card.querySelector('strong, [class*="name"]');
                    const subtitleEl = card.querySelector('[class*="subtitle"], [class*="headline"]');
                    const imgEl = card.querySelector('img');
                    const linkEl = card.querySelector('a[href*="/in/"]');
                    
                    const name = nameEl?.innerText.trim() || '';
                    const headline = subtitleEl?.innerText.trim() || '';
                    const profileUrl = linkEl?.href || '';
                    const profileImage = imgEl?.src || '';
                    
                    if (name) {
                        items.push({
                            name,
                            headline,
                            profile_url: profileUrl,
                            profile_image: profileImage,
                        });
                    }
                }
                
                // Also try the main list format
                const listItems = document.querySelectorAll('main ul li');
                for (const li of listItems) {
                    const text = li.innerText;
                    const linkEl = li.querySelector('a[href*="/in/"]');
                    const acceptBtn = li.querySelector('button[aria-label*="Accept"]');
                    
                    if (acceptBtn && linkEl) {
                        const lines = text.split('\\n').filter(l => l.trim());
                        items.push({
                            name: lines[0] || '',
                            headline: lines[1] || '',
                            profile_url: linkEl.href,
                        });
                    }
                }
                
                return items;
            }""")

            # Remove duplicates
            seen = set()
            unique_invitations = []
            for inv in invitations:
                key = inv.get("profile_url") or inv.get("name")
                if key and key not in seen:
                    seen.add(key)
                    unique_invitations.append(inv)

            return {
                "status": "success",
                "count": len(unique_invitations),
                "invitations": unique_invitations,
            }

        except Exception as e:
            logger.error("Error getting pending invitations: %s", e)
            return {
                "status": "error",
                "message": f"Error getting invitations: {str(e)}",
            }

    async def accept_connection(self, person_name: str) -> dict[str, Any]:
        """Accept a pending connection invitation.

        Args:
            person_name: Name of the person to accept (partial match supported).

        Returns:
            Status dictionary with result.
        """
        network_url = "https://www.linkedin.com/mynetwork/invitation-manager/"

        try:
            await self._page.goto(network_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Find and click Accept button for matching person
            result = await self._page.evaluate("""(personName) => {
                const searchName = personName.toLowerCase();
                const buttons = document.querySelectorAll('button');
                
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    
                    if (ariaLabel.includes('accept') && ariaLabel.includes(searchName)) {
                        btn.click();
                        return { found: true, accepted: true, name: ariaLabel };
                    }
                }
                
                // Try finding by card content
                const cards = document.querySelectorAll('li');
                for (const card of cards) {
                    const cardText = card.innerText.toLowerCase();
                    if (cardText.includes(searchName)) {
                        const acceptBtn = card.querySelector('button[aria-label*="Accept"]');
                        if (acceptBtn) {
                            acceptBtn.click();
                            return { found: true, accepted: true };
                        }
                    }
                }
                
                return { found: false };
            }""", person_name)

            await asyncio.sleep(1)

            if result.get("accepted"):
                return {
                    "status": "success",
                    "message": f"Connection accepted for {person_name}",
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"No pending invitation found from {person_name}",
                }

        except Exception as e:
            logger.error("Error accepting connection from %s: %s", person_name, e)
            return {
                "status": "error",
                "message": f"Error accepting connection: {str(e)}",
            }

    async def reject_connection(self, person_name: str) -> dict[str, Any]:
        """Ignore/reject a pending connection invitation.

        Args:
            person_name: Name of the person to reject (partial match supported).

        Returns:
            Status dictionary with result.
        """
        network_url = "https://www.linkedin.com/mynetwork/invitation-manager/"

        try:
            await self._page.goto(network_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Find and click Ignore button for matching person
            result = await self._page.evaluate("""(personName) => {
                const searchName = personName.toLowerCase();
                const buttons = document.querySelectorAll('button');
                
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    
                    if (ariaLabel.includes('ignore') && ariaLabel.includes(searchName)) {
                        btn.click();
                        return { found: true, ignored: true, name: ariaLabel };
                    }
                }
                
                // Try finding by card content
                const cards = document.querySelectorAll('li');
                for (const card of cards) {
                    const cardText = card.innerText.toLowerCase();
                    if (cardText.includes(searchName)) {
                        const ignoreBtn = card.querySelector('button[aria-label*="Ignore"]');
                        if (ignoreBtn) {
                            ignoreBtn.click();
                            return { found: true, ignored: true };
                        }
                    }
                }
                
                return { found: false };
            }""", person_name)

            await asyncio.sleep(1)

            if result.get("ignored"):
                return {
                    "status": "success",
                    "message": f"Connection invitation ignored from {person_name}",
                }
            else:
                return {
                    "status": "not_found",
                    "message": f"No pending invitation found from {person_name}",
                }

        except Exception as e:
            logger.error("Error rejecting connection from %s: %s", person_name, e)
            return {
                "status": "error",
                "message": f"Error rejecting connection: {str(e)}",
            }

    async def get_connections(self, limit: int = 50) -> dict[str, Any]:
        """Get list of current connections.

        Args:
            limit: Maximum number of connections to retrieve (default 50).

        Returns:
            Dictionary with connections list.
        """
        connections_url = "https://www.linkedin.com/mynetwork/invite-connect/connections/"

        try:
            await self._page.goto(connections_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Scroll to load more connections if needed
            for _ in range(min(limit // 10, 10)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract connection data
            connections = await self._page.evaluate("""(maxCount) => {
                const items = [];
                const cards = document.querySelectorAll('li[class*="connection"], [class*="mn-connection-card"]');
                
                for (const card of cards) {
                    if (items.length >= maxCount) break;
                    
                    const nameEl = card.querySelector('[class*="name"], strong, a[href*="/in/"]');
                    const subtitleEl = card.querySelector('[class*="subtitle"], [class*="occupation"]');
                    const linkEl = card.querySelector('a[href*="/in/"]');
                    const imgEl = card.querySelector('img');
                    const timeEl = card.querySelector('[class*="time"], time');
                    
                    const name = nameEl?.innerText.trim() || '';
                    const headline = subtitleEl?.innerText.trim() || '';
                    const profileUrl = linkEl?.href || '';
                    const profileImage = imgEl?.src || '';
                    const connectedTime = timeEl?.innerText.trim() || '';
                    
                    if (name && profileUrl) {
                        const username = profileUrl.match(/\\/in\\/([^\\/\\?]+)/)?.[1] || '';
                        items.push({
                            name,
                            headline,
                            username,
                            profile_url: profileUrl,
                            profile_image: profileImage,
                            connected_time: connectedTime,
                        });
                    }
                }
                
                // Alternative: extract from any list with profile links
                if (items.length === 0) {
                    const listItems = document.querySelectorAll('main li');
                    for (const li of listItems) {
                        if (items.length >= maxCount) break;
                        
                        const linkEl = li.querySelector('a[href*="/in/"]');
                        if (linkEl) {
                            const text = li.innerText;
                            const lines = text.split('\\n').filter(l => l.trim());
                            const username = linkEl.href.match(/\\/in\\/([^\\/\\?]+)/)?.[1] || '';
                            
                            items.push({
                                name: lines[0] || '',
                                headline: lines[1] || '',
                                username,
                                profile_url: linkEl.href,
                            });
                        }
                    }
                }
                
                return items;
            }""", limit)

            # Get total count
            total_count = await self._page.evaluate("""() => {
                const countEl = document.querySelector('[class*="connections-count"], h1, h2');
                if (countEl) {
                    const match = countEl.innerText.match(/([\\d,]+)/);
                    return match ? parseInt(match[1].replace(',', '')) : 0;
                }
                return 0;
            }""")

            return {
                "status": "success",
                "total_connections": total_count,
                "retrieved": len(connections),
                "connections": connections,
            }

        except Exception as e:
            logger.error("Error getting connections: %s", e)
            return {
                "status": "error",
                "message": f"Error getting connections: {str(e)}",
            }

    async def remove_connection(self, linkedin_username: str) -> dict[str, Any]:
        """Remove an existing connection.

        Args:
            linkedin_username: LinkedIn username of the connection to remove.

        Returns:
            Status dictionary with result.
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click More button to reveal options
            more_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    const text = btn.innerText.toLowerCase();
                    if (ariaLabel.includes('more actions') || text === 'more' || 
                        ariaLabel.includes('more options')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }""")

            if more_clicked:
                await asyncio.sleep(1)

                # Click Remove connection option
                remove_clicked = await self._page.evaluate("""() => {
                    const items = document.querySelectorAll('[role="menuitem"], [role="option"], button, a');
                    for (const item of items) {
                        const text = item.innerText.toLowerCase();
                        if (text.includes('remove connection') || text.includes('remove ')) {
                            item.click();
                            return true;
                        }
                    }
                    return false;
                }""")

                if remove_clicked:
                    await asyncio.sleep(1)

                    # Confirm removal in dialog if present
                    await self._page.evaluate("""() => {
                        const dialogs = document.querySelectorAll('[role="dialog"], [role="alertdialog"]');
                        for (const dialog of dialogs) {
                            const buttons = dialog.querySelectorAll('button');
                            for (const btn of buttons) {
                                const text = btn.innerText.toLowerCase();
                                if (text.includes('remove') || text.includes('confirm')) {
                                    btn.click();
                                    return true;
                                }
                            }
                        }
                        return false;
                    }""")

                    await asyncio.sleep(1)

                    return {
                        "status": "success",
                        "message": f"Connection removed: {linkedin_username}",
                        "profile_url": profile_url,
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Could not find 'Remove connection' option. User might not be a connection.",
                        "profile_url": profile_url,
                    }
            else:
                return {
                    "status": "error",
                    "message": "Could not find More actions button",
                    "profile_url": profile_url,
                }

        except Exception as e:
            logger.error("Error removing connection %s: %s", linkedin_username, e)
            return {
                "status": "error",
                "message": f"Error removing connection: {str(e)}",
                "profile_url": profile_url,
            }

    # ===== NOTIFICATIONS METHODS =====

    async def get_notifications(
        self,
        filter_type: str = "all",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get LinkedIn notifications.

        Args:
            filter_type: Filter notifications by type. Options:
                - "all" (default)
                - "jobs" (job alerts)
                - "my_posts" (reactions/comments on your posts)
                - "mentions" (when you're mentioned)
            limit: Maximum number of notifications to retrieve.

        Returns:
            Dictionary with notifications list.
        """
        # Map filter types to LinkedIn URL parameters
        filter_map = {
            "all": "all",
            "jobs": "JOB",
            "my_posts": "MY_POSTS",
            "mentions": "MENTIONS",
        }
        
        filter_param = filter_map.get(filter_type, "all")
        notifications_url = f"https://www.linkedin.com/notifications/?filter={filter_param}"

        try:
            await self._page.goto(notifications_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click the appropriate filter radio button
            if filter_type != "all":
                await self._page.evaluate("""(filterType) => {
                    const radios = document.querySelectorAll('input[type="radio"]');
                    const labelMap = {
                        'jobs': 'Jobs',
                        'my_posts': 'My posts',
                        'mentions': 'Mentions'
                    };
                    const targetLabel = labelMap[filterType];
                    
                    for (const radio of radios) {
                        const label = document.querySelector(`label[for="${radio.id}"]`);
                        if (label?.innerText.trim() === targetLabel) {
                            radio.click();
                            return true;
                        }
                    }
                    return false;
                }""", filter_type)
                await asyncio.sleep(2)

            # Scroll to load more notifications
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract notifications
            notifications = await self._page.evaluate("""(maxCount) => {
                const items = [];
                const notifElements = document.querySelectorAll('main a[href*="linkedin.com"]');
                
                const seen = new Set();
                
                for (const el of notifElements) {
                    if (items.length >= maxCount) break;
                    
                    const parent = el.closest('li') || el.parentElement?.parentElement;
                    if (!parent) continue;
                    
                    const text = el.innerText.trim();
                    if (!text || text.length < 10 || seen.has(text)) continue;
                    seen.add(text);
                    
                    // Find time
                    const timeEl = parent.querySelector('time') || 
                        Array.from(parent.querySelectorAll('span')).find(s => /^\\d+[hdwm]$/.test(s.innerText.trim()));
                    const time = timeEl?.innerText.trim() || '';
                    
                    // Determine notification type
                    let type = 'unknown';
                    const lowerText = text.toLowerCase();
                    if (lowerText.includes('job') || lowerText.includes('opportunities')) {
                        type = 'job_alert';
                    } else if (lowerText.includes('posted')) {
                        type = 'post';
                    } else if (lowerText.includes('reacted') || lowerText.includes('liked')) {
                        type = 'reaction';
                    } else if (lowerText.includes('commented')) {
                        type = 'comment';
                    } else if (lowerText.includes('mentioned')) {
                        type = 'mention';
                    } else if (lowerText.includes('hiring')) {
                        type = 'hiring';
                    } else if (lowerText.includes('congratulate') || lowerText.includes('new position')) {
                        type = 'job_change';
                    } else if (lowerText.includes('birthday')) {
                        type = 'birthday';
                    } else if (lowerText.includes('endorsed')) {
                        type = 'endorsement';
                    }
                    
                    // Check if unread
                    const isUnread = parent.innerText.toLowerCase().includes('unread');
                    
                    items.push({
                        text: text.substring(0, 500),
                        type,
                        time,
                        url: el.href,
                        is_unread: isUnread,
                    });
                }
                
                return items;
            }""", limit)

            return {
                "status": "success",
                "filter": filter_type,
                "count": len(notifications),
                "notifications": notifications,
            }

        except Exception as e:
            logger.error("Error getting notifications: %s", e)
            return {
                "status": "error",
                "message": f"Error getting notifications: {str(e)}",
            }

    # ===== JOBS ADVANCED METHODS =====

    async def get_saved_jobs(self, limit: int = 50) -> dict[str, Any]:
        """Get saved jobs from My Jobs page.

        Args:
            limit: Maximum number of jobs to retrieve.

        Returns:
            Dictionary with saved jobs list.
        """
        url = "https://www.linkedin.com/my-items/saved-jobs/"

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click on Saved tab to ensure we're on the right tab
            await self._page.evaluate("""() => {
                const tabs = document.querySelectorAll('button[role="tab"]');
                for (const tab of tabs) {
                    if (tab.innerText.toLowerCase().includes('saved')) {
                        tab.click();
                        return true;
                    }
                }
                return false;
            }""")
            await asyncio.sleep(2)

            # Scroll to load more jobs
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract saved jobs
            jobs = await self._page.evaluate(r"""(maxCount) => {
                const items = [];
                const jobCards = document.querySelectorAll('li[class*="list"]');
                
                for (const card of jobCards) {
                    if (items.length >= maxCount) break;
                    
                    const titleEl = card.querySelector('a[href*="/jobs/view/"]');
                    const companyEl = card.querySelector('a[href*="/company/"]') || 
                        card.querySelector('span[class*="company"]');
                    const locationEl = Array.from(card.querySelectorAll('span'))
                        .find(s => s.innerText.includes(',') || /[A-Z][a-z]+ ?\(/.test(s.innerText));
                    
                    if (!titleEl) continue;
                    
                    const jobUrl = titleEl.href;
                    const jobIdMatch = jobUrl.match(/jobs\/view\/(\d+)/);
                    const jobId = jobIdMatch ? jobIdMatch[1] : null;
                    
                    // Check for saved date
                    const dateEl = Array.from(card.querySelectorAll('span, time'))
                        .find(el => el.innerText.match(/saved|ago|posted/i));
                    
                    // Check job status indicators
                    const cardText = card.innerText.toLowerCase();
                    const isActive = !cardText.includes('no longer accepting') && 
                        !cardText.includes('job closed');
                    
                    items.push({
                        job_id: jobId,
                        title: titleEl.innerText.trim(),
                        company: companyEl?.innerText.trim() || '',
                        location: locationEl?.innerText.trim() || '',
                        job_url: jobUrl,
                        saved_date: dateEl?.innerText.trim() || '',
                        is_active: isActive,
                    });
                }
                
                return items;
            }""", limit)

            return {
                "status": "success",
                "count": len(jobs),
                "jobs": jobs,
                "url": url,
            }

        except Exception as e:
            logger.error("Error getting saved jobs: %s", e)
            return {
                "status": "error",
                "message": f"Error getting saved jobs: {str(e)}",
            }

    async def get_applied_jobs(self, limit: int = 50) -> dict[str, Any]:
        """Get applied jobs from My Jobs page.

        Args:
            limit: Maximum number of jobs to retrieve.

        Returns:
            Dictionary with applied jobs list and application status.
        """
        url = "https://www.linkedin.com/my-items/saved-jobs/"

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click on Applied tab
            tab_clicked = await self._page.evaluate("""() => {
                const tabs = document.querySelectorAll('button[role="tab"]');
                for (const tab of tabs) {
                    if (tab.innerText.toLowerCase().includes('applied')) {
                        tab.click();
                        return true;
                    }
                }
                return false;
            }""")
            
            if not tab_clicked:
                return {
                    "status": "error",
                    "message": "Could not find Applied tab",
                }
            
            await asyncio.sleep(2)

            # Scroll to load more jobs
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract applied jobs
            jobs = await self._page.evaluate(r"""(maxCount) => {
                const items = [];
                const jobCards = document.querySelectorAll('li[class*="list"]');
                
                for (const card of jobCards) {
                    if (items.length >= maxCount) break;
                    
                    const titleEl = card.querySelector('a[href*="/jobs/view/"]');
                    const companyEl = card.querySelector('a[href*="/company/"]') || 
                        card.querySelector('span[class*="company"]');
                    const locationEl = Array.from(card.querySelectorAll('span'))
                        .find(s => s.innerText.includes(',') || /[A-Z][a-z]+ ?\(/.test(s.innerText));
                    
                    if (!titleEl) continue;
                    
                    const jobUrl = titleEl.href;
                    const jobIdMatch = jobUrl.match(/jobs\/view\/(\d+)/);
                    const jobId = jobIdMatch ? jobIdMatch[1] : null;
                    
                    // Find application date and status
                    const cardText = card.innerText;
                    const dateMatch = cardText.match(/Applied (\\d+ \\w+ ago|on \\w+ \\d+)/i);
                    const appliedDate = dateMatch ? dateMatch[1] : '';
                    
                    // Determine application status
                    let status = 'applied';
                    const lowerText = cardText.toLowerCase();
                    if (lowerText.includes('viewed') || lowerText.includes('seen')) {
                        status = 'viewed';
                    } else if (lowerText.includes('in progress')) {
                        status = 'in_progress';
                    } else if (lowerText.includes('not selected') || lowerText.includes('rejected')) {
                        status = 'rejected';
                    } else if (lowerText.includes('hired') || lowerText.includes('offer')) {
                        status = 'offered';
                    }
                    
                    items.push({
                        job_id: jobId,
                        title: titleEl.innerText.trim(),
                        company: companyEl?.innerText.trim() || '',
                        location: locationEl?.innerText.trim() || '',
                        job_url: jobUrl,
                        applied_date: appliedDate,
                        status: status,
                    });
                }
                
                return items;
            }""", limit)

            return {
                "status": "success",
                "count": len(jobs),
                "jobs": jobs,
                "url": url,
            }

        except Exception as e:
            logger.error("Error getting applied jobs: %s", e)
            return {
                "status": "error",
                "message": f"Error getting applied jobs: {str(e)}",
            }

    async def get_job_alerts(self) -> dict[str, Any]:
        """Get all job alerts configured by the user.

        Returns:
            Dictionary with job alerts list.
        """
        url = "https://www.linkedin.com/jobs/jam/"

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click "Show more" buttons to load all alerts
            for _ in range(3):
                show_more_clicked = await self._page.evaluate("""() => {
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.innerText.toLowerCase().includes('show') && 
                            btn.innerText.match(/\\d+ more/i)) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }""")
                if not show_more_clicked:
                    break
                await asyncio.sleep(1)

            # Extract job alerts
            alerts = await self._page.evaluate("""() => {
                const items = [];
                const alertItems = document.querySelectorAll('li');
                
                for (const item of alertItems) {
                    const text = item.innerText;
                    
                    // Look for alert patterns like "software engineer" + location
                    const editBtn = item.querySelector('button[aria-label*="Edit"]') ||
                        item.querySelector('button[aria-label*="edit"]');
                    
                    if (!editBtn) continue;
                    
                    const ariaLabel = editBtn.getAttribute('aria-label') || '';
                    const labelMatch = ariaLabel.match(/Edit (.*?) job alert in (.*)/i);
                    
                    let keywords = '';
                    let location = '';
                    
                    if (labelMatch) {
                        keywords = labelMatch[1];
                        location = labelMatch[2];
                    } else {
                        // Try to extract from text content
                        const lines = text.split('\\n').filter(l => l.trim());
                        if (lines.length >= 2) {
                            keywords = lines[0].trim();
                            location = lines[1].trim();
                        }
                    }
                    
                    if (keywords) {
                        items.push({
                            keywords: keywords,
                            location: location,
                            raw_text: text.substring(0, 200),
                        });
                    }
                }
                
                return items;
            }""")

            return {
                "status": "success",
                "count": len(alerts),
                "alerts": alerts,
                "url": url,
            }

        except Exception as e:
            logger.error("Error getting job alerts: %s", e)
            return {
                "status": "error",
                "message": f"Error getting job alerts: {str(e)}",
            }

    async def create_job_alert(
        self,
        keywords: str,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Create a new job alert.

        Args:
            keywords: Search keywords for the alert (e.g., "software engineer")
            location: Optional location for the alert (e.g., "Remote", "San Francisco")

        Returns:
            Dictionary with creation status.
        """
        # Start from jobs search page with the keywords
        search_url = f"https://www.linkedin.com/jobs/search/?keywords={quote_plus(keywords)}"
        if location:
            search_url += f"&location={quote_plus(location)}"

        try:
            await self._page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Look for "Set alert" or "Get notified" button
            alert_created = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.innerText.toLowerCase();
                    const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
                    
                    if (text.includes('set alert') || text.includes('get notified') ||
                        ariaLabel.includes('alert') || ariaLabel.includes('notif')) {
                        // Check if already set
                        if (text.includes('alert on') || ariaLabel.includes('turn off')) {
                            return { status: 'already_exists' };
                        }
                        btn.click();
                        return { status: 'clicked' };
                    }
                }
                return { status: 'not_found' };
            }""")

            if alert_created.get("status") == "already_exists":
                return {
                    "status": "already_exists",
                    "message": f"Job alert for '{keywords}' already exists",
                    "keywords": keywords,
                    "location": location,
                }
            elif alert_created.get("status") == "clicked":
                await asyncio.sleep(2)
                return {
                    "status": "success",
                    "message": f"Job alert created for '{keywords}'" + (f" in {location}" if location else ""),
                    "keywords": keywords,
                    "location": location,
                }
            else:
                return {
                    "status": "not_found",
                    "message": "Could not find alert button. Try searching for jobs first.",
                    "keywords": keywords,
                    "location": location,
                }

        except Exception as e:
            logger.error("Error creating job alert: %s", e)
            return {
                "status": "error",
                "message": f"Error creating job alert: {str(e)}",
            }

    # ===== COMPANY ADVANCED METHODS =====

    async def get_company_jobs(
        self,
        company_name: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get job openings from a specific company.

        Args:
            company_name: LinkedIn company identifier (e.g., "google", "microsoft")
            limit: Maximum number of jobs to retrieve.

        Returns:
            Dictionary with company jobs list.
        """
        url = f"https://www.linkedin.com/company/{company_name}/jobs/"

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Get total job count
            total_jobs = await self._page.evaluate("""() => {
                const header = document.querySelector('h2');
                if (header) {
                    const match = header.innerText.match(/(\\d[\\d,]*) job/i);
                    if (match) return parseInt(match[1].replace(',', ''));
                }
                return 0;
            }""")

            # Scroll to load more jobs
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract company jobs
            jobs = await self._page.evaluate(r"""(maxCount) => {
                const items = [];
                const jobCards = document.querySelectorAll('li');
                
                for (const card of jobCards) {
                    if (items.length >= maxCount) break;
                    
                    const titleEl = card.querySelector('a[href*="/jobs/"]');
                    if (!titleEl) continue;
                    
                    const title = titleEl.innerText.trim();
                    if (!title || title.length < 3) continue;
                    
                    const jobUrl = titleEl.href;
                    const jobIdMatch = jobUrl.match(/jobs\/view\/(\d+)/) || 
                        jobUrl.match(/jobs\/.*?\/(\d+)/);
                    const jobId = jobIdMatch ? jobIdMatch[1] : null;
                    
                    // Get location and posted time
                    const spans = card.querySelectorAll('span');
                    let location = '';
                    let postedTime = '';
                    
                    for (const span of spans) {
                        const text = span.innerText.trim();
                        if (text.includes(',') && !location) {
                            location = text;
                        } else if (text.match(/ago|posted|hour|day|week|month/i) && !postedTime) {
                            postedTime = text;
                        }
                    }
                    
                    // Check for Easy Apply badge
                    const hasEasyApply = card.innerText.toLowerCase().includes('easy apply');
                    
                    items.push({
                        job_id: jobId,
                        title: title,
                        location: location,
                        posted_time: postedTime,
                        job_url: jobUrl,
                        easy_apply: hasEasyApply,
                    });
                }
                
                return items;
            }""", limit)

            return {
                "status": "success",
                "company": company_name,
                "total_jobs": total_jobs,
                "count": len(jobs),
                "jobs": jobs,
                "url": url,
            }

        except Exception as e:
            logger.error("Error getting company jobs: %s", e)
            return {
                "status": "error",
                "message": f"Error getting company jobs: {str(e)}",
            }

    async def get_company_employees(
        self,
        company_name: str,
        role_filter: str | None = None,
        location_filter: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get employees from a specific company.

        Args:
            company_name: LinkedIn company identifier (e.g., "google", "microsoft")
            role_filter: Optional filter for job role (e.g., "Engineer", "Manager")
            location_filter: Optional filter for location (e.g., "San Francisco")
            limit: Maximum number of employees to retrieve.

        Returns:
            Dictionary with company employees list.
        """
        url = f"https://www.linkedin.com/company/{company_name}/people/"

        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Get total employee count
            total_employees = await self._page.evaluate("""() => {
                const spans = document.querySelectorAll('span');
                for (const span of spans) {
                    const text = span.innerText;
                    const match = text.match(/(\\d[\\d,]*) associated members/i);
                    if (match) return parseInt(match[1].replace(',', ''));
                }
                return 0;
            }""")

            # Apply filters if provided
            if role_filter:
                await self._page.evaluate("""(roleFilter) => {
                    const inputs = document.querySelectorAll('input[placeholder*="function"], input[aria-label*="function"]');
                    for (const input of inputs) {
                        input.value = roleFilter;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    // Try clicking "What they do" filter
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.innerText.toLowerCase().includes('what they do')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }""", role_filter)
                await asyncio.sleep(2)

            if location_filter:
                await self._page.evaluate("""(locationFilter) => {
                    const inputs = document.querySelectorAll('input[placeholder*="location"], input[aria-label*="location"]');
                    for (const input of inputs) {
                        input.value = locationFilter;
                        input.dispatchEvent(new Event('input', { bubbles: true }));
                        return true;
                    }
                    // Try clicking "Where they live" filter
                    const buttons = document.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.innerText.toLowerCase().includes('where they live')) {
                            btn.click();
                            return true;
                        }
                    }
                    return false;
                }""", location_filter)
                await asyncio.sleep(2)

            # Scroll to load more employees
            for _ in range(min(limit // 10, 5)):
                await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)

            # Extract employees
            employees = await self._page.evaluate("""(maxCount) => {
                const items = [];
                const cards = document.querySelectorAll('li');
                const seen = new Set();
                
                for (const card of cards) {
                    if (items.length >= maxCount) break;
                    
                    const nameEl = card.querySelector('a[href*="/in/"]');
                    if (!nameEl) continue;
                    
                    const name = nameEl.innerText.trim();
                    if (!name || name.length < 2 || seen.has(name)) continue;
                    seen.add(name);
                    
                    const profileUrl = nameEl.href;
                    const usernameMatch = profileUrl.match(/\\/in\\/([^/]+)/);
                    const username = usernameMatch ? usernameMatch[1] : null;
                    
                    // Get title/role
                    const titleEl = card.querySelector('span[dir="ltr"]') ||
                        Array.from(card.querySelectorAll('span')).find(s => 
                            s.innerText.length > 5 && !s.innerText.includes('degree'));
                    const title = titleEl?.innerText.trim() || '';
                    
                    // Get location
                    const locationEl = Array.from(card.querySelectorAll('span')).find(s =>
                        s.innerText.includes(',') || /[A-Z][a-z]+( Area)?$/.test(s.innerText.trim()));
                    const location = locationEl?.innerText.trim() || '';
                    
                    // Check connection status
                    const cardText = card.innerText.toLowerCase();
                    let connectionStatus = 'none';
                    if (cardText.includes('1st')) {
                        connectionStatus = '1st';
                    } else if (cardText.includes('2nd')) {
                        connectionStatus = '2nd';
                    } else if (cardText.includes('3rd')) {
                        connectionStatus = '3rd';
                    }
                    
                    // Check for mutual connections
                    const mutualMatch = card.innerText.match(/(\\d+) mutual connection/i);
                    const mutualConnections = mutualMatch ? parseInt(mutualMatch[1]) : 0;
                    
                    items.push({
                        name: name,
                        username: username,
                        title: title,
                        location: location,
                        profile_url: profileUrl,
                        connection_degree: connectionStatus,
                        mutual_connections: mutualConnections,
                    });
                }
                
                return items;
            }""", limit)

            return {
                "status": "success",
                "company": company_name,
                "total_employees": total_employees,
                "count": len(employees),
                "employees": employees,
                "url": url,
                "filters_applied": {
                    "role": role_filter,
                    "location": location_filter,
                },
            }

        except Exception as e:
            logger.error("Error getting company employees: %s", e)
            return {
                "status": "error",
                "message": f"Error getting company employees: {str(e)}",
            }

    # ===== EASY APPLY COMPLETE FLOW =====

    async def easy_apply_complete(
        self,
        job_id: str,
        phone_number: str | None = None,
        answers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Complete the Easy Apply flow for a job.

        This method handles the entire Easy Apply process including:
        - Contact information (uses existing profile data)
        - Resume (uses already uploaded resume on LinkedIn)
        - Additional questions
        - Review and submit

        Args:
            job_id: The LinkedIn job ID.
            phone_number: Optional phone number to use (with country code).
            answers: Optional dict mapping question keywords to answers.
                    Example: {"years experience": "5", "work authorization": "yes"}

        Returns:
            Dict with application result.
        """
        job_url = f"https://www.linkedin.com/jobs/view/{job_id}/"
        answers = answers or {}

        try:
            await self._page.goto(job_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Click Easy Apply button
            easy_apply_clicked = await self._page.evaluate("""() => {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.innerText.toLowerCase();
                    if (text.includes('easy apply')) {
                        btn.click();
                        return true;
                    }
                }
                return false;
            }""")

            if not easy_apply_clicked:
                # Check if it's an external application
                external_url = await self._page.evaluate("""() => {
                    const links = document.querySelectorAll('a');
                    for (const link of links) {
                        if (link.innerText.toLowerCase().includes('apply')) {
                            return link.href;
                        }
                    }
                    return null;
                }""")
                if external_url:
                    return {
                        "status": "external",
                        "message": "This job requires external application",
                        "job_id": job_id,
                        "external_url": external_url,
                    }
                return {
                    "status": "not_available",
                    "message": "No Easy Apply button found",
                    "job_id": job_id,
                }

            await asyncio.sleep(2)

            # Process Easy Apply modal steps
            max_steps = 10
            current_step = 0
            steps_completed = []

            while current_step < max_steps:
                current_step += 1

                # Check if modal is open
                modal_info = await self._page.evaluate("""() => {
                    const dialogs = document.querySelectorAll('[role="dialog"]');
                    for (const dialog of dialogs) {
                        const text = dialog.innerText;
                        const progress = dialog.querySelector('[role="progressbar"]');
                        const progressValue = progress?.getAttribute('aria-valuenow') || '0';
                        
                        // Find buttons
                        const buttons = Array.from(dialog.querySelectorAll('button'))
                            .map(b => ({ text: b.innerText.trim().toLowerCase(), disabled: b.disabled }));
                        
                        const hasNext = buttons.some(b => b.text.includes('next') || b.text.includes('continue'));
                        const hasSubmit = buttons.some(b => b.text.includes('submit'));
                        const hasReview = buttons.some(b => b.text.includes('review'));
                        
                        return {
                            found: true,
                            progress: parseInt(progressValue),
                            text: text,
                            buttons: buttons,
                            has_next: hasNext,
                            has_submit: hasSubmit,
                            has_review: hasReview,
                        };
                    }
                    return { found: false };
                }""")

                if not modal_info.get("found"):
                    # Check if we completed successfully
                    success_message = await self._page.evaluate("""() => {
                        const body = document.body.innerText.toLowerCase();
                        return body.includes('application submitted') || 
                               body.includes('applied successfully') ||
                               body.includes('your application was sent');
                    }""")
                    if success_message:
                        return {
                            "status": "success",
                            "message": "Application submitted successfully",
                            "job_id": job_id,
                            "job_url": job_url,
                            "steps_completed": steps_completed,
                        }
                    break

                step_text = modal_info.get("text", "")[:500].lower()

                # Handle phone number field
                if phone_number and ("phone" in step_text or "mobile" in step_text):
                    await self._page.evaluate("""(phoneNumber) => {
                        const inputs = document.querySelectorAll('input[type="tel"], input[name*="phone"], input[id*="phone"]');
                        for (const input of inputs) {
                            input.value = phoneNumber;
                            input.dispatchEvent(new Event('input', { bubbles: true }));
                            input.dispatchEvent(new Event('change', { bubbles: true }));
                            return true;
                        }
                        return false;
                    }""", phone_number)
                    steps_completed.append("phone_number")
                    await asyncio.sleep(0.5)

                # Handle resume selection (choose first available)
                if "resume" in step_text or "cv" in step_text:
                    await self._page.evaluate("""() => {
                        // Try to select the first radio button for resume
                        const radios = document.querySelectorAll('input[type="radio"]');
                        if (radios.length > 0) {
                            radios[0].click();
                            return true;
                        }
                        return false;
                    }""")
                    steps_completed.append("resume_selected")
                    await asyncio.sleep(0.5)

                # Handle additional questions
                if answers:
                    await self._page.evaluate("""(answers) => {
                        const labels = document.querySelectorAll('label');
                        const inputs = document.querySelectorAll('input, select, textarea');
                        
                        for (const [keyword, answer] of Object.entries(answers)) {
                            const keywordLower = keyword.toLowerCase();
                            
                            // Find matching label
                            for (const label of labels) {
                                if (label.innerText.toLowerCase().includes(keywordLower)) {
                                    // Find associated input
                                    const inputId = label.getAttribute('for');
                                    const input = inputId ? document.getElementById(inputId) : 
                                        label.parentElement?.querySelector('input, select, textarea');
                                    
                                    if (input) {
                                        if (input.tagName === 'SELECT') {
                                            // Find matching option
                                            for (const opt of input.options) {
                                                if (opt.text.toLowerCase().includes(answer.toLowerCase())) {
                                                    input.value = opt.value;
                                                    break;
                                                }
                                            }
                                        } else if (input.type === 'radio' || input.type === 'checkbox') {
                                            // Find matching radio/checkbox
                                            const group = document.querySelectorAll(`input[name="${input.name}"]`);
                                            for (const radio of group) {
                                                const radioLabel = document.querySelector(`label[for="${radio.id}"]`);
                                                if (radioLabel?.innerText.toLowerCase().includes(answer.toLowerCase())) {
                                                    radio.click();
                                                    break;
                                                }
                                            }
                                        } else {
                                            input.value = answer;
                                        }
                                        input.dispatchEvent(new Event('input', { bubbles: true }));
                                        input.dispatchEvent(new Event('change', { bubbles: true }));
                                    }
                                }
                            }
                        }
                    }""", answers)
                    steps_completed.append("questions_answered")
                    await asyncio.sleep(0.5)

                # Click next/continue/submit button
                if modal_info.get("has_submit"):
                    # Final step - submit
                    submitted = await self._page.evaluate("""() => {
                        const buttons = document.querySelectorAll('[role="dialog"] button');
                        for (const btn of buttons) {
                            if (btn.innerText.toLowerCase().includes('submit') && !btn.disabled) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    if submitted:
                        steps_completed.append("submitted")
                        await asyncio.sleep(3)
                        return {
                            "status": "success",
                            "message": "Application submitted successfully",
                            "job_id": job_id,
                            "job_url": job_url,
                            "steps_completed": steps_completed,
                        }
                elif modal_info.get("has_review"):
                    # Review step
                    await self._page.evaluate("""() => {
                        const buttons = document.querySelectorAll('[role="dialog"] button');
                        for (const btn of buttons) {
                            if (btn.innerText.toLowerCase().includes('review') && !btn.disabled) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    steps_completed.append("review")
                    await asyncio.sleep(2)
                elif modal_info.get("has_next"):
                    # Continue to next step
                    await self._page.evaluate("""() => {
                        const buttons = document.querySelectorAll('[role="dialog"] button');
                        for (const btn of buttons) {
                            const text = btn.innerText.toLowerCase();
                            if ((text.includes('next') || text.includes('continue')) && !btn.disabled) {
                                btn.click();
                                return true;
                            }
                        }
                        return false;
                    }""")
                    steps_completed.append(f"step_{current_step}")
                    await asyncio.sleep(2)
                else:
                    # No action button found
                    break

            return {
                "status": "incomplete",
                "message": "Easy Apply flow did not complete. May need manual intervention.",
                "job_id": job_id,
                "job_url": job_url,
                "steps_completed": steps_completed,
            }

        except Exception as e:
            logger.error("Error in Easy Apply: %s", e)
            return {
                "status": "error",
                "message": f"Error in Easy Apply: {str(e)}",
                "job_id": job_id,
            }

    # ===== PROFILE ADVANCED =====

    async def add_certification(
        self,
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
        """Add a new certification/license to the profile.

        Args:
            name: Certification name (required)
            issuing_organization: Organization that issued the certification (required)
            issue_month: Month issued, e.g. "January" (optional)
            issue_year: Year issued, e.g. "2024" (optional)
            expiration_month: Expiration month (optional)
            expiration_year: Expiration year (optional)
            has_expiration: Whether the certification expires (default True)
            credential_id: Credential ID (max 80 chars, optional)
            credential_url: URL to verify credential (optional)

        Returns:
            Status dictionary with success/error info.
        """
        add_cert_url = "https://www.linkedin.com/in/me/edit/forms/certification/new/"

        try:
            await self._page.goto(add_cert_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Fill certification name
            name_input = self._page.locator('input[id*="name"], input[aria-label*="Name"]').first
            if await name_input.count() > 0:
                await name_input.fill(name)
                await asyncio.sleep(0.3)

            # Fill issuing organization
            org_input = self._page.locator('input[id*="issuing"], input[aria-label*="Issuing organization"]')
            if await org_input.count() > 0:
                await org_input.first.fill(issuing_organization)
                await asyncio.sleep(0.5)
                # Select from autocomplete if available
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")

            # Handle "does not expire" checkbox if no expiration
            if not has_expiration:
                await self._page.evaluate(r"""() => {
                    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                    for (const cb of checkboxes) {
                        const label = cb.closest('label') || document.querySelector(`label[for="${cb.id}"]`);
                        const labelText = (label?.innerText || '').toLowerCase();
                        if (labelText.includes('expire') || labelText.includes('expiration')) {
                            if (!cb.checked) cb.click();
                            return true;
                        }
                    }
                    return false;
                }""")
                await asyncio.sleep(0.3)

            # Set issue date if provided
            if issue_year or issue_month:
                await self._page.evaluate(r"""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const container = select.closest('.artdeco-text-input--container') || select.parentElement;
                        const label = container?.querySelector('label') || 
                                     document.querySelector(`label[for="${select.id}"]`) ||
                                     select.previousElementSibling;
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        // Issue year
                        if ((labelText.includes('issue') || labelText.includes('start')) && 
                            (select.querySelector('option[value*="202"]') || labelText.includes('year'))) {
                            if (year) {
                                for (const opt of select.options) {
                                    if (opt.value === year || opt.innerText === year) {
                                        select.value = opt.value;
                                        select.dispatchEvent(new Event('change', { bubbles: true }));
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Issue month
                        if ((labelText.includes('issue') || labelText.includes('start')) && 
                            labelText.includes('month') && month) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", issue_year, issue_month)
                await asyncio.sleep(0.3)

            # Set expiration date if provided and has_expiration is True
            if has_expiration and (expiration_year or expiration_month):
                await self._page.evaluate(r"""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const container = select.closest('.artdeco-text-input--container') || select.parentElement;
                        const label = container?.querySelector('label') || 
                                     document.querySelector(`label[for="${select.id}"]`) ||
                                     select.previousElementSibling;
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        // Expiration year
                        if ((labelText.includes('expir') || labelText.includes('end')) && 
                            (select.querySelector('option[value*="202"]') || labelText.includes('year'))) {
                            if (year) {
                                for (const opt of select.options) {
                                    if (opt.value === year || opt.innerText === year) {
                                        select.value = opt.value;
                                        select.dispatchEvent(new Event('change', { bubbles: true }));
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Expiration month
                        if ((labelText.includes('expir') || labelText.includes('end')) && 
                            labelText.includes('month') && month) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", expiration_year, expiration_month)
                await asyncio.sleep(0.3)

            # Fill credential ID if provided
            if credential_id:
                cred_id_input = self._page.locator('input[id*="credential-id"], input[aria-label*="Credential ID"]')
                if await cred_id_input.count() > 0:
                    await cred_id_input.first.fill(credential_id[:80])  # Max 80 chars
                    await asyncio.sleep(0.3)

            # Fill credential URL if provided
            if credential_url:
                cred_url_input = self._page.locator('input[id*="credential-url"], input[aria-label*="Credential URL"], input[type="url"]')
                if await cred_url_input.count() > 0:
                    await cred_url_input.first.fill(credential_url)
                    await asyncio.sleep(0.3)

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Certification added: {name} from {issuing_organization}",
            }

        except Exception as e:
            logger.error("Error adding certification: %s", e)
            return {
                "status": "error",
                "message": f"Error adding certification: {str(e)}",
            }

    async def add_project(
        self,
        name: str,
        description: str | None = None,
        start_month: str | None = None,
        start_year: str | None = None,
        end_month: str | None = None,
        end_year: str | None = None,
        is_current: bool = False,
        project_url: str | None = None,
    ) -> dict[str, Any]:
        """Add a new project to the profile.

        Args:
            name: Project name (required)
            description: Project description (max 2000 chars, optional)
            start_month: Start month name, e.g. "January" (optional)
            start_year: Start year, e.g. "2024" (optional)
            end_month: End month name (optional, ignored if is_current=True)
            end_year: End year (optional, ignored if is_current=True)
            is_current: Whether currently working on this project
            project_url: URL to the project (optional)

        Returns:
            Status dictionary with success/error info.
        """
        add_project_url = "https://www.linkedin.com/in/me/edit/forms/project/new/"

        try:
            await self._page.goto(add_project_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Fill project name
            name_input = self._page.locator('input[id*="name"], input[aria-label*="Project name"], input[aria-label*="Name"]').first
            if await name_input.count() > 0:
                await name_input.fill(name)
                await asyncio.sleep(0.3)

            # Set "I am currently working on this project" checkbox if current
            if is_current:
                await self._page.evaluate(r"""() => {
                    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
                    for (const cb of checkboxes) {
                        const label = cb.closest('label') || document.querySelector(`label[for="${cb.id}"]`);
                        const labelText = (label?.innerText || '').toLowerCase();
                        if (labelText.includes('current')) {
                            if (!cb.checked) cb.click();
                            return true;
                        }
                    }
                    return false;
                }""")
                await asyncio.sleep(0.3)

            # Set start date if provided
            if start_year or start_month:
                await self._page.evaluate(r"""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const container = select.closest('.artdeco-text-input--container') || select.parentElement;
                        const label = container?.querySelector('label') || 
                                     document.querySelector(`label[for="${select.id}"]`);
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (labelText.includes('start') && (select.querySelector('option[value*="202"]') || labelText.includes('year'))) {
                            if (year) {
                                for (const opt of select.options) {
                                    if (opt.value === year || opt.innerText === year) {
                                        select.value = opt.value;
                                        select.dispatchEvent(new Event('change', { bubbles: true }));
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if (labelText.includes('start') && labelText.includes('month') && month) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", start_year, start_month)
                await asyncio.sleep(0.3)

            # Set end date if provided and not current
            if not is_current and (end_year or end_month):
                await self._page.evaluate(r"""(year, month) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const container = select.closest('.artdeco-text-input--container') || select.parentElement;
                        const label = container?.querySelector('label') || 
                                     document.querySelector(`label[for="${select.id}"]`);
                        const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (labelText.includes('end') && (select.querySelector('option[value*="202"]') || labelText.includes('year'))) {
                            if (year) {
                                for (const opt of select.options) {
                                    if (opt.value === year || opt.innerText === year) {
                                        select.value = opt.value;
                                        select.dispatchEvent(new Event('change', { bubbles: true }));
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if (labelText.includes('end') && labelText.includes('month') && month) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(month.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", end_year, end_month)
                await asyncio.sleep(0.3)

            # Fill project URL if provided
            if project_url:
                url_input = self._page.locator('input[type="url"], input[id*="url"], input[aria-label*="URL"]')
                if await url_input.count() > 0:
                    await url_input.first.fill(project_url)
                    await asyncio.sleep(0.3)

            # Fill description if provided
            if description:
                desc_input = self._page.locator('textarea, [role="textbox"]').filter(has_text=lambda el: el)
                if await desc_input.count() > 0:
                    await desc_input.last.click()
                    await self._page.keyboard.type(description[:2000], delay=5)  # Max 2000 chars
                    await asyncio.sleep(0.3)

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Project added: {name}",
            }

        except Exception as e:
            logger.error("Error adding project: %s", e)
            return {
                "status": "error",
                "message": f"Error adding project: {str(e)}",
            }

    async def add_language(
        self,
        language: str,
        proficiency: str = "Professional working",
    ) -> dict[str, Any]:
        """Add a new language to the profile.

        Args:
            language: Language name (required, e.g. "Spanish", "French")
            proficiency: Proficiency level. Options:
                - "Elementary" or "Elementary proficiency"
                - "Limited working" or "Limited working proficiency"
                - "Professional working" or "Professional working proficiency" (default)
                - "Full professional" or "Full professional proficiency"
                - "Native or bilingual" or "Native or bilingual proficiency"

        Returns:
            Status dictionary with success/error info.
        """
        add_lang_url = "https://www.linkedin.com/in/me/edit/forms/language/new/"

        try:
            await self._page.goto(add_lang_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Fill language name
            lang_input = self._page.locator('input[id*="language"], input[aria-label*="Language"]').first
            if await lang_input.count() > 0:
                await lang_input.fill(language)
                await asyncio.sleep(0.5)
                # Select from autocomplete
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")
                await asyncio.sleep(0.3)

            # Select proficiency level
            await self._page.evaluate(r"""(proficiency) => {
                const prof = proficiency.toLowerCase();
                const selects = document.querySelectorAll('select');
                for (const select of selects) {
                    const label = document.querySelector(`label[for="${select.id}"]`);
                    const labelText = (label?.innerText || select.getAttribute('aria-label') || '').toLowerCase();
                    if (labelText.includes('proficiency') || labelText.includes('level')) {
                        for (const opt of select.options) {
                            const optText = opt.innerText.toLowerCase();
                            if (optText.includes(prof) || prof.includes(optText.split(' ')[0])) {
                                select.value = opt.value;
                                select.dispatchEvent(new Event('change', { bubbles: true }));
                                return true;
                            }
                        }
                    }
                }
                return false;
            }""", proficiency)
            await asyncio.sleep(0.3)

            await asyncio.sleep(1)

            # Click Save button
            save_button = self._page.locator('button:has-text("Save")')
            if await save_button.count() > 0:
                await save_button.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Language added: {language} ({proficiency})",
            }

        except Exception as e:
            logger.error("Error adding language: %s", e)
            return {
                "status": "error",
                "message": f"Error adding language: {str(e)}",
            }

    async def get_profile_views(self, limit: int = 20) -> dict[str, Any]:
        """Get the list of people who viewed your profile.

        Args:
            limit: Maximum number of viewers to return (default 20)

        Returns:
            Dict with total_views and viewers list.
            Each viewer has: name, headline, time, profile_url.
            Note: Some viewer details may be hidden (LinkedIn Premium feature).
        """
        views_url = "https://www.linkedin.com/analytics/profile-views/"

        try:
            await self._page.goto(views_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Scroll to load more viewers
            await scroll_to_bottom(self._page)
            await asyncio.sleep(1)

            # Extract viewer data
            viewers_data = await self._page.evaluate(r"""(limit) => {
                const viewers = [];
                let totalViews = 0;
                
                // Try to find total views count
                const headings = document.querySelectorAll('h1, h2, .t-bold');
                for (const h of headings) {
                    const text = h.innerText;
                    const match = text.match(/(\d+)\s*(profile\s+view|viewer)/i);
                    if (match) {
                        totalViews = parseInt(match[1], 10);
                        break;
                    }
                }
                
                // Find viewer cards/list items
                const viewerElements = document.querySelectorAll(
                    '[data-view-name*="viewer"], .profile-viewer-card, ' +
                    'li[class*="viewer"], .pvs-list__item--one-column'
                );
                
                for (const el of viewerElements) {
                    if (viewers.length >= limit) break;
                    
                    const linkEl = el.querySelector('a[href*="/in/"]');
                    const nameEl = el.querySelector('.t-bold, .t-16, [class*="name"]') || linkEl;
                    const headlineEl = el.querySelector('.t-normal, .t-14, [class*="headline"]');
                    const timeEl = el.querySelector('.t-black--light, time, [class*="time"]');
                    
                    const name = nameEl?.innerText?.trim();
                    if (!name || name.toLowerCase().includes('linkedin member')) {
                        // Anonymous viewer - still include
                        viewers.push({
                            name: 'LinkedIn Member',
                            headline: 'Profile hidden',
                            time: timeEl?.innerText?.trim() || null,
                            profile_url: null,
                            is_anonymous: true,
                        });
                    } else if (name) {
                        viewers.push({
                            name: name.split('\n')[0].trim(),
                            headline: headlineEl?.innerText?.trim() || null,
                            time: timeEl?.innerText?.trim() || null,
                            profile_url: linkEl?.href || null,
                            is_anonymous: false,
                        });
                    }
                }
                
                // If no viewers found via cards, try innerText parsing
                if (viewers.length === 0) {
                    const mainContent = document.querySelector('main')?.innerText || '';
                    const lines = mainContent.split('\n').filter(l => l.trim());
                    
                    let i = 0;
                    while (viewers.length < limit && i < lines.length) {
                        const line = lines[i].trim();
                        // Look for patterns like name followed by headline and time
                        if (line && !line.includes('profile view') && !line.includes('Premium')) {
                            const nextLine = lines[i + 1]?.trim() || '';
                            const timeLine = lines[i + 2]?.trim() || '';
                            
                            if (timeLine.match(/\d+[hmd]\s*ago|yesterday|today/i) ||
                                nextLine.match(/at\s+\w+|engineer|manager|developer/i)) {
                                viewers.push({
                                    name: line,
                                    headline: nextLine,
                                    time: timeLine.match(/\d+[hmd]\s*ago|yesterday|today/i) ? 
                                          timeLine : null,
                                    profile_url: null,
                                    is_anonymous: false,
                                });
                                i += 3;
                                continue;
                            }
                        }
                        i++;
                    }
                }
                
                return {
                    total_views: totalViews,
                    viewers: viewers.slice(0, limit),
                };
            }""", limit)

            return {
                "status": "success",
                "total_views": viewers_data.get("total_views", 0),
                "viewers_count": len(viewers_data.get("viewers", [])),
                "viewers": viewers_data.get("viewers", []),
                "note": "Some viewers may be hidden. LinkedIn Premium shows all viewers.",
            }

        except Exception as e:
            logger.error("Error getting profile views: %s", e)
            return {
                "status": "error",
                "message": f"Error getting profile views: {str(e)}",
            }

    # ===== MESSAGES ADVANCED =====

    async def get_conversation(
        self,
        linkedin_username: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get the full conversation/messages with a specific person.

        Args:
            linkedin_username: LinkedIn username of the person
            limit: Maximum number of messages to retrieve (default 50)

        Returns:
            Dict with conversation details and messages list.
            Each message has: sender, text, time, is_you.
        """
        try:
            # First navigate to messaging to find the conversation
            messages_url = "https://www.linkedin.com/messaging/"
            await self._page.goto(messages_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Search for the conversation
            search_input = self._page.locator('input[placeholder*="Search messages"], input[aria-label*="Search"]')
            if await search_input.count() > 0:
                await search_input.first.fill(linkedin_username)
                await asyncio.sleep(1)

            # Click on the conversation if found
            conv_clicked = await self._page.evaluate(r"""(username) => {
                const conversations = document.querySelectorAll(
                    '[class*="msg-conversation-card"], li[class*="message"], ' +
                    '[data-control-name*="conversation"]'
                );
                
                for (const conv of conversations) {
                    const text = conv.innerText.toLowerCase();
                    const link = conv.querySelector('a')?.href || '';
                    if (text.includes(username.toLowerCase()) || link.includes(username)) {
                        conv.click();
                        return true;
                    }
                }
                
                // Try clicking directly on search result
                const results = document.querySelectorAll('[role="option"], [role="listitem"]');
                for (const r of results) {
                    if (r.innerText.toLowerCase().includes(username.toLowerCase())) {
                        r.click();
                        return true;
                    }
                }
                
                return false;
            }""", linkedin_username)

            if not conv_clicked:
                return {
                    "status": "error",
                    "message": f"Conversation with {linkedin_username} not found",
                }

            await asyncio.sleep(2)

            # Scroll up to load older messages
            await self._page.evaluate(r"""() => {
                const msgContainer = document.querySelector(
                    '[class*="msg-s-message-list"], [class*="conversation-content"]'
                );
                if (msgContainer) {
                    for (let i = 0; i < 5; i++) {
                        msgContainer.scrollTop = 0;
                    }
                }
            }""")
            await asyncio.sleep(1)

            # Extract messages
            messages_data = await self._page.evaluate(r"""(limit, targetUsername) => {
                const messages = [];
                
                const msgElements = document.querySelectorAll(
                    '[class*="msg-s-message-list__event"], [class*="message-event"], ' +
                    '.msg-s-event-listitem'
                );
                
                for (const msg of msgElements) {
                    if (messages.length >= limit) break;
                    
                    const senderEl = msg.querySelector(
                        '[class*="msg-s-message-group__name"], .msg-s-message-group__profile-link, ' +
                        '[class*="actor-name"]'
                    );
                    const textEl = msg.querySelector(
                        '[class*="msg-s-event-listitem__body"], .msg-s-event__content, ' +
                        '[class*="message-body"], p'
                    );
                    const timeEl = msg.querySelector(
                        'time, [class*="msg-s-message-group__timestamp"], [class*="time"]'
                    );
                    
                    const sender = senderEl?.innerText?.trim();
                    const text = textEl?.innerText?.trim();
                    
                    if (text) {
                        const isYou = msg.classList.toString().includes('outgoing') ||
                                     msg.querySelector('[class*="outgoing"]') !== null ||
                                     (sender && sender.toLowerCase() === 'you');
                        
                        messages.push({
                            sender: isYou ? 'You' : (sender || targetUsername),
                            text: text,
                            time: timeEl?.innerText?.trim() || timeEl?.getAttribute('datetime') || null,
                            is_you: isYou,
                        });
                    }
                }
                
                // If structured extraction failed, try innerText
                if (messages.length === 0) {
                    const threadEl = document.querySelector(
                        '[class*="msg-s-message-list"], [class*="conversation"]'
                    );
                    if (threadEl) {
                        const text = threadEl.innerText;
                        return {
                            raw_text: text.slice(0, 5000),
                            messages: [],
                        };
                    }
                }
                
                return { messages: messages.reverse() };  // Oldest first
            }""", limit, linkedin_username)

            return {
                "status": "success",
                "conversation_with": linkedin_username,
                "messages_count": len(messages_data.get("messages", [])),
                "messages": messages_data.get("messages", []),
                "raw_text": messages_data.get("raw_text"),
            }

        except Exception as e:
            logger.error("Error getting conversation: %s", e)
            return {
                "status": "error",
                "message": f"Error getting conversation: {str(e)}",
            }

    async def archive_conversation(self, linkedin_username: str) -> dict[str, Any]:
        """Archive a conversation with a specific person.

        Args:
            linkedin_username: LinkedIn username of the person

        Returns:
            Status dictionary with success/error info.
        """
        try:
            # Navigate to messaging
            messages_url = "https://www.linkedin.com/messaging/"
            await self._page.goto(messages_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Find and click the conversation
            conv_found = await self._page.evaluate(r"""(username) => {
                const conversations = document.querySelectorAll(
                    '[class*="msg-conversation-card"], li[class*="conversation"]'
                );
                
                for (const conv of conversations) {
                    const text = conv.innerText.toLowerCase();
                    const link = conv.querySelector('a')?.href || '';
                    if (text.includes(username.toLowerCase()) || link.includes(username)) {
                        // Find and click the options menu
                        const optionsBtn = conv.querySelector(
                            'button[aria-label*="option"], button[aria-label*="more"], ' +
                            '[data-control-name*="option"]'
                        );
                        if (optionsBtn) {
                            optionsBtn.click();
                            return true;
                        }
                        // Try right-clicking or hover menu
                        conv.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true }));
                        return true;
                    }
                }
                return false;
            }""", linkedin_username)

            if not conv_found:
                return {
                    "status": "error",
                    "message": f"Conversation with {linkedin_username} not found",
                }

            await asyncio.sleep(1)

            # Click Archive option
            archived = await self._page.evaluate(r"""() => {
                const menuItems = document.querySelectorAll(
                    '[role="menuitem"], [role="option"], button, [class*="dropdown__item"]'
                );
                
                for (const item of menuItems) {
                    const text = item.innerText.toLowerCase();
                    if (text.includes('archive')) {
                        item.click();
                        return true;
                    }
                }
                return false;
            }""")

            if not archived:
                return {
                    "status": "error",
                    "message": "Archive option not found in menu",
                }

            await asyncio.sleep(1)

            return {
                "status": "success",
                "message": f"Conversation with {linkedin_username} archived",
            }

        except Exception as e:
            logger.error("Error archiving conversation: %s", e)
            return {
                "status": "error",
                "message": f"Error archiving conversation: {str(e)}",
            }

    # ===== ENDORSEMENTS & RECOMMENDATIONS =====

    async def endorse_skill(
        self,
        linkedin_username: str,
        skill_name: str,
    ) -> dict[str, Any]:
        """Endorse a skill on someone's profile.

        Args:
            linkedin_username: LinkedIn username of the person to endorse
            skill_name: Name of the skill to endorse

        Returns:
            Status dictionary with success/error info.
        """
        profile_url = f"https://www.linkedin.com/in/{linkedin_username}/"

        try:
            await self._page.goto(profile_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Scroll to find skills section
            await scroll_to_bottom(self._page)
            await asyncio.sleep(1)

            # Try to find and click endorse button for the skill
            endorsed = await self._page.evaluate(r"""(skillName) => {
                const skill = skillName.toLowerCase();
                
                // Find skills section
                const skillsSection = document.querySelector(
                    '#skills, [id*="skills"], section[class*="skills"]'
                );
                
                // Look for skill items
                const skillItems = document.querySelectorAll(
                    '[class*="skill-card"], [class*="pv-skill-entity"], ' +
                    'li[class*="skill"], [data-field="skill"]'
                );
                
                for (const item of skillItems) {
                    const itemText = item.innerText.toLowerCase();
                    if (itemText.includes(skill)) {
                        // Look for endorse button (+ or "Endorse")
                        const endorseBtn = item.querySelector(
                            'button[aria-label*="ndorse"], button:has-text("Endorse"), ' +
                            'button[class*="endorse"], .pv-skill-entity__featured-endorse-button-shared'
                        );
                        
                        if (endorseBtn && !endorseBtn.disabled) {
                            endorseBtn.click();
                            return { found: true, endorsed: true };
                        }
                        
                        // Try the plus icon
                        const plusBtn = item.querySelector('button svg, button[class*="icon"]');
                        if (plusBtn) {
                            plusBtn.closest('button')?.click();
                            return { found: true, endorsed: true };
                        }
                        
                        return { found: true, endorsed: false, reason: 'Endorse button not available' };
                    }
                }
                
                return { found: false };
            }""", skill_name)

            if not endorsed.get("found"):
                # Try navigating to skills details page
                skills_url = f"https://www.linkedin.com/in/{linkedin_username}/details/skills/"
                await self._page.goto(skills_url, wait_until="domcontentloaded", timeout=30000)
                await asyncio.sleep(2)

                endorsed = await self._page.evaluate(r"""(skillName) => {
                    const skill = skillName.toLowerCase();
                    const items = document.querySelectorAll('li, [class*="skill"]');
                    
                    for (const item of items) {
                        if (item.innerText.toLowerCase().includes(skill)) {
                            const btn = item.querySelector('button');
                            if (btn && btn.innerText.toLowerCase().includes('endorse')) {
                                btn.click();
                                return { found: true, endorsed: true };
                            }
                        }
                    }
                    return { found: false };
                }""", skill_name)

            if endorsed.get("endorsed"):
                await asyncio.sleep(1)
                return {
                    "status": "success",
                    "message": f"Endorsed {skill_name} for {linkedin_username}",
                }
            elif endorsed.get("found"):
                return {
                    "status": "error",
                    "message": endorsed.get("reason", "Could not endorse skill"),
                }
            else:
                return {
                    "status": "error",
                    "message": f"Skill '{skill_name}' not found on {linkedin_username}'s profile",
                }

        except Exception as e:
            logger.error("Error endorsing skill: %s", e)
            return {
                "status": "error",
                "message": f"Error endorsing skill: {str(e)}",
            }

    async def request_recommendation(
        self,
        linkedin_username: str,
        message: str | None = None,
        relationship: str | None = None,
        position_at_time: str | None = None,
    ) -> dict[str, Any]:
        """Request a recommendation from a connection.

        Args:
            linkedin_username: LinkedIn username of the person to request from
            message: Custom message for the request (optional)
            relationship: How you know them, e.g. "colleague", "manager" (optional)
            position_at_time: Your position when you worked with them (optional)

        Returns:
            Status dictionary with success/error info.
        """
        request_url = "https://www.linkedin.com/in/me/details/recommendations/edit/request/"

        try:
            await self._page.goto(request_url, wait_until="domcontentloaded", timeout=30000)
            await detect_rate_limit(self._page)
            await asyncio.sleep(2)

            # Search for the person
            search_input = self._page.locator('input[placeholder*="Search"], input[aria-label*="Search for people"]')
            if await search_input.count() > 0:
                await search_input.first.fill(linkedin_username)
                await asyncio.sleep(1.5)

                # Select from dropdown
                await self._page.keyboard.press("ArrowDown")
                await self._page.keyboard.press("Enter")
                await asyncio.sleep(1)

            # Click Continue/Next to go to step 2
            continue_btn = self._page.locator('button:has-text("Continue"), button:has-text("Next")')
            if await continue_btn.count() > 0:
                await continue_btn.first.click()
                await asyncio.sleep(2)

            # Fill relationship if provided (step 2)
            if relationship:
                await self._page.evaluate(r"""(rel) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`);
                        if (label?.innerText.toLowerCase().includes('relationship')) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(rel.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", relationship)
                await asyncio.sleep(0.3)

            # Fill position if provided
            if position_at_time:
                await self._page.evaluate(r"""(pos) => {
                    const selects = document.querySelectorAll('select');
                    for (const select of selects) {
                        const label = document.querySelector(`label[for="${select.id}"]`);
                        if (label?.innerText.toLowerCase().includes('position')) {
                            for (const opt of select.options) {
                                if (opt.innerText.toLowerCase().includes(pos.toLowerCase())) {
                                    select.value = opt.value;
                                    select.dispatchEvent(new Event('change', { bubbles: true }));
                                    break;
                                }
                            }
                        }
                    }
                }""", position_at_time)
                await asyncio.sleep(0.3)

            # Fill custom message if provided
            if message:
                msg_input = self._page.locator('textarea, [role="textbox"]')
                if await msg_input.count() > 0:
                    await msg_input.last.fill("")
                    await msg_input.last.type(message, delay=5)
                    await asyncio.sleep(0.3)

            # Click Send button
            send_btn = self._page.locator('button:has-text("Send"), button:has-text("Request")')
            if await send_btn.count() > 0:
                await send_btn.first.click()
                await asyncio.sleep(2)

            return {
                "status": "success",
                "message": f"Recommendation request sent to {linkedin_username}",
            }

        except Exception as e:
            logger.error("Error requesting recommendation: %s", e)
            return {
                "status": "error",
                "message": f"Error requesting recommendation: {str(e)}",
            }
