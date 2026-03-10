import asyncio
import json
import os.path
import random
from playwright.async_api import Page

DATE_FORMAT = '%Y-%m-%d'

VIEWPORTS = [
    {'width': 1024, 'height': 1366},
]

def update_session(name: str, new_data: dict) -> None:
    """Updates or creates a session file."""
    os.makedirs("sessions", exist_ok=True)
    session_file_path = os.path.join("sessions", f"{name}.json")
    try:
        with open(session_file_path, "w") as file:
            json.dump(new_data, file, indent=4)
        print(f"Session {name} successfully updated.")
    except Exception as e:
        print(f"Failed to update session {name}: {e}")


async def perform_random_mouse_movements(page: Page, min_actions: int = 2, max_actions: int = 5,
                                         min_pause: float = 0.5, max_pause: float = 1.0):
    """
    Performs random mouse movements and scrolling actions on a Playwright page
    without any clicking.
    """
    # Get page dimensions
    dimensions = await page.evaluate('''() => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            scrollHeight: document.documentElement.scrollHeight
        }
    }''')

    viewport_width = dimensions['width']
    viewport_height = dimensions['height']
    scroll_height = dimensions['scrollHeight']

    # Determine number of actions to perform
    num_actions = random.randint(min_actions, max_actions)

    for _ in range(num_actions):
        # Choose a random action (no clicks)
        action = random.choice(['move', 'hover_element', 'scroll_down', 'scroll_up', 'pause'])

        # Random pause between actions
        await asyncio.sleep(random.uniform(min_pause, max_pause))

        if action == 'move':
            # Move to random position
            x = random.randint(0, viewport_width)
            y = random.randint(0, viewport_height)
            await page.mouse.move(x, y)

        elif action == 'hover_element':
            # Find a random clickable element and hover over it
            clickable_elements = await page.evaluate('''() => {
                const elements = Array.from(document.querySelectorAll('a, button, [role="button"], input[type="submit"]'));
                return elements.map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        width: rect.width,
                        height: rect.height,
                        visible: rect.width > 0 && rect.height > 0 &&
                                rect.top >= 0 && rect.left >= 0 &&
                                rect.bottom <= window.innerHeight &&
                                rect.right <= window.innerWidth
                    };
                }).filter(el => el.visible);
            }''')

            if clickable_elements:
                element = random.choice(clickable_elements)
                await page.mouse.move(element['x'], element['y'])

        elif action == 'scroll_down':
            # Random scroll down amount (between 100 and 800 pixels)
            scroll_amount = random.randint(100, 800)
            current_scroll = await page.evaluate('window.scrollY')

            # Ensure we don't scroll past the bottom
            max_scroll = scroll_height - viewport_height
            if current_scroll < max_scroll:
                # Use smooth scrolling
                await page.evaluate(f'window.scrollBy({{ top: {scroll_amount}, behavior: "smooth" }})')

        elif action == 'scroll_up':
            # Random scroll up amount (between 100 and 400 pixels)
            if await page.evaluate('window.scrollY') > 0:
                scroll_amount = random.randint(100, 400)
                await page.evaluate(f'window.scrollBy({{ top: -{scroll_amount}, behavior: "smooth" }})')

        elif action == 'pause':
            # Just pause for a random time (already handled above)
            pass

    # Final random pause
    await asyncio.sleep(random.uniform(min_pause, max_pause))


async def scroll_down(page: Page, button_selector: str, scroll_amount: int) -> None:
    try:
        await page.mouse.wheel(0, scroll_amount)
        await asyncio.sleep(random.uniform(1, 3))
        await page.click(button_selector, timeout=1000)
    except Exception as e:
        print(f"failed to press button: {e}")


def read_session(session_name: str) -> dict | None:
    """Reads a specific session file."""
    session_file_path = os.path.join("sessions", f"{session_name}.json")
    try:
        if not os.path.exists(session_file_path):
            print(
                f"Session file for {session_name} not found, not using session."
            )
            return None
        with open(session_file_path, "r") as file:
            content = file.read()
            if not content.strip():
                print(
                    f"Session file for {session_name} is empty, not using session."
                )
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        print(
            f"Failed to decode session {session_name}, not using session."
        )
        return None
    except Exception as e:
        print(f"Unexpected error reading session {session_name}: {e}")
        return None


def read_cookies(cookies_name: str) -> dict | None:
    """Reads a specific session file."""
    cookies_file_path = os.path.join("cookies", f"{cookies_name}-cookies.json")
    try:
        if not os.path.exists(cookies_file_path):
            print(
                f"Cookies file for {cookies_name} not found, not using cookies."
            )
            return None
        with open(cookies_file_path, "r") as file:
            content = file.read()
            if not content.strip():
                print(
                    f"Cookies file for {cookies_name} is empty, not using cookies."
                )
                return None
            return json.loads(content)
    except json.JSONDecodeError:
        print(
            f"Failed to decode cookies {cookies_name}, not using cookies."
        )
        return None
    except Exception as e:
        print(f"Unexpected error reading cookies {cookies_name}: {e}")
        return None