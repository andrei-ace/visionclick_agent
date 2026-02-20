import time
import base64
import ast
import os
from datetime import datetime
import pyautogui
from PIL import Image, ImageDraw
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage

# ---------------------------------------------------------
# 1. State and Helper Variables
# ---------------------------------------------------------
CURRENT_SCREENSHOT = "current_screen.png"

# ---------------------------------------------------------
# Run tracing
# ---------------------------------------------------------
_RUN_DIR: str = ""
_SCREENSHOT_COUNTER: int = 0


def _init_run_dir() -> str:
    global _RUN_DIR, _SCREENSHOT_COUNTER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    _RUN_DIR = run_dir
    _SCREENSHOT_COUNTER = 0
    return run_dir


def _next_screenshot_index() -> int:
    global _SCREENSHOT_COUNTER
    idx = _SCREENSHOT_COUNTER
    _SCREENSHOT_COUNTER += 1
    return idx


def take_screenshot(out_path=CURRENT_SCREENSHOT):
    img = pyautogui.screenshot()
    img.save(out_path)
    return out_path, img.size


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _save_trace_screenshot(index: int) -> str:
    """Take a screenshot, save it to the run dir, return the run-dir path."""
    path = os.path.join(_RUN_DIR, f"screenshot_{index}.png")
    img = pyautogui.screenshot()
    img.save(path)
    # Also refresh the shared current screenshot
    img.save(CURRENT_SCREENSHOT)
    return path


def _save_annotated_screenshot(index: int, bbox: tuple[int, int, int, int]) -> str:
    """
    Draw a red bounding box on the existing screenshot_{index}.png
    and save it as screenshot_{index}_annotated.png.

    bbox: (xmin, ymin, xmax, ymax) in model-space pixels (0-1000 scale).
    """
    src_path = os.path.join(_RUN_DIR, f"screenshot_{index}.png")
    dst_path = os.path.join(_RUN_DIR, f"screenshot_{index}_annotated.png")

    img = Image.open(src_path).convert("RGB")
    w, h = img.size

    # Scale from model 0-1000 space to actual image pixels
    xmin, ymin, xmax, ymax = bbox
    sx = w / 1000.0
    sy = h / 1000.0
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [xmin * sx, ymin * sy, xmax * sx, ymax * sy],
        outline="red",
        width=4,
    )
    img.save(dst_path)
    return dst_path


def _save_description(index: int, description: str) -> str:
    """Save the vision model description to screenshot_{index}.txt."""
    path = os.path.join(_RUN_DIR, f"screenshot_{index}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(description)
    return path


def _save_prompt(prompt: str) -> str:
    """Save the user prompt to prompt.txt in the run directory."""
    path = os.path.join(_RUN_DIR, "prompt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt)
    return path


# Initialize vision model for the tools to share
vision_model = ChatOllama(model="qwen3-vl:30b", temperature=0, num_ctx=8192)
# Initialize the reasoning LLM
llm = ChatOllama(model="gpt-oss", temperature=0, num_ctx=16384)


# ---------------------------------------------------------
# 2. Define the Agent Tools
# ---------------------------------------------------------
@tool
def describe_webpage(intention: str = "general") -> str:
    """
    Screenshot the current screen and summarize the visible UI state so an automation agent can decide the next action.

    Args:
        intention: What the agent is trying to accomplish (e.g., "set date range",
                   "dismiss blockers", "find search box", "locate next button").
                   Used only to prioritize what details to include (facts only).

    Returns:
        A short UI STATE REPORT (facts only) with blockers, open panels, navigation controls,
        focus indication, and key visible text relevant to the intention.
    """
    print(f"\n-> describe_webpage(intention='{intention}')")

    index = _next_screenshot_index()
    image_path = _save_trace_screenshot(index)
    base64_img = encode_image(image_path)

    prompt_text = f"""
Inspect the screenshot carefully. Think silently.

- Use this intention only to prioritize what details to include.
- Report facts only (what is visible). Do NOT give steps or advice.

Return a UI STATE REPORT using EXACTLY the sections below, in this order.
Formatting rules:
- Each section header must be on its own line and end with a colon.
- Each line inside a section must start with "- ".
- Prefer exact visible UI text in quotes. If unreadable, describe the element and its location.

BLOCKERS:
- "none" OR describe any overlay/modal/banner blocking interaction.
- Include visible dismiss/accept/close controls (exact text if possible).

PAGE_CONTEXT:
- Briefly describe what this screen appears to be (site/app + page type).
- State the primary purpose of the visible area (form, results, settings, etc.).

PRIMARY_INTERACTION:
- What appears to be the main interactive region right now (form section, open panel, modal, results list).
- Mention any element that looks active (caret, highlighted field, selected item).

OPEN_PANELS:
- List all open panels/popups/widgets (dropdown, modal, picker, calendar, sidebar, menu, tooltip).
- For each panel include:
  - TYPE: (calendar / dropdown / modal / sidebar / menu / picker / tooltip / other)
  - TITLE/HEADER: exact visible title text if any (in quotes) else "none"
  - ANCHOR: what it appears attached to (e.g., under a field, in center modal) if obvious
  - POSITION: where on screen (center / left / right / below top bar / etc.)

PANEL_LAYOUT:
- If no open panel has multiple panes/columns/sections: "- none"
- Otherwise for the most prominent open panel, describe its structure:
  - LAYOUT: single / two-pane / multi-pane / split view / columns
  - PANE_TITLES: list pane titles/headers in left-to-right order (exact text in quotes if possible)
  - PANE_ROLES: what each pane contains (e.g., "left: month grid", "right: month grid", "left: categories", "right: options")
  - REPEATED_ITEMS: note if the same labels/numbers/options appear in more than one pane (e.g., repeated day numbers, repeated options)

PANEL_DETAILS:
- For the most prominent open panel (or "- none" if no panels open), report:
  - CONTENT: what it contains at a high level (grid/list/options)
  - CONTROLS: buttons/tabs/chips within the panel (exact text if possible)
  - NAVIGATION: any prev/next/paging controls and where they are relative to the panel header
  - CONFIRM: any Apply/Done/OK/Close controls (exact text) or "none"
- If PANEL_LAYOUT is not "none", ALSO include:
  - LEFT_PANE: summary of what is in the left pane + its header text (if any)
  - RIGHT_PANE: summary of what is in the right pane + its header text (if any)

VISIBLE_FIELDS_AND_VALUES:
- List important visible fields and their current values/placeholders (exact text in quotes).

KEY_ACTIONS:
- List prominent actionable controls (primary buttons, top actions) with exact text + general location.

NOTES:
- Any ambiguity that affects targeting (duplicate labels, repeated numbers, similar buttons).
- Visually distinctive anchors (icons/shapes/placement), but no coordinates.

Rules recap:
- No coordinates.
- No tool instructions.
- No imperative verbs.
- Facts only, screenshot is the source of truth.

INTENTION: {intention}
Now produce the UI STATE REPORT.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
        ]
    )

    response = vision_model.invoke([message])
    description = str(response.content).strip()

    _save_description(index, description)
    print(f"Vision model response:\n{description}\n")
    print(f"[trace] saved screenshot_{index}.png + screenshot_{index}.txt → {_RUN_DIR}")
    return description


@tool
def get_coordinates_for(query: str) -> dict:
    """
    Locate ONE visible UI element in the current screenshot and return a screen click point.

    Use this for:
    - Clickable controls: buttons, links, icons, tabs, menu items.
    - Inputs: text fields, pickers, search boxes.
    - Items inside containers: a specific row in a list, a specific cell in a grid, a specific option in a dropdown.

    Writing a good query (make the target uniquely identifiable on the screenshot):
    - Prefer exact visible text when available: "Search", "Done", "Apply".
    - For icon-only controls, describe shape + placement: 'close "X" top-right of modal', 'right chevron in panel header'.
    - When the UI is split into panes/columns/sections OR repeated items exist, include a pane hint:
        * "left pane", "right pane", "second column", "top section"
        * or anchor to a nearby header/title: 'under header "December 2025"', 'within section "Filters"'
      This is critical for grids/pickers where the same label/number may appear multiple times.

    What NOT to ask for:
    - Avoid targeting a container title if you intend to click an item inside that container.
      Prefer describing the clickable item itself plus its container/pane context.

    Vision model output contract (STRICT):
    - Output ONLY one Python list: [xmin, ymin, xmax, ymax]
    - If not found: [-1, -1, -1, -1]
    - No other text.

    Returns:
      {"status":"success","x":int,"y":int}    bbox center in SCREEN pixels
      {"status":"not_found"}                  if bbox is [-1,-1,-1,-1]
      {"status":"error","message":str}        empty output or parse failure
    """
    print(f"\n-> get_coordinates_for('{query}')")

    # Reuse the last screenshot index (the most recent describe_webpage screenshot)
    # but take a fresh screenshot so annotation matches current state
    index = _SCREENSHOT_COUNTER - 1 if _SCREENSHOT_COUNTER > 0 else _next_screenshot_index()
    image_path = os.path.join(_RUN_DIR, f"screenshot_{index}.png")

    # If no describe_webpage was called yet, take a fresh screenshot now
    if not os.path.exists(image_path):
        image_path = _save_trace_screenshot(index)

    base64_img = encode_image(image_path)
    screen_w, screen_h = pyautogui.size()

    prompt_text = f"""
You are a visual UI locator. Carefully inspect the screenshot
You may reason silently, but your final output MUST be ONLY one Python list: [xmin, ymin, xmax, ymax]
No other text.
If you cannot confidently find the target, output exactly: [-1, -1, -1, -1]
Target to locate: {query}
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
        ]
    )

    raw = str(vision_model.invoke([message]).content).strip()
    if not raw:
        return {"status": "error", "message": "empty output"}

    try:
        xmin, ymin, xmax, ymax = ast.literal_eval(raw)
        if xmin == ymin == xmax == ymax == -1:
            return {"status": "not_found"}

        # Save annotated screenshot
        annotated_path = _save_annotated_screenshot(index, (xmin, ymin, xmax, ymax))
        print(f"[trace] saved {os.path.basename(annotated_path)} → {_RUN_DIR}")

        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        return {
            "status": "success",
            "x": int(cx / 1000.0 * screen_w),
            "y": int(cy / 1000.0 * screen_h),
        }
    except Exception:
        return {"status": "error", "message": f"parse failed: {raw}"}


@tool
def click(x: int, y: int) -> str:
    """
    Moves the mouse to the specified (x, y) coordinates and clicks the left mouse button.
    Args:
        x: The horizontal screen coordinate.
        y: The vertical screen coordinate.
    """
    print(f"\n-> click({x}, {y})")
    pyautogui.moveTo(x, y, duration=0.5)
    pyautogui.click()
    time.sleep(0.5)
    return f"Clicked successfully at ({x}, {y})."


@tool
def write(text: str, press_enter: bool = False) -> str:
    """
    Types the specified text at the current cursor location.
    Args:
        text: The string of text to type.
        press_enter: If True, simulates pressing the Enter key after typing the text.
    """
    print(f"\n-> write('{text}', press_enter={press_enter})")
    pyautogui.write(text, interval=0.05)
    if press_enter:
        time.sleep(0.2)
        pyautogui.press("enter")

    time.sleep(2)  # allow time for the page to react after pressing enter
    return f"Typed '{text}' successfully."


# ---------------------------------------------------------
# 3. Middleware
# ---------------------------------------------------------
class DeprecateOldScreenshotsMiddleware(AgentMiddleware):
    """
    Before every model call, replace the content of all previous
    describe_webpage ToolMessages (all but the most recent) with
    "[DEPRECATED]" so the LLM only sees the freshest screenshot
    description and the context stays small.
    """

    @staticmethod
    def _deprecate_messages(messages: list) -> list:
        """Replace all but the most recent describe_webpage ToolMessage with [DEPRECATED]."""
        dw_indices = [
            i
            for i, m in enumerate(messages)
            if isinstance(m, ToolMessage) and getattr(m, "name", None) == "describe_webpage"
        ]
        for i in dw_indices[:-1]:
            m = messages[i]
            messages[i] = ToolMessage(
                content="[DEPRECATED - superseded by a more recent screenshot]",
                tool_call_id=m.tool_call_id,
                name=m.name,
            )
        return messages

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        return handler(request.override(messages=self._deprecate_messages(list(request.messages))))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        return await handler(request.override(messages=self._deprecate_messages(list(request.messages))))


# ---------------------------------------------------------
# 4. Assemble and Run the Agent
# ---------------------------------------------------------
if __name__ == "__main__":
    _init_run_dir()
    print(f"Run traces will be saved to: {_RUN_DIR}")

    tools = [describe_webpage, get_coordinates_for, click, write]

    system_prompt = """
You are a UI automation agent that completes tasks in a web browser.

--- WORKFLOW ---
1. ASSESS — Call describe_webpage to understand the current UI state.
2. LOCATE — Call get_coordinates_for to find the next element to interact with.
3. ACT — Call click using the coordinates returned in step 2, then write if text input is needed.
4. VERIFY — Call describe_webpage to confirm the action took effect before moving on.
5. Repeat steps 2–4 for each remaining field or interaction.
6. SUBMIT — Once all fields are filled, locate and click the final submit/search button.

--- RULES ---
- Never guess coordinates. Only call click(x,y) with values from the immediately preceding get_coordinates_for().
- On not_found/error, call describe_webpage to re-assess before retrying.
- Prefer clicking over typing (dropdowns, pickers, suggestions).
- Stop immediately after the final submit action.
"""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[DeprecateOldScreenshotsMiddleware()],
    )

    print("Starting agent...")
    time.sleep(3)

    task = "Search for accommodation in Paris, check-in April 22 2026, check-out April 27 2026."

    _save_prompt(task)
    print(f"User task:\n{task}\n")
    print(f"[trace] saved prompt.txt → {_RUN_DIR}")

    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    print("\nFinal Result:")
    print(result["messages"][-1].content)
