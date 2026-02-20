# VisionClick Agent

VisionClick Agent is a local, vision-driven UI automation agent. It controls a desktop browser by looking at screenshots, reasoning about what it sees, and issuing mouse clicks and keyboard input — no browser extensions, no DOM access required.

---

## Architecture overview

```
User task (natural language)
        │
        ▼
  Reasoning LLM  ◄──── full message history (tool calls + results)
        │
   decides which tool to call
        │
   ┌────┴──────────────────────────────────────┐
   │                  Tools                    │
   │  describe_webpage   get_coordinates_for   │
   │  click              write                 │
   └────┬──────────────────────────────────────┘
        │ tool results injected back into history
        ▼
  Reasoning LLM (next step)
        │
        ▼
   ... loop until task complete
```

---

## Models

Two separate local models are used via [Ollama](https://ollama.com), each with a distinct role.

### Vision model — `qwen3-vl:30b`

Used exclusively inside the `describe_webpage` and `get_coordinates_for` tools. It receives a raw base64-encoded screenshot and processes it visually.

- In `describe_webpage` it produces a structured **UI STATE REPORT** — a text description of what is currently on screen: blockers, open panels, visible fields, key actions, and layout details.
- In `get_coordinates_for` it locates a single described UI element and returns a bounding box `[xmin, ymin, xmax, ymax]` in a 0–1000 coordinate space that is then scaled to real screen pixels.

The vision model is stateless: it sees only one screenshot per call and returns a plain-text result. It has no memory of previous calls.

Configuration: `temperature=0`, `num_ctx=8192`.

### Reasoning LLM — `gpt-oss`

The orchestrator. It receives the full conversation history (user task + all previous tool calls and their results) and decides which tool to invoke next and with which arguments. It follows a fixed workflow:

1. **Assess** — call `describe_webpage` to understand current UI state.
2. **Locate** — call `get_coordinates_for` to find the target element.
3. **Act** — call `click` (and optionally `write`) to interact.
4. **Verify** — call `describe_webpage` again to confirm the action took effect.
5. Repeat until the task is complete.

Configuration: `temperature=0`, `num_ctx=16384` (larger context to hold the growing history).

---

## Tools

| Tool | Purpose |
|---|---|
| `describe_webpage(intention)` | Takes a screenshot, sends it to the vision model, returns a structured text description of the current UI state. Saves the screenshot and description to the run directory for tracing. |
| `get_coordinates_for(query)` | Takes the most recent screenshot, asks the vision model to locate one UI element described by `query`, returns `{"status": "success", "x": int, "y": int}` in screen pixels (or `not_found`/`error`). Also saves an annotated screenshot with the bounding box drawn in red. |
| `click(x, y)` | Moves the mouse to `(x, y)` and clicks. Coordinates must always come from the immediately preceding `get_coordinates_for` call — never guessed. |
| `write(text, press_enter)` | Types text at the current cursor position. Optionally presses Enter afterwards. |

The reasoning LLM never has direct access to screenshots — it only ever sees the text results returned by the tools.

---

## How context is kept

The agent maintains a **single growing message list** that is passed to the reasoning LLM on every step. This list contains:

- The initial user task (`HumanMessage`).
- Every tool call the LLM has made (as assistant messages with tool-call payloads).
- Every tool result (as `ToolMessage` entries), including the full text output of each `describe_webpage` call.

This means the LLM can always see what it has already done, what the UI looked like at each step, and what the outcomes were.

### Why context matters

UI automation is inherently sequential and stateful. Each action changes the screen. Without seeing the previous steps, the model cannot know:

- Whether a field was already filled.
- Whether a panel opened or closed as expected.
- Which part of a multi-step interaction is still pending.
- Whether an earlier action failed silently.

Keeping the full history allows the reasoning LLM to recover from unexpected states, avoid repeating actions, and build up a coherent understanding of progress toward the goal.

### The `DeprecateOldScreenshotsMiddleware`

Screenshot descriptions are verbose. If every past `describe_webpage` result were kept at full length, the context window would fill up quickly and older (now stale) descriptions would mislead the model.

`DeprecateOldScreenshotsMiddleware` runs **before every LLM call** and replaces the content of all `describe_webpage` tool results except the most recent one with:

```
[DEPRECATED - superseded by a more recent screenshot]
```

This keeps the history structurally intact (the LLM can still see that a describe call happened and what action followed it) while aggressively shrinking its token footprint. The most recent description — the only one that reflects the actual current state of the screen — is always kept in full.

---

## Example run

The [docs/example-run/](example-run/) directory contains a recorded trace of the agent completing the following task:

> *Search for accommodation in Paris, check-in April 22 2026, check-out April 27 2026.*

The trace files:

| File | Contents |
|---|---|
| `prompt.txt` | The original user task. |
| `screenshot_0.txt` | Vision model description of the initial screen: Booking.com home page, empty search form. |
| `screenshot_1.txt` | Screen state after "Paris" was typed into the destination field. |
| `screenshot_2.txt` | Screen state after the Paris autocomplete suggestion was clicked. |
| `screenshot_3.txt` | Screen state after the date field was clicked — a two-pane calendar is open showing March and April 2026. The vision model correctly identifies the two-pane layout and flags that day numbers are repeated across both panes (important for the locator to use pane hints). |
| `screenshot_4.txt` | Screen state after the check-in date (April 22) was selected. |
| `screenshot_5.txt` | Screen state after the check-out date (April 27) was selected. |
| `screenshot_6.txt` | Final state after the Search button was clicked — results page loaded. |

Each `.txt` file is the raw output of the vision model for that step: a structured UI STATE REPORT covering blockers, open panels, panel layout, visible field values, and available actions. The reasoning LLM read these descriptions (not the images) to decide every subsequent action.

The annotated screenshots (`screenshot_N_annotated.png`, stored in `runs/` during a live run) show the red bounding box the vision model drew around the element targeted by each `get_coordinates_for` call.

---

## Run tracing

Every execution creates a timestamped directory under `runs/` (e.g., `runs/run-20260220_105913/`). It contains:

- `prompt.txt` — the task given to the agent.
- `screenshot_N.png` — raw screenshot taken at step N.
- `screenshot_N_annotated.png` — same screenshot with the located element highlighted.
- `screenshot_N.txt` — vision model UI STATE REPORT for step N.

This trace is useful for debugging agent decisions, inspecting what the vision model reported at each step, and replaying runs.

---

## Dependencies

- `pyautogui` — mouse/keyboard control and screen capture.
- `Pillow` — image annotation.
- `langchain`, `langchain-core`, `langchain-ollama` — agent framework, tool definitions, message types.
- `pydantic` — data validation.
- A running [Ollama](https://ollama.com) instance with `qwen3-vl:30b` and `gpt-oss` pulled.
