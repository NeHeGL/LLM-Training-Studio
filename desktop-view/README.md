# LLM Training Studio — PyQt6 Desktop App

Opens the LLM Training Studio in a native Windows desktop window using
**PyQt6 + QtWebEngine** (Chromium-based rendering — same engine as Chrome).
Shows a dark splash screen while the Flask server starts, then displays the
full Studio UI with no browser chrome or address bar.

> **Note:** Originally planned as a `pywebview` wrapper, but pywebview's
> Windows backend (`pythonnet`) doesn't yet support Python 3.14.
> PyQt6-WebEngine is used instead — it's actually a better fit: full
> Chromium engine, proper event loop integration, and works on Python 3.14+.

## How it works

- `start_app.bat` checks if a server is already running on port 5001
  - **If yes** — launches the desktop app and attaches to the existing server
  - **If no** — starts the Flask server (`train/server.py --no-open`) in a console
    window, then launches this script via `pythonw` (no console)
- Shows a matching dark splash screen while the server initialises
- Opens a `QWebEngineView` window pointing at `http://localhost:5001`
- The UI is pixel-identical to the browser version — it IS the same `studio.html`
- Works alongside `start_web.bat` — both can run at the same time sharing one server

## Launch

Double-click **`start_app.bat`** in the project root.

It will automatically install PyQt6 if missing, then launch the desktop app with no console window.

Or directly (server must already be running on port 5001):

```
python desktop-view\launch_app.py
```

## Notes

- QtWebEngine is a full Chromium build — the UI renders identically to Chrome
- The existing `start_web.bat` / browser workflow is completely untouched
- Both the browser and this desktop app can be used interchangeably
- **No console window** — all subprocess calls use `CREATE_NO_WINDOW` so no black
  console flashes appear during training, export, or package updates
- **Restart behaviour** — "Restart Server" in the Updates tab reloads the UI page
  immediately (the server console window stays running) rather than trying to spawn a new process
