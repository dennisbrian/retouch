#!/usr/bin/env python3
"""Desktop application wrapper for Pro Max Retouch Studio using PyWebView."""

import threading
import webview
from gui import app


def start_gradio():
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    # Start Gradio backend on a daemon thread
    threading.Thread(
        target=start_gradio,
        daemon=True,
    ).start()

    # Create native OS window pointing to the local Gradio server
    webview.create_window(
        "Pro Max Retouch Studio",
        "http://127.0.0.1:7860",
        width=1600,
        height=1000,
    )
    webview.start()
