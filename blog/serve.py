#!/usr/bin/env python3
"""Serve the local blog preview on http://localhost:8000.

Usage:
    python blog/serve.py           # port 8000
    python blog/serve.py 9000      # custom port
"""

import http.server
import os
import socketserver
import sys


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving blog at http://localhost:{PORT}")
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping.")


if __name__ == "__main__":
    main()
