"""Shim for `uvicorn app:app`; canonical app is `server.app`."""

from server.app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
