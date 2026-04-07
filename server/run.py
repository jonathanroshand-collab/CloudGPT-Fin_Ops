from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "2"))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
