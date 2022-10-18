from __future__ import annotations

from fastapi import FastAPI

# from api.v1.main import router as v1_router
from api.v2.main import router as v2_router

app = FastAPI(
    title="ALREADYME AI Serving",
    description="""
Backend APIs for ALREADYME AI Service

[ALREADYME.md](https://github.com/readme-generator) is a service to help writing
`README.md` file through AI. AI model will read the source codes in the repository
and suggests various README contents.

This API is for serving our large-scale language model which is trained for
generating proper `README.md` content from the repository files. We recommend not to
access this API from frontend (e.g. website, application, desktop and etc.)
directly. Instead, you should use another proxy-server to relay the inputs and
outputs.
    """,
    version="0.2.6",
)
# app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
