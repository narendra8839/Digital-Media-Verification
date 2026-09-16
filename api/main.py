"""FastAPI service layer for Digital Media Verification system."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.verify import router as verify_router
from api.dependencies.pipeline_dep import init_pipeline

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dmv_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan manager pre-warming AI models at server startup to eliminate per-request cold starts."""
    logger.info("Application startup: Pre-warming MultimodalPipeline...")
    init_pipeline(device="cpu")
    logger.info("Application startup: MultimodalPipeline ready.")
    yield
    logger.info("Application shutdown: Cleaning up resources.")


app = FastAPI(
    title="Digital Media Verification API",
    description=(
        "REST API service layer for multimodal media verification. "
        "Intelligently routes Image, Video, and Text inputs through deepfake detection, "
        "propaganda analysis, hate speech classification, Explainable AI (Grad-CAM, Token Attribution), "
        "and Monte Carlo Dropout Uncertainty Quantification (T=20)."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for local development and integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handlers for Structured Error Output
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Structured handler for intentional HTTPExceptions."""
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request Error",
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Structured handler for Pydantic/FastAPI request validation errors (422)."""
    error_messages = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "invalid")
        error_messages.append(f"{field}: {msg}")
    detail_str = "; ".join(error_messages)
    logger.warning(f"Validation failure on {request.url.path}: {detail_str}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": detail_str,
            "status_code": 422
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unhandled server exceptions (500), avoiding stack trace leakage."""
    logger.error(f"Unhandled server error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected internal server error occurred while processing the request.",
            "status_code": 500
        }
    )


# Mount application routers
app.include_router(health_router)
app.include_router(verify_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
