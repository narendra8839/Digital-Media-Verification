"""Digital Media Verification FastAPI package."""

def __getattr__(name: str):
    if name == "app":
        from api.main import app
        return app
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["app"]
