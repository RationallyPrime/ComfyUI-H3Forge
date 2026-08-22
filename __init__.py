if not __package__:
    # Pytest can collect this file as a top-level module rather than a package.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    try:
        from .h3forge.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except ModuleNotFoundError as exc:
        if exc.name != "comfy":
            raise
        # Standalone test tooling imports the repository package outside ComfyUI.
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
