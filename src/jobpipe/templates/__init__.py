"""Task template registry — one entry per TaskSpec.task value.

A template is a module exposing: staging_repo(spec), driver(spec), image(spec),
job_shell(spec, driver_filename, rank), stage_command(spec, driver_filename),
launch_commands(spec, driver_filename), publish_command(spec, owner).
The seed ships exactly one: embeddings.
"""

from __future__ import annotations

from jobpipe.templates import embeddings, generation, ocr

_TEMPLATES = {"embeddings": embeddings, "generation": generation, "ocr": ocr}


def get_template(task: str):
    try:
        return _TEMPLATES[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r}; available: {sorted(_TEMPLATES)}") from None
