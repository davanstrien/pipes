"""jobpipe: run a model over a whole dataset on Hugging Face Jobs.

Three layers: saturate is the low-level batch client (a serving engine in,
resumable parquet out); the uv-scripts recipes are tested single-file scripts
built on it; this package resolves a short request into a pinned recipe plus
its arguments, launches one Job per shard, and reads progress back from
storage. It generates no code.

    from jobpipe import embed
    run = embed("biglam/europeana_newspapers", keep=["language"], world=4)
    run.status()
"""

from jobpipe.api import Run, embed, extract, ocr
from jobpipe.resolve import Plan

__all__ = ["Plan", "Run", "embed", "extract", "ocr", "status"]


def status(output_uri, **kwargs):
    from jobpipe.status import status as _status

    return _status(output_uri, **kwargs)
