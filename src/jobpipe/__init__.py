"""jobpipe: task verbs -> saturate driver compiler + launcher + status.

The layer between a UI and saturate: run models over datasets on HF Jobs.

Three-layer picture: saturate is the low-level client (serving -> datasets/
buckets in, resumable parquet out; knows nothing about Jobs or hardware);
THIS layer covers common tasks on Jobs behind one python call / CLI verb with
sensible defaults; datatrove is the flexible DAG-shaped library above, for
workloads that are actual pipelines.

    from jobpipe import embed
    run = embed("HuggingFaceFW/fineweb-edu", column="text", model="BAAI/bge-m3",
                output="hf://buckets/me/fw-embeds", world=4)
    run.status(); run.publish("me/fineweb-edu-embeddings")

Design intent: templates
are functions; script emission is a projection of them, not a parallel source
of truth. The planned execution path is a generic runner from a pinned wheel;
ejected scripts default THIN (importing task functions), with full inlining an
explicit eject-and-mutate fork. Flip timing: design-sprint material — its
trigger (a second task template) has fired; blocked only on naming/packaging
decisions the sprint owns.
"""

from jobpipe.api import Run, embed, generate
from jobpipe.compiler import CompiledRun, compile
from jobpipe.spec import TaskSpec

__all__ = [
    "CompiledRun",
    "Run",
    "TaskSpec",
    "compile",
    "embed",
    "generate",
    "launch",
    "status",
]


def launch(spec, **kwargs):  # lazy import: keeps `compile` usable offline
    from jobpipe.launcher import launch as _launch

    return _launch(spec, **kwargs)


def status(output_uri, **kwargs):
    from jobpipe.status import status as _status

    return _status(output_uri, **kwargs)
