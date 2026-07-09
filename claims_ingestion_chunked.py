"""
Medical Claims Intelligence — Ingestion for a SINGLE S3 document,
with CHUNK-LEVEL fan-out (analyze + write as separate nodes, per chunk)
========================================================================

One claim document (one S3 object) still processed at a time — no document
fan-out yet (that's a separate, later concern). What's NEW here: the old
"comprehend_analyze(list_of_chunks)" — which internally looped and wrote an
annotation per chunk — is now split into two real LangGraph nodes,
`analyze_chunk` and `write_chunk_annotation`, run once per chunk. Because
each chunk's Comprehend call and annotation write are independent of every
other chunk (no shared state, order doesn't matter), they're fanned out in
PARALLEL via Send — same rationale as document-level fan-out, one level down.

Graph shape:

        START
          |
     check_cache ──(chunk annotations already exist)──► END
          |
     fetch_metadata            STEP 2  (get_metadata)
          |
     ingest_document            STEP 3  (data_ingestion / Textract inside)
          |
     [fan out: one Send per chunk] ──────────────────┐
          |                                            |
     process_chunk (parallel, one per chunk)  ◄────────┘
       runs a 2-node SUBGRAPH per chunk:
          analyze_chunk           STEP 4  (comprehend detect_entities, ONE chunk)
               |
          write_chunk_annotation  STEP 5  (enhance + write ONE chunk's annotation)
          |
     finalize_document (fan-in, runs once after all chunks complete)
          |
         END

IMPORTANT ASSUMPTION (confirm against your real write_annotation):
  Each chunk gets its OWN annotation object (e.g. keyed by
  f"{key}#chunk-{index}"), not a single shared document-level object being
  incrementally updated. This is what makes PARALLEL writes safe — there's no
  shared mutable resource for concurrent branches to race on. If your real
  write_annotation instead does a read-modify-write on ONE object per
  document, do NOT parallelize this loop — keep it sequential instead
  (ask me for that variant).

Contract change vs. your existing code:
  comprehend_analyze used to take the FULL chunk list and loop+write inside
  itself. Here it must be refactored to analyze a SINGLE chunk of text and
  return ONLY the analysis (no looping, no writing) — the loop now lives in
  the graph (via Send), and the write is a separate node/function call.

Install:  pip install langgraph pydantic boto3
Requires: AWS credentials with S3 + Comprehend Medical.
This is PHI — keep everything in-account and scope IAM tightly.
"""

from __future__ import annotations

import json
import logging
import operator
import os
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

try:
    from botocore.exceptions import (
        ClientError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
    )
except Exception:  # pragma: no cover
    ClientError = EndpointConnectionError = ConnectTimeoutError = ReadTimeoutError = ()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claims_ingestion_chunked")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
COMPREHEND_MAX_CHARS = 18_000
MIN_ENTITY_SCORE = 0.50
MAX_CHUNK_CONCURRENCY = 8  # cap parallel Comprehend calls per document (TPS limits)


# --------------------------------------------------------------------------- #
# ADAPT THESE — your existing functions.                                     #
# NOTE the changed contract on comprehend_analyze (single chunk, analysis only)
# --------------------------------------------------------------------------- #
def get_metadata(bucket: str, key: str) -> dict:
    raise NotImplementedError("wire in your get_metadata")


def data_ingestion(metadata: dict) -> list[str]:
    raise NotImplementedError("wire in your data_ingestion")


def comprehend_analyze(chunk_text: str) -> dict:
    """CHANGED CONTRACT: analyzes ONE chunk of text only. No looping, no
    writing — both of those now happen elsewhere. Expected to return a dict
    with an 'Entities' list (Category, Type, Text, Score, Traits)."""
    raise NotImplementedError("wire in your comprehend_analyze (single-chunk version)")


def write_annotation(bucket: str, key: str, annotation: dict) -> str:
    """Persist ONE annotation object and return a reference/URI. Called once
    per chunk here, with a chunk-specific key."""
    raise NotImplementedError("wire in your write_annotation")


def read_annotation(bucket: str, key: str) -> Optional[dict]:
    raise NotImplementedError("wire in your read_annotation")


def list_annotation(bucket: str, prefix: Optional[str] = None) -> list[str]:
    """Used here for the cache check — lists existing chunk annotation keys
    under this document's prefix."""
    raise NotImplementedError("wire in your list_annotation")


# --------------------------------------------------------------------------- #
# Retry policy for AWS-calling step nodes                                     #
# --------------------------------------------------------------------------- #
_TRANSIENT_AWS_CODES = {
    "ThrottlingException", "Throttling", "TooManyRequestsException",
    "ProvisionedThroughputExceededException", "RequestLimitExceeded",
    "ServiceUnavailable", "ServiceUnavailableException",
    "InternalServerError", "InternalServerException", "RequestTimeout",
}


def _is_transient_aws_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError,
         ConnectionError, TimeoutError),
    ):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in _TRANSIENT_AWS_CODES
    return False


AWS_RETRY = RetryPolicy(
    max_attempts=4,
    retry_on=_is_transient_aws_error,
    initial_interval=0.5,
    backoff_factor=2.0,
    max_interval=20.0,
    jitter=True,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def chunk_annotation_key(document_key: str, chunk_index: int) -> str:
    """Naming convention for a chunk's own annotation object. Adjust to match
    whatever key scheme your real write_annotation/list_annotation expect."""
    return f"{document_key}#chunk-{chunk_index:04d}"


def _split_for_comprehend(text: str, limit: int = COMPREHEND_MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _build_chunk_annotation(
    bucket: str, key: str, chunk_index: int, metadata: Optional[dict], analysis: dict
) -> dict:
    """Enhance one chunk's raw Comprehend analysis with claims metadata,
    filtered/deduped for LLM-friendliness. This is the 'enhance with claims
    metadata' step from your description, folded into the write node."""
    entities = []
    for ent in analysis.get("Entities", []):
        if ent.get("Score", 0) < MIN_ENTITY_SCORE:
            continue
        entities.append(
            {
                "text": ent.get("Text"),
                "type": ent.get("Type"),
                "category": ent.get("Category"),
                "score": round(ent.get("Score", 0.0), 3),
                "traits": [t.get("Name") for t in ent.get("Traits", [])],
                "icd10": ent.get("ICD10CMConcepts"),
                "rxnorm": ent.get("RxNormConcepts"),
            }
        )
    return {
        "source_bucket": bucket,
        "source_key": key,
        "chunk_index": chunk_index,
        "claims_metadata": metadata,
        "entities": entities,
    }


# --------------------------------------------------------------------------- #
# CHUNK SUBGRAPH — analyze_chunk and write_chunk_annotation as separate nodes  #
# --------------------------------------------------------------------------- #
class ChunkState(BaseModel):
    bucket: str
    key: str
    chunk_index: int
    chunk_text: str
    metadata: Optional[dict] = None
    analysis: Optional[dict] = None
    annotation: Optional[dict] = None


def analyze_chunk(state: ChunkState) -> dict:              # STEP 4
    """Analyze ONE chunk. If a chunk is itself too large for the Comprehend
    sync ceiling, sub-split and merge the entity lists back together."""
    merged_entities: list[dict] = []
    for sub in _split_for_comprehend(state.chunk_text):
        result = comprehend_analyze(sub)
        merged_entities.extend(result.get("Entities", []))
    return {"analysis": {"Entities": merged_entities}}


def write_chunk_annotation(state: ChunkState) -> dict:      # STEP 5
    """Enhance with claims metadata and persist ONE chunk's annotation."""
    annotation = _build_chunk_annotation(
        state.bucket, state.key, state.chunk_index, state.metadata, state.analysis
    )
    ref = write_annotation(
        state.bucket, chunk_annotation_key(state.key, state.chunk_index), annotation
    )
    annotation["annotation_ref"] = ref
    return {"annotation": annotation}


def build_chunk_pipeline():
    g = StateGraph(ChunkState)
    g.add_node("analyze_chunk", analyze_chunk, retry_policy=AWS_RETRY)
    g.add_node("write_chunk_annotation", write_chunk_annotation, retry_policy=AWS_RETRY)
    g.add_edge(START, "analyze_chunk")
    g.add_edge("analyze_chunk", "write_chunk_annotation")
    g.add_edge("write_chunk_annotation", END)
    return g.compile()


CHUNK_PIPELINE = build_chunk_pipeline()


# --------------------------------------------------------------------------- #
# PARENT (document) GRAPH                                                     #
# --------------------------------------------------------------------------- #
class ClaimDocumentState(BaseModel):
    bucket: str
    key: str
    force_reprocess: bool = False
    metadata: Optional[dict] = None
    chunks: list[str] = Field(default_factory=list)
    # accumulators — reducers required because chunk branches write in parallel
    chunk_annotations: Annotated[list[dict], operator.add] = Field(default_factory=list)
    chunk_errors: Annotated[list[dict], operator.add] = Field(default_factory=list)
    status: Optional[str] = None  # "ok" | "skipped" | "partial_error" | "error"


def check_cache(state: ClaimDocumentState) -> dict:
    """Idempotency guard using list_annotation (per-document prefix) rather
    than a single read_annotation, since annotations are now one-per-chunk."""
    if state.force_reprocess:
        return {}
    try:
        existing_refs = list_annotation(state.bucket, prefix=f"{state.key}#chunk-")
    except Exception:
        logger.warning("Cache check failed for %s; will reprocess", state.key)
        return {}
    if not existing_refs:
        return {}
    logger.info("Found %d existing chunk annotation(s) for %s; skipping", len(existing_refs), state.key)
    return {"status": "skipped"}


def route_after_cache(state: ClaimDocumentState) -> str:
    return "done" if state.status == "skipped" else "fetch_metadata"


def fetch_metadata(state: ClaimDocumentState) -> dict:      # STEP 2
    return {"metadata": get_metadata(state.bucket, state.key)}


def ingest_document(state: ClaimDocumentState) -> dict:     # STEP 3 (Textract inside)
    return {"chunks": data_ingestion(state.metadata)}


def fan_out_chunks(state: ClaimDocumentState):
    """Map step: one parallel branch per chunk (or straight to finalize if none)."""
    if not state.chunks:
        return "finalize_document"
    return [
        Send(
            "process_chunk",
            {
                "bucket": state.bucket,
                "key": state.key,
                "chunk_index": idx,
                "chunk_text": chunk,
                "metadata": state.metadata,
            },
        )
        for idx, chunk in enumerate(state.chunks)
    ]


def process_chunk(raw) -> dict:
    """Fan-out target: runs the 2-node chunk subgraph for ONE chunk. Catches
    everything so one bad chunk can't fail the whole parallel superstep.

    Reached via Send, so — same nuance as document-level fan-out earlier —
    it receives the raw payload dict uncoerced; we construct ChunkState
    ourselves rather than relying on automatic validation."""
    chunk_state = raw if isinstance(raw, ChunkState) else ChunkState(**raw)
    try:
        result = CHUNK_PIPELINE.invoke(chunk_state.model_dump())
    except Exception as exc:  # noqa: BLE001 — intentional per-chunk isolation
        logger.exception("Chunk %d failed for %s", chunk_state.chunk_index, chunk_state.key)
        return {"chunk_errors": [{"chunk_index": chunk_state.chunk_index, "error": str(exc)}]}

    annotation = result.get("annotation")
    if not annotation:
        return {
            "chunk_errors": [
                {"chunk_index": chunk_state.chunk_index, "error": "no annotation produced"}
            ]
        }
    return {"chunk_annotations": [annotation]}


def finalize_document(state: ClaimDocumentState) -> dict:
    """Fan-in: runs once after all chunk branches complete."""
    n_ok, n_err = len(state.chunk_annotations), len(state.chunk_errors)
    logger.info("Document %s: %d chunk(s) annotated, %d chunk error(s)", state.key, n_ok, n_err)
    if n_ok and not n_err:
        status = "ok"
    elif n_ok and n_err:
        status = "partial_error"
    else:
        status = "error"
    return {"status": status}


def build_ingestion_graph():
    g = StateGraph(ClaimDocumentState)
    g.add_node("check_cache", check_cache)
    g.add_node("fetch_metadata", fetch_metadata, retry_policy=AWS_RETRY)
    g.add_node("ingest_document", ingest_document, retry_policy=AWS_RETRY)
    g.add_node("process_chunk", process_chunk)
    g.add_node("finalize_document", finalize_document)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges(
        "check_cache", route_after_cache, {"fetch_metadata": "fetch_metadata", "done": END}
    )
    g.add_edge("fetch_metadata", "ingest_document")
    g.add_conditional_edges(
        "ingest_document", fan_out_chunks, ["process_chunk", "finalize_document"]
    )
    g.add_edge("process_chunk", "finalize_document")
    g.add_edge("finalize_document", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_ingestion(bucket: str, key: str, force_reprocess: bool = False) -> dict:
    graph = build_ingestion_graph()
    return graph.invoke(
        {"bucket": bucket, "key": key, "force_reprocess": force_reprocess},
        config={"max_concurrency": MAX_CHUNK_CONCURRENCY},
    )


# --------------------------------------------------------------------------- #
# Demo — runs the graph end-to-end right now with FAKE AWS calls              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _fake_store: dict[str, dict] = {}

    def _fake_get_metadata(bucket, key):
        return {"filename": key, "content_type": "application/pdf", "pages": 3}

    def _fake_data_ingestion(metadata):
        return [
            "Patient presents with hypertension and type 2 diabetes mellitus.",
            "Prescribed metformin 500mg twice daily. Follow-up in 3 months.",
            "History of myocardial infarction in 2019, currently stable.",
        ]

    def _fake_comprehend_analyze(chunk_text):
        if "hypertension" in chunk_text:
            return {"Entities": [
                {"Category": "MEDICAL_CONDITION", "Type": "DX_NAME", "Text": "hypertension", "Score": 0.97, "Traits": []},
                {"Category": "MEDICAL_CONDITION", "Type": "DX_NAME", "Text": "type 2 diabetes mellitus", "Score": 0.95, "Traits": []},
            ]}
        if "metformin" in chunk_text:
            return {"Entities": [
                {"Category": "MEDICATION", "Type": "GENERIC_NAME", "Text": "metformin", "Score": 0.93, "Traits": [{"Name": "DOSAGE"}]},
            ]}
        return {"Entities": [
            {"Category": "MEDICAL_CONDITION", "Type": "DX_NAME", "Text": "myocardial infarction", "Score": 0.91, "Traits": [{"Name": "PAST_HISTORY"}]},
        ]}

    def _fake_write_annotation(bucket, key, annotation):
        _fake_store[key] = annotation
        return f"s3://{bucket}/annotations/{key}.json"

    def _fake_read_annotation(bucket, key):
        return _fake_store.get(key)

    def _fake_list_annotation(bucket, prefix=None):
        return [k for k in _fake_store if not prefix or k.startswith(prefix)]

    globals()["get_metadata"] = _fake_get_metadata
    globals()["data_ingestion"] = _fake_data_ingestion
    globals()["comprehend_analyze"] = _fake_comprehend_analyze
    globals()["write_annotation"] = _fake_write_annotation
    globals()["read_annotation"] = _fake_read_annotation
    globals()["list_annotation"] = _fake_list_annotation

    BUCKET, KEY = "my-claims-bucket", "claims/12345/discharge_summary.pdf"

    print("\n===== FIRST RUN (should process 3 chunks in parallel) =====")
    result = run_ingestion(bucket=BUCKET, key=KEY)
    print(f"status: {result['status']}")
    print(f"chunk_annotations: {len(result['chunk_annotations'])}, chunk_errors: {len(result['chunk_errors'])}")
    for ann in sorted(result["chunk_annotations"], key=lambda a: a["chunk_index"]):
        print(json.dumps(ann, indent=2, default=str))

    print("\n===== SECOND RUN (should hit cache and skip) =====")
    result2 = run_ingestion(bucket=BUCKET, key=KEY)
    print(f"status: {result2['status']}")

    # To visualise the graph:
    #   build_ingestion_graph().get_graph(xray=True).draw_mermaid()
