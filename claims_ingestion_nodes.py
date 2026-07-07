"""
Medical Claims Intelligence — Ingestion flow (functional, steps as separate nodes)
==================================================================================

Ingestion ONLY (no summarize / no chat yet).

Two graphs compose:

  PARENT (orchestrator)          PER-DOCUMENT SUBGRAPH (the "steps")
  ---------------------          ----------------------------------
  list_documents                 check_cache
       |  (fan out, 1 Send/doc)       |  (skip if annotation exists)
       v                              v
  process_document  ── invokes ──> fetch_metadata     (step 2, get_metadata)
       |                              v
       |                          ingest_document      (step 3, data_ingestion / Textract)
       |                              v
       |                          analyze_entities     (step 4, comprehend_analyze)
       |                              v
       |                          persist_annotation   (step 5, write_annotation)
       v
  collect_results   (fan-in: runs once after all documents complete)

Why a subgraph? A `Send` branch can target only ONE node. To fan each document
through a multi-node pipeline, that node hosts the pipeline as a subgraph. The
four steps are now genuine, separate, individually-retryable nodes.

Key properties:
  * Each AWS-calling step node has its own RetryPolicy (transient errors only:
    throttling, 5xx, connection/timeout). Only the failing node retries.
  * process_document wraps the subgraph invoke in try/except so a single bad
    document can't fail the whole parallel superstep (superstep failures are
    atomic — one raise loses every branch's writes).
  * Accumulator channels use Annotated[list, operator.add] reducers, required
    for parallel writes.
  * Idempotency: check_cache short-circuits documents already annotated.

Install:  pip install langgraph langchain-aws boto3
Requires: AWS credentials with S3 + Comprehend Medical (+ optional Infer*).
This is PHI — keep everything in-account and scope IAM tightly.

-------------------------------------------------------------------------------
ADAPT THIS SECTION: wire in your six existing functions. Signatures are ASSUMED.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import operator
import os
from typing import Annotated, Optional

import boto3
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

try:  # botocore ships with boto3; used for AWS-aware retry classification
    from botocore.exceptions import (
        ClientError,
        ConnectTimeoutError,
        EndpointConnectionError,
        ReadTimeoutError,
    )
except Exception:  # pragma: no cover
    ClientError = EndpointConnectionError = ConnectTimeoutError = ReadTimeoutError = ()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claims_ingestion")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
COMPREHEND_MAX_CHARS = 18_000  # stay under the sync limit (verify current quota)
MIN_ENTITY_SCORE = 0.50        # drop low-confidence entities
MAX_CONCURRENCY = 5            # cap parallel Textract/Comprehend calls (TPS)


# --------------------------------------------------------------------------- #
# ADAPT THESE — your existing functions. Replace bodies with real ones.       #
# --------------------------------------------------------------------------- #
def get_metadata(bucket: str, key: str) -> dict:
    """Return metadata for one S3 object (assumed signature)."""
    raise NotImplementedError("wire in your get_metadata")


def data_ingestion(metadata: dict) -> list[str]:
    """Textract the document (internally) and return a list of text chunks."""
    raise NotImplementedError("wire in your data_ingestion")


def comprehend_analyze(text: str) -> dict:
    """Run Comprehend Medical on one text chunk; return its raw analysis dict.
    Expected to contain an 'Entities' list (Category, Type, Text, Score, Traits).
    Merge InferICD10CM / InferRxNorm output here if you add coded diagnoses/meds.
    """
    raise NotImplementedError("wire in your comprehend_analyze")


def write_annotation(bucket: str, key: str, annotation: dict) -> str:
    """Persist the annotation (e.g. to S3) and return a reference/URI."""
    raise NotImplementedError("wire in your write_annotation")


def read_annotation(bucket: str, key: str) -> Optional[dict]:
    """Return an existing annotation for (bucket, key) or None."""
    raise NotImplementedError("wire in your read_annotation")


def list_annotation(bucket: str, prefix: Optional[str] = None) -> list[str]:
    """List annotation references under an optional prefix."""
    raise NotImplementedError("wire in your list_annotation")


# --------------------------------------------------------------------------- #
# Retry policy for the AWS-calling step nodes                                 #
# --------------------------------------------------------------------------- #
_TRANSIENT_AWS_CODES = {
    "ThrottlingException", "Throttling", "TooManyRequestsException",
    "ProvisionedThroughputExceededException", "RequestLimitExceeded",
    "ServiceUnavailable", "ServiceUnavailableException",
    "InternalServerError", "InternalServerException", "RequestTimeout",
}


def _is_transient_aws_error(exc: BaseException) -> bool:
    """Retry only transient AWS failures — never programming bugs or AccessDenied."""
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
def list_s3_documents(
    bucket: str, prefix: Optional[str] = None, suffixes: tuple[str, ...] = (".pdf",)
) -> list[str]:
    """List document keys in a bucket (paginated; handles >1000 keys)."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            if suffixes and not k.lower().endswith(suffixes):
                continue
            keys.append(k)
    return keys


def _split_for_comprehend(text: str, limit: int = COMPREHEND_MAX_CHARS) -> list[str]:
    """Sub-split a chunk if it exceeds the Comprehend Medical sync ceiling."""
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def _build_annotation(
    bucket: str, key: str, metadata: dict, chunk_analyses: list[dict]
) -> dict:
    """Compact verbose Comprehend output into a deduped, LLM-friendly rollup
    with provenance (source_key)."""
    categories: dict[str, list[dict]] = {}
    for ca in chunk_analyses:
        for ent in ca["analysis"].get("Entities", []):
            if ent.get("Score", 0) < MIN_ENTITY_SCORE:
                continue
            cat = ent.get("Category", "OTHER")
            item = {
                "text": ent.get("Text"),
                "type": ent.get("Type"),
                "score": round(ent.get("Score", 0.0), 3),
                "traits": [t.get("Name") for t in ent.get("Traits", [])],
                "icd10": ent.get("ICD10CMConcepts"),    # populated if you add InferICD10CM
                "rxnorm": ent.get("RxNormConcepts"),     # populated if you add InferRxNorm
            }
            bucket_list = categories.setdefault(cat, [])
            if not any(
                x["text"] == item["text"] and x["type"] == item["type"]
                for x in bucket_list
            ):
                bucket_list.append(item)

    return {
        "source_bucket": bucket,
        "source_key": key,
        "document_metadata": metadata,
        "entities_by_category": categories,
        "num_chunks": len({ca["chunk_index"] for ca in chunk_analyses}),
    }


# --------------------------------------------------------------------------- #
# PER-DOCUMENT SUBGRAPH — the four steps as separate nodes                    #
# --------------------------------------------------------------------------- #
class DocState(BaseModel):
    bucket: str
    key: str
    force_reprocess: bool = False
    metadata: Optional[dict] = None
    chunks: list[str] = Field(default_factory=list)
    chunk_analyses: list[dict] = Field(default_factory=list)
    annotation: Optional[dict] = None
    status: Optional[str] = None  # "ok" | "skipped"


def check_cache(state: DocState) -> dict:
    """Idempotency guard: skip documents that already have an annotation."""
    if state.force_reprocess:
        return {}
    try:
        existing = read_annotation(state.bucket, state.key)
    except Exception:  # a cache-read failure shouldn't block ingestion
        logger.warning("Cache check failed for %s; will reprocess", state.key)
        return {}
    if existing:
        return {"annotation": existing, "status": "skipped"}
    return {}


def route_after_cache(state: DocState) -> str:
    return "done" if state.annotation else "fetch_metadata"


def fetch_metadata(state: DocState) -> dict:           # STEP 2
    return {"metadata": get_metadata(state.bucket, state.key)}


def ingest_document(state: DocState) -> dict:          # STEP 3 (Textract inside)
    return {"chunks": data_ingestion(state.metadata)}


def analyze_entities(state: DocState) -> dict:         # STEP 4 (Comprehend Medical)
    analyses: list[dict] = []
    for idx, chunk in enumerate(state.chunks):
        for sub in _split_for_comprehend(chunk):
            analyses.append({"chunk_index": idx, "analysis": comprehend_analyze(sub)})
    return {"chunk_analyses": analyses}


def persist_annotation(state: DocState) -> dict:       # STEP 5 (write annotation)
    ann = _build_annotation(
        state.bucket, state.key, state.metadata, state.chunk_analyses
    )
    ann["annotation_ref"] = write_annotation(state.bucket, state.key, ann)
    return {"annotation": ann, "status": "ok"}


def build_document_pipeline():
    g = StateGraph(DocState)
    g.add_node("check_cache", check_cache)
    g.add_node("fetch_metadata", fetch_metadata, retry_policy=AWS_RETRY)
    g.add_node("ingest_document", ingest_document, retry_policy=AWS_RETRY)
    g.add_node("analyze_entities", analyze_entities, retry_policy=AWS_RETRY)
    g.add_node("persist_annotation", persist_annotation, retry_policy=AWS_RETRY)

    g.add_edge(START, "check_cache")
    g.add_conditional_edges(
        "check_cache",
        route_after_cache,
        {"fetch_metadata": "fetch_metadata", "done": END},
    )
    g.add_edge("fetch_metadata", "ingest_document")
    g.add_edge("ingest_document", "analyze_entities")
    g.add_edge("analyze_entities", "persist_annotation")
    g.add_edge("persist_annotation", END)
    return g.compile()


DOC_PIPELINE = build_document_pipeline()


# --------------------------------------------------------------------------- #
# PARENT GRAPH — discover documents, fan out, collect                        #
# --------------------------------------------------------------------------- #
class IngestState(BaseModel):
    bucket: str
    prefix: Optional[str] = None
    force_reprocess: bool = False
    key: Optional[str] = None  # per-branch working field (set by the Send fan-out)
    object_keys: list[str] = Field(default_factory=list)
    # accumulators — reducers required for parallel writes
    annotations: Annotated[list[dict], operator.add] = Field(default_factory=list)
    errors: Annotated[list[dict], operator.add] = Field(default_factory=list)
    processed_keys: Annotated[list[str], operator.add] = Field(default_factory=list)
    skipped_keys: Annotated[list[str], operator.add] = Field(default_factory=list)


def list_documents(state: IngestState) -> dict:        # STEP 1
    keys = list_s3_documents(state.bucket, state.prefix)
    logger.info("Discovered %d document(s)", len(keys))
    return {"object_keys": keys}


def fan_out_documents(state: IngestState):
    """Map step: one parallel branch per document (or straight to collect if none)."""
    keys = state.object_keys
    if not keys:
        return "collect_results"
    return [
        Send(
            "process_document",
            {
                "bucket": state.bucket,
                "key": key,
                "force_reprocess": state.force_reprocess,
            },
        )
        for key in keys
    ]


def process_document(state: IngestState) -> dict:
    """Fan-out target: runs the per-document subgraph for ONE document and maps
    its result to the parent accumulators. Catches everything so one bad
    document cannot fail the parallel superstep.

    NOTE ON PYDANTIC + Send: LangGraph only validates/constructs the Pydantic
    state instance for the FIRST node of a graph invocation. A node reached via
    Send(...) receives the raw payload dict exactly as built in fan_out_documents,
    with no coercion. We coerce it here so the rest of this function (and any
    reader of this code) can rely on attribute access like a normal node.
    """
    if isinstance(state, dict):
        state = IngestState(**state)
    key = state.key
    try:
        result = DOC_PIPELINE.invoke(
            {
                "bucket": state.bucket,
                "key": key,
                "force_reprocess": state.force_reprocess,
            }
        )
    except Exception as exc:  # noqa: BLE001 — intentional per-document isolation
        logger.exception("Document pipeline failed for %s", key)
        return {"errors": [{"key": key, "error": str(exc)}]}

    # DOC_PIPELINE.invoke() returns a dict-like result (LangGraph graph output is
    # never a pydantic instance, even when the graph's state_schema is one), so
    # dict-style access here is correct.
    annotation = result.get("annotation")
    if not annotation:
        return {"errors": [{"key": key, "error": "pipeline produced no annotation"}]}
    if result.get("status") == "skipped":
        return {"annotations": [annotation], "skipped_keys": [key]}
    return {"annotations": [annotation], "processed_keys": [key]}


def collect_results(state: IngestState) -> dict:
    """Fan-in: runs once after all document branches complete."""
    logger.info(
        "Ingestion complete: %d processed, %d skipped, %d errors",
        len(state.processed_keys),
        len(state.skipped_keys),
        len(state.errors),
    )
    return {}


def build_ingestion_graph():
    g = StateGraph(IngestState)
    g.add_node("list_documents", list_documents)
    g.add_node("process_document", process_document)
    g.add_node("collect_results", collect_results)

    g.add_edge(START, "list_documents")
    g.add_conditional_edges(
        "list_documents", fan_out_documents, ["process_document", "collect_results"]
    )
    g.add_edge("process_document", "collect_results")
    g.add_edge("collect_results", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_ingestion(
    bucket: str, prefix: Optional[str] = None, force_reprocess: bool = False
) -> dict:
    graph = build_ingestion_graph()
    return graph.invoke(
        {
            "bucket": bucket,
            "prefix": prefix,
            "force_reprocess": force_reprocess,
            "key": None,
            "object_keys": [],
            "annotations": [],
            "errors": [],
            "processed_keys": [],
            "skipped_keys": [],
        },
        config={"max_concurrency": MAX_CONCURRENCY},
    )


if __name__ == "__main__":
    result = run_ingestion(bucket="my-claims-bucket", prefix="claims/12345/")
    print(
        f"\nprocessed={len(result['processed_keys'])} "
        f"skipped={len(result['skipped_keys'])} "
        f"errors={len(result['errors'])}\n"
    )
    if result["annotations"]:
        print("First annotation:")
        print(json.dumps(result["annotations"][0], indent=2, default=str))
    if result["errors"]:
        print("\nErrors:", result["errors"])

    # To visualise the pipeline:
    #   build_ingestion_graph().get_graph(xray=True).draw_mermaid()  # includes subgraph nodes
