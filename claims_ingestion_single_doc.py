"""
Medical Claims Intelligence — Ingestion flow for a SINGLE S3 document
======================================================================

No fan-out, no fan-in, no Send, no parent/subgraph split, no reducers.
One claim = one S3 object for now. Multiple documents per claim come later.

Graph shape (flat, linear, with one short-circuit branch):

        START
          |
     check_cache  ──(annotation already exists)──► END
          |
     (else) fetch_metadata        STEP 2  (get_metadata)
          |
     ingest_document              STEP 3  (data_ingestion / Textract inside)
          |
     analyze_entities             STEP 4  (comprehend_analyze)
          |
     persist_annotation           STEP 5  (write_annotation)
          |
         END

Why this is simpler than the fan-out version:
  * State fields are plain scalars/lists that get OVERWRITTEN by each node,
    not accumulated — so no Annotated[..., operator.add] reducers are needed.
    Reducers exist only to resolve concurrent writes from parallel branches;
    with one document processed at a time, there's only ever one writer.
  * No Send() means no dynamic branching, so LangGraph validates/constructs
    the Pydantic state instance normally for every node (the earlier
    "Send bypasses coercion" gotcha does not apply here).
  * process_document / DOC_PIPELINE / collect_results are gone entirely —
    the four steps are just top-level nodes in a straight line.

Install:  pip install langgraph pydantic boto3
Requires: AWS credentials with S3 + Comprehend Medical.
This is PHI — keep everything in-account and scope IAM tightly.

-------------------------------------------------------------------------------
ADAPT THIS SECTION: wire in your six existing functions. Signatures are ASSUMED.
The __main__ block at the bottom monkey-patches these with FAKE in-memory
versions purely so you can run this file right now and watch the graph
execute end-to-end before your real AWS-backed implementations exist.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

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
logger = logging.getLogger("claims_ingestion_single_doc")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
COMPREHEND_MAX_CHARS = 18_000  # stay under the sync limit (verify current quota)
MIN_ENTITY_SCORE = 0.50        # drop low-confidence entities


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
    """
    raise NotImplementedError("wire in your comprehend_analyze")


def write_annotation(bucket: str, key: str, annotation: dict) -> str:
    """Persist the annotation (e.g. to S3) and return a reference/URI."""
    raise NotImplementedError("wire in your write_annotation")


def read_annotation(bucket: str, key: str) -> Optional[dict]:
    """Return an existing annotation for (bucket, key) or None."""
    raise NotImplementedError("wire in your read_annotation")


def list_annotation(bucket: str, prefix: Optional[str] = None) -> list[str]:
    """List annotation references under an optional prefix. Not used in this
    single-document flow yet — kept for interface parity with your six
    existing functions."""
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
                "icd10": ent.get("ICD10CMConcepts"),
                "rxnorm": ent.get("RxNormConcepts"),
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
# STATE — single document, plain overwrite fields, no reducers needed         #
# --------------------------------------------------------------------------- #
class ClaimDocumentState(BaseModel):
    bucket: str
    key: str
    force_reprocess: bool = False
    metadata: Optional[dict] = None
    chunks: list[str] = Field(default_factory=list)
    chunk_analyses: list[dict] = Field(default_factory=list)
    annotation: Optional[dict] = None
    status: Optional[str] = None  # "ok" | "skipped"


# --------------------------------------------------------------------------- #
# NODES — the five steps                                                     #
# --------------------------------------------------------------------------- #
def check_cache(state: ClaimDocumentState) -> dict:
    """Idempotency guard: skip if this document was already annotated."""
    if state.force_reprocess:
        return {}
    try:
        existing = read_annotation(state.bucket, state.key)
    except Exception:  # a cache-read failure shouldn't block ingestion
        logger.warning("Cache check failed for %s; will reprocess", state.key)
        return {}
    if existing:
        logger.info("Annotation already exists for %s; skipping", state.key)
        return {"annotation": existing, "status": "skipped"}
    return {}


def route_after_cache(state: ClaimDocumentState) -> str:
    return "done" if state.annotation else "fetch_metadata"


def fetch_metadata(state: ClaimDocumentState) -> dict:            # STEP 2
    logger.info("Fetching metadata for %s", state.key)
    return {"metadata": get_metadata(state.bucket, state.key)}


def ingest_document(state: ClaimDocumentState) -> dict:            # STEP 3 (Textract inside)
    logger.info("Ingesting (Textract) %s", state.key)
    return {"chunks": data_ingestion(state.metadata)}


def analyze_entities(state: ClaimDocumentState) -> dict:          # STEP 4 (Comprehend Medical)
    logger.info("Analyzing %d chunk(s) with Comprehend Medical", len(state.chunks))
    analyses: list[dict] = []
    for idx, chunk in enumerate(state.chunks):
        for sub in _split_for_comprehend(chunk):
            analyses.append({"chunk_index": idx, "analysis": comprehend_analyze(sub)})
    return {"chunk_analyses": analyses}


def persist_annotation(state: ClaimDocumentState) -> dict:        # STEP 5 (write annotation)
    ann = _build_annotation(state.bucket, state.key, state.metadata, state.chunk_analyses)
    ann["annotation_ref"] = write_annotation(state.bucket, state.key, ann)
    logger.info("Annotation written for %s", state.key)
    return {"annotation": ann, "status": "ok"}


# --------------------------------------------------------------------------- #
# GRAPH                                                                        #
# --------------------------------------------------------------------------- #
def build_ingestion_graph():
    g = StateGraph(ClaimDocumentState)
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


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def run_ingestion(bucket: str, key: str, force_reprocess: bool = False) -> dict:
    graph = build_ingestion_graph()
    return graph.invoke(
        {"bucket": bucket, "key": key, "force_reprocess": force_reprocess}
    )


# --------------------------------------------------------------------------- #
# Demo — runs the graph end-to-end right now with FAKE AWS calls              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # These monkey-patches exist ONLY so you can run this file immediately and
    # watch the graph execute correctly. Delete this block once you've wired
    # the real get_metadata / data_ingestion / comprehend_analyze /
    # write_annotation / read_annotation functions in the ADAPT section above.
    _fake_store: dict[str, dict] = {}

    def _fake_get_metadata(bucket, key):
        return {"filename": key, "content_type": "application/pdf", "pages": 3}

    def _fake_data_ingestion(metadata):
        return [
            "Patient presents with hypertension and type 2 diabetes mellitus.",
            "Prescribed metformin 500mg twice daily. Follow-up in 3 months.",
        ]

    def _fake_comprehend_analyze(text):
        if "hypertension" in text:
            return {
                "Entities": [
                    {"Category": "MEDICAL_CONDITION", "Type": "DX_NAME",
                     "Text": "hypertension", "Score": 0.97, "Traits": []},
                    {"Category": "MEDICAL_CONDITION", "Type": "DX_NAME",
                     "Text": "type 2 diabetes mellitus", "Score": 0.95, "Traits": []},
                ]
            }
        return {
            "Entities": [
                {"Category": "MEDICATION", "Type": "GENERIC_NAME",
                 "Text": "metformin", "Score": 0.93,
                 "Traits": [{"Name": "DOSAGE"}]},
            ]
        }

    def _fake_write_annotation(bucket, key, annotation):
        _fake_store[key] = annotation
        return f"s3://{bucket}/annotations/{key}.json"

    def _fake_read_annotation(bucket, key):
        return _fake_store.get(key)

    globals()["get_metadata"] = _fake_get_metadata
    globals()["data_ingestion"] = _fake_data_ingestion
    globals()["comprehend_analyze"] = _fake_comprehend_analyze
    globals()["write_annotation"] = _fake_write_annotation
    globals()["read_annotation"] = _fake_read_annotation

    BUCKET, KEY = "my-claims-bucket", "claims/12345/discharge_summary.pdf"

    print("\n===== FIRST RUN (should process) =====")
    result = run_ingestion(bucket=BUCKET, key=KEY)
    print(f"status: {result['status']}")
    print(json.dumps(result["annotation"], indent=2, default=str))

    print("\n===== SECOND RUN (should hit cache and skip) =====")
    result2 = run_ingestion(bucket=BUCKET, key=KEY)
    print(f"status: {result2['status']}")

    # To visualise the graph:
    #   build_ingestion_graph().get_graph().draw_mermaid()
