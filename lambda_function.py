"""
Medical Document Pipeline - Lambda Durable Function
====================================================
Workflow:
  S3 upload trigger
    → [1] Call Data Ingestion Service (Textract internally) → get chunks
    → [2] Write chunks as S3 annotation on the source object
    → [3] For each chunk → Comprehend Medical detect_entities_v2
    → [4] Write entities as S3 annotation on the source object

Author: PoC
Runtime: Python 3.13 or 3.14 (required for durable functions)
"""

import json
import logging
import urllib.request
import urllib.error

import boto3
from aws_durable_execution_sdk_python import (
    DurableContext,
    StepContext,
    durable_execution,
    durable_step,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── AWS clients (initialised once per container, reused across warm invocations)
s3_client = boto3.client("s3")
comprehend_medical_client = boto3.client("comprehendmedical")

# ── REPLACE this with your real Data Ingestion Service endpoint
DATA_INGESTION_API_URL = "https://your-data-ingestion-service.example.com/ingest"

# ── Comprehend Medical text limit per call (hard AWS limit)
COMPREHEND_MAX_BYTES = 19_000   # stay safely under the 20 000-byte hard limit


# ─────────────────────────────────────────────────────────────
# STEP 1 — Call Data Ingestion Service
# ─────────────────────────────────────────────────────────────
@durable_step
def call_ingestion_service(step_context: StepContext, bucket: str, key: str) -> list:
    """
    POST { bucket, key } to the Data Ingestion Service.
    The service runs Textract internally and returns a list of text chunks.

    Expected response shape:
        { "chunks": ["chunk text 1", "chunk text 2", ...] }
    """
    logger.info("Calling Data Ingestion Service for s3://%s/%s", bucket, key)

    payload = json.dumps({"bucket": bucket, "key": key}).encode("utf-8")
    req = urllib.request.Request(
        DATA_INGESTION_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            chunks = body.get("chunks", [])
            logger.info("Received %d chunks from ingestion service", len(chunks))
            return chunks
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(
            f"Data Ingestion Service returned HTTP {e.code}: {error_body}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to call Data Ingestion Service: {e}") from e


# ─────────────────────────────────────────────────────────────
# STEP 2 — Write chunks as S3 annotation
# ─────────────────────────────────────────────────────────────
@durable_step
def write_chunks_annotation(
    step_context: StepContext, bucket: str, key: str, chunks: list
) -> str:
    """
    Persists the full chunk list as a single S3 annotation named 'doc_chunks'.
    Annotation payload is stored as JSON so it's queryable via Athena later.
    """
    logger.info("Writing %d chunks as S3 annotation on s3://%s/%s", len(chunks), bucket, key)

    annotation_payload = json.dumps(
        {
            "total_chunks": len(chunks),
            "chunks": chunks,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    s3_client.put_object_annotation(
        Bucket=bucket,
        Key=key,
        AnnotationName="doc_chunks",          # name must NOT start with 'aws' or 's3'
        AnnotationPayload=annotation_payload,
    )

    logger.info("Successfully wrote 'doc_chunks' annotation")
    return "doc_chunks"


# ─────────────────────────────────────────────────────────────
# STEP 3 — Comprehend Medical for a single chunk
# ─────────────────────────────────────────────────────────────
@durable_step
def detect_entities_for_chunk(
    step_context: StepContext, chunk_index: int, chunk_text: str
) -> dict:
    """
    Calls Comprehend Medical detect_entities_v2 on a single chunk.
    Splits the chunk if it exceeds the 20 000-byte limit.

    Returns:
        {
          "chunk_index": int,
          "entities": [ { Text, Category, Type, Score, ... }, ... ]
        }
    """
    logger.info("Running Comprehend Medical on chunk %d", chunk_index)

    all_entities = []
    encoded = chunk_text.encode("utf-8")

    # Split into sub-segments if chunk exceeds the byte limit
    segments = []
    for start in range(0, len(encoded), COMPREHEND_MAX_BYTES):
        segment_bytes = encoded[start : start + COMPREHEND_MAX_BYTES]
        segments.append(segment_bytes.decode("utf-8", errors="ignore"))

    for seg_idx, segment in enumerate(segments):
        logger.info(
            "  → Chunk %d, segment %d/%d (%d bytes)",
            chunk_index, seg_idx + 1, len(segments), len(segment.encode("utf-8")),
        )
        response = comprehend_medical_client.detect_entities_v2(Text=segment)
        entities = [
            {
                "Text": e["Text"],
                "Category": e["Category"],
                "Type": e["Type"],
                "Score": round(e["Score"], 4),
                "Traits": [t["Name"] for t in e.get("Traits", [])],
            }
            for e in response.get("Entities", [])
        ]
        all_entities.extend(entities)

    return {"chunk_index": chunk_index, "entities": all_entities}


# ─────────────────────────────────────────────────────────────
# STEP 4 — Write all entities as S3 annotation
# ─────────────────────────────────────────────────────────────
@durable_step
def write_entities_annotation(
    step_context: StepContext, bucket: str, key: str, all_chunk_results: list
) -> str:
    """
    Consolidates entity results from all chunks into a single S3 annotation
    named 'medical_entities'.

    Stores per-chunk breakdown AND a flat deduplicated entity list.
    """
    logger.info(
        "Writing medical entities annotation for s3://%s/%s", bucket, key
    )

    # Build flat unique entity list (Text + Category) for easy querying
    seen = set()
    flat_entities = []
    for chunk_result in all_chunk_results:
        for entity in chunk_result.get("entities", []):
            key_tuple = (entity["Text"].lower(), entity["Category"])
            if key_tuple not in seen:
                seen.add(key_tuple)
                flat_entities.append(entity)

    annotation_payload = json.dumps(
        {
            "total_entities": len(flat_entities),
            "unique_entities": flat_entities,
            "by_chunk": all_chunk_results,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    s3_client.put_object_annotation(
        Bucket=bucket,
        Key=key,
        AnnotationName="medical_entities",
        AnnotationPayload=annotation_payload,
    )

    logger.info(
        "Successfully wrote 'medical_entities' annotation (%d unique entities)",
        len(flat_entities),
    )
    return "medical_entities"


# ─────────────────────────────────────────────────────────────
# MAIN DURABLE HANDLER
# ─────────────────────────────────────────────────────────────
@durable_execution
def lambda_handler(event, context: DurableContext):
    """
    Entry point. Triggered by:
      (a) S3 Event Notification  → event has 'Records'
      (b) Manual test invocation → event has 'bucket' and 'key' directly
    """

    # ── Parse bucket/key from either S3 trigger or manual test ──────────────
    if "Records" in event:
        # S3 trigger
        record = event["Records"][0]
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]
    else:
        # Manual test invocation
        bucket = event["bucket"]
        key = event["key"]

    logger.info("Pipeline started for s3://%s/%s", bucket, key)

    # ── STEP 1: Call Data Ingestion Service ─────────────────────────────────
    chunks = context.step(call_ingestion_service(bucket, key))
    logger.info("Step 1 complete — got %d chunks", len(chunks))

    # ── STEP 2: Write chunks as S3 annotation ───────────────────────────────
    context.step(write_chunks_annotation(bucket, key, chunks))
    logger.info("Step 2 complete — chunks annotation written")

    # ── STEP 3: Comprehend Medical per chunk ────────────────────────────────
    all_chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_result = context.step(
            detect_entities_for_chunk(i, chunk),
            name=f"comprehend-chunk-{i}",   # unique name per chunk for idempotency
        )
        all_chunk_results.append(chunk_result)
    logger.info("Step 3 complete — Comprehend Medical done for all %d chunks", len(chunks))

    # ── STEP 4: Write entities as S3 annotation ─────────────────────────────
    context.step(write_entities_annotation(bucket, key, all_chunk_results))
    logger.info("Step 4 complete — medical entities annotation written")

    return {
        "status": "SUCCEEDED",
        "bucket": bucket,
        "key": key,
        "total_chunks": len(chunks),
        "total_entities": sum(len(r["entities"]) for r in all_chunk_results),
    }
