import json
import logging
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

s3_client = boto3.client("s3")

DATA_INGESTION_API_URL = "https://your-data-ingestion-service.example.com/ingest"


def lambda_handler(event, context):
    bucket = event["bucket"]
    key = event["key"]
    logger.info("Processing s3://%s/%s", bucket, key)

    # --- Step 1: Call Data Ingestion Service ---
    payload = json.dumps({"bucket": bucket, "key": key}).encode("utf-8")
    req = urllib.request.Request(
        DATA_INGESTION_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    chunks = body.get("chunks", [])
    logger.info("Got %d chunks", len(chunks))

    # --- Step 2: Save chunks as S3 annotation ---
    annotation_payload = json.dumps(
        {"total_chunks": len(chunks), "chunks": chunks}
    ).encode("utf-8")

    s3_client.put_object_annotation(
        Bucket=bucket,
        Key=key,
        AnnotationName="doc_chunks",
        AnnotationPayload=annotation_payload,
    )
    logger.info("Annotation 'doc_chunks' written")

    return {"status": "done", "total_chunks": len(chunks)}
