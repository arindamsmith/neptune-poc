"""
S3 Annotation - Write and Read
================================
Two simple functions:
  - write_annotation(): writes basic metadata as annotation on an S3 object
  - read_annotation(): reads the annotation back and returns the data
"""

import json
import boto3
from datetime import datetime, timezone

s3_client = boto3.client("s3")

BUCKET_NAME = "YOUR-BUCKET-NAME"
OBJECT_KEY  = "documents/sample.pdf"


# ─────────────────────────────────────────────
# WRITE annotation
# ─────────────────────────────────────────────
def write_annotation(bucket: str, key: str) -> dict:
    """
    Writes basic metadata as an S3 annotation on the given object.
    Annotation name: 'doc_metadata'
    """
    metadata = {
        "source":       "clinical-intelligence-platform",
        "document_type": "medical-record",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "status":       "processed",
        "tags":         ["LTD", "claim", "medical"],
        "page_count":   1,
        "notes":        "PoC annotation — will be replaced with Textract output"
    }

    payload = json.dumps(metadata, indent=2).encode("utf-8")

    s3_client.put_object_annotation(
        Bucket=bucket,
        Key=key,
        AnnotationName="doc_metadata",
        AnnotationPayload=payload,
    )

    print(f"✅ Annotation written to s3://{bucket}/{key}")
    print(f"   Annotation name : doc_metadata")
    print(f"   Payload size    : {len(payload)} bytes")
    return metadata


# ─────────────────────────────────────────────
# READ annotation
# ─────────────────────────────────────────────
def read_annotation(bucket: str, key: str) -> dict:
    """
    Reads the 'doc_metadata' annotation from the given S3 object
    and returns it as a Python dict.
    """
    response = s3_client.get_object_annotation(
        Bucket=bucket,
        Key=key,
        AnnotationName="doc_metadata",
    )

    raw = response["AnnotationPayload"].read()
    data = json.loads(raw.decode("utf-8"))

    print(f"✅ Annotation retrieved from s3://{bucket}/{key}")
    print(json.dumps(data, indent=2))
    return data


# ─────────────────────────────────────────────
# MAIN — run both
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("── WRITE ──────────────────────────────")
    write_annotation(BUCKET_NAME, OBJECT_KEY)

    print("\n── READ ───────────────────────────────")
    data = read_annotation(BUCKET_NAME, OBJECT_KEY)
