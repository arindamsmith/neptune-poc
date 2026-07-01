"""
Dispatcher Lambda
=================
S3 triggers cannot directly invoke a durable function with a qualified ARN.
This tiny dispatcher bridges that gap:
  S3 Event Notification → Dispatcher Lambda → Durable Function (qualified ARN)

Why needed:
  Durable functions MUST be invoked via a versioned/aliased ARN, not $LATEST.
  S3 triggers always call $LATEST, so we use this dispatcher to invoke
  the durable function with the correct qualified ARN.

Deploy this as a SEPARATE, standard (non-durable) Lambda function.
"""

import json
import os
import logging
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

lambda_client = boto3.client("lambda")

# Set this env variable to your durable function's versioned ARN, e.g.:
# arn:aws:lambda:us-east-1:123456789012:function:MedicalDocPipeline:1
DURABLE_FUNCTION_ARN = os.environ["DURABLE_FUNCTION_ARN"]


def lambda_handler(event, context):
    """
    Receives S3 event, fires off the durable function asynchronously.
    Uses the object key as a unique execution name to prevent duplicates
    if S3 retries the event delivery.
    """
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        # Use bucket+key as execution name for idempotency
        # (same file uploaded twice will not re-run if execution is still active)
        execution_name = f"{bucket}-{key}".replace("/", "-").replace(".", "-")[:128]

        logger.info(
            "Dispatching durable execution for s3://%s/%s (execution: %s)",
            bucket, key, execution_name,
        )

        lambda_client.invoke(
            FunctionName=DURABLE_FUNCTION_ARN,
            InvocationType="Event",           # async — fire and forget
            Payload=json.dumps({
                "executionName": execution_name,
                "Records": [record],          # pass original S3 record through
            }),
        )

    return {"statusCode": 200}
