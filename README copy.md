# Medical Document Pipeline — PoC Setup Guide
## Lambda Durable Functions + S3 Annotations

---

## Files in this Project

```
medical-doc-pipeline/
├── lambda_function/
│   └── lambda_function.py   ← Main durable function (4-step workflow)
├── dispatcher/
│   └── dispatcher.py        ← Tiny Lambda that bridges S3 trigger → durable function
├── iam_policy.json          ← IAM permissions to attach to execution role
└── test_event.json          ← JSON to paste into Lambda console for manual test
```

---

## PART 1 — AWS Console Setup (Do this first)

### Step 1 — Check your region supports both features
Go to: https://aws.amazon.com/about-aws/whats-new/
- Lambda Durable Functions: available in ~31 regions as of June 2026
- S3 Annotations: available in all regions
- Recommended region: us-east-1 (N. Virginia) — safest bet for both

---

### Step 2 — Create the S3 Bucket
1. Go to **S3 Console** → Create bucket
2. Bucket name: e.g. `medical-docs-poc`
3. Region: match your Lambda region
4. Leave all other settings default → Create

---

### Step 3 — Create the IAM Role for the Durable Function

1. Go to **IAM Console** → Roles → Create role
2. Trusted entity: **AWS Service** → **Lambda** → Next
3. Skip attaching policies here → Next
4. Role name: `MedicalPipelineLambdaRole` → Create role
5. Open the created role → **Add permissions** → **Create inline policy**
6. Switch to **JSON** tab → paste the contents of `iam_policy.json`
7. Replace `YOUR-BUCKET-NAME` with your actual bucket name
8. Policy name: `MedicalPipelinePolicy` → Create policy

---

### Step 4 — Create the Durable Lambda Function

1. Go to **Lambda Console** → Create function
2. **Author from scratch**
3. Function name: `MedicalDocPipeline`
4. Runtime: **Python 3.13** (or 3.14)
5. **☑ Enable durable execution** — tick this checkbox
   - Execution timeout: `1 hour` (sufficient for this PoC)
   - Retention period: `14 days` (default is fine)
6. Execution role: **Use an existing role** → select `MedicalPipelineLambdaRole`
7. → **Create function**

**Paste the code:**
8. On the Code tab → open `lambda_function.py` in the editor
9. **Replace ALL the default code** with the contents of `lambda_function/lambda_function.py`
10. Update line: `DATA_INGESTION_API_URL = "https://your-data-ingestion-service..."`
    → replace with your real endpoint
11. → **Deploy**

**Publish a version (required for durable functions):**
12. Actions → **Publish new version** → description: "v1" → Publish
13. Note the versioned ARN shown at the top — it ends in `:1`
    Example: `arn:aws:lambda:us-east-1:123456789012:function:MedicalDocPipeline:1`
    ← **Save this ARN — you need it in Step 5**

---

### Step 5 — Create the Dispatcher Lambda

The dispatcher is needed because S3 triggers always call $LATEST,
but durable functions require a qualified (versioned) ARN.

1. Lambda Console → Create function
2. Function name: `MedicalDocDispatcher`
3. Runtime: **Python 3.13**
4. Execution role: same `MedicalPipelineLambdaRole`
   (needs lambda:InvokeFunction permission — add to IAM policy if needed)
5. → Create function
6. Paste the contents of `dispatcher/dispatcher.py` into the code editor → Deploy
7. Go to **Configuration** → **Environment variables** → Edit → Add:
   - Key: `DURABLE_FUNCTION_ARN`
   - Value: the versioned ARN from Step 4 (ends in `:1`)
8. → Save

---

### Step 6 — Add S3 Trigger to Dispatcher

1. Open the **Dispatcher Lambda** (`MedicalDocDispatcher`)
2. Configuration → Triggers → Add trigger
3. Source: **S3**
4. Bucket: your bucket (`medical-docs-poc`)
5. Event types: **PUT** (or "All object create events")
6. Suffix: `.pdf` (optional — limits to PDFs only)
7. → Add

---

## PART 2 — Testing

### Test A — Manual test (no S3 upload needed, fastest way to verify)

1. Open the **MedicalDocPipeline** durable function
2. Go to **Versions** tab → click on version `1`
   ⚠️ You MUST test from the versioned function, not $LATEST
3. Click **Test** tab → Create new test event
4. Event name: `ManualTest`
5. Paste the contents of `test_event.json`:
   ```json
   {
     "bucket": "medical-docs-poc",
     "key": "documents/sample.pdf"
   }
   ```
   (Upload a sample.pdf to your bucket first)
6. → **Test**

**What to look for:**
- The test returns quickly — durable functions return immediately and execute asynchronously
- Go to the **Durable executions** tab to see the execution status
- Click the execution to see each step: call-ingestion, write-chunks, comprehend-chunk-0, comprehend-chunk-1... write-entities

---

### Test B — End-to-end via S3 trigger

1. Upload a PDF to your S3 bucket:
   - S3 Console → your bucket → Upload → select a PDF
2. This fires the S3 trigger → Dispatcher Lambda → Durable Function
3. Monitor in Lambda Console → MedicalDocPipeline → **Durable executions** tab

---

### Verify annotations were written

After a successful run, check via AWS CLI:

```bash
# List all annotations on the object
aws s3api list-object-annotations \
  --bucket medical-docs-poc \
  --key documents/sample.pdf

# Read the chunks annotation
aws s3api get-object-annotation \
  --bucket medical-docs-poc \
  --key documents/sample.pdf \
  --annotation-name doc_chunks \
  /tmp/chunks.json
cat /tmp/chunks.json

# Read the medical entities annotation
aws s3api get-object-annotation \
  --bucket medical-docs-poc \
  --key documents/sample.pdf \
  --annotation-name medical_entities \
  /tmp/entities.json
cat /tmp/entities.json
```

---

## PART 3 — Monitoring & Debugging

### CloudWatch Logs
- Each step logs its progress
- Go to CloudWatch → Log groups → `/aws/lambda/MedicalDocPipeline`

### Durable Executions Tab
- Lambda Console → MedicalDocPipeline → Durable executions
- See: RUNNING / SUCCEEDED / FAILED status per execution
- Drill in to see which step failed and why

### Common errors and fixes

| Error | Likely cause | Fix |
|-------|-------------|-----|
| `DurableExecutionNotEnabled` | Invoked $LATEST | Use versioned ARN `:1` |
| `AccessDeniedException` on S3 annotation | Missing IAM permission | Add `s3:PutObjectAnnotation` to role |
| `AccessDeniedException` on Comprehend Medical | Missing IAM permission | Add `comprehendmedical:DetectEntitiesV2` |
| `TextSizeLimitExceededException` | Chunk > 20 000 bytes | Code handles this with auto-split — verify chunk sizes |
| Ingestion service returns empty chunks | Check your API endpoint | Log the raw response from step 1 in CloudWatch |

---

## Key things to remember

1. Always invoke the durable function using a **versioned ARN** (ends in `:1`, `:2`, etc.), never $LATEST
2. Each time you deploy new code, **publish a new version** and update the Dispatcher's env variable
3. S3 annotations names must NOT start with `aws` or `s3` (reserved prefixes)
4. `context.step(...)` calls must use **unique names** — the loop uses `comprehend-chunk-{i}` to ensure this
5. Do NOT put side effects outside `context.step()` — they'll re-run on every replay


## Some info
1. first list all the annotations with prefix "comprehend_analysis" from the object

i want to collate and aggregate all the comprehend analyzed chunks into a single file so that it will be easier to refer. but as comprehend has the 20k limit, we can't directly analyze the whole file and hence we analyze separate chunks. now that we have all the chunks analyzed and created as annotation to the object, let's read all the annotations and collate into a single file and store it as an object in s3.

2. also the data_ingestion list of chunk that is saved locally is also saved in the s3 bucket as an object

3. take these 2 objects, the comnbined chunks and the combined medical entities and create the context for llm.

4. write the system prompt to use this context to extract meaningful medical summary utilizing the comprehend medical detect entities. this summary will be used by the claims manager. it is a very important summary which will form the basis of the crucial decision made by the claims manager about claims raised.

5. take this summary and store it in dynamo db

i would want to see this first as functional program, as you have been doing now in a .py file. after this code runs fine, i would like to convert all these functions and make it into a langgrpah flow where the nodes will be the function calls as you have done for the previous flow.
