"""
Medical Claims Intelligence — LangGraph PoC (functional style)
==============================================================

Two graphs, deliberately kept separate because they have different lifecycles:

  1) INGESTION graph  (runs once per claim, fans out over documents):
        list_documents  ->  [process_document x N in parallel]  ->  summarize
     process_document does, for ONE document:
        Step 2: get_metadata      -> file metadata
        Step 3: data_ingestion    -> Textract inside, returns text chunks
        Step 4: comprehend_analyze -> Comprehend Medical per chunk
        Step 5: write_annotation  -> persist compacted annotation to S3

  2) CHAT graph  (interactive, multi-turn, checkpointed):
        the claims manager asks questions; answers are grounded ONLY in the
        annotations + summary produced by the ingestion graph.

Design notes baked into this code:
  * process_document NEVER raises -> protects the parallel superstep (one bad
    PDF can otherwise wipe the whole batch's writes).
  * accumulator channels use Annotated[list, operator.add] reducers, or parallel
    writes silently overwrite each other.
  * idempotency: skip a document if an annotation already exists (avoids
    re-billing Textract + Comprehend).
  * Comprehend Medical synchronous size ceiling (~20k UTF-8 chars) is respected
    by _split_for_comprehend.
  * the LLM is told to use ONLY provided findings, cite source docs, and NOT to
    make an approve/deny decision (that stays with the human).

Install:  pip install langgraph langchain-aws boto3
Requires: AWS credentials with S3 + Comprehend Medical (+ optional Infer*) +
          Bedrock model access. This is PHI — keep everything in-account.

-------------------------------------------------------------------------------
ADAPT THIS SECTION: wire in your six existing functions. The signatures below
are ASSUMED — change them to match your real implementations, or write thin
adapters. Everything downstream calls only these names.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import operator
import os
from typing import Annotated, Any, Optional, TypedDict

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claims_poc")

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# Set to a Bedrock model id available in YOUR region/account:
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)
# Stay comfortably under the Comprehend Medical sync limit (verify current quota):
COMPREHEND_MAX_CHARS = 18_000
# Drop low-confidence entities before they reach the LLM:
MIN_ENTITY_SCORE = 0.50
# Cap parallel Textract/Comprehend calls to respect TPS limits:
MAX_CONCURRENCY = 5


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

    Expected to contain at least an 'Entities' list of dicts with keys like
    Category, Type, Text, Score, Traits. If you add InferICD10CM / InferRxNorm,
    merge their outputs into the returned dict (see _build_annotation).
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
# Small helpers (not your functions)                                          #
# --------------------------------------------------------------------------- #
def list_s3_documents(
    bucket: str, prefix: Optional[str] = None, suffixes: tuple[str, ...] = (".pdf",)
) -> list[str]:
    """List document keys in a bucket (paginated, so it handles >1000 keys)."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix or ""):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):  # skip folder placeholders
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
    """Compact verbose Comprehend output into an LLM-friendly, deduped rollup
    with provenance. This is what keeps token cost and noise down."""
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
                # If you add InferICD10CM/InferRxNorm, attach codes here:
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


_LLM: Optional[ChatBedrockConverse] = None


def _get_llm() -> ChatBedrockConverse:
    global _LLM
    if _LLM is None:
        # temperature=0 for factual, reproducible claims work
        _LLM = ChatBedrockConverse(
            model=BEDROCK_MODEL_ID, region_name=AWS_REGION, temperature=0
        )
    return _LLM


# --------------------------------------------------------------------------- #
# INGESTION GRAPH                                                             #
# --------------------------------------------------------------------------- #
class IngestState(TypedDict):
    # inputs / config
    bucket: str
    prefix: Optional[str]
    force_reprocess: bool
    # per-branch working field (set by the Send fan-out, one per document)
    key: Optional[str]
    # discovery
    object_keys: list[str]
    # accumulators — MUST have reducers because branches write in parallel
    annotations: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]
    processed_keys: Annotated[list[str], operator.add]
    # output
    case_summary: str


def list_documents(state: IngestState) -> dict:
    keys = list_s3_documents(state["bucket"], state.get("prefix"))
    logger.info("Discovered %d document(s)", len(keys))
    return {"object_keys": keys}


def fan_out_documents(state: IngestState):
    """Map step: one parallel branch per document. Falls through to summarize
    when there is nothing to process."""
    keys = state["object_keys"]
    if not keys:
        return "summarize"
    return [
        Send(
            "process_document",
            {
                "bucket": state["bucket"],
                "key": key,
                "force_reprocess": state.get("force_reprocess", False),
            },
        )
        for key in keys
    ]


def process_document(state: IngestState) -> dict:
    """Runs once per document, in parallel. NEVER raises: on failure it records
    an error in state so the superstep (and the other branches) survive."""
    bucket, key = state["bucket"], state["key"]
    try:
        # idempotency: skip if we already analysed this document
        if not state.get("force_reprocess", False):
            existing = read_annotation(bucket, key)
            if existing:
                logger.info("Skipping %s (annotation exists)", key)
                return {"annotations": [existing], "processed_keys": [key]}

        metadata = get_metadata(bucket, key)                       # step 2
        chunks = data_ingestion(metadata)                          # step 3 (Textract inside)

        chunk_analyses: list[dict] = []                            # step 4
        for idx, chunk in enumerate(chunks):
            for sub in _split_for_comprehend(chunk):
                chunk_analyses.append(
                    {"chunk_index": idx, "analysis": comprehend_analyze(sub)}
                )

        annotation = _build_annotation(bucket, key, metadata, chunk_analyses)
        annotation["annotation_ref"] = write_annotation(bucket, key, annotation)  # step 5

        logger.info("Processed %s (%d chunks)", key, annotation["num_chunks"])
        return {"annotations": [annotation], "processed_keys": [key]}

    except Exception as exc:  # noqa: BLE001 — intentional: isolate branch failure
        logger.exception("Failed processing %s", key)
        return {"errors": [{"key": key, "error": str(exc)}]}


CLAIMS_SUMMARY_SYSTEM_PROMPT = """You are a clinical claims analyst assistant. You are given STRUCTURED medical
findings that were automatically extracted from the documents attached to an
insurance claim. Produce a case summary for a human claims manager.

Hard rules:
- Use ONLY the findings provided. Do not invent, infer, or add outside medical
  knowledge. If something is not in the findings, say so.
- For every material clinical point, cite the source document (its source_key).
- DO NOT state or recommend an approve/deny decision. That is the claims
  manager's responsibility. Your job is to summarise and surface issues.

Structure the summary as:
1. Documents overview (what was received)
2. Key medical conditions / diagnoses
3. Treatments, procedures, and medications
4. Relevant dates / timeline (if present)
5. Potential red flags or items to verify (e.g. pre-existing indicators,
   negation/uncertainty traits, low-confidence extractions, conflicting info)
6. Information gaps (what a claims decision would still need)
"""


def summarize(state: IngestState) -> dict:
    annotations = state.get("annotations", [])
    if not annotations:
        return {"case_summary": "No documents were processed; no summary available."}

    context = _format_context_for_llm(annotations)
    messages = [
        SystemMessage(content=CLAIMS_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Structured findings extracted from the claim documents:\n\n"
                f"{context}\n\nProduce the claim case summary."
            )
        ),
    ]
    resp = _get_llm().invoke(messages)
    return {"case_summary": resp.content}


def _format_context_for_llm(annotations: list[dict]) -> str:
    blocks: list[str] = []
    for ann in annotations:
        lines = [f"### Document: {ann.get('source_key')}"]
        meta = ann.get("document_metadata") or {}
        if meta:
            lines.append(f"Metadata: {json.dumps(meta, default=str)[:500]}")
        for cat, items in (ann.get("entities_by_category") or {}).items():
            rendered = "; ".join(
                item["text"]
                + (f" [{', '.join(item['traits'])}]" if item.get("traits") else "")
                for item in items
                if item.get("text")
            )
            lines.append(f"- {cat}: {rendered}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_ingestion_graph():
    g = StateGraph(IngestState)
    g.add_node("list_documents", list_documents)
    g.add_node("process_document", process_document)
    g.add_node("summarize", summarize)

    g.add_edge(START, "list_documents")
    # conditional edge = the map step; targets it can route to:
    g.add_conditional_edges(
        "list_documents", fan_out_documents, ["process_document", "summarize"]
    )
    # fan-in: summarize runs once, after ALL process_document branches complete
    g.add_edge("process_document", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# CHAT GRAPH                                                                  #
# --------------------------------------------------------------------------- #
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str  # summary + findings, injected once, persisted by checkpointer


CLAIMS_CHAT_SYSTEM_PROMPT = """You answer a claims manager's questions about ONE insurance claim, grounded
strictly in the context below (a generated summary plus the extracted findings).

Rules:
- Answer ONLY from the provided context. If the answer is not there, say
  "That is not available in the provided claim documents."
- Cite the source document(s) for factual claims.
- Do not make an approve/deny decision and do not give medical or legal advice.
- Be concise.

CLAIM CONTEXT:
{context}
"""


def chat_node(state: ChatState) -> dict:
    system = SystemMessage(
        content=CLAIMS_CHAT_SYSTEM_PROMPT.format(context=state["context"])
    )
    resp = _get_llm().invoke([system, *state["messages"]])
    return {"messages": [resp]}


def build_chat_graph():
    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    # MemorySaver for the PoC; swap for PostgresSaver/DynamoDB in production
    return g.compile(checkpointer=MemorySaver())


# --------------------------------------------------------------------------- #
# Orchestration / usage                                                       #
# --------------------------------------------------------------------------- #
def run_ingestion(
    bucket: str, prefix: Optional[str] = None, force_reprocess: bool = False
) -> dict:
    graph = build_ingestion_graph()
    result = graph.invoke(
        {
            "bucket": bucket,
            "prefix": prefix,
            "force_reprocess": force_reprocess,
            "key": None,
            "object_keys": [],
            "annotations": [],
            "errors": [],
            "processed_keys": [],
            "case_summary": "",
        },
        config={"max_concurrency": MAX_CONCURRENCY},
    )
    return result


def make_chat_asker(ingestion_result: dict, thread_id: str = "claim-001"):
    """Returns ask(question) -> str, with conversation memory per thread_id."""
    chat_graph = build_chat_graph()
    context = (
        "CASE SUMMARY:\n"
        + ingestion_result.get("case_summary", "")
        + "\n\nDETAILED FINDINGS:\n"
        + _format_context_for_llm(ingestion_result.get("annotations", []))
    )
    cfg = {"configurable": {"thread_id": thread_id}}

    def ask(question: str) -> str:
        out = chat_graph.invoke(
            {"messages": [HumanMessage(content=question)], "context": context}, cfg
        )
        return out["messages"][-1].content

    return ask


if __name__ == "__main__":
    # 1) Ingest a claim's documents from S3
    result = run_ingestion(bucket="my-claims-bucket", prefix="claims/12345/")
    print("\n===== CASE SUMMARY =====\n")
    print(result["case_summary"])
    if result["errors"]:
        print("\n===== ERRORS =====\n", result["errors"])

    # 2) Let the claims manager interrogate it
    ask = make_chat_asker(result, thread_id="claim-12345")
    print(ask("What are the primary diagnoses and are any flagged as pre-existing?"))
    print(ask("What medications are documented?"))  # remembers prior turn
