from langchain_experimental.graph_transformers import LLMGraphTransformer

# Define your medical/claims schema explicitly
# LLM can ONLY extract these — nothing else gets into the graph
ALLOWED_NODES = [
    "Patient", "Doctor", "Hospital",
    "Diagnosis", "Procedure", "Claim", "Insurer", "Medication"
]

ALLOWED_RELATIONSHIPS = [
    ("Patient",   "FILED",       "Claim"),
    ("Patient",   "TREATED_BY",  "Doctor"),
    ("Patient",   "ADMITTED_TO", "Hospital"),
    ("Patient",   "HAS_DIAGNOSIS","Diagnosis"),
    ("Claim",     "COVERS",      "Diagnosis"),
    ("Claim",     "INCLUDES",    "Procedure"),
    ("Claim",     "PROCESSED_BY","Insurer"),
    ("Doctor",    "WORKS_AT",    "Hospital"),
    ("Doctor",    "PRESCRIBED",  "Medication"),
]

transformer = LLMGraphTransformer(
    llm                    = llm,
    allowed_nodes          = ALLOWED_NODES,
    allowed_relationships  = ALLOWED_RELATIONSHIPS,
    strict_mode            = True,     # filters out anything not in schema
    node_properties        = ["name", "age", "gender", "amount",
                               "status", "date", "code"],
    relationship_properties = True
)


def validate_graph_document(graph_documents, source_texts):
    """
    Validates extracted GraphDocuments against source text.
    Checks completeness, schema compliance, and key entity coverage.
    Runs BEFORE writing to Neptune — catches errors early.
    """

    print(f"\n{'═'*55}")
    print("  GRAPH EXTRACTION VALIDATION REPORT")
    print(f"{'═'*55}")

    for doc_idx, (doc, source_text) in enumerate(
        zip(graph_documents, source_texts)
    ):
        print(f"\n[Document {doc_idx+1}]")
        print(f"  Source text length : {len(source_text)} chars")
        print(f"  Nodes extracted    : {len(doc.nodes)}")
        print(f"  Relationships      : {len(doc.relationships)}")

        # ── Check 1: Were any nodes extracted at all? ─────────────────
        if not doc.nodes:
            print(f"  ❌ WARNING: No nodes extracted — LLM may have failed")
            continue

        # ── Check 2: Print all extracted nodes ────────────────────────
        print(f"\n  Extracted Nodes:")
        for node in doc.nodes:
            print(f"    ({node.type}) id='{node.id}' props={node.properties}")

        # ── Check 3: Print all extracted relationships ─────────────────
        print(f"\n  Extracted Relationships:")
        for rel in doc.relationships:
            print(f"    ({rel.source.id})-[{rel.type}]->({rel.target.id})")

        # ── Check 4: Orphan node detection ────────────────────────────
        # Nodes with no relationships are likely extraction errors
        connected_ids = set()
        for rel in doc.relationships:
            connected_ids.add(rel.source.id)
            connected_ids.add(rel.target.id)

        orphans = [n for n in doc.nodes if n.id not in connected_ids]
        if orphans:
            print(f"\n  ⚠️  Orphan nodes (no relationships):")
            for o in orphans:
                print(f"    ({o.type}) '{o.id}'")

        # ── Check 5: Key entity presence check ────────────────────────
        # Check if important keywords from source appear as node IDs
        source_lower  = source_text.lower()
        node_ids_lower = [n.id.lower() for n in doc.nodes]

        # These are domain keywords you expect to appear as nodes
        # Adjust for your medical/claims domain
        important_keywords = [
            "patient", "doctor", "hospital", "claim",
            "diagnosis", "insurance", "procedure"
        ]
        missing_concepts = []
        for keyword in important_keywords:
            if keyword in source_lower:
                if not any(keyword in nid for nid in node_ids_lower):
                    missing_concepts.append(keyword)

        if missing_concepts:
            print(f"\n  ⚠️  Keywords in text but NOT found as nodes:")
            for kw in missing_concepts:
                print(f"    '{kw}' — possibly missed or named differently")
        else:
            print(f"\n  ✅ All expected concept keywords found as nodes")

        # ── Check 6: Relationship completeness ratio ───────────────────
        # A well-formed graph should have roughly 1+ relationship per node
        ratio = len(doc.relationships) / max(len(doc.nodes), 1)
        if ratio < 0.5:
            print(f"\n  ⚠️  Low relationship ratio: {ratio:.2f} rels/node")
            print(f"      Graph may be incomplete")
        else:
            print(f"\n  ✅ Relationship ratio: {ratio:.2f} rels/node — looks complete")

    print(f"\n{'═'*55}")
    print("  VALIDATION COMPLETE — Review warnings above before writing to Neptune")
    print(f"{'═'*55}")



def llm_judge_extraction(llm, source_text, graph_document):
    
    """
    Uses LLM as a judge to verify extracted graph against source text.

    For each extracted relationship (triple), the judge LLM checks:
      - Is this triple actually stated or implied in the source text?
      - Is the entity name accurate?
      - Is the relationship type correct?

    Returns a scored validation report.

    Args:
        llm            : ChatBedrock LLM instance
        source_text    : original document text
        graph_document : GraphDocument from LLMGraphTransformer

    Returns:
        dict : validation report with scores and flagged issues
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json

    # Build list of extracted triples
    triples = []
    for rel in graph_document.relationships:
        triples.append(
            f"({rel.source.type}:{rel.source.id})"
            f"-[{rel.type}]->"
            f"({rel.target.type}:{rel.target.id})"
        )

    if not triples:
        return {"score": 0, "issues": ["No relationships extracted"]}

    JUDGE_PROMPT = PromptTemplate(
        input_variables=["source_text", "triples"],
        template="""You are a medical data quality expert.
Below is a source document and a list of knowledge graph triples
extracted from it by an AI system.

Your job: verify each triple against the source document.

SOURCE DOCUMENT:
{source_text}

EXTRACTED TRIPLES:
{triples}

For each triple, respond with:
  CORRECT   — if the triple is clearly supported by the source text
  INCORRECT — if the triple contradicts the source text
  INFERRED  — if the triple is implied but not explicitly stated
  HALLUCINATED — if the triple has no basis in the source text

Then provide:
  MISSING: List any important entities or relationships from the
           source text that were NOT extracted as triples.
  SCORE: Overall extraction quality score 0-100.

Respond in this exact JSON format:
{{
  "triple_results": [
    {{"triple": "...", "verdict": "CORRECT|INCORRECT|INFERRED|HALLUCINATED", "reason": "..."}}
  ],
  "missing_entities": ["entity1", "entity2"],
  "missing_relationships": ["relationship1"],
  "score": 85,
  "summary": "brief overall assessment"
}}"""
    )

    chain  = JUDGE_PROMPT | llm | StrOutputParser()
    result = chain.invoke({
        "source_text": source_text[:3000],  # truncate for token limit
        "triples":     "\n".join(triples)
    })

    # Parse JSON response
    try:
        cleaned = result.replace("```json", "").replace("```", "").strip()
        report  = json.loads(cleaned)
    except Exception:
        report = {"raw_response": result, "score": -1, "error": "parse failed"}

    # Print report
    print(f"\n{'═'*55}")
    print("  LLM JUDGE VALIDATION REPORT")
    print(f"{'═'*55}")

    if "triple_results" in report:
        correct      = 0
        hallucinated = 0
        for item in report["triple_results"]:
            verdict = item.get("verdict", "?")
            icon    = {"CORRECT":"✅","INCORRECT":"❌",
                       "INFERRED":"⚠️","HALLUCINATED":"🚨"}.get(verdict, "?")
            print(f"  {icon} [{verdict}] {item.get('triple','')}")
            print(f"       → {item.get('reason','')}")
            if verdict == "CORRECT":
                correct += 1
            if verdict == "HALLUCINATED":
                hallucinated += 1

    if report.get("missing_entities"):
        print(f"\n  📋 Missing entities  : {report['missing_entities']}")
    if report.get("missing_relationships"):
        print(f"  📋 Missing relations : {report['missing_relationships']}")

    print(f"\n  🎯 Extraction Score  : {report.get('score', 'N/A')}/100")
    print(f"  📝 Summary           : {report.get('summary', '')}")

    return report



def verify_neptune_write(client, graph_documents):
    """
    After writing to Neptune, queries back node and relationship
    counts and compares against what GraphDocuments contained.
    Detects write failures and data loss.
    """
    # Count what was in GraphDocuments
    expected_nodes = sum(len(d.nodes) for d in graph_documents)
    expected_rels  = sum(len(d.relationships) for d in graph_documents)

    # Count what Neptune actually has
    node_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (n)
            RETURN labels(n)[0] AS Type, count(n) AS Count
            ORDER BY Count DESC
        """
    )
    rel_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH ()-[r]->()
            RETURN type(r) AS Type, count(r) AS Count
            ORDER BY Count DESC
        """
    )

    actual_nodes = sum(r["Count"] for r in node_result.get("results",[]))
    actual_rels  = sum(r["Count"] for r in rel_result.get("results",[]))

    print(f"\n{'═'*55}")
    print("  POST-WRITE NEPTUNE VERIFICATION")
    print(f"{'═'*55}")
    print(f"\n  {'':25} {'Expected':>10} {'In Neptune':>10} {'Status':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Nodes':<25} {expected_nodes:>10} {actual_nodes:>10} "
          f"{'✅' if actual_nodes >= expected_nodes else '⚠️':>8}")
    print(f"  {'Relationships':<25} {expected_rels:>10} {actual_rels:>10} "
          f"{'✅' if actual_rels >= expected_rels else '⚠️':>8}")

    print(f"\n  Node breakdown in Neptune:")
    for row in node_result.get("results", []):
        print(f"    {row['Type']:<20}: {row['Count']}")

    print(f"\n  Relationship breakdown in Neptune:")
    for row in rel_result.get("results", []):
        print(f"    {row['Type']:<25}: {row['Count']}")

    if actual_nodes < expected_nodes:
        print(f"\n  ⚠️  {expected_nodes - actual_nodes} nodes may be "
              f"de-duplicated (MERGE) or failed to write")
    if actual_rels < expected_rels:
        print(f"\n  ⚠️  {expected_rels - actual_rels} relationships "
              f"may have failed — check orphaned nodes")
        

# Recommended usage
# Step 1: Extract with schema constraints
# transformer     = LLMGraphTransformer(
#     llm                   = llm,
#     allowed_nodes         = ALLOWED_NODES,
#     allowed_relationships = ALLOWED_RELATIONSHIPS,
#     strict_mode           = True
# )
# graph_documents = transformer.convert_to_graph_documents(docs)

# # Step 2: Programmatic validation (fast, always run)
# validate_graph_document(graph_documents, texts)

# # Step 3: LLM-as-Judge (run on sample or first doc — catches hallucinations)
# for doc, text in zip(graph_documents[:2], texts[:2]):
#     llm_judge_extraction(llm, text, doc)

# # Step 4: Write to Neptune
# write_graph_documents_option_a(client_a, texts)

# # Step 5: Post-write verification (always run)
# verify_neptune_write(client_a, graph_documents)


def rectify_missing_data(client_a, graph_documents):
    """
    Detects and re-inserts nodes and relationships that failed
    to write in the original write_graph_documents call.

    Strategy:
      Step 1 — For every node in GraphDocuments, check if it exists
               in Neptune. Re-insert if missing.
      Step 2 — For every relationship in GraphDocuments, check if it
               exists in Neptune. Re-insert if missing.
      Step 3 — Run verify_neptune_write again to confirm tally matches.

    Args:
        client_a        : boto3 neptunedata client from connect_option_a()
        graph_documents : list of GraphDocument objects from LLMGraphTransformer

    Returns:
        dict : summary of how many nodes and relationships were rectified
    """

    print(f"\n{'═'*55}")
    print("  RECTIFICATION — Re-inserting missing nodes and relationships")
    print(f"{'═'*55}")

    total_nodes_checked    = 0
    total_nodes_rectified  = 0
    total_rels_checked     = 0
    total_rels_rectified   = 0
    failed_nodes           = []
    failed_rels            = []

    # ─────────────────────────────────────────────────────────────────
    # STEP 1 — Check and re-insert every node
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Checking all nodes...")

    for doc in graph_documents:
        for node in doc.nodes:
            total_nodes_checked += 1

            # Check if node already exists in Neptune
            check_query = f"""
                MATCH (n:{node.type})
                WHERE toLower(n.id) = toLower('{node.id}')
                RETURN count(n) AS found
            """
            try:
                result = client_a.execute_open_cypher_query(
                    openCypherQuery=check_query
                )
                found = result.get("results", [{}])[0].get("found", 0)
            except Exception as e:
                print(f"  ⚠️  Check failed for ({node.type}|{node.id}): {e}")
                found = 0

            if found > 0:
                # Node exists — skip
                print(f"  ✅ EXISTS    : ({node.type} | {node.id})")
                continue

            # Node missing — re-insert using MERGE
            print(f"  ❌ MISSING   : ({node.type} | {node.id}) — reinserting...")

            props     = node.properties or {}
            props_str = ", ".join(f"n.{k} = '{v}'" for k, v in props.items()
                                  if v is not None)
            set_clause = f"SET {props_str}" if props_str else ""

            insert_query = f"""
                MERGE (n:{node.type} {{id: '{node.id}'}})
                {set_clause}
            """
            try:
                client_a.execute_open_cypher_query(openCypherQuery=insert_query)
                print(f"  ✅ INSERTED  : ({node.type} | {node.id})")
                total_nodes_rectified += 1
            except Exception as e:
                print(f"  ❌ INSERT FAILED: ({node.type} | {node.id}) → {e}")
                failed_nodes.append({
                    "type": node.type,
                    "id":   node.id,
                    "error": str(e)
                })

    # ─────────────────────────────────────────────────────────────────
    # STEP 2 — Check and re-insert every relationship
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 2] Checking all relationships...")

    for doc in graph_documents:
        for rel in doc.relationships:
            total_rels_checked += 1

            # Check if relationship already exists in Neptune
            check_query = f"""
                MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}})-[r:{rel.type}]->(b:{rel.target.type} {{id: '{rel.target.id}'}})
                RETURN count(r) AS found
            """
            try:
                result = client_a.execute_open_cypher_query(
                    openCypherQuery=check_query
                )
                found = result.get("results", [{}])[0].get("found", 0)
            except Exception as e:
                print(f"  ⚠️  Check failed for relationship: {e}")
                found = 0

            rel_label = (f"({rel.source.id})"
                         f"-[{rel.type}]->"
                         f"({rel.target.id})")

            if found > 0:
                print(f"  ✅ EXISTS    : {rel_label}")
                continue

            # Relationship missing — first ensure both nodes exist
            print(f"  ❌ MISSING   : {rel_label} — reinserting...")

            # Ensure source node exists
            ensure_source = f"""
                MERGE (n:{rel.source.type} {{id: '{rel.source.id}'}})
            """
            # Ensure target node exists
            ensure_target = f"""
                MERGE (n:{rel.target.type} {{id: '{rel.target.id}'}})
            """
            # Insert the relationship
            insert_rel = f"""
                MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}})
                MATCH (b:{rel.target.type} {{id: '{rel.target.id}'}})
                MERGE (a)-[r:{rel.type}]->(b)
            """
            try:
                client_a.execute_open_cypher_query(openCypherQuery=ensure_source)
                client_a.execute_open_cypher_query(openCypherQuery=ensure_target)
                client_a.execute_open_cypher_query(openCypherQuery=insert_rel)
                print(f"  ✅ INSERTED  : {rel_label}")
                total_rels_rectified += 1
            except Exception as e:
                print(f"  ❌ INSERT FAILED: {rel_label} → {e}")
                failed_rels.append({
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "type":   rel.type,
                    "error":  str(e)
                })

    # ─────────────────────────────────────────────────────────────────
    # STEP 3 — Print rectification summary
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  RECTIFICATION SUMMARY")
    print(f"{'═'*55}")
    print(f"  Nodes checked      : {total_nodes_checked}")
    print(f"  Nodes rectified    : {total_nodes_rectified}")
    print(f"  Nodes failed       : {len(failed_nodes)}")
    print(f"  Rels checked       : {total_rels_checked}")
    print(f"  Rels rectified     : {total_rels_rectified}")
    print(f"  Rels failed        : {len(failed_rels)}")

    if failed_nodes:
        print(f"\n  ❌ Nodes that could not be inserted:")
        for f in failed_nodes:
            print(f"    ({f['type']} | {f['id']}) → {f['error'][:80]}")

    if failed_rels:
        print(f"\n  ❌ Relationships that could not be inserted:")
        for f in failed_rels:
            print(f"    ({f['source']})-[{f['type']}]->({f['target']}) → {f['error'][:80]}")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4 — Re-run verification to confirm tally now matches
    # ─────────────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Re-running verification after rectification...")
    verify_neptune_write(client_a, graph_documents)

    return {
        "nodes_checked":    total_nodes_checked,
        "nodes_rectified":  total_nodes_rectified,
        "nodes_failed":     failed_nodes,
        "rels_checked":     total_rels_checked,
        "rels_rectified":   total_rels_rectified,
        "rels_failed":      failed_rels
    }


# How to use
# Normal flow
# graph_documents = transformer.convert_to_graph_documents(docs)
# write_graph_documents_option_a(client_a, texts)

# # Verify
# verify_neptune_write(client_a, graph_documents)

# # If tally doesn't match — rectify
# rectify_missing_data(client_a, graph_documents)