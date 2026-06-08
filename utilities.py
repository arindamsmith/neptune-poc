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

# Query by ID filter
def query_by_id(client_a, entity_type, entity_id, depth=1):
    """
    Queries Neptune for a specific entity and all its connected nodes
    up to a given depth. Works for any entity type and ID.

    Args:
        client_a    : boto3 neptunedata client from connect_option_a()
        entity_type : node label e.g. 'Patient', 'Claim', 'Doctor'
        entity_id   : the id value e.g. 'Ravi Sharma', 'CLM001'
        depth       : how many hops to traverse (default 1)
                      1 = direct connections only
                      2 = connections of connections

    Returns:
        dict : {
            "entity"        : the matched node properties,
            "connections"   : list of directly connected nodes,
            "full_path"     : all nodes and relationships up to depth
        }

    Usage:
        query_by_id(client_a, 'Patient', 'Ravi Sharma')
        query_by_id(client_a, 'Claim',   'CLM001')
        query_by_id(client_a, 'Doctor',  'Dr. Kapoor', depth=2)
    """

    def run(cypher):
        result = client_a.execute_open_cypher_query(openCypherQuery=cypher)
        return result.get("results", [])

    def display(title, rows):
        print(f"\n  {'─'*52}")
        print(f"  {title}")
        print(f"  {'─'*52}")
        if not rows:
            print("  (no results)")
            return
        headers = list(rows[0].keys())
        print("  " + " | ".join(h[:18].ljust(18) for h in headers))
        print("  " + "─" * (21 * len(headers)))
        for row in rows:
            print("  " + " | ".join(
                str(row.get(h, ""))[:18].ljust(18) for h in headers
            ))
        print(f"  {len(rows)} row(s)")

    print(f"\n{'═'*55}")
    print(f"  QUERY: {entity_type} — '{entity_id}'")
    print(f"{'═'*55}")

    # ── Q1: Fetch the entity node itself ──────────────────────────────
    entity_rows = run(f"""
        MATCH (n:{entity_type})
        WHERE toLower(n.id)   CONTAINS toLower('{entity_id}')
           OR toLower(n.name) CONTAINS toLower('{entity_id}')
        RETURN labels(n)[0] AS Type, n.id AS ID, n.name AS Name,
               n.status AS Status, n.amount AS Amount,
               n.age AS Age, n.city AS City,
               n.specialty AS Specialty, n.filedDate AS FiledDate
        LIMIT 5
    """)
    display(f"{entity_type} Details", entity_rows)

    if not entity_rows:
        print(f"\n  ❌ No {entity_type} found with id containing '{entity_id}'")
        return {}

    # ── Q2: All direct connections (depth 1) ─────────────────────────
    direct_rows = run(f"""
        MATCH (n:{entity_type})-[r]-(connected)
        WHERE toLower(n.id)   CONTAINS toLower('{entity_id}')
           OR toLower(n.name) CONTAINS toLower('{entity_id}')
        RETURN labels(n)[0]         AS FromType,
               n.id                 AS From,
               type(r)              AS Relationship,
               labels(connected)[0] AS ToType,
               connected.id         AS To,
               connected.name       AS ToName,
               connected.status     AS ToStatus,
               connected.amount     AS ToAmount
        ORDER BY type(r)
        LIMIT 100
    """)
    display(f"Direct Connections (depth 1)", direct_rows)

    # ── Q3: Depth 2 — connections of connections ──────────────────────
    if depth >= 2:
        depth2_rows = run(f"""
            MATCH (n:{entity_type})-[r1]-(hop1)-[r2]-(hop2)
            WHERE (toLower(n.id)   CONTAINS toLower('{entity_id}')
               OR toLower(n.name)  CONTAINS toLower('{entity_id}'))
              AND hop2 <> n
            RETURN labels(hop1)[0] AS Via,
                   hop1.id         AS ViaID,
                   type(r2)        AS Relationship,
                   labels(hop2)[0] AS ToType,
                   hop2.id         AS To
            ORDER BY Via
            LIMIT 100
        """)
        display(f"Depth 2 Connections (via intermediaries)", depth2_rows)

    # ── Q4: Relationship type summary ────────────────────────────────
    summary_rows = run(f"""
        MATCH (n:{entity_type})-[r]-(connected)
        WHERE toLower(n.id)   CONTAINS toLower('{entity_id}')
           OR toLower(n.name) CONTAINS toLower('{entity_id}')
        RETURN type(r)              AS RelationshipType,
               labels(connected)[0] AS ConnectedNodeType,
               count(connected)     AS Count
        ORDER BY Count DESC
    """)
    display(f"Connection Summary", summary_rows)

    return {
        "entity":      entity_rows,
        "connections": direct_rows,
        "summary":     summary_rows
    }

# Visualize by ID filter
def visualize_by_id(client_a, entity_type, entity_id,
                    depth=1, output_file=None):
    """
    Visualizes the subgraph around a specific entity using PyVis.
    Only shows nodes and relationships connected to the given entity.

    Args:
        client_a    : boto3 neptunedata client from connect_option_a()
        entity_type : node label e.g. 'Patient', 'Claim', 'Doctor'
        entity_id   : the id value e.g. 'Ravi Sharma', 'CLM001'
        depth       : 1 = direct connections, 2 = two hops
        output_file : HTML filename (auto-generated if None)

    Usage:
        visualize_by_id(client_a, 'Patient', 'Ravi Sharma')
        visualize_by_id(client_a, 'Claim',   'CLM001', depth=2)
        visualize_by_id(client_a, 'Doctor',  'Dr. Kapoor')
    """
    from pyvis.network import Network
    import re

    if output_file is None:
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', entity_id)
        output_file = f"graph_{entity_type}_{safe_id}.html"

    print(f"\n[VISUALIZE] {entity_type} '{entity_id}' "
          f"(depth={depth}) → {output_file}")

    # ── Fetch the anchor node ─────────────────────────────────────────
    anchor_result = client_a.execute_open_cypher_query(
        openCypherQuery=f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.id)   CONTAINS toLower('{entity_id}')
               OR toLower(n.name) CONTAINS toLower('{entity_id}')
            RETURN id(n) AS node_id, labels(n)[0] AS label, n AS node_obj
            LIMIT 1
        """
    )
    anchor_rows = anchor_result.get("results", [])
    if not anchor_rows:
        print(f"  ❌ No {entity_type} found with id '{entity_id}'")
        return

    # ── Fetch depth 1: anchor + direct connections ────────────────────
    depth1_nodes = client_a.execute_open_cypher_query(
        openCypherQuery=f"""
            MATCH (n:{entity_type})-[r]-(connected)
            WHERE toLower(n.id)   CONTAINS toLower('{entity_id}')
               OR toLower(n.name) CONTAINS toLower('{entity_id}')
            RETURN id(n)         AS from_id,
                   labels(n)[0]  AS from_label,
                   n             AS from_obj,
                   id(connected) AS to_id,
                   labels(connected)[0] AS to_label,
                   connected     AS to_obj,
                   type(r)       AS rel_type,
                   id(r)         AS rel_id
            LIMIT 200
        """
    ).get("results", [])

    # ── Fetch depth 2 if requested ────────────────────────────────────
    depth2_nodes = []
    if depth >= 2:
        depth2_nodes = client_a.execute_open_cypher_query(
            openCypherQuery=f"""
                MATCH (n:{entity_type})-[r1]-(hop1)-[r2]-(hop2)
                WHERE (toLower(n.id)   CONTAINS toLower('{entity_id}')
                   OR toLower(n.name)  CONTAINS toLower('{entity_id}'))
                  AND hop2 <> n
                RETURN id(hop1)         AS from_id,
                       labels(hop1)[0]  AS from_label,
                       hop1             AS from_obj,
                       id(hop2)         AS to_id,
                       labels(hop2)[0]  AS to_label,
                       hop2             AS to_obj,
                       type(r2)         AS rel_type,
                       id(r2)           AS rel_id
                LIMIT 200
            """
        ).get("results", [])

    all_rows = depth1_nodes + depth2_nodes

    if not all_rows:
        print(f"  ⚠️  Entity found but has no connections.")
        return

    # ── Colour palette — auto-assigned per label ──────────────────────
    PALETTE = [
        "#4A90D9", "#27AE60", "#E74C3C", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#2ECC71", "#3498DB", "#E91E63",
    ]
    all_labels   = sorted({
        row.get("from_label", "Unknown") for row in all_rows
    } | {
        row.get("to_label", "Unknown") for row in all_rows
    })
    label_colour = {
        lbl: PALETTE[i % len(PALETTE)]
        for i, lbl in enumerate(all_labels)
    }

    # ── Helper: extract display name from node object ─────────────────
    def get_display_name(node_obj):
        if not isinstance(node_obj, dict):
            return str(node_obj)
        props = {k: v for k, v in node_obj.items()
                 if not k.startswith("~") and v is not None}
        return (props.get("name") or
                props.get("id")   or
                props.get("title") or
                next((str(v) for v in props.values()
                      if isinstance(v, str)), None) or
                "unknown")

    def get_tooltip(label, node_obj, node_id):
        if not isinstance(node_obj, dict):
            return f"Type: {label}"
        props = {k: v for k, v in node_obj.items()
                 if not k.startswith("~") and v is not None}
        lines = [f"Type    : {label}", f"Int. ID : {node_id}", "─────────"]
        lines += [f"{k}: {v}" for k, v in props.items()]
        return "\n".join(lines)

    # ── Build PyVis network ───────────────────────────────────────────
    net = Network(
        height     = "820px",
        width      = "100%",
        bgcolor    = "#1a1a2e",
        font_color = "white",
        notebook   = False
    )
    net.barnes_hut(
        gravity         = -12000,
        central_gravity = 0.1,
        spring_length   = 120,
        spring_strength = 0.04,
        damping         = 0.95,
        overlap         = 1
    )

    added_node_ids = set()

    def add_node(node_id, label, node_obj, is_anchor=False):
        nid = str(node_id)
        if nid in added_node_ids:
            return
        colour = label_colour.get(label, "#95A5A6")
        name   = get_display_name(node_obj)
        tip    = get_tooltip(label, node_obj, nid)

        net.add_node(
            nid,
            label       = name,
            title       = tip,
            color       = {
                "background": "#FFD700" if is_anchor else colour,
                "border":     "#FFFFFF",
                "highlight":  {
                    "background": "#FFFFFF",
                    "border":     "#FFD700" if is_anchor else colour
                }
            },
            size        = 40 if is_anchor else 25,
            font        = {
                "size":        16 if is_anchor else 14,
                "color":       "white",
                "strokeWidth": 3,
                "strokeColor": "#000000",
                "bold":        is_anchor
            },
            borderWidth         = 4 if is_anchor else 2,
            borderWidthSelected = 6,
            shadow      = True
        )
        added_node_ids.add(nid)

    # Add anchor node (highlighted in gold)
    anchor = anchor_rows[0]
    add_node(
        anchor["node_id"],
        anchor["label"],
        anchor.get("node_obj", {}),
        is_anchor=True
    )

    # Add all connected nodes and edges
    added_edge_ids = set()
    for row in all_rows:
        from_id  = str(row.get("from_id",  ""))
        to_id    = str(row.get("to_id",    ""))
        rel_type = str(row.get("rel_type", ""))
        rel_id   = str(row.get("rel_id",   ""))

        add_node(from_id, row.get("from_label","?"), row.get("from_obj",{}))
        add_node(to_id,   row.get("to_label","?"),   row.get("to_obj",{}))

        edge_key = f"{from_id}_{rel_type}_{to_id}"
        if edge_key not in added_edge_ids:
            net.add_edge(
                from_id, to_id,
                label  = rel_type,
                title  = rel_type,
                color  = {"color": "#aaaaaa", "highlight": "#ffffff"},
                arrows = "to",
                width  = 2,
                font   = {
                    "size":        12,
                    "color":       "#eeeeee",
                    "strokeWidth": 2,
                    "strokeColor": "#000000",
                    "align":       "middle"
                }
            )
            added_edge_ids.add(edge_key)

    # ── Legend ────────────────────────────────────────────────────────
    net.add_node(
        "__legend_anchor",
        label   = f"★ {entity_type} (anchor)",
        color   = {"background": "#FFD700", "border": "#FFFFFF"},
        size    = 15,
        x       = -700, y = -400,
        physics = False,
        fixed   = {"x": True, "y": True},
        font    = {"size": 13, "color": "white",
                   "strokeWidth": 2, "strokeColor": "#000000"},
        shape   = "dot"
    )
    for i, (lbl, colour) in enumerate(label_colour.items()):
        net.add_node(
            f"__legend_{lbl}",
            label   = lbl,
            color   = {"background": colour, "border": "#ffffff"},
            size    = 15,
            x       = -700, y = -340 + (i * 50),
            physics = False,
            fixed   = {"x": True, "y": True},
            font    = {"size": 13, "color": "white",
                       "strokeWidth": 2, "strokeColor": "#000000"},
            shape   = "dot"
        )

    # ── Options and freeze ────────────────────────────────────────────
    net.set_options("""
    {
      "nodes": { "shape": "dot", "shadow": true },
      "edges": { "shadow": true, "selectionWidth": 3 },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "tooltipDelay": 80
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.1,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.95,
          "avoidOverlap": 1
        },
        "stabilization": { "enabled": true, "iterations": 300, "fit": true }
      }
    }
    """)

    net.save_graph(output_file)

    # Inject freeze script
    with open(output_file, "r") as f:
        html = f.read()
    freeze = """<script>
    window.addEventListener("load", function() {
        network.on("stabilizationIterationsDone", function() {
            network.setOptions({ physics: { enabled: false } });
        });
    });
    </script>"""
    html = html.replace("</body>", freeze + "</body>")
    with open(output_file, "w") as f:
        f.write(html)

    print(f"  ✅ Saved → {output_file}")
    print(f"  Nodes : {len(added_node_ids)}  |  Edges : {len(added_edge_ids)}")
    webbrowser.open(f"file://{os.path.abspath(output_file)}")


# Verify whether the written data is present in the graph db along with properties
def verify(client):
    """
    Verifies Neptune graph contents including:
      - Node counts by type
      - Relationship counts by type
      - Node properties actually stored (samples one node per type)
      - Flags any node type that has no properties written
    """
    import json

    print("\n" + "═"*55)
    print("  [VERIFY] Neptune Graph Contents")
    print("═"*55)

    # ── Node counts ───────────────────────────────────────────────────
    node_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (n)
            RETURN labels(n)[0] AS Type, count(n) AS Count
            ORDER BY Count DESC
        """
    )
    print("\n  NODE COUNTS:")
    print(f"  {'Type':<25} {'Count':>8}")
    print(f"  {'─'*35}")
    node_count = 0
    for row in node_result.get("results", []):
        print(f"  {row.get('Type','?'):<25} {row.get('Count',0):>8}")
        node_count += row.get("Count", 0)
    print(f"  {'─'*35}")
    print(f"  {'TOTAL':<25} {node_count:>8}")

    # ── Relationship counts ───────────────────────────────────────────
    rel_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH ()-[r]->()
            RETURN type(r) AS Relationship, count(r) AS Count
            ORDER BY Count DESC
        """
    )
    print("\n  RELATIONSHIP COUNTS:")
    print(f"  {'Type':<30} {'Count':>8}")
    print(f"  {'─'*40}")
    rel_count = 0
    for row in rel_result.get("results", []):
        print(f"  {row.get('Relationship','?'):<30} {row.get('Count',0):>8}")
        rel_count += row.get("Count", 0)
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<30} {rel_count:>8}")

    # # ── Property verification — sample one node per type ───────────── old code not verifying properties correctly
    # print("\n  NODE PROPERTIES (sample per type):")
    # print(f"  {'─'*55}")

    # label_result = client.execute_open_cypher_query(
    #     openCypherQuery="""
    #         MATCH (n)
    #         RETURN DISTINCT labels(n)[0] AS label
    #         ORDER BY label
    #     """
    # )
    # labels = [
    #     r["label"]
    #     for r in label_result.get("results", [])
    #     if r.get("label")
    # ]

    # nodes_with_no_props = []

    # for label in labels:
    #     # Fetch one sample node with all its properties
    #     sample_result = client.execute_open_cypher_query(
    #         openCypherQuery=f"""
    #             MATCH (n:{label})
    #             RETURN n.id AS id, keys(n) AS propKeys, n AS node
    #             LIMIT 1
    #         """
    #     )
    #     rows = sample_result.get("results", [])
    #     if not rows:
    #         print(f"\n  [{label}] — no nodes found")
    #         continue

    #     row      = rows[0]
    #     node_id  = row.get("id", "unknown")
    #     prop_keys = row.get("propKeys", [])
    #     node_obj  = row.get("node", {})

    #     # Remove internal 'id' key from display — always present
    #     display_keys = [k for k in prop_keys if k != "id"]

    #     print(f"\n  [{label}] sample node: '{node_id}'")

    #     if not display_keys:
    #         print(f"    ⚠️  NO PROPERTIES WRITTEN — only 'id' present")
    #         nodes_with_no_props.append(label)
    #     else:
    #         for k in display_keys:
    #             # Read actual value from node object
    #             val = ""
    #             if isinstance(node_obj, dict):
    #                 val = node_obj.get(k, "")
    #             print(f"    ✅  {k:<20} : {val}")

    # # ── Relationship property verification ────────────────────────────
    # print(f"\n  RELATIONSHIP PROPERTIES (sample per type):")
    # print(f"  {'─'*55}")

    # rel_type_result = client.execute_open_cypher_query(
    #     openCypherQuery="""
    #         MATCH ()-[r]->()
    #         RETURN DISTINCT type(r) AS relType
    #         ORDER BY relType
    #     """
    # )
    # rel_types = [
    #     r["relType"]
    #     for r in rel_type_result.get("results", [])
    #     if r.get("relType")
    # ]

    # for rel_type in rel_types:
    #     sample = client.execute_open_cypher_query(
    #         openCypherQuery=f"""
    #             MATCH (a)-[r:{rel_type}]->(b)
    #             RETURN a.id AS from, b.id AS to,
    #                    keys(r) AS propKeys, r AS rel
    #             LIMIT 1
    #         """
    #     )
    #     rows = sample.get("results", [])
    #     if not rows:
    #         continue

    #     row       = rows[0]
    #     from_id   = row.get("from", "?")
    #     to_id     = row.get("to",   "?")
    #     prop_keys = row.get("propKeys", [])
    #     rel_obj   = row.get("rel", {})

    #     print(f"\n  [{rel_type}] sample: ({from_id})→({to_id})")
    #     if not prop_keys:
    #         print(f"    (no properties)")
    #     else:
    #         for k in prop_keys:
    #             val = rel_obj.get(k, "") if isinstance(rel_obj, dict) else ""
    #             print(f"    ✅  {k:<20} : {val}")

    # ── Property verification — sample one node per type ─────────────
    print("\n  NODE PROPERTIES (sample per type):")
    print(f"  {'─'*55}")

    label_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (n)
            RETURN DISTINCT labels(n)[0] AS label
            ORDER BY label
        """
    )
    labels = [
        r["label"]
        for r in label_result.get("results", [])
        if r.get("label")
    ]

    nodes_with_no_props = []

    for label in labels:
        # ── KEY FIX: use properties(n) which returns flat dict ────────
        sample_result = client.execute_open_cypher_query(
            openCypherQuery=f"""
                MATCH (n:{label})
                RETURN n.id          AS id,
                       properties(n) AS props
                LIMIT 1
            """
        )
        rows = sample_result.get("results", [])
        if not rows:
            print(f"\n  [{label}] — no nodes found")
            continue

        row     = rows[0]
        node_id = row.get("id", "unknown")
        props   = row.get("props", {})    # ← flat dict, no nesting

        # Filter out internal id for display
        display_props = {k: v for k, v in props.items() if k != "id"}

        print(f"\n  [{label}] sample node: '{node_id}'")

        if not display_props:
            print(f"    ⚠️  NO PROPERTIES WRITTEN — only 'id' present")
            nodes_with_no_props.append(label)
        else:
            for k, v in display_props.items():
                print(f"    ✅  {k:<20} : {v}")

    # ── Relationship property verification ────────────────────────────
    print(f"\n  RELATIONSHIP PROPERTIES (sample per type):")
    print(f"  {'─'*55}")

    rel_type_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH ()-[r]->()
            RETURN DISTINCT type(r) AS relType
            ORDER BY relType
        """
    )
    rel_types = [
        r["relType"]
        for r in rel_type_result.get("results", [])
        if r.get("relType")
    ]

    for rel_type in rel_types:
        # ── Same fix: properties(r) instead of r ─────────────────────
        sample = client.execute_open_cypher_query(
            openCypherQuery=f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN a.id          AS fromId,
                       b.id          AS toId,
                       properties(r) AS props
                LIMIT 1
            """
        )
        rows = sample.get("results", [])
        if not rows:
            continue

        row     = rows[0]
        from_id = row.get("fromId", "?")
        to_id   = row.get("toId",   "?")
        props   = row.get("props",  {})    # ← flat dict

        print(f"\n  [{rel_type}] sample: ({from_id})→({to_id})")
        if not props:
            print(f"    (no properties)")
        else:
            for k, v in props.items():
                print(f"    ✅  {k:<20} : {v}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n  {'═'*55}")
    print(f"  SUMMARY")
    print(f"  {'═'*55}")
    print(f"  Total nodes         : {node_count}")
    print(f"  Total relationships : {rel_count}")

    if nodes_with_no_props:
        print(f"\n  ⚠️  These node types have NO properties written:")
        for label in nodes_with_no_props:
            print(f"    - {label}")
        print(f"\n  Possible causes:")
        print(f"    1. LLMGraphTransformer extracted no properties for these types")
        print(f"    2. write_graph_documents_option_a failed silently on SET clause")
        print(f"    3. node_properties=True not set on the transformer")
    else:
        print(f"\n  ✅ All node types have properties written correctly")


# query the graph db and dispaly nodes and relationships along with properties
def query_neptune(client):
    """
    Queries Neptune and prints:
      - Node type summary (counts)
      - Relationship type summary (counts)
      - All nodes with ALL their properties
      - All connections between nodes
      - All connections with relationship properties
    """
    print("\n[QUERY NEPTUNE] Querying graph contents via boto3...\n")

    # ── Internal helpers ──────────────────────────────────────────────

    def run(cypher):
        result = client.execute_open_cypher_query(openCypherQuery=cypher)
        return result.get("results", [])

    def display(title, rows):
        """Prints any result set as a formatted table."""
        print(f"  {'─'*60}")
        print(f"  {title}")
        print(f"  {'─'*60}")
        if not rows:
            print("  (no results)")
            return
        headers = list(rows[0].keys())
        col_w   = 22
        print("  " + " | ".join(h[:col_w].ljust(col_w) for h in headers))
        print("  " + "─" * ((col_w + 3) * len(headers)))
        for row in rows:
            print("  " + " | ".join(
                str(row.get(h, ""))[:col_w].ljust(col_w) for h in headers
            ))
        print(f"  {len(rows)} row(s)\n")

    def display_nodes_with_properties(label, rows):
        """
        Prints each node as a block showing all its properties.
        Used for Q3 — more readable than a wide flat table
        when nodes have many properties.
        """
        print(f"  {'─'*60}")
        print(f"  All [{label}] nodes with properties")
        print(f"  {'─'*60}")
        if not rows:
            print("  (no nodes found)")
            return
        for row in rows:
            node_id   = row.get("id", "unknown")
            node_obj  = row.get("node", {})
            prop_keys = row.get("propKeys", [])

            print(f"\n  ● id: {node_id}")
            display_keys = [k for k in prop_keys if k != "id"]
            if not display_keys:
                print(f"    (no additional properties)")
            else:
                for k in display_keys:
                    val = node_obj.get(k, "") if isinstance(node_obj, dict) else ""
                    print(f"    {k:<22} : {val}")
        print(f"\n  {len(rows)} node(s)\n")

    # ── Q1: Node type summary ─────────────────────────────────────────
    display(
        "Q1 — Node Types and Counts",
        run("""
            MATCH (n)
            RETURN labels(n)[0] AS NodeType, count(n) AS Count
            ORDER BY Count DESC
        """)
    )

    # ── Q2: Relationship type summary ─────────────────────────────────
    display(
        "Q2 — Relationship Types and Counts",
        run("""
            MATCH ()-[r]->()
            RETURN type(r) AS Relationship, count(r) AS Count
            ORDER BY Count DESC
        """)
    )

    # ── Q3: All nodes with ALL properties — grouped by label ──────────
    label_rows = run("""
        MATCH (n)
        RETURN DISTINCT labels(n)[0] AS label
        ORDER BY label
    """)
    labels = [r["label"] for r in label_rows if r.get("label")]

    print(f"  {'─'*60}")
    print(f"  Q3 — All Nodes with All Properties (grouped by type)")
    print(f"  {'─'*60}")

    for label in labels:
        # ── KEY FIX: properties(n) returns flat dict ──────────────────
        rows = run(f"""
            MATCH (n:{label})
            RETURN n.id          AS id,
                   properties(n) AS props
            ORDER BY n.id
        """)
        print(f"\n  [{label}] — {len(rows)} node(s)")
        for row in rows:
            node_id = row.get("id", "unknown")
            props   = row.get("props", {})
            display_props = {k: v for k, v in props.items() if k != "id"}

            print(f"\n  ● id: {node_id}")
            if not display_props:
                print(f"    (no additional properties)")
            else:
                for k, v in display_props.items():
                    print(f"    {k:<22} : {v}")

    # ── Q4: All connections ───────────────────────────────────────────
    display(
        "Q4 — All Connections (who is connected to whom)",
        run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS FromType,
                   a.id         AS From,
                   type(r)      AS Via,
                   labels(b)[0] AS ToType,
                   b.id         AS To
            ORDER BY FromType, From
            LIMIT 50
        """)
    )

    # ── Q5: All connections with relationship properties ──────────────
    rel_with_props = run("""
        MATCH (a)-[r]->(b)
        WHERE size(keys(r)) > 0
        RETURN a.id          AS fromId,
               type(r)       AS relType,
               b.id          AS toId,
               properties(r) AS props
        LIMIT 50
    """)

    print(f"  {'─'*60}")
    print(f"  Q5 — Relationships with Properties")
    print(f"  {'─'*60}")
    if not rel_with_props:
        print(f"  (no relationship properties found)\n")
    else:
        for row in rel_with_props:
            from_id  = row.get("fromId",  "?")
            rel_type = row.get("relType", "?")
            to_id    = row.get("toId",    "?")
            props    = row.get("props",   {})

            print(f"\n  ● ({from_id})-[{rel_type}]→({to_id})")
            for k, v in props.items():
                print(f"    {k:<22} : {v}")
        print(f"\n  {len(rel_with_props)} relationship(s) with properties\n")