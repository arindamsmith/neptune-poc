"""
╔══════════════════════════════════════════════════════════════════╗
║         AWS NEPTUNE — COMPLETE POC (Medical & Claims)           ║
║                                                                  ║
║  Sections:                                                       ║
║   1. CONNECTION   — Option A (boto3) | Option B (LangChain)     ║
║   2. WRITE DATA   — Option A         | Option B                 ║
║   3. QUERY DATA   — Option A         | Option B                 ║
║   4. VISUALIZE    — PyVis (HTML)                                ║
║   5. NLP QUERY    — create_neptune_opencypher_qa_chain          ║
║                                                                  ║
║  HOW TO USE:                                                     ║
║   - Fill in your endpoint in .env                               ║
║   - Run: python main.py                                         ║
║   - Comment/uncomment sections at the bottom to choose          ║
║     which options to run                                         ║
╚══════════════════════════════════════════════════════════════════╝

Official Docs:
  boto3 neptunedata  : https://boto3.amazonaws.com/v1/documentation/api/latest/
                       reference/services/neptunedata/client/execute_open_cypher_query.html
  LangChain Neptune  : https://python.langchain.com/api_reference/aws/graphs/
                       langchain_aws.graphs.neptune_graph.NeptuneGraph.html
  NLP QA Chain       : https://python.langchain.com/docs/integrations/graphs/
                       amazon_neptune_open_cypher/
"""

import os
import json
import webbrowser
import config

# ── Suppress SSL warnings for PoC ────────────────────────────────────
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ═══════════════════════════════════════════════════════════════════════
#  SAMPLE DATA
#  Texts that represent medical/claims documents.
#  LLMGraphTransformer will convert these into nodes + relationships.
# ═══════════════════════════════════════════════════════════════════════
SAMPLE_TEXTS = [
    "Patient Ravi Sharma, age 45, was admitted to Apollo Hospitals in Mumbai on January 15, 2024.",
    "Ravi Sharma was treated by Dr. Anil Kapoor, a Cardiologist at Apollo Hospitals.",
    "Ravi Sharma was diagnosed with Acute Myocardial Infarction (ICD-I21), a critical condition.",
    "Dr. Anil Kapoor performed a Coronary Angioplasty procedure on Ravi Sharma, costing Rs. 250000.",
    "Ravi Sharma filed insurance claim CLM001 for Rs. 240000 with Star Health Insurance.",
    "Claim CLM001 was approved by Star Health Insurance on January 22, 2024.",
    "Patient Anjali Mehta, age 32, visited AIIMS Delhi and was treated by Dr. Sunita Rao.",
    "Dr. Sunita Rao is an Orthopedics specialist working at AIIMS Delhi.",
    "Anjali Mehta was diagnosed with Osteoarthritis of Hip (ICD-M16).",
    "Anjali Mehta filed claim CLM002 for Rs. 340000 with HDFC ERGO Health Insurance.",
    "Patient Suresh Nair, age 58, was admitted to Fortis Healthcare in Bangalore.",
    "Suresh Nair was treated by Dr. Ramesh Gupta, a Neurologist at Fortis Healthcare.",
    "Suresh Nair was diagnosed with Multiple Sclerosis (ICD-G35).",
    "Suresh Nair filed claim CLM003 for Rs. 410000 which is currently under review.",
]


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1A — CONNECTION via boto3 (Option A)
#
#  Uses AWS SDK directly. Recommended for production.
#  No LLM involved — just a raw database client.
#  Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/
#       services/neptunedata/client/execute_open_cypher_query.html
# ═══════════════════════════════════════════════════════════════════════
def connect_option_a():
    """
    Creates and returns a boto3 neptunedata client.
    This is the AWS-native way to connect to Neptune.
    IAM SigV4 auth is handled automatically by boto3 using
    credentials from .env / ~/.aws/credentials / IAM Role.
    """
    import boto3
    from botocore.config import Config

    print("\n[OPTION A] Connecting via boto3 neptunedata client...")
    client = boto3.client(
        "neptunedata",
        endpoint_url=config.NEPTUNE_URL,
        region_name=config.AWS_REGION,
        config=Config(read_timeout=None, retries={"total_max_attempts": 1})
    )

    # Test connection
    result = client.execute_open_cypher_query(
        openCypherQuery="MATCH (n) RETURN count(n) AS total"
    )
    total = result["results"][0]["total"]
    print(f"  ✅ Option A connected. Nodes in graph: {total}")
    return client


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1B — CONNECTION via LangChain NeptuneGraph (Option B)
#
#  Uses LangChain wrapper. Automatically reads graph schema on connect.
#  Required for NLP querying (Section 5).
#  Ref: https://python.langchain.com/api_reference/aws/graphs/
#       langchain_aws.graphs.neptune_graph.NeptuneGraph.html
# ═══════════════════════════════════════════════════════════════════════
def connect_option_b():
    """
    Creates and returns a LangChain NeptuneGraph object.
    On init it auto-discovers the graph schema (node labels +
    relationship types) — this schema is later used by the LLM
    to write openCypher queries in the NLP section.
    """
    from langchain_aws.graphs import NeptuneGraph

    print("\n[OPTION B] Connecting via LangChain NeptuneGraph...")
    graph = NeptuneGraph(
        host      = config.NEPTUNE_ENDPOINT,
        port      = config.NEPTUNE_PORT,
        use_https = True
    )
    print("  ✅ Option B connected.")
    print(f"  📐 Schema discovered:\n{graph.get_schema}")
    return graph


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2A — WRITE GRAPH DOCUMENTS via boto3 (Option A)
#
#  LLMGraphTransformer converts plain text → GraphDocument objects
#  (nodes + relationships). We then loop over them and write each
#  node/relationship into Neptune using MERGE (upsert — no duplicates).
# ═══════════════════════════════════════════════════════════════════════
def write_graph_documents_option_a(client, texts: list):
    """
    Converts text to graph using LLMGraphTransformer (Bedrock Claude),
    then writes all nodes and relationships to Neptune via boto3.

    Uses MERGE (not CREATE) so re-running is safe — no duplicates.

    Args:
        client : boto3 neptunedata client from connect_option_a()
        texts  : list of text strings to convert and store
    """
    from langchain_core.documents import Document
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_aws import ChatBedrock

    print("\n[OPTION A] Converting texts to graph and writing to Neptune...")

    # Step 1 — Convert texts to GraphDocuments using LLM
    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )
    transformer    = LLMGraphTransformer(llm=llm)
    docs           = [Document(page_content=t) for t in texts]
    graph_documents = transformer.convert_to_graph_documents(docs)

    print(f"  LLM extracted {sum(len(d.nodes) for d in graph_documents)} nodes "
          f"and {sum(len(d.relationships) for d in graph_documents)} relationships")

    total_nodes = 0
    total_rels  = 0
    failed_nodes = []
    failed_rels  = []

    for doc in graph_documents:

        # Write nodes
        for node in doc.nodes:
            props     = node.properties or {}
            props_str = ", ".join(f"n.{k} = '{v}'" for k, v in props.items())
            set_clause = f"SET {props_str}" if props_str else ""
            cypher = f"MERGE (n:{node.type} {{id: '{node.id}'}}) {set_clause}"
            try:
                client.execute_open_cypher_query(openCypherQuery=cypher)
                total_nodes += 1
                print(f"  ✅ Node  : ({node.type} | {node.id})")
            except Exception as e:
                failed_nodes.append(node.id)
                print(f"  ❌ Node FAILED : ({node.type} | {node.id}) → {e}")

        # Write relationships
        for rel in doc.relationships:
            cypher = f"""
                MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}})
                MATCH (b:{rel.target.type} {{id: '{rel.target.id}'}})
                MERGE (a)-[r:{rel.type}]->(b)
            """
            try:
                client.execute_open_cypher_query(openCypherQuery=cypher)
                total_rels += 1
                print(f"  ✅ Rel   : ({rel.source.id})-[{rel.type}]->({rel.target.id})")
            except Exception as e:
                failed_rels.append(f"{rel.source.id}->{rel.target.id}")
                print(f"  ❌ Rel FAILED  : ({rel.source.id})-[{rel.type}]->({rel.target.id}) → {e}")

    print(f"\n  WRITE SUMMARY (Option A):")
    print(f"    Nodes written     : {total_nodes}  |  Failed: {len(failed_nodes)}")
    print(f"    Relations written : {total_rels}  |  Failed: {len(failed_rels)}")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2B — WRITE GRAPH DOCUMENTS via LangChain NeptuneGraph (Option B)
#
#  Same LLMGraphTransformer step, but uses graph.query() to write
#  instead of boto3. Simpler code — NeptuneGraph wraps the HTTP call.
# ═══════════════════════════════════════════════════════════════════════
def write_graph_documents_option_b(graph, texts: list):
    """
    Converts text to graph using LLMGraphTransformer (Bedrock Claude),
    then writes all nodes and relationships to Neptune via LangChain
    NeptuneGraph.query().

    Args:
        graph  : NeptuneGraph object from connect_option_b()
        texts  : list of text strings to convert and store
    """
    from langchain_core.documents import Document
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_aws import ChatBedrock

    print("\n[OPTION B] Converting texts to graph and writing to Neptune...")

    # Step 1 — Convert texts to GraphDocuments using LLM
    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )
    transformer     = LLMGraphTransformer(llm=llm)
    docs            = [Document(page_content=t) for t in texts]
    graph_documents = transformer.convert_to_graph_documents(docs)

    print(f"  LLM extracted {sum(len(d.nodes) for d in graph_documents)} nodes "
          f"and {sum(len(d.relationships) for d in graph_documents)} relationships")

    total_nodes = 0
    total_rels  = 0
    failed_nodes = []
    failed_rels  = []

    for doc in graph_documents:

        # Write nodes
        for node in doc.nodes:
            cypher = f"MERGE (n:{node.type} {{id: '{node.id}'}})"
            try:
                graph.query(cypher)
                total_nodes += 1
                print(f"  ✅ Node  : ({node.type} | {node.id})")
            except Exception as e:
                failed_nodes.append(node.id)
                print(f"  ❌ Node FAILED : ({node.type} | {node.id}) → {e}")

        # Write relationships
        for rel in doc.relationships:
            cypher = f"""
                MATCH (a:{rel.source.type} {{id: '{rel.source.id}'}})
                MATCH (b:{rel.target.type} {{id: '{rel.target.id}'}})
                MERGE (a)-[r:{rel.type}]->(b)
            """
            try:
                graph.query(cypher)
                total_rels += 1
                print(f"  ✅ Rel   : ({rel.source.id})-[{rel.type}]->({rel.target.id})")
            except Exception as e:
                failed_rels.append(f"{rel.source.id}->{rel.target.id}")
                print(f"  ❌ Rel FAILED  : ({rel.source.id})-[{rel.type}]->({rel.target.id}) → {e}")

    print(f"\n  WRITE SUMMARY (Option B):")
    print(f"    Nodes written     : {total_nodes}  |  Failed: {len(failed_nodes)}")
    print(f"    Relations written : {total_rels}  |  Failed: {len(failed_rels)}")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3A — QUERY DATA via boto3 (Option A)
#
#  Runs openCypher queries directly via boto3.
#  Returns structured data — ideal for application logic.
# ═══════════════════════════════════════════════════════════════════════
def query_option_a(client):
    """
    Runs a set of openCypher queries using boto3 and prints results.
    Each query demonstrates a different retrieval pattern.
    """
    print("\n[OPTION A] Querying Neptune via boto3...\n")

    def run(cypher):
        result = client.execute_open_cypher_query(openCypherQuery=cypher)
        return result.get("results", [])

    def display(title, rows):
        print(f"  {'─'*55}")
        print(f"  {title}")
        print(f"  {'─'*55}")
        if not rows:
            print("  (no results)")
            return
        headers = list(rows[0].keys())
        print("  " + " | ".join(h[:20].ljust(20) for h in headers))
        print("  " + "─" * (23 * len(headers)))
        for row in rows:
            print("  " + " | ".join(str(row.get(h,""))[:20].ljust(20) for h in headers))
        print(f"  {len(rows)} row(s)\n")

    # Q1 — Graph summary: what is in the db?
    display("Graph Summary — All Node Types",
        run("MATCH (n) RETURN labels(n)[0] AS NodeType, count(n) AS Count ORDER BY Count DESC"))

    # Q2 — All relationships
    display("All Relationship Types",
        run("MATCH ()-[r]->() RETURN type(r) AS Relationship, count(r) AS Count ORDER BY Count DESC"))

    # Q3 — All nodes with their IDs
    display("All Nodes (sample)",
        run("MATCH (n) RETURN labels(n)[0] AS Type, n.id AS ID LIMIT 20"))

    # Q4 — All connections: who is connected to whom?
    display("All Connections in Graph",
        run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS From, a.id AS FromID,
                   type(r)      AS Via,
                   labels(b)[0] AS To,   b.id AS ToID
            LIMIT 30
        """))

    # Q5 — Neighbours of a specific node (change ID as needed)
    display("Neighbours of 'Ravi Sharma'",
        run("""
            MATCH (n {id: 'Ravi Sharma'})-[r]->(neighbour)
            RETURN type(r) AS Relationship, labels(neighbour)[0] AS Type, neighbour.id AS Neighbour
        """))


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3B — QUERY DATA via LangChain NeptuneGraph (Option B)
#
#  Same openCypher queries, sent via graph.query().
#  Also shows the schema that NeptuneGraph auto-discovered.
# ═══════════════════════════════════════════════════════════════════════
def query_option_b(graph):
    """
    Runs openCypher queries via LangChain NeptuneGraph.query().
    Also prints the auto-discovered graph schema — which is what
    the LLM reads before writing queries in the NLP section.
    """
    print("\n[OPTION B] Querying Neptune via LangChain NeptuneGraph...\n")

    def run(cypher):
        result = graph.query(cypher)
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        if isinstance(result, list):
            return result
        return []

    def display(title, rows):
        print(f"  {'─'*55}")
        print(f"  {title}")
        print(f"  {'─'*55}")
        if not rows:
            print("  (no results)")
            return
        headers = list(rows[0].keys())
        print("  " + " | ".join(h[:20].ljust(20) for h in headers))
        print("  " + "─" * (23 * len(headers)))
        for row in rows:
            print("  " + " | ".join(str(row.get(h,""))[:20].ljust(20) for h in headers))
        print(f"  {len(rows)} row(s)\n")

    # Auto-discovered schema
    print(f"  {'─'*55}")
    print(f"  Graph Schema (auto-discovered by LangChain NeptuneGraph)")
    print(f"  {'─'*55}")
    print(f"  {graph.get_schema}\n")

    # Q1 — Graph summary
    display("Graph Summary — All Node Types",
        run("MATCH (n) RETURN labels(n)[0] AS NodeType, count(n) AS Count ORDER BY Count DESC"))

    # Q2 — All connections
    display("All Connections in Graph",
        run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS From, a.id AS FromID,
                   type(r)      AS Via,
                   labels(b)[0] AS To,   b.id AS ToID
            LIMIT 30
        """))

    # Q3 — Most connected nodes
    display("Most Connected Nodes",
        run("""
            MATCH (n)-[r]-()
            RETURN labels(n)[0] AS Type, n.id AS Node, count(r) AS Connections
            ORDER BY Connections DESC LIMIT 10
        """))


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 — VISUALIZE using PyVis
#
#  Pulls all nodes and edges from Neptune via boto3, builds an
#  interactive graph, and saves it as an HTML file that opens
#  automatically in your browser.
#
#  pip install pyvis
# ═══════════════════════════════════════════════════════════════════════

# Old working code
# def visualize_graph(client, output_file="neptune_graph.html"):
#     """
#     Generic interactive graph visualizer.
#     Works for ANY nodes and edges in Neptune — not hardcoded to any domain.
#     Auto-assigns colours dynamically based on whatever labels exist.
#     Saves as HTML and opens in browser.
#     """
#     from pyvis.network import Network
#     import hashlib

#     print(f"\n[VISUALIZE] Building interactive graph → {output_file}...")

#     # ── Step 1: Pull nodes with explicit property return ──────────────
#     # KEY FIX: Instead of returning the full node object (n),
#     # explicitly return labels and all properties as separate columns.
#     # Neptune's node object format varies — explicit columns are reliable.
#     node_result = client.execute_open_cypher_query(
#         openCypherQuery="""
#             MATCH (n)
#             RETURN labels(n)[0]  AS label,
#                    id(n)         AS node_id,
#                    keys(n)       AS prop_keys,
#                    n             AS node_obj
#             LIMIT 300
#         """
#     )

#     # ── Step 2: Pull relationships with explicit columns ───────────────
#     rel_result = client.execute_open_cypher_query(
#         openCypherQuery="""
#             MATCH (a)-[r]->(b)
#             RETURN id(a)    AS from_id,
#                    id(b)    AS to_id,
#                    type(r)  AS rel_type
#             LIMIT 500
#         """
#     )

#     nodes_raw = node_result.get("results", [])
#     rels_raw  = rel_result.get("results",  [])

#     print(f"  Raw nodes fetched : {len(nodes_raw)}")
#     print(f"  Raw rels fetched  : {len(rels_raw)}")

#     if not nodes_raw:
#         print("  ⚠️  No nodes found in Neptune. Load data first.")
#         return

#     # ── Step 3: Discover all unique labels and auto-assign colours ─────
#     # Generate a distinct colour for every label found in the graph.
#     # Uses a hash of the label name → consistent colour every run.
#     PALETTE = [
#         "#4A90D9", "#27AE60", "#E74C3C", "#F39C12", "#9B59B6",
#         "#1ABC9C", "#E67E22", "#2ECC71", "#3498DB", "#E91E63",
#         "#FF5722", "#607D8B", "#795548", "#009688", "#673AB7"
#     ]

#     all_labels   = list({row.get("label", "Unknown") for row in nodes_raw})
#     label_colour = {
#         label: PALETTE[i % len(PALETTE)]
#         for i, label in enumerate(sorted(all_labels))
#     }

#     print(f"  Labels found      : {all_labels}")
#     print(f"  Colour map        : {label_colour}")

#     # ── Step 4: Build a node_id → display info map ────────────────────
#     # node_id from id(n) is Neptune's internal element ID — reliable.
#     # We find the best display label from the node's own properties.
#     node_map = {}   # node_id → {label, display_name, tooltip, colour}

#     for row in nodes_raw:
#         label    = row.get("label", "Unknown")
#         node_id  = str(row.get("node_id", ""))
#         node_obj = row.get("node_obj", {})
#         prop_keys = row.get("prop_keys", [])

#         if not node_id:
#             continue

#         # Extract all properties from the node object
#         props = {}
#         if isinstance(node_obj, dict):
#             props = {
#                 k: v for k, v in node_obj.items()
#                 if not k.startswith("~") and v is not None
#             }

#         # Pick the best human-readable display name from properties
#         # Priority: name > id > first string property > node_id
#         display_name = (
#             props.get("name")   or
#             props.get("id")     or
#             props.get("title")  or
#             next((str(v) for v in props.values() if isinstance(v, str)), None) or
#             f"{label}_{node_id}"
#         )

#         # Build tooltip: show label + all properties on hover
#         tooltip_lines = [f"Label: {label}"]
#         for k, v in props.items():
#             tooltip_lines.append(f"{k}: {v}")
#         tooltip = "\n".join(tooltip_lines)

#         node_map[node_id] = {
#             "label":        label,
#             "display_name": str(display_name),
#             "tooltip":      tooltip,
#             "colour":       label_colour.get(label, "#95A5A6")
#         }

#     # ── Step 5: Build PyVis network ───────────────────────────────────
#     net = Network(
#         height     = "800px",
#         width      = "100%",
#         bgcolor    = "#1a1a2e",
#         font_color = "white",
#         notebook   = False
#     )
#     # Barnes-hut physics: nodes repel, edges attract — clean auto-layout
#     net.barnes_hut(
#         gravity=-8000,
#         central_gravity=0.3,
#         spring_length=150,
#         spring_strength=0.05,
#         damping=0.9
#     )

#     # Add nodes to PyVis
#     added_node_ids = set()
#     for node_id, info in node_map.items():
#         net.add_node(
#             node_id,
#             label = info["display_name"],   # text shown ON the node
#             title = info["tooltip"],         # text shown on HOVER
#             color = info["colour"],
#             size  = 25,
#             font  = {"size": 14, "color": "white", "strokeWidth": 2, "strokeColor": "#000000"},
#             borderWidth = 2,
#             borderWidthSelected = 4
#         )
#         added_node_ids.add(node_id)

#     # Add edges to PyVis
#     edge_count = 0
#     for row in rels_raw:
#         from_id  = str(row.get("from_id", ""))
#         to_id    = str(row.get("to_id",   ""))
#         rel_type = str(row.get("rel_type",""))

#         if from_id in added_node_ids and to_id in added_node_ids:
#             net.add_edge(
#                 from_id, to_id,
#                 label  = rel_type,           # relationship name shown on edge
#                 title  = rel_type,           # shown on hover
#                 color  = {"color": "#aaaaaa", "highlight": "#ffffff"},
#                 arrows = "to",
#                 font   = {"size": 11, "color": "#dddddd", "strokeWidth": 0},
#                 width  = 1.5
#             )
#             edge_count += 1

#     print(f"  Nodes added to graph : {len(added_node_ids)}")
#     print(f"  Edges added to graph : {edge_count}")

#     # ── Step 6: Add legend for node labels ────────────────────────────
#     # Adds invisible legend nodes in the top-left so viewer knows
#     # which colour = which label
#     legend_x = -600
#     legend_y = -400
#     for i, (label, colour) in enumerate(label_colour.items()):
#         legend_id = f"__legend_{label}"
#         net.add_node(
#             legend_id,
#             label   = f"  {label}",
#             color   = colour,
#             size    = 15,
#             x       = legend_x,
#             y       = legend_y + (i * 50),
#             physics = False,          # legend nodes don't move
#             fixed   = True,
#             font    = {"size": 13, "color": "white"},
#             shape   = "dot",
#             title   = f"Node type: {label}"
#         )

#     # ── Step 7: Configure display options ─────────────────────────────
#     net.set_options("""
#     {
#       "nodes": {
#         "shape": "dot",
#         "scaling": { "min": 20, "max": 30 }
#       },
#       "edges": {
#         "smooth": { "type": "curvedCW", "roundness": 0.2 },
#         "font":   { "align": "middle" }
#       },
#       "interaction": {
#         "hover":          true,
#         "navigationButtons": true,
#         "keyboard":       true,
#         "tooltipDelay":   100
#       },
#       "physics": {
#         "enabled": true,
#         "stabilization": { "iterations": 150 }
#       }
#     }
#     """)

#     # ── Step 8: Save and open ─────────────────────────────────────────
#     net.save_graph(output_file)
#     print(f"  ✅ Graph saved → {output_file}")
#     webbrowser.open(f"file://{os.path.abspath(output_file)}")

# Fixed the lumping of nodes and the display of  node names instead of types
def visualize_graph(client, output_file="neptune_graph.html"):
    """
    Generic interactive graph visualizer — fixed version.
    - Nodes show NAME (not type/id) as display label
    - Full details (type, id, all properties) shown only on hover
    - Legend separated and pinned — does not interfere with graph
    - Physics tuned so nodes spread out and stay separated
    """
    from pyvis.network import Network

    print(f"\n[VISUALIZE] Building interactive graph → {output_file}...")

    # ── Fetch nodes ───────────────────────────────────────────────────
    node_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (n)
            RETURN labels(n)[0] AS label,
               id(n)        AS node_id,
               n.id         AS id_prop,
               n.name       AS name_prop,
               n.title      AS title_prop
            LIMIT 300
        """
    )

    # ── Fetch relationships ───────────────────────────────────────────
    rel_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (a)-[r]->(b)
            RETURN id(a)   AS from_id,
                   id(b)   AS to_id,
                   type(r) AS rel_type
            LIMIT 500
        """
    )

    nodes_raw = node_result.get("results", [])
    rels_raw  = rel_result.get("results",  [])

    print(f"  Nodes fetched : {len(nodes_raw)}")
    print(f"  Rels fetched  : {len(rels_raw)}")

    if not nodes_raw:
        print("  ⚠️  No nodes found. Load data first.")
        return

    # ── Auto-assign colours — one per label ──────────────────────────
    PALETTE = [
        "#4A90D9", "#27AE60", "#E74C3C", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#2ECC71", "#3498DB", "#E91E63",
        "#FF5722", "#607D8B", "#795548", "#009688", "#673AB7"
    ]
    all_labels   = sorted({row.get("label", "Unknown") for row in nodes_raw})
    label_colour = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(all_labels)}
    print(f"  Labels found  : {all_labels}")

    # ── Build node map ────────────────────────────────────────────────
    node_map = {}

    # for row in nodes_raw:
    #     label    = row.get("label", "Unknown")
    #     node_id  = str(row.get("node_id", ""))
    #     node_obj = row.get("node_obj", {})
    #     if not node_id:
    #         continue

    #     # Extract all properties cleanly
    #     props = {
    #         k: v for k, v in (node_obj if isinstance(node_obj, dict) else {}).items()
    #         if not k.startswith("~") and v is not None
    #     }

    #     # ── FIX 2: Display NAME on the node, everything else on hover ─
    #     # Priority for display label: name > title > id property > first string value
    #     display_name = (
    #         props.get("name")   or
    #         props.get("title")  or
    #         props.get("id")     or
    #         next((str(v) for v in props.values() if isinstance(v, str)), None) or
    #         f"{label}_{node_id}"
    #     )

    #     # Hover tooltip — shows ALL details including type and internal id
    #     tooltip_lines = [
    #         f"Type    : {label}",
    #         f"Intern ID: {node_id}",
    #         "─────────────────",
    #     ]
    #     for k, v in props.items():
    #         tooltip_lines.append(f"{k}: {v}")
    #     tooltip = "\n".join(tooltip_lines)

    #     node_map[node_id] = {
    #         "label":        label,
    #         "display_name": str(display_name),   # human name on node
    #         "tooltip":      tooltip,              # full detail on hover
    #         "colour":       label_colour.get(label, "#95A5A6")
    #     }
    for row in nodes_raw:
        label      = row.get("label",      "Unknown")
        node_id    = str(row.get("node_id", ""))
        id_prop    = row.get("id_prop")          # value of n.id property
        name_prop  = row.get("name_prop")        # value of n.name property
        title_prop = row.get("title_prop")       # value of n.title property

        if not node_id:
            continue

        # Display name — explicitly from queried columns, no object parsing
        display_name = (
            name_prop   or    # n.name  e.g. "Ravi Sharma"
            title_prop  or    # n.title
            id_prop     or    # n.id    e.g. "Alice" (set by LLMGraphTransformer)
            f"{label}_{node_id}"
        )

        # Hover tooltip — type + id + all known properties
        tooltip = "\n".join([
            f"Type : {label}",
            f"ID   : {node_id}",
            f"name : {name_prop or '—'}",
            f"id   : {id_prop   or '—'}",
        ])

        node_map[node_id] = {
            "label":        label,
            "display_name": str(display_name),
            "tooltip":      tooltip,
            "colour":       label_colour.get(label, "#95A5A6")
        }

    # ── Build PyVis network ───────────────────────────────────────────
    net = Network(
        height     = "860px",
        width      = "100%",
        bgcolor    = "#1a1a2e",
        font_color = "white",
        notebook   = False
    )

    # ── FIX 1A: Physics tuned so nodes repel strongly and spread out ──
    # High gravitational repulsion + longer spring = nodes stay apart
    net.barnes_hut(
        gravity         = -25000,   # strong repulsion between nodes
        central_gravity = 0.05,     # very weak pull to centre — nodes spread freely
        spring_length   = 220,      # longer springs = more space between connected nodes
        spring_strength = 0.04,     # weak spring = doesn't pull nodes together too hard
        damping         = 0.95,     # high damping = settles quickly, less bouncing
        overlap         = 1         # avoid node overlap
    )

    # Add data nodes
    added_node_ids = set()
    for node_id, info in node_map.items():
        net.add_node(
            node_id,
            label       = info["display_name"],   # ← NAME shown on node
            title       = info["tooltip"],         # ← full detail on hover only
            color       = {
                "background": info["colour"],
                "border":     "#ffffff",
                "highlight":  {"background": "#ffffff", "border": info["colour"]}
            },
            size        = 28,
            font        = {
                "size":        15,
                "color":       "white",
                "strokeWidth": 3,
                "strokeColor": "#000000"   # black outline makes text readable on any bg
            },
            borderWidth         = 2,
            borderWidthSelected = 5,
            shadow      = True
        )
        added_node_ids.add(node_id)

    # Add edges
    edge_count = 0
    for row in rels_raw:
        from_id  = str(row.get("from_id",  ""))
        to_id    = str(row.get("to_id",    ""))
        rel_type = str(row.get("rel_type", ""))

        if from_id in added_node_ids and to_id in added_node_ids:
            net.add_edge(
                from_id, to_id,
                label  = rel_type,
                title  = rel_type,
                color  = {"color": "#aaaaaa", "highlight": "#ffffff", "opacity": 0.8},
                arrows = "to",
                width  = 2,
                font   = {
                    "size":        12,
                    "color":       "#eeeeee",
                    "strokeWidth": 2,
                    "strokeColor": "#000000",
                    "align":       "middle"
                },
                smooth = {"type": "curvedCW", "roundness": 0.15}
            )
            edge_count += 1

    print(f"  Nodes in graph : {len(added_node_ids)}")
    print(f"  Edges in graph : {edge_count}")

    # ── FIX 1B: Legend pinned far left, outside graph area ───────────
    # Physics=False + fixed=True means legend nodes NEVER move,
    # even when you drag or physics runs. They stay locked in place.
    LEGEND_X = -900          # far left, well outside the graph cluster area
    LEGEND_Y = -350
    GAP      = 55

    # Invisible anchor node — gives legend a stable top reference point
    net.add_node(
        "__legend_title",
        label   = "── NODE TYPES ──",
        color   = {"background": "#2c2c4a", "border": "#4A90D9"},
        size    = 5,
        x       = LEGEND_X,
        y       = LEGEND_Y - 30,
        physics = False,
        fixed   = {"x": True, "y": True},
        font    = {"size": 13, "color": "#aaaaff"},
        shape   = "box"
    )

    for i, (label, colour) in enumerate(label_colour.items()):
        net.add_node(
            f"__legend_{label}",
            label   = label,            # ← legend shows TYPE name (correct for legend)
            color   = {"background": colour, "border": "#ffffff"},
            size    = 18,
            x       = LEGEND_X,
            y       = LEGEND_Y + (i * GAP),
            physics = False,            # ← never moves
            fixed   = {"x": True, "y": True},
            font    = {"size": 13, "color": "white", "strokeWidth": 2, "strokeColor": "#000000"},
            shape   = "dot",
            title   = f"Node type: {label}"
        )

    # ── Global interaction and display options ────────────────────────
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "shadow": true
      },
      "edges": {
        "shadow": true,
        "selectionWidth": 3
      },
      "interaction": {
        "hover":             true,
        "navigationButtons": true,
        "keyboard":          true,
        "tooltipDelay":      80,
        "zoomView":          true,
        "dragView":          true
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -25000,
          "centralGravity":        0.05,
          "springLength":          220,
          "springConstant":        0.04,
          "damping":               0.95,
          "avoidOverlap":          1
        },
        "stabilization": {
          "enabled":    true,
          "iterations": 300,
          "fit":        true
        }
      }
    }
    """)

    # net.save_graph(output_file)
    # print(f"  ✅ Saved → {output_file}")
    # webbrowser.open(f"file://{os.path.abspath(output_file)}")

    net.save_graph(output_file)

    # ── Inject JS: disable physics once stabilization completes ───────
    # After the initial layout run, physics turns off automatically.
    # Nodes then hold position when you drag them — they never snap back.
    with open(output_file, "r") as f:
        html = f.read()

    freeze_script = """
    <script>
    window.addEventListener("load", function() {
        network.on("stabilizationIterationsDone", function () {
            network.setOptions({ physics: { enabled: false } });
            console.log("Physics disabled — nodes are now frozen in place.");
        });
    });
    </script>
    """
    # Inject just before closing </body> tag
    html = html.replace("</body>", freeze_script + "</body>")

    with open(output_file, "w") as f:
        f.write(html)

    print(f"   Saved → {output_file}")
    print(f"   Nodes will auto-freeze after initial layout. Drag freely.")
    webbrowser.open(f"file://{os.path.abspath(output_file)}")

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 5 — NLP QUERY using create_neptune_opencypher_qa_chain
#
#  What happens under the hood for each question you ask:
#    1. NeptuneGraph already has the schema (from connect_option_b)
#    2. LLM reads schema + your English question
#    3. LLM writes the openCypher query
#    4. Chain runs the query against Neptune
#    5. LLM reads the raw results + original question
#    6. LLM writes a natural language answer back to you
#
#  Ref: https://python.langchain.com/docs/integrations/graphs/
#       amazon_neptune_open_cypher/
# ═══════════════════════════════════════════════════════════════════════
def nlp_query(graph):
    """
    Query Neptune in plain English using LangChain QA chain.
    The LLM converts your question → openCypher → runs it → answers in English.

    Args:
        graph : NeptuneGraph object from connect_option_b()
                (must be Option B — it holds the schema the LLM needs)
    """
    from langchain_aws import ChatBedrock
    from langchain_community.chains.graph_qa.neptune_cypher import (
        create_neptune_opencypher_qa_chain
    )

    print("\n[NLP QUERY] Setting up natural language query chain...")

    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )

    # Create the chain
    # return_intermediate_steps=True → shows generated query + raw Neptune result
    # verbose=True                   → prints each internal step to console
    chain = create_neptune_opencypher_qa_chain(
        llm                       = llm,
        graph                     = graph,
        return_intermediate_steps = True,
        verbose                   = True
    )

    # ── Your NLP questions ─────────────────────────────────────────────
    # Add / change / remove questions as needed for your demo
    questions = [
        "Who are all the patients in the database?",
        "Which doctor treated Ravi Sharma?",
        "What hospital was Ravi Sharma admitted to?",
        "Which claims are currently under review?",
        "What diagnosis does Anjali Mehta have?",
        "Which patient filed the highest value claim?",
        "Show me all relationships involving Suresh Nair.",
    ]

    print(f"\n{'═'*60}")
    print("  NLP QUERY RESULTS")
    print(f"{'═'*60}")

    for question in questions:
        print(f"\n  ❓ QUESTION : {question}")
        try:
            response = chain.invoke({"query": question})

            # Natural language answer
            print(f"  💬 ANSWER   : {response['result']}")

            # Show intermediate steps — generated query + raw data
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if "query" in step:
                        print(f"  🔍 QUERY    : {step['query'].strip()}")
                    if "context" in step:
                        print(f"  📦 RAW DATA : {str(step['context'])[:200]}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

        print()


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 6 — VERIFY (cross-check what was written vs what's in Neptune)
# ═══════════════════════════════════════════════════════════════════════
def verify(client):
    """Quick verification — counts nodes and relationships in Neptune."""
    print("\n[VERIFY] Checking Neptune graph contents...")

    node_result = client.execute_open_cypher_query(
        openCypherQuery="MATCH (n) RETURN labels(n)[0] AS Type, count(n) AS Count ORDER BY Count DESC"
    )
    rel_result = client.execute_open_cypher_query(
        openCypherQuery="MATCH ()-[r]->() RETURN type(r) AS Relationship, count(r) AS Count ORDER BY Count DESC"
    )

    print("\n  Node counts:")
    for row in node_result.get("results", []):
        print(f"    {row.get('Type','?'):<20}: {row.get('Count',0)}")

    print("\n  Relationship counts:")
    for row in rel_result.get("results", []):
        print(f"    {row.get('Relationship','?'):<25}: {row.get('Count',0)}")


# ═══════════════════════════════════════════════════════════════════════
#  SECTION 7 — CLEAN (wipe all data — use to reset between test runs)
# ═══════════════════════════════════════════════════════════════════════
def clean_graph(client):
    """Deletes ALL nodes and relationships from Neptune. Use carefully."""
    confirm = input("\n⚠️  Delete ALL data from Neptune? Type 'yes' to confirm: ").strip()
    if confirm.lower() == "yes":
        client.execute_open_cypher_query(openCypherQuery="MATCH (n) DETACH DELETE n")
        print("  ✅ Graph cleared.")
    else:
        print("  Cancelled.")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — Run the PoC
#
#  HOW TO USE:
#  Comment / uncomment the blocks below to choose what to run.
#  For a full end-to-end demo, run everything in order (top to bottom).
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("\n" + "★"*60)
    print("   AWS NEPTUNE FULL POC — MEDICAL & CLAIMS GRAPH")
    print("★"*60)

    # ──────────────────────────────────────────────────────────
    # STEP 1 — Connect
    # Run ONE or BOTH. client_a is used in Steps 2A,3A,4,6,7.
    # graph_b  is used in Steps 2B,3B,5.
    # ──────────────────────────────────────────────────────────
    client_a = connect_option_a()           # boto3    — for write/query/visualize
    graph_b  = connect_option_b()           # LangChain — for NLP querying

    # ──────────────────────────────────────────────────────────
    # STEP 2 — Write Graph Documents
    # Choose Option A OR Option B — both do the same thing,
    # just via different clients. Comment out the one you don't need.
    # ──────────────────────────────────────────────────────────
    write_graph_documents_option_a(client_a, SAMPLE_TEXTS)   # boto3
    # write_graph_documents_option_b(graph_b,  SAMPLE_TEXTS) # LangChain

    # ──────────────────────────────────────────────────────────
    # STEP 3 — Verify data was written correctly
    # ──────────────────────────────────────────────────────────
    verify(client_a)

    # ──────────────────────────────────────────────────────────
    # STEP 4 — Query & Display
    # Run BOTH to show your manager the two connection approaches.
    # ──────────────────────────────────────────────────────────
    query_option_a(client_a)    # boto3 — structured query results
    query_option_b(graph_b)     # LangChain — same queries + schema display

    # ──────────────────────────────────────────────────────────
    # STEP 5 — Visualize (opens browser with interactive graph)
    # ──────────────────────────────────────────────────────────
    visualize_graph(client_a, output_file="neptune_graph.html")

    # ──────────────────────────────────────────────────────────
    # STEP 6 — NLP Query (plain English → openCypher → answer)
    # Requires Option B connection (graph_b) for schema access.
    # ──────────────────────────────────────────────────────────
    nlp_query(graph_b)

    # ──────────────────────────────────────────────────────────
    # STEP 7 — Clean (reset — only use between test runs)
    # Uncomment this when you want to wipe the graph.
    # ──────────────────────────────────────────────────────────
    # clean_graph(client_a)

    print("\n" + "★"*60)
    print("   POC COMPLETE")
    print("★"*60)
