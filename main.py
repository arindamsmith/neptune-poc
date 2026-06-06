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
    print(f"\n\n results - total nodes - {node_result}")
    print("\n  Node counts:")
    node_count=0
    for row in node_result.get("results", []):
        print(f"    {row.get('Type','?'):<20}: {row.get('Count',0)}")
        node_count+=row.get('Count',0)
    print(f"Total node count={node_count}")

    print(f"\n\n results - total Relationship - {rel_result}")
    print("\n  Relationship counts:")
    rel_count=0
    for row in rel_result.get("results", []):
        print(f"    {row.get('Relationship','?'):<25}: {row.get('Count',0)}")
        rel_count+=row.get('Count',0)
    print(f"Total relationship count={rel_count}")


def get_nlp_query_chain(graph):
    """
    Builds and returns a reusable NeptuneOpenCypherQAChain with a
    generic, foolproof custom prompt.

    Handles ANY graph content — medical, claims, social, financial, etc.
    Tolerates spelling mistakes, case differences, partial names,
    implicit relationships, and reverse relationship directions.

    Args:
        graph : NeptuneGraph object from connect_option_b()

    Returns:
        chain : invoke with chain.invoke({"query": "your question"})

    Usage:
        chain  = get_nlp_query_chain(graph)
        result = chain.invoke({"query": "What does Alice like?"})
        print(result["result"])
    """
    from langchain_aws import ChatBedrock
    from langchain.prompts import PromptTemplate
    from langchain_community.chains.graph_qa.neptune_cypher import (
        create_neptune_opencypher_qa_chain
    )

    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )

    # ─────────────────────────────────────────────────────────────────
    # GENERIC FOOLPROOF CYPHER GENERATION PROMPT
    #
    # Designed to handle:
    #   - Any node types and relationship types (not hardcoded to domain)
    #   - Case insensitivity  : "bob" matches "Bob", "BOB"
    #   - Spelling tolerance  : CONTAINS instead of exact match
    #   - Property ambiguity  : searches id, name, title, description
    #   - Implicit questions  : "what does X like?" infers relationship
    #   - Reverse direction   : undirected match catches both directions
    #   - Missing data        : OPTIONAL MATCH avoids empty results
    #   - Aggregation queries : count, sum, avg, max, min
    #   - List queries        : "show all", "list all" type questions
    # ─────────────────────────────────────────────────────────────────
    CYPHER_GENERATION_TEMPLATE = """
You are an expert openCypher query generator for Amazon Neptune graph database.
Convert the user's natural language question into a valid Neptune openCypher query.

Graph Schema:
{schema}

════════════════════════════════════════════════════════
NEPTUNE-SPECIFIC RULES — STRICTLY FOLLOW THESE:
════════════════════════════════════════════════════════

RULE 1 — NEVER USE coalesce() IN A WHERE CLAUSE
  Neptune does NOT support coalesce() inside WHERE filters.
  This is INVALID and will throw MalformedQueryException:
    WHERE toLower(coalesce(n.id,'')) CONTAINS toLower('alice')  ← WRONG

  Instead use direct OR conditions on individual properties:
    WHERE toLower(n.id) CONTAINS toLower('alice')
       OR toLower(n.name) CONTAINS toLower('alice')             ← CORRECT

RULE 2 — ALWAYS USE toLower() ON BOTH SIDES FOR CASE-INSENSITIVE MATCH
  WHERE toLower(n.id) CONTAINS toLower('search_term')
  This handles: 'alice', 'Alice', 'ALICE' — all match correctly.

RULE 3 — SEARCH ACROSS MULTIPLE PROPERTIES USING OR
  Nodes may store the display value under id, name, or title.
  Always check all using separate OR conditions:
    WHERE toLower(n.id)    CONTAINS toLower('term')
       OR toLower(n.name)  CONTAINS toLower('term')
       OR toLower(n.title) CONTAINS toLower('term')

RULE 4 — USE UNDIRECTED RELATIONSHIPS (no arrow)
  Use (a)-[r]-(b) not (a)-[r]->(b) unless direction is 100% certain.
  This handles reverse relationships automatically.

RULE 5 — USE OPTIONAL MATCH FOR RELATIONSHIP TRAVERSALS
  OPTIONAL MATCH (a)-[r]-(b) ensures partial data returns results
  instead of an empty list.

RULE 6 — RETURN PROPERTY VALUES NOT NODE OBJECTS
  WRONG : RETURN n
  CORRECT: RETURN n.id AS Name, n.name AS Name, labels(n)[0] AS Type

RULE 7 — coalesce() IS ONLY VALID IN RETURN CLAUSE
  coalesce() works fine in RETURN but NOT in WHERE.
  VALID  : RETURN coalesce(n.name, n.id) AS DisplayName
  INVALID: WHERE coalesce(n.id,'') CONTAINS 'alice'

RULE 8 — INFER RELATIONSHIP FROM QUESTION CONTEXT
  If the user does not name a relationship, infer from keywords:
  'friend/friends' → FRIENDS_WITH or FRIEND
  'work/works at' → WORKS_AT or EMPLOYED_BY
  'like/likes'    → LIKES
  'treat/treated' → TREATED_BY or TREATS
  'filed/claim'   → FILED
  'diagnose'      → DIAGNOSED_WITH or HAS_DIAGNOSIS
  Use: toLower(type(r)) CONTAINS toLower('inferred_keyword')
  Or omit relationship filter entirely and return all connections.

RULE 9 — ALWAYS ADD LIMIT 50

RULE 10 — ALWAYS ALIAS EVERY RETURN COLUMN WITH AS

════════════════════════════════════════════════════════
CORRECT QUERY PATTERNS FOR NEPTUNE:
════════════════════════════════════════════════════════

PATTERN A — Find a specific entity by name (MOST COMMON):
  MATCH (n)
  WHERE toLower(n.id)    CONTAINS toLower('alice')
     OR toLower(n.name)  CONTAINS toLower('alice')
  RETURN labels(n)[0] AS Type,
         n.id           AS ID,
         n.name         AS Name
  LIMIT 50

PATTERN B — Find ALL connections of an entity:
  MATCH (a)
  WHERE toLower(a.id)   CONTAINS toLower('alice')
     OR toLower(a.name) CONTAINS toLower('alice')
  OPTIONAL MATCH (a)-[r]-(b)
  RETURN a.id            AS Entity,
         type(r)          AS Relationship,
         b.id             AS ConnectedTo,
         labels(b)[0]     AS ConnectedType
  LIMIT 50

PATTERN C — Find relationship between TWO specific entities:
  MATCH (a)
  WHERE toLower(a.id)   CONTAINS toLower('entity_one')
     OR toLower(a.name) CONTAINS toLower('entity_one')
  OPTIONAL MATCH (a)-[r]-(b)
  WHERE toLower(b.id)   CONTAINS toLower('entity_two')
     OR toLower(b.name) CONTAINS toLower('entity_two')
  RETURN a.id   AS From,
         type(r) AS Relationship,
         b.id    AS To
  LIMIT 50

PATTERN D — Find by relationship type keyword:
  MATCH (a)-[r]-(b)
  WHERE toLower(type(r)) CONTAINS toLower('relationship_keyword')
  RETURN a.id   AS From,
         type(r) AS Relationship,
         b.id    AS To
  LIMIT 50

PATTERN E — Find entity connections filtered by relationship keyword:
  MATCH (a)
  WHERE toLower(a.id)   CONTAINS toLower('entity_name')
     OR toLower(a.name) CONTAINS toLower('entity_name')
  OPTIONAL MATCH (a)-[r]-(b)
  WHERE toLower(type(r)) CONTAINS toLower('relationship_keyword')
  RETURN a.id    AS Entity,
         type(r)  AS Relationship,
         b.id     AS ConnectedTo
  LIMIT 50

PATTERN F — List all nodes of a type:
  MATCH (n:NodeLabel)
  RETURN n.id           AS ID,
         n.name         AS Name,
         labels(n)[0]   AS Type
  ORDER BY n.id
  LIMIT 50

PATTERN G — Count / aggregate query:
  MATCH (a)-[r]-(b)
  WHERE toLower(a.id)   CONTAINS toLower('entity_name')
     OR toLower(a.name) CONTAINS toLower('entity_name')
  RETURN a.id          AS Entity,
         type(r)        AS Relationship,
         count(b)       AS Count
  ORDER BY Count DESC
  LIMIT 50

PATTERN H — Full graph overview:
  MATCH (a)-[r]-(b)
  RETURN labels(a)[0] AS FromType,
         a.id          AS From,
         type(r)       AS Relationship,
         labels(b)[0]  AS ToType,
         b.id          AS To
  LIMIT 50

════════════════════════════════════════════════════════
THINK STEP BY STEP:
  1. Identify entity names in the question
  2. Identify what relationship or property is being asked
  3. Is aggregation needed? (how many, total, highest, most)
  4. Pick the matching PATTERN above
  5. Apply ALL rules — especially RULE 1 (no coalesce in WHERE)
════════════════════════════════════════════════════════

Question: {question}

openCypher Query:"""

    cypher_prompt = PromptTemplate(
        input_variables = ["schema", "question"],
        template        = CYPHER_GENERATION_TEMPLATE
    )

    chain = create_neptune_opencypher_qa_chain(
        llm                       = llm,
        graph                     = graph,
        cypher_prompt             = cypher_prompt,
        return_intermediate_steps = True,
        verbose                   = True
    )

    print("✅ NLP Query Chain ready.")
    return chain

# Build the chain once — reuse it for all questions
chain = get_nlp_query_chain(graph_b)

# Invoke for any question
def ask(chain, question: str):
    print(f"\n❓ {question}")
    response = chain.invoke({"query": question})
    print(f"💬 {response['result']}")

    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            if "query" in step:
                print(f"🔍 {step['query'].strip()}")
    return response["result"]

# Use anywhere
ask(chain, "What does Alice like?")
ask(chain, "where does bob work")
ask(chain, "Which claims are under review?")
ask(chain, "Who treated Ravi Sharma?")
ask(chain, "What is the total claim amount for all approved claims?")
ask(chain, "Which hospital has the most patients?")


import re
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ═══════════════════════════════════════════════════════════════════════
# NLP QUERY — SEQUENTIAL FUNCTIONS
# Call order: get_llm → get_schema → generate_cypher → clean_cypher
#             → run_neptune_query → generate_answer → ask
# ═══════════════════════════════════════════════════════════════════════

def get_llm():
    """
    Initialises and returns the Bedrock LLM.
    Call once and reuse for both generate_cypher and generate_answer.
    """
    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )
    print("✅ LLM initialised.")
    return llm


def get_schema(graph):
    """
    Reads and returns the graph schema from Neptune.
    NeptuneGraph auto-discovers all node labels and relationship types
    on connect. The LLM uses this schema to write correct Cypher queries.
    """
    schema = graph.get_schema
    print(f"✅ Schema loaded:\n{schema}")
    return schema


def generate_cypher(llm, schema, question):
    """
    Step 1 — LLM reads schema + question and generates an openCypher query.
    Output is raw LLM text — may contain markdown fences or reasoning text.
    QueryGuard (clean_cypher) will fix it before Neptune sees it.

    Args:
        llm      : ChatBedrock instance from get_llm()
        schema   : graph schema string from get_schema()
        question : plain English question from user

    Returns:
        str : raw LLM output containing the generated Cypher
    """
    CYPHER_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template="""You are an openCypher expert for Amazon Neptune.
Generate ONE openCypher query for the question below.

Schema:
{schema}

RULES:
1. Use ONE flat MATCH clause only — never split into two MATCH clauses
2. Put ALL conditions in ONE WHERE block
3. Never use OPTIONAL MATCH
4. Use undirected relationship: (a)-[r]-(b)
5. Case-insensitive: toLower(n.id) CONTAINS toLower('term')
6. Search both id and name: toLower(n.id) CONTAINS toLower('x') OR toLower(n.name) CONTAINS toLower('x')
7. No spaces inside functions: toLower() not to Lower() and not toLower ()
8. RETURN property values not node objects: RETURN n.id not RETURN n
9. End with LIMIT 50
10. Output ONLY the raw Cypher — no explanation, no markdown, no commentary

CORRECT example for "what does alice like":
MATCH (a)-[r]-(b)
WHERE (toLower(a.id) CONTAINS toLower('alice') OR toLower(a.name) CONTAINS toLower('alice'))
AND toLower(type(r)) CONTAINS toLower('like')
RETURN a.id AS Person, type(r) AS Relationship, b.id AS What
LIMIT 50

CORRECT example for "who is alice friend":
MATCH (a)-[r]-(b)
WHERE (toLower(a.id) CONTAINS toLower('alice') OR toLower(a.name) CONTAINS toLower('alice'))
AND toLower(type(r)) CONTAINS toLower('friend')
RETURN a.id AS From, type(r) AS Relationship, b.id AS Friend
LIMIT 50

CORRECT example for "show all patients":
MATCH (n:Patient)
RETURN n.id AS ID, n.name AS Name
LIMIT 50

Question: {question}
Cypher Query:"""
    )

    chain      = CYPHER_PROMPT | llm | StrOutputParser()
    raw_query  = chain.invoke({"schema": schema, "question": question})
    print(f"\n🤖 RAW LLM OUTPUT:\n{raw_query}")
    return raw_query


def clean_cypher(raw_query):
    """
    Step 2 — QueryGuard: cleans the LLM-generated Cypher before Neptune sees it.

    Fixes these known LLM failure patterns:
      a) Strips markdown fences and reasoning text — extracts only Cypher
      b) Fixes spaces inside function calls: to Lower() → toLower()
      c) Replaces OPTIONAL MATCH with plain MATCH
      d) Merges two MATCH blocks into one flat MATCH — the core fix

    The LLM always generates a 2-block pattern:
        MATCH (a)                     ← block 1: find entity
        WHERE condition1
        OPTIONAL MATCH (a)-[r]-(b)   ← block 2: follow relationship
        WHERE condition2

    Neptune only accepts one flat MATCH:
        MATCH (a)-[r]-(b)
        WHERE condition1
        AND condition2

    Args:
        raw_query : raw LLM output string from generate_cypher()

    Returns:
        str : clean, Neptune-safe Cypher query
    """

    # ── a) Extract Cypher — strip markdown and reasoning text ─────────
    match = re.search(r'```(?:cypher)?\s*(.*?)```', raw_query, re.DOTALL | re.IGNORECASE)
    if match:
        query = match.group(1).strip()
    else:
        # No markdown fence — extract lines starting from first Cypher keyword
        lines        = raw_query.split('\n')
        cypher_lines = []
        started      = False
        for line in lines:
            stripped = line.strip()
            if re.match(
                r'^(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|WHERE|UNWIND)',
                stripped, re.IGNORECASE
            ):
                started = True
            if started:
                # Stop when explanation text begins after the query
                if stripped.startswith(('-', '*', 'This', 'The query', 'The user')):
                    break
                cypher_lines.append(line)
        query = '\n'.join(cypher_lines).strip() if cypher_lines else raw_query.strip()

    # ── b) Fix spaces inside Neptune function calls ───────────────────
    function_fixes = [
        (r'to\s+[Ll]ower\s*\(',                        'toLower('),
        (r'toLower\s+\(',                               'toLower('),
        (r'to\s+[Uu]pper\s*\(',                        'toUpper('),
        (r'labels\s*\(\s*(\w+)\s*\)\s*\[\s*0\s*\]',   r'labels(\1)[0]'),
        (r'type\s*\(\s*(\w+)\s*\)',                    r'type(\1)'),
    ]
    for pattern, replacement in function_fixes:
        query = re.sub(pattern, replacement, query)

    # ── c) Replace OPTIONAL MATCH with plain MATCH ────────────────────
    query = re.sub(r'OPTIONAL\s+MATCH', 'MATCH', query, flags=re.IGNORECASE)

    # ── d) Merge two MATCH blocks into one flat MATCH ─────────────────
    # Handles:
    #   MATCH (a)
    #   WHERE cond1
    #   MATCH (a)-[r]-(b)
    #   WHERE cond2
    # →
    #   MATCH (a)-[r]-(b)
    #   WHERE cond1
    #   AND cond2
    query = re.sub(
        r'MATCH\s+(\(\w+\))\s*\n\s*WHERE\s+([\s\S]+?)\n\s*MATCH\s+(\(\w+\)-\[[^\]]*\]-\(\w+\))\s*\n\s*WHERE\s+',
        r'MATCH \3\nWHERE \2\nAND ',
        query,
        flags=re.IGNORECASE
    )

    print(f"\n✅ CLEANED QUERY (sent to Neptune):\n{query}")
    return query


# def run_neptune_query(graph, clean_query, question):
#     """
#     Step 3 — Runs the cleaned Cypher query against Neptune.
#     If it still fails, runs a safe broad fallback query.

#     The fallback extracts the most meaningful word from the question
#     and searches across all node id and name properties — always valid.

#     Args:
#         graph       : NeptuneGraph object from connect_option_b()
#         clean_query : cleaned Cypher string from clean_cypher()
#         question    : original user question (used to build fallback)

#     Returns:
#         list : list of result dicts from Neptune, or [] if nothing found
#     """
#     # Try cleaned query first
#     try:
#         raw     = graph.query(clean_query)
#         results = raw if isinstance(raw, list) else raw.get("results", [])
#         print(f"\n📦 NEPTUNE RESULT: {str(results)[:300]}")
#         return results

#     except Exception as e:
#         print(f"\n❌ Cleaned query failed: {str(e)[:200]}")

#     # Build and run safe fallback
#     stop_words = {
#         "who", "what", "where", "when", "how", "is", "are",
#         "the", "a", "an", "of", "in", "at", "by", "for",
#         "with", "to", "do", "does", "did", "was", "were"
#     }
#     words   = [
#         w.strip("?.,!") for w in question.lower().split()
#         if w.strip("?.,!") not in stop_words and len(w.strip("?.,!")) > 2
#     ]
#     term    = words[0] if words else "a"

#     fallback = f"""MATCH (a)-[r]-(b)
# WHERE toLower(a.id) CONTAINS toLower('{term}')
#    OR toLower(a.name) CONTAINS toLower('{term}')
#    OR toLower(b.id) CONTAINS toLower('{term}')
#    OR toLower(b.name) CONTAINS toLower('{term}')
# RETURN a.id AS From, type(r) AS Relationship, b.id AS To
# LIMIT 50"""

#     print(f"\n⚙️  Running fallback query:\n{fallback}")
#     try:
#         raw     = graph.query(fallback)
#         results = raw if isinstance(raw, list) else raw.get("results", [])
#         print(f"\n📦 FALLBACK RESULT: {str(results)[:300]}")
#         return results
#     except Exception as fe:
#         print(f"❌ Fallback also failed: {str(fe)[:200]}")
#         return []
    

def run_neptune_query(graph, clean_query, question):
    """
    Step 3 — Runs cleaned Cypher against Neptune.

    Fix: When results are empty, checks if the entities mentioned
    in the question actually exist in the graph.
    - If both exist but no relationship → returns explicit "not related" context
    - If one or both don't exist      → returns explicit "not found" context
    - This ensures generate_answer always has meaningful context to work with

    Args:
        graph       : NeptuneGraph object from connect_option_b()
        clean_query : cleaned Cypher string from clean_cypher()
        question    : original user question

    Returns:
        list : result dicts from Neptune, or meaningful context list if empty
    """

    # ── Try the cleaned query ─────────────────────────────────────────
    try:
        raw     = graph.query(clean_query)
        results = raw if isinstance(raw, list) else raw.get("results", [])
        print(f"\n📦 NEPTUNE RESULT: {str(results)[:300]}")

        if results:
            return results
        # Results empty — fall through to existence check below

    except Exception as e:
        print(f"\n❌ Cleaned query failed: {str(e)[:200]}")

        # ── Fallback broad query ──────────────────────────────────────
        stop_words = {
            "who", "what", "where", "when", "how", "is", "are",
            "the", "a", "an", "of", "in", "at", "by", "for",
            "with", "to", "do", "does", "did", "was", "were",
            "and", "related", "relation", "between", "neighbors",
            "neighbor", "friend", "friends"
        }
        words = [
            w.strip("?.,!") for w in question.lower().split()
            if w.strip("?.,!") not in stop_words and len(w.strip("?.,!")) > 2
        ]
        term = words[0] if words else "a"

        fallback = f"""MATCH (a)-[r]-(b)
WHERE toLower(a.id) CONTAINS toLower('{term}')
   OR toLower(a.name) CONTAINS toLower('{term}')
   OR toLower(b.id) CONTAINS toLower('{term}')
   OR toLower(b.name) CONTAINS toLower('{term}')
RETURN a.id AS From, type(r) AS Relationship, b.id AS To
LIMIT 50"""

        print(f"\n⚙️  Running fallback query:\n{fallback}")
        try:
            raw     = graph.query(fallback)
            results = raw if isinstance(raw, list) else raw.get("results", [])
            print(f"\n📦 FALLBACK RESULT: {str(results)[:300]}")
            if results:
                return results
        except Exception as fe:
            print(f"❌ Fallback also failed: {str(fe)[:200]}")

    # ── Results are empty — run existence check ───────────────────────
    # Extract all meaningful words from question as potential entity names
    print(f"\n🔎 Results empty — running entity existence check...")

    stop_words_exist = {
        "who", "what", "where", "when", "how", "is", "are", "the",
        "a", "an", "of", "in", "at", "by", "for", "with", "to",
        "do", "does", "did", "was", "were", "and", "related",
        "relation", "between", "neighbors", "neighbor", "friend",
        "friends", "have", "has", "many", "much"
    }
    candidate_entities = [
        w.strip("?.,!") for w in question.split()
        if w.strip("?.,!").lower() not in stop_words_exist
        and len(w.strip("?.,!")) > 2
    ]

    found_entities = []
    for entity in candidate_entities:
        existence_query = f"""MATCH (n)
WHERE toLower(n.id) CONTAINS toLower('{entity}')
   OR toLower(n.name) CONTAINS toLower('{entity}')
RETURN n.id AS NodeID
LIMIT 1"""
        try:
            res = graph.query(existence_query)
            res = res if isinstance(res, list) else res.get("results", [])
            if res:
                found_entities.append(entity)
                print(f"   ✅ Entity found in graph: '{entity}'")
            else:
                print(f"   ❌ Entity NOT in graph  : '{entity}'")
        except Exception:
            pass

    # Build meaningful context based on existence check results
    if len(found_entities) >= 2:
        context = [{
            "info": f"Both '{found_entities[0]}' and '{found_entities[1]}' "
                    f"exist in the graph but have NO direct relationship between them."
        }]
    elif len(found_entities) == 1:
        context = [{
            "info": f"'{found_entities[0]}' exists in the graph "
                    f"but no related data was found for the question asked."
        }]
    else:
        context = [{
            "info": f"None of the entities mentioned in the question "
                    f"were found in the graph."
        }]

    print(f"\n📦 EXISTENCE CHECK CONTEXT: {context}")
    return context


# def generate_answer(llm, question, results):
#     """
#     Step 4 — LLM reads the Neptune results and writes a natural language answer.

#     Args:
#         llm      : ChatBedrock instance from get_llm()
#         question : original user question
#         results  : list of result dicts from run_neptune_query()

#     Returns:
#         str : natural language answer
#     """
#     if not results:
#         return "No information found in the graph for that question."

#     ANSWER_PROMPT = PromptTemplate(
#         input_variables=["question", "context"],
#         template="""Answer the question using only the data provided below.
# Be concise and direct. If the data is empty say "No information found."

# Question: {question}
# Data: {context}

# Answer:"""
#     )

#     chain  = ANSWER_PROMPT | llm | StrOutputParser()
#     answer = chain.invoke({"question": question, "context": str(results)})
#     print(f"\n💬 ANSWER: {answer}")
#     return answer

def generate_answer(llm, question, results):
    """
    Step 4 — LLM writes a natural language answer from Neptune results.

    Fix 1: Prompt now explicitly instructs LLM to list EVERY row —
           no summarising, no skipping.
    Fix 2: Empty results now return "not related" instead of blank,
           after verifying both entities exist in the graph.

    Args:
        llm      : ChatBedrock instance from get_llm()
        question : original user question
        results  : list of result dicts from run_neptune_query()

    Returns:
        str : natural language answer
    """
    if not results:
        return "No information found in the graph for that question."

    ANSWER_PROMPT = PromptTemplate(
        input_variables=["question", "context"],
        template="""Answer the question using ONLY the data provided below.

STRICT RULES:
1. List EVERY item from the data — never skip or omit any row
2. If there are multiple rows, mention ALL of them in your answer
3. Be concise and direct
4. Do not add information not present in the data

Question: {question}
Data (ALL rows must be included in your answer): {context}

Answer:"""
    )

    chain  = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": str(results)})
    print(f"\n💬 ANSWER: {answer}")
    return answer


def ask(llm, graph, schema, question):
    """
    Master function — calls all 4 steps in sequence.

    Args:
        llm      : ChatBedrock instance from get_llm()
        graph    : NeptuneGraph object from connect_option_b()
        schema   : schema string from get_schema()
        question : plain English question

    Returns:
        str : natural language answer
    """
    print(f"\n{'═'*55}")
    print(f"❓ QUESTION: {question}")
    print(f"{'═'*55}")

    raw_query   = generate_cypher(llm, schema, question)     # Step 1
    clean_query = clean_cypher(raw_query)                    # Step 2
    results     = run_neptune_query(graph, clean_query, question)  # Step 3
    answer      = generate_answer(llm, question, results)    # Step 4

    return answer


def test_chain_invoke_with_custom_prompt(graph, question):
    """
    TEST FUNCTION — validates whether custom prompt alone prevents
    OPTIONAL MATCH and 2-block patterns in NeptuneOpenCypherQAChain.

    This is a controlled test — does NOT affect your working ask() flow.
    Run this, check the printed intermediate query, and compare with
    what clean_cypher would have produced.

    Args:
        graph    : NeptuneGraph object from connect_option_b()
        question : plain English question to test

    Returns:
        None — prints all intermediate steps for inspection
    """
    from langchain_aws import ChatBedrock
    from langchain_core.prompts import PromptTemplate
    from langchain_aws.chains import create_neptune_opencypher_qa_chain
    from langchain.callbacks.base import BaseCallbackHandler

    # ── Callback to capture generated query BEFORE Neptune runs it ────
    class QueryCaptureCallback(BaseCallbackHandler):
        def __init__(self):
            self.captured_query = "NOT CAPTURED YET"

        def on_llm_end(self, response, **kwargs):
            try:
                self.captured_query = response.generations[0][0].text.strip()
            except Exception:
                pass

    llm = ChatBedrock(
        model_id    = config.BEDROCK_MODEL_ID,
        region_name = config.AWS_REGION
    )

    # ── Custom prompt — Neptune-specific, no OPTIONAL MATCH ───────────
    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["schema", "question"],
        template="""You are an openCypher expert for Amazon Neptune graph database.
Generate ONE valid openCypher query for the question below.

Schema:
{schema}

STRICT NEPTUNE RULES — every rule is mandatory:
1. ONE flat MATCH clause only — never two MATCH clauses
2. ALL conditions in ONE WHERE block after ONE MATCH
3. NEVER use OPTIONAL MATCH — it causes MalformedQueryException in Neptune
4. Use undirected relationship (a)-[r]-(b) — no arrow direction
5. Case-insensitive: toLower(n.id) CONTAINS toLower('term')
6. Search both properties: toLower(n.id) CONTAINS toLower('x') OR toLower(n.name) CONTAINS toLower('x')
7. No spaces inside functions: toLower() not to Lower() not toLower ()
8. RETURN property values not node objects: n.id not n
9. Always end with LIMIT 50
10. Output ONLY raw Cypher — no explanation, no markdown, no commentary

CORRECT PATTERNS:

Find entity connections:
MATCH (a)-[r]-(b)
WHERE (toLower(a.id) CONTAINS toLower('alice') OR toLower(a.name) CONTAINS toLower('alice'))
RETURN a.id AS From, type(r) AS Relationship, b.id AS To
LIMIT 50

Find entity + filter by relationship:
MATCH (a)-[r]-(b)
WHERE (toLower(a.id) CONTAINS toLower('alice') OR toLower(a.name) CONTAINS toLower('alice'))
AND toLower(type(r)) CONTAINS toLower('friend')
RETURN a.id AS From, type(r) AS Relationship, b.id AS To
LIMIT 50

Find relationship between two entities:
MATCH (a)-[r]-(b)
WHERE (toLower(a.id) CONTAINS toLower('alice') OR toLower(a.name) CONTAINS toLower('alice'))
AND (toLower(b.id) CONTAINS toLower('bob') OR toLower(b.name) CONTAINS toLower('bob'))
RETURN a.id AS From, type(r) AS Relationship, b.id AS To
LIMIT 50

WRONG PATTERNS — never generate these:
MATCH (a)                        <- standalone MATCH with no relationship
OPTIONAL MATCH (a)-[r]-(b)      <- OPTIONAL MATCH not supported in Neptune
WHERE condition                  <- WHERE as separate clause after OPTIONAL MATCH

Question: {question}
Cypher Query:"""
    )

    chain    = create_neptune_opencypher_qa_chain(
        llm                       = llm,
        graph                     = graph,
        cypher_prompt             = CUSTOM_PROMPT,
        return_intermediate_steps = True,
        verbose                   = True
    )

    callback = QueryCaptureCallback()

    print(f"\n{'═'*60}")
    print(f"  TEST — chain.invoke() with custom prompt")
    print(f"  Question: {question}")
    print(f"{'═'*60}")

    try:
        response = chain.invoke(
            {"query": question},
            config={"callbacks": [callback]}
        )

        # ── Print captured query (from callback — before Neptune ran it)
        print(f"\n🤖 CAPTURED QUERY (from LLM, before Neptune):")
        print(f"{'─'*50}")
        print(callback.captured_query)
        print(f"{'─'*50}")

        # ── Print intermediate steps (from chain — after Neptune ran it)
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if "query" in step:
                    print(f"\n✅ QUERY THAT NEPTUNE EXECUTED:")
                    print(f"{'─'*50}")
                    print(step["query"].strip())
                    print(f"{'─'*50}")
                if "context" in step:
                    print(f"\n📦 NEPTUNE RESULT:")
                    print(f"  {str(step['context'])[:300]}")

        print(f"\n💬 ANSWER: {response.get('result', 'No answer')}")

        # ── Compare captured vs executed ──────────────────────────────
        print(f"\n{'─'*50}")
        print(f"🔍 COMPARISON:")
        captured = callback.captured_query
        executed = ""
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if "query" in step:
                    executed = step["query"].strip()

        if "OPTIONAL MATCH" in captured.upper():
            print(f"  ⚠️  LLM generated OPTIONAL MATCH — custom prompt NOT sufficient")
            print(f"  ✅ clean_cypher() is still needed")
        else:
            print(f"  ✅ LLM did NOT generate OPTIONAL MATCH — custom prompt worked")
            print(f"  🤔 Run more questions to confirm consistency")

        if captured.strip() == executed.strip():
            print(f"  ✅ Captured query = Executed query (no chain modification)")
        else:
            print(f"  ⚠️  Chain modified the query before sending to Neptune")

    except Exception as e:
        print(f"\n❌ chain.invoke() FAILED")
        print(f"\n🤖 QUERY CAPTURED BEFORE ERROR:")
        print(f"{'─'*50}")
        print(callback.captured_query)
        print(f"{'─'*50}")
        print(f"\n💥 ERROR: {str(e)[:300]}")

        if "OPTIONAL MATCH" in callback.captured_query.upper():
            print(f"\n⚠️  OPTIONAL MATCH found in captured query — this caused the error")
            print(f"✅ Confirms clean_cypher() is needed even with custom prompt")
        else:
            print(f"\n🔎 OPTIONAL MATCH not in query — error is from something else")




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
