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
def visualize_graph(client, output_file="neptune_graph.html"):
    """
    Reads all nodes and relationships from Neptune and renders
    an interactive visual graph using PyVis.
    Saves output as an HTML file and opens it in the browser.

    Args:
        client      : boto3 neptunedata client
        output_file : name of the HTML file to save
    """
    from pyvis.network import Network

    print(f"\n[VISUALIZE] Building interactive graph → {output_file}...")

    # Colour per node label
    COLOUR_MAP = {
        "Patient":      "#4A90D9",
        "Doctor":       "#27AE60",
        "Hospital":     "#E67E22",
        "Claim":        "#E74C3C",
        "Diagnosis":    "#9B59B6",
        "Procedure":    "#F39C12",
        "Insurer":      "#1ABC9C",
        "Person":       "#4A90D9",
        "Organization": "#E67E22",
    }
    DEFAULT_COLOUR = "#95A5A6"

    # Fetch all nodes
    node_result = client.execute_open_cypher_query(
        openCypherQuery="MATCH (n) RETURN labels(n)[0] AS label, n AS node LIMIT 200"
    )
    # Fetch all relationships
    rel_result = client.execute_open_cypher_query(
        openCypherQuery="""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS from_label, a AS from_node,
                   type(r)      AS rel_type,
                   labels(b)[0] AS to_label,   b AS to_node
            LIMIT 300
        """
    )

    # Build PyVis network
    net = Network(
        height    = "750px",
        width     = "100%",
        bgcolor   = "#1a1a2e",
        font_color= "white",
        notebook  = False
    )
    net.barnes_hut()    # physics layout — nodes naturally repel each other

    added_nodes = set()

    # Add nodes
    for row in node_result.get("results", []):
        label    = row.get("label", "Unknown")
        node_obj = row.get("node", {})
        if not isinstance(node_obj, dict):
            continue
        props    = {k: v for k, v in node_obj.items() if not k.startswith("~")}
        node_id  = str(props.get("id", list(props.values())[0] if props else id(node_obj)))
        tooltip  = "\n".join(f"{k}: {v}" for k, v in props.items())

        if node_id not in added_nodes:
            net.add_node(
                node_id,
                label = node_id,
                title = tooltip,            # shown on hover
                color = COLOUR_MAP.get(label, DEFAULT_COLOUR),
                size  = 22,
                font  = {"size": 13, "color": "white"}
            )
            added_nodes.add(node_id)

    # Add edges
    for row in rel_result.get("results", []):
        from_obj = row.get("from_node", {})
        to_obj   = row.get("to_node",   {})
        rel_type = row.get("rel_type",  "")
        if not isinstance(from_obj, dict) or not isinstance(to_obj, dict):
            continue
        from_props = {k: v for k, v in from_obj.items() if not k.startswith("~")}
        to_props   = {k: v for k, v in to_obj.items()   if not k.startswith("~")}
        from_id    = str(from_props.get("id", list(from_props.values())[0] if from_props else None))
        to_id      = str(to_props.get("id",   list(to_props.values())[0]   if to_props   else None))

        if from_id and to_id and from_id in added_nodes and to_id in added_nodes:
            net.add_edge(from_id, to_id, label=rel_type, color="#aaaaaa", arrows="to")

    net.save_graph(output_file)
    print(f"  ✅ Graph saved → {output_file}")
    print(f"  Opening in browser...")
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
