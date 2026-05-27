# AWS Neptune Full PoC — Medical & Claims Graph

## Project Structure
```
neptune_full_poc/
├── main.py           ← All PoC code — run this
├── config.py         ← Reads .env
├── .env              ← Your Neptune endpoint and AWS credentials
├── requirements.txt  ← pip dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

Edit `.env` — set your Neptune endpoint and AWS credentials.

## Run
```bash
python main.py
```

## Sections in main.py

| Section | Function | What it does |
|---------|----------|--------------|
| 1A | connect_option_a()  | Connect via boto3 neptunedata |
| 1B | connect_option_b()  | Connect via LangChain NeptuneGraph |
| 2A | write_graph_documents_option_a() | Write via boto3 |
| 2B | write_graph_documents_option_b() | Write via LangChain |
| 3A | query_option_a()    | Query via boto3 |
| 3B | query_option_b()    | Query via LangChain |
| 4  | visualize_graph()   | PyVis HTML visualization |
| 5  | nlp_query()         | Plain English → openCypher → answer |
| 6  | verify()            | Cross-check node/rel counts |
| 7  | clean_graph()       | Wipe all data (reset) |

## Official Docs
- boto3 neptunedata  : https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/neptunedata/client/execute_open_cypher_query.html
- LangChain Neptune  : https://python.langchain.com/api_reference/aws/graphs/langchain_aws.graphs.neptune_graph.NeptuneGraph.html
- NLP QA Chain       : https://python.langchain.com/docs/integrations/graphs/amazon_neptune_open_cypher/
