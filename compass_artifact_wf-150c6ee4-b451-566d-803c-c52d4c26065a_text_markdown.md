# Replacing AWS Comprehend Medical for ICD-10-CM / RxNorm Inference in an LTD Clinical Intelligence Platform

## TL;DR
- **Comprehend Medical's low, often-wrong ICD-10-CM/RxNorm scores are a known, structural limitation, not a misconfiguration**: its ontology `Score` is an uncalibrated relevance/likelihood value returned as a ranked top-5 candidate list, and independent testing shows large accuracy variability across cloud clinical-NLP services. The fix is not another single black-box API but a **hybrid architecture**: clinical NER for span detection → retrieval over the official ICD-10-CM/RxNorm terminology files → a cross-encoder/LLM reranker → calibrated confidence → human-in-the-loop for low-agreement codes.
- **The single best recommendation is a self-hosted/in-VPC hybrid pipeline anchored on John Snow Labs Healthcare NLP (or an open MedCAT/GLiNER-biomed stack) for NER + entity resolution, plus a terminology-RAG reranker**, keeping Textract (with a VLM fallback for handwriting) and Claude-on-Bedrock as narrator. This beats Comprehend Medical on accuracy, calibration, auditability, customizability, and — at LTD volumes — cost, because Comprehend Medical bills **separately per API**, tripling per-character cost when you run DetectEntitiesV2 + InferICD10CM + InferRxNorm.
- **Do not adopt zero-shot LLM coding or "autonomous coding" RCM vendors as the source of truth**: in the Mount Sinai NEJM AI study, GPT-4 scored only 33.9% exact-match on ICD-10-CM zero-shot, and Fathom/Nym/CodaMetrix are full-service billing engines, not API components, built for provider revenue-cycle coding rather than payer LTD adjudication.

## Key Findings

1. **Why Comprehend Medical scores run low.** AWS documents that `ICD10CMConcept.Score` / `RxNormConcept.Score` is "the level of confidence that Amazon Comprehend Medical has that the entity is accurately linked to an ICD-10-CM/RxNorm concept" — returned as a **descending-ranked list of candidate concepts** (the top-5 pattern), not a calibrated probability. There is also a separate `LOW_CONFIDENCE` *trait* that AWS explicitly states "is not directly related to the confidence scores provided." Two design facts explain your 5–60% observations: (a) ontology linking is a hard 70,000+-class disambiguation problem, so probability mass is spread across near-neighbor codes; and (b) you are reading only the top-1 score when the correct code is frequently ranked #2–#5. Practitioners on AWS re:Post and Microsoft Q&A report the identical pattern on Azure — a documented case returned F32.9 for "major depressive disorder, recurrent, moderate," which should be F33.1 (splitting the phrase into three entities and mis-linking the first). **Wrong-code behavior is common to all the cloud clinical-NLP services**, not unique to AWS.

2. **Independent accuracy evidence is sobering and variable.** The most trustworthy neutral comparison — Hegde, Ninan, Dillman, Somasundaram et al., *"Evaluating Clinical NLP Services for Chest Radiograph Report Labeling,"* Journal of Imaging Informatics in Medicine (2026, DOI 10.1007/s10278-026-02148-y), on 95,008 pediatric chest-radiograph reports — found **assertion accuracy ranged from 50% (AWS Comprehend Medical) to 76% (John Snow Labs Spark NLP), with CheXpert/CheXbert at 56%**, concluding there is "substantial performance variability, emphasizing the need for validation." John Snow Labs' own (vendor, therefore biased) benchmarks claim ICD-10 entity-resolution top-3 accuracy of 82.7% vs Amazon 55.8% vs GPT-4 8.9% on RxNorm, and ICD-10-CM code extraction of 76% vs GPT-4 36% / GPT-3.5 26%; treat these skeptically because JSL wins its own tests, but the *direction* is corroborated by neutral literature.

3. **Zero-shot LLMs are poor direct coders.** Soroush, Glicksberg, Nadkarni, Klang et al., *"Large Language Models Are Poor Medical Coders,"* NEJM AI 1(5), 2024, evaluated 27,000+ codes from the Mount Sinai EHR and found **GPT-4's exact-match rate was 45.9% (ICD-9-CM), 33.9% (ICD-10-CM), and 49.8% (CPT)**, with models "often generating codes conveying imprecise or fabricated information." **Retrieval-augmented approaches change the picture dramatically:** Keith Kwan, *"Large Language Models Are Good Medical Coders, If Provided With Tools"* (arXiv 2407.12849), reports that a Retrieve-Rank ColBERT-v2 system "achieved 100% accuracy in predicting correct ICD-10-CM codes... outperforming the Vanilla LLM (GPT-3.5-turbo), which achieved only 6%" — though this was on a simplified 100-item single-term set and should be read as a proof-of-direction, not a production number. More realistically, a RAG-enhanced ED-coding study improved exact-match from 0.8%→17.6% (Qwen-2-7B) and had physician reviewers favor RAG-GPT-4 codes over provider-assigned codes; and *"Validation of 13,102 ICD-10-CM Codes Using a Large Language Model-Based System"* (medRxiv 2025; PubMed 41670956), on 865,079 MIMIC-IV codes, showed a GPT-4o validator **"achieving 93.6% accuracy, 95.4% sensitivity, and 85.2% specificity"** at judging whether an assigned code is correct. This is the core evidence base for the recommended hybrid pattern.

4. **Fine-tuned and open models are viable in-VPC.** MedCAT (CogStack) reports UMLS extraction F1 of ~0.45–0.79 out of the box and **>0.94 after hospital-specific fine-tuning**; GLiNER-biomed reaches state-of-the-art **zero-shot biomedical NER micro-F1 of 59.77%** and is lightweight/self-hostable; a Llama-3-70B fine-tuned on synthetic policy-aware data hit exact ICD-10 F1 0.704 (0.629 with evidence localization). These support a defensible, auditable, self-hosted stack.

5. **The handwriting problem is partly upstream at OCR.** In a 100-document PDF-parser benchmark, **Google Document AI scored 74.8% vs Textract 71.2% on handwritten content** — both "not reliable for handwritten content without significant human verification." Modern vision-language models are now competitive-to-superior on messy medical forms: Bassett et al. (WITS MIND Institute), *"From Handwriting to Structured Data: Benchmarking AI Digitisation of Handwritten Forms"* (arXiv 2604.16504), tested 17 models on real handwritten medical forms and found **the latest Google/OpenAI models "reach accuracies around 85% with weighted F1 scores ≃90% across the discrete or predefined fields"** (Gemini 3.1 best overall; GPT 5.4 lowest hallucination rate at 6%). A fine-tuned VLM (RAPTOR+) achieved ~96% reading accuracy **with grounded evidence bounding boxes**, preserving citation traceability.

6. **Cost: Comprehend Medical's per-API billing is the hidden driver.** Comprehend Medical bills per unit (100 chars) *per operation*; running DetectEntitiesV2 + InferICD10CM + InferRxNorm means **paying for the same characters three times**. NERe is $0.01/unit (first 1M/month), $0.005 (1M–2M), $0.001 (over 2M); SNOMED CT is $0.0075/$0.00375/$0.00075; ICD-10-CM and RxNorm ontology linking are separately billed line items priced above NERe (read the exact current per-unit rate off the live AWS pricing table / Pricing Calculator before finalizing budget). Tier boundaries are 1M/2M units, not the 10M used by standard Comprehend.

7. **HIPAA/deployment.** Comprehend Medical, Google Cloud Healthcare NLP API, and Azure Text Analytics for Health are all HIPAA-eligible under a BAA; Comprehend Medical does not train on customer data. Only **Azure (containers), John Snow Labs, and fully self-hosted MedCAT/GLiNER/scispaCy** can run entirely inside your VPC/on-prem with no PHI leaving your account. **Amazon A2I is no longer open to new customers**, so plan a Step Functions + custom review-UI (or SageMaker Ground Truth) human-in-the-loop instead.

## Details

### Landscape survey (by category)

**Incumbent cloud clinical-NLP APIs**
- **AWS Comprehend Medical** — NERe, PHI, and ICD-10-CM/RxNorm/SNOMED-CT linking; HIPAA-eligible; no custom models; ranked top-5 candidates; documented low/uncalibrated ontology scores; per-API billing. English only.
- **Google Cloud Healthcare Natural Language API** — maps to ICD-10-CM, SNOMED CT (US only), RxNorm, MeSH; returns UMLS CUIs with temporal/certainty/subject assessments; HIPAA BAA; English only; Google now steers new work toward Gemini/MedLM for extraction.
- **Azure AI Language – Text Analytics for Health** — NER, relation extraction, UMLS entity linking (surfaces ICD-10-CM, RxNorm, SNOMED CT via UMLS), assertion detection, FHIR output; **available as a container for in-VPC/on-prem**; requires you to separately license source vocabularies; documented wrong-code cases.

**Vendor NLP libraries**
- **John Snow Labs Healthcare NLP (Spark NLP)** — 100+ clinical NER models, 60+ entity-resolution models across 10+ terminologies (ICD-10-CM, ICD-10-PCS, CPT, RxNorm, SNOMED, LOINC, HCC), assertion status, relation extraction, trainable, deployable in-VPC via AWS/Azure Marketplace; fixed/PAYG license (AWS Marketplace hourly listings range from about $1.86 to $253.56/hr plus AWS usage, depending on tier). Highest published entity-resolution accuracy — but benchmarks are vendor-run; validate independently.

**Amazon-native adjacencies**
- **AWS HealthScribe** — ambient speech→SOAP with transcript-grounded citations and medical-term extraction; built for audio consults, not PDF/handwritten claim documents; not a coding engine.
- **AWS HealthLake** — FHIR store with integrated medical-NLP; useful as a normalized data layer downstream, not the extractor.

**Open-source / self-hosted**
- **MedCAT/CogStack** (self-supervised NER+L to UMLS/SNOMED, F1 up to >0.94 fine-tuned), **GLiNER-biomed** (zero-shot NER micro-F1 59.77%), **scispaCy/medspaCy** (UMLS linking, lighter accuracy: MedMentions strict F1 ~0.37), **Stanza biomedical**, **cTAKES/MetaMap/QuickUMLS/CLAMP** (mature but rules-based, lower ceiling). All run in-VPC, no per-character fees.

**Normalization/validation layer (not extractors)**
- **UMLS Metathesaurus / UTS**, **RxNav/RxNorm API**, **NLM ICD-10-CM tooling**, **CMS/CDC ICD-10-CM files** — use these to build the terminology index and to *validate* that any inferred code is real, billable, and current.

**Purpose-built autonomous-coding vendors** (Fathom, Nym, CodaMetrix, Solventum/3M 360 Encompass, Optum, AKASA, Regard, IMO, Averbis, Lexigram, Melax, Clinical Architecture) — mostly **full-service RCM engines**, priced per-chart/claim (Nym publicly ~$0.50–$1.00/claim; vendors report 90–98% automation/accuracy on *provider billing* charts), EHR-integrated, and oriented to reimbursement, not payer LTD adjudication. IMO and Clinical Architecture/Lexigram/Averbis are more API/terminology-oriented. These fit only if you want to outsource the whole coding function; they don't fit a "grounding layer for an LLM summarizer" role and reduce auditability/control.

**LLM-based extraction** — GPT/Claude/Gemini and medical models (MedGemma, Med-PaLM 2, Palmyra-Med, OpenBioLLM, Meditron, BioMistral). Strong at *span extraction and narration*, weak at *direct code assignment* zero-shot; strong again when wrapped in retrieval + reranking.

### Handwriting / OCR layer
Keep Textract for born-digital and clean scans (native S3/Lambda integration, handwriting included at no surcharge, bounding boxes for citation grounding). Add a **VLM fallback** (Claude/Gemini/GPT vision on the page image) for pages Textract flags as low-confidence or handwriting-heavy. A VLM-first approach is now competitive-to-superior on messy forms but must be constrained to **preserve bounding-box/offset grounding** (as RAPTOR+ demonstrates with grounded evidence boxes) so the downstream Claude Citations API stays auditable. Do not go VLM-only across the whole corpus — cost, latency, and grounding risk are higher.

### Confidence-score handling
- Comprehend Medical/Azure/Google scores are **not comparable across vendors** and are not calibrated probabilities; do not threshold them as if they were.
- Build calibration on *your* gold set (Platt scaling / isotonic regression) so a "0.9" means 90% empirical precision.
- Use **top-k, not top-1**: retrieve candidate codes, rerank, and treat the margin between #1 and #2 as an uncertainty signal.
- **Ensemble/cross-validation**: agreement between two independent extractors (e.g., your NER+RAG stack and Comprehend Medical, or NER+RAG and an LLM validator) raises effective confidence; disagreement routes to human review. The GPT-4o validator result (93.6% at judging code correctness) supports using an independent second model as a checker.
- **Keep the LLM honest**: pass each fact to Claude as a structured object `{text_span, char_offsets, page, bbox, code, code_description, confidence_band}` and instruct it to render uncertainty explicitly (e.g., "possible diagnosis, low confidence" / omit if below threshold), never to assert a low-confidence code as fact. Suppress or flag anything below the calibrated threshold; route it to human review rather than laundering it into authoritative prose.

### Compliance & enterprise constraints
PHI for a US insurer requires a BAA and, ideally, no PHI leaving your VPC. Self-hosted (JSL/MedCAT/GLiNER) and Azure containers meet the strictest "PHI never leaves our account" bar; Comprehend Medical/Google/Bedrock are HIPAA-eligible under BAA but are managed services. For claims adjudication, coding must be **defensible and auditable**: every code must trace to a text span → page → bounding box, carry a calibrated confidence, and be validated against the official current-year terminology. Low-confidence or wrong codes on an LTD benefit decision are a material compliance and litigation risk; the architecture must fail safe to human review.

## Recommendations

### Comparison table (top options)

| Option | Code accuracy (evidence) | Ontologies | Handwriting/OCR | Confidence quality | Scale/throughput | Cost @ LTD volume | HIPAA/deploy | Integration w/ AWS | Auditability | Lock-in |
|---|---|---|---|---|---|---|---|---|---|---|
| **Comprehend Medical (incumbent)** | Low/variable; assertion ~50% (indep. CXR study); wrong-code reports | ICD-10-CM, RxNorm, SNOMED | Relies on Textract; no native handwriting coding | Uncalibrated top-5 relevance | Managed, async multi-min latency | **High — triple per-API billing** | HIPAA-eligible, managed (PHI in AWS acct) | Native | Medium (spans+scores) | High (AWS) |
| **Hybrid NER + terminology-RAG + reranker (RECOMMENDED)** | Highest ceiling; RAG lifts exact-match markedly; validator ~93.6% | Any (you index ICD-10-CM/RxNorm/SNOMED/LOINC/CPT) | Textract + VLM fallback w/ grounding | Calibrated on your data; top-k + margin | Self-hosted GPUs, batch | **Lowest at volume (fixed infra)** | Fully in-VPC | High (build on AWS) | **Highest (span→code→source, validated)** | Low |
| **John Snow Labs Healthcare NLP** | Highest *published* (vendor-run) resolution accuracy; independent CXR assertion 76% | ICD-10-CM/PCS, RxNorm, SNOMED, CPT, LOINC, HCC | Visual NLP OCR add-on | Trainable/calibratable | In-VPC Spark, scalable | License + GPU (flat) | In-VPC (Marketplace) | High | High (trainable, transparent) | Medium |
| **Azure Text Analytics for Health** | Comparable to AWS; wrong-code cases documented | UMLS→ICD-10-CM/RxNorm/SNOMED | Azure Doc Intelligence | Per-entity scores | Managed or container | Per-record; container option | HIPAA; **container in-VPC** | Cross-cloud effort | Medium | Medium |
| **Google Healthcare NLP API** | Comparable; fewer entities extracted in indep. test | ICD-10, SNOMED(US), RxNorm, MeSH | Google Document AI | Temporal/certainty scores | Managed | Per-unit | HIPAA BAA | Cross-cloud effort | Medium | High (GCP) |
| **Zero-shot LLM (GPT/Claude/Gemini)** | Poor direct coding (GPT-4 33.9% ICD-10-CM) | Any (unreliable) | VLM native | Uncalibrated/hallucination risk | API | Token-based | Bedrock HIPAA | Native (Bedrock) | Low unless grounded | Low |
| **Autonomous coding vendors (Fathom/Nym/CodaMetrix)** | 90–98% *on provider billing charts* (vendor) | ICD-10, CPT, HCC | Vendor pipeline | Vendor-internal | SaaS | Per-claim (~$0.50–1.00) | BAA; SaaS | Integration project | Vendor audit trail | High (vendor) |

### Top 3 ranked
1. **Hybrid NER → terminology-RAG → reranker → calibrated confidence → HITL** (best accuracy ceiling, auditability, cost, control).
2. **John Snow Labs Healthcare NLP as the NER+resolution engine inside that hybrid** (fastest path to production accuracy, in-VPC, trainable) — the pragmatic way to implement #1 without building NER from scratch.
3. **Retain Comprehend Medical as a second ensemble voter / cold-start baseline** (cheap agreement signal to raise effective confidence and de-risk migration), not as source of truth.

### The single best approach — and why it beats Comprehend Medical
**Recommended:** a self-hosted, in-VPC **hybrid entity-resolution pipeline**: (1) clinical **NER** (John Snow Labs Healthcare NLP, or open MedCAT/GLiNER-biomed) detects condition/medication spans with character offsets; (2) each span is embedded and used to **retrieve top-k candidate codes from an index built on the official current-year ICD-10-CM and RxNorm files** (OpenSearch or Aurora pgvector); (3) a **cross-encoder or LLM reranker** (Claude on Bedrock) selects and justifies the best code using the surrounding sentence context; (4) codes are **validated** against RxNav/UMLS/CMS files (real, billable, current); (5) confidence is **calibrated** on your gold set, and low-agreement/low-confidence codes route to **human review**; (6) Claude narrates SOAP summaries grounded only on the structured, code-linked, offset-carrying facts.

**Head-to-head vs Comprehend Medical, for this exact LTD use case:**
- **Accuracy & wrong codes**: replaces a fixed 70k-class classifier (top-1 in the ~34–56% regimes seen in NEJM AI and the pediatric-CXR study) with retrieval that *guarantees the candidate set contains real codes* plus a context-aware reranker — directly attacking your "codes are simply wrong" problem, and adding a validator that independent MIMIC-IV evidence shows reaches ~93.6% accuracy at catching wrong codes.
- **Confidence**: swaps AWS's uncalibrated relevance score for a probability calibrated on your own documents, plus a top-1/top-2 margin signal — so a "90%" threshold is meaningful.
- **Cost**: eliminates triple per-API billing; at LTD volume, fixed GPU + license economics beat per-character SaaS (see cost model below).
- **Auditability**: every code carries span→page→bbox→candidate-list→reranker-rationale→validation status — defensible for a benefit decision, which Comprehend Medical's opaque score cannot match.
- **Control**: trainable on your APS/handwritten-form corpus; Comprehend Medical offers no custom models.
- **What Comprehend Medical still wins**: zero-ops, native integration, no GPU management, and a genuinely useful *cheap second opinion* — which is why it stays in the ensemble.

### Target architecture (fits your AWS stack)
1. **Ingest**: PDFs → S3 → Step Functions orchestration.
2. **OCR**: Textract (async, retain page/line/bbox); Lambda routes low-confidence/handwriting pages to a **VLM (Claude/Gemini vision on Bedrock)** with a grounding prompt that returns text + approximate boxes.
3. **NER**: JSL Healthcare NLP (or MedCAT/GLiNER) on GPU (SageMaker endpoint or ECS/EKS on g5 instances) → spans with char offsets mapped back to page/bbox.
4. **Terminology retrieval**: embed spans; hybrid BM25 + dense ANN search over ICD-10-CM/RxNorm/SNOMED indexes in **OpenSearch or Aurora pgvector**; return top-k.
5. **Rerank & validate**: Claude reranker selects code + rationale from the retrieved candidate set only; validate against RxNav/UMLS/CMS; attach calibrated confidence.
6. **Merge**: canonical model merges by overlapping offsets, flattens attributes, applies calibrated thresholds; optionally add Comprehend Medical as a second voter (agreement → boost, disagreement → review).
7. **HITL**: below-threshold or disagreement → Step Functions + custom review UI (or SageMaker Ground Truth), since A2I is closed to new customers; reviewed codes feed back as training data.
8. **Persist**: DynamoDB (fact store), Kafka/MSK (events), S3 (artifacts).
9. **Summarize**: Claude Citations API consumes structured facts `{span, offsets, page, bbox, code, description, confidence_band, evidence}` so page-level citations remain exact; the prompt enforces uncertainty language and suppression rules.

### Migration & validation path
- **Gold set**: have certified coders annotate 300–1,000 representative pages (physician notes, APS, handwritten forms, med lists, labs) with span-level entities and *correct* ICD-10-CM/RxNorm codes, plus page/bbox.
- **Metrics**: span-level P/R/F1; code-level exact-match and top-k accuracy (k=1,3,5); calibration (reliability curves/ECE); handwriting-subset accuracy; and downstream summary faithfulness.
- **Bake-off**: run Comprehend Medical (baseline), JSL, Azure container, and your hybrid over the gold set; compare accuracy, calibration, cost/claim, and latency. Commit only when the hybrid clears your ~90% precision bar at an acceptable review rate.
- **Rollout**: shadow-mode the hybrid alongside Comprehend Medical, measure agreement, then cut over with Comprehend Medical retained as a voter.

### Worked cost model (state your own numbers)
Assumptions (substitute yours): 20 documents/claim × 5 pages/doc = **100 pages/claim**; 1,700 chars/page = **170,000 chars = 1,700 units/claim**; **10,000 claims/month** → **1,000,000 pages/month** and **17,000,000 units/month per API**.
- **Comprehend Medical, triple API** (NERe + ICD-10-CM + RxNorm): at the *NERe* tier rates alone, one API over 17M units ≈ 1M×$0.01 + 1M×$0.005 + 15M×$0.001 = **$30,000/month**; three APIs ≈ **$90,000/month** (ICD-10-CM and RxNorm are priced above NERe, so the true figure is higher — verify live rates). Plus Textract basic OCR 1M pages × $1.50/1k = **$1,500/month** (or $50/1k = $50,000/month if AnalyzeDocument Forms is needed).
- **Self-hosted hybrid**: e.g., 4× g5.2xlarge (A10G) at ~$1.212/hr ≈ **$3,500/month** running 24×7 (fewer with batch/autoscaling) + JSL license (flat) + embedding/reranker Bedrock tokens (small, batchable at −50%; Claude 3.5/3.7 Sonnet ~$3/1M input, $15/1M output; Haiku ~$0.80–1.00/1M input) + OpenSearch/pgvector. Even generously, this lands **well under Comprehend Medical's triple-API bill at this volume** — break-even is roughly a few hundred thousand claims/year, below which managed APIs may still be cheaper.
- **Cost levers**: entity-based routing (only send condition/med spans to retrieval), dedup/caching of resolved codes (LTD corpora are highly repetitive), batch inference, VLM only on flagged pages, and tiering (cheap NER first, expensive reranker only on ambiguous spans).

## Caveats
- **Vendor benchmarks are biased**: John Snow Labs' and autonomous-coding vendors' accuracy figures come from their own tests; the independent pediatric-CXR study (JIIM 2026) is the most trustworthy neutral comparison and shows only 50–76% assertion accuracy across all tools — treat any "95%+" claim as unverified until you reproduce it on your corpus.
- **The Retrieve-Rank "100%" figure** is on a simplified 100-item single-term set and is a proof-of-direction, not a production expectation; real multi-condition coding is materially harder.
- **The exact ICD-10-CM/RxNorm ontology-linking per-unit rates must be read off the live AWS pricing table/Calculator**; they are separate line items above NERe and drive the cost case even harder than modeled here.
- **Self-hosting shifts burden to your team** (MLOps, GPU capacity, model updates each ICD-10-CM annual release); if you lack that capability, JSL-managed-in-VPC or Azure containers reduce it.
- **What would change the recommendation**: if your monthly volume is small (managed per-API cost stays modest) *and* your accuracy bar can tolerate human review on most codes, staying on Comprehend Medical + HITL is simplest; if you want to *outsource coding entirely* and can accept a SaaS black box, an autonomous-coding vendor may win; if handwriting dominates and grounding can be relaxed, a fine-tuned VLM-first pipeline could displace the OCR+NER split.
- **A2I is closed to new customers** — budget for a custom HITL build.
- **LLM reranker hallucination** is bounded but not zero; constrain it to choose only from the retrieved, validated candidate set and log its rationale.