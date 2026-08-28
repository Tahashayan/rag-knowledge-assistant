import sys, types
import os
from dotenv import load_dotenv

load_dotenv()

# --- stub must come before ANY ragas import ---
import langchain_community.chat_models as chat_models
fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
fake_vertexai.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai
chat_models.vertexai = fake_vertexai
# ------------------------------------------------

import asyncio
import time
import pandas as pd
from openai import AsyncOpenAI as OpenAICompatClient, RateLimitError  # used to hit Groq's OpenAI-compatible endpoint
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
import qdrant_client
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision

cohere_api_key = os.environ["COHERE_API"]

# ==========================================================
# 1. SETUP THE "STUDENT" (Your LlamaIndex Pipeline)
# ==========================================================
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Groq(model="openai/gpt-oss-120b", api_key=os.environ["GROQ_API_KEY"])

# ==========================================================
# 2. SETUP THE "TEACHER" (Ragas judge LLM)
# ==========================================================
# ragas' collections metrics (Faithfulness/ContextPrecision) require the new
# InstructorLLM interface, built via llm_factory + a native provider client.
# LangChain wrappers like ChatGroq are no longer accepted directly, and the
# installed ragas version's "groq" adapter is currently broken (it tries to
# patch the client as if it were Anthropic's messages.create API). Groq
# exposes an OpenAI-compatible endpoint, so we route through that instead
# using provider="openai" — this path is well supported by Instructor.
groq_openai_compat_client = OpenAICompatClient(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
judge_llm = llm_factory("openai/gpt-oss-120b", provider="openai", client=groq_openai_compat_client)


# ==========================================================
# 3. CONNECT TO DATABASE (Outside the loop!)
# ==========================================================
client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
    grpc_port=6334,
    prefer_grpc=True,  
    timeout=600,        
)
collection_name = "enterprise_docs_v2"
vector_store = QdrantVectorStore(client=client, collection_name=collection_name, enable_hybrid=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

filters = MetadataFilters(filters=[MetadataFilter(key="tenant_id", value="company_A")])
query_engine = index.as_retriever(filters=filters, similarity_top_k=20)
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=5)


# ==========================================================
# 4. TAKE THE EXAM
# ==========================================================
test_data = [
    {
        "question": "What are the six pillars of the AWS Well-Architected Framework?",
        "ground_truth": "The six pillars are Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, and Sustainability."
    },
    {
        "question": "What were Apple's total net sales in 2023?",
        "ground_truth": "I'm sorry, I don't have enough information in the company documents to answer that."    
    }
]

data_samples = []

for item in test_data:
    current_question = item["question"]
    current_ground_truth = item["ground_truth"]
    
    print(f"Testing Question: {current_question}")
    
    nodes = query_engine.retrieve(current_question)
    reranked_nodes = cohere_rerank.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=current_question))

    contexts_list = []
    context_str_for_prompt = ""
    
    for node in reranked_nodes:
        contexts_list.append(node.text)
        context_str_for_prompt += f"Content: {node.text}\n\n" 
    
    prompt = f"""You are a helpful corporate assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say 'I do not know'. 
    CONTEXT: 
    {context_str_for_prompt}
    QUESTION: 
    {current_question} 
    ANSWER:
    """
    
    response = Settings.llm.complete(prompt)
    ai_answer = response.text
    
    data_samples.append({
        "user_input": current_question,
        "response": ai_answer,
        "retrieved_contexts": contexts_list,
        "reference": current_ground_truth
    })
    
print("\nAll questions tested! Handing over to RAGAS for grading...")

# ==========================================================
# 5. GRADE THE EXAM (Outside the loop!)
# ==========================================================
# This installed ragas version's evaluate() runner does an isinstance check
# that the new `collections` metric classes fail (a version-mismatch bug
# between `ragas.metrics.collections` and `ragas.evaluate`). So instead of
# going through evaluate(), we score each sample directly with the metrics'
# own async .ascore() method - the same API ragas' own docs use.
faithfulness_metric = Faithfulness(llm=judge_llm)
context_precision_metric = ContextPrecision(llm=judge_llm)

async def score_with_retry(coro_fn, max_attempts=6, base_delay=5):
    """Retry an ascore() call with exponential backoff on Groq's TPM rate limit."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except RateLimitError as e:
            if attempt == max_attempts:
                raise
            wait_time = base_delay * attempt
            print(f"  Rate limited (attempt {attempt}/{max_attempts}). Waiting {wait_time}s...")
            await asyncio.sleep(wait_time)

async def grade_all(samples):
    graded_rows = []
    for sample in samples:
        faithfulness_result = await score_with_retry(
            lambda: faithfulness_metric.ascore(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            )
        )
        # small pause between calls so we don't immediately re-trip the TPM limit
        await asyncio.sleep(3)
        context_precision_result = await score_with_retry(
            lambda: context_precision_metric.ascore(
                user_input=sample["user_input"],
                retrieved_contexts=sample["retrieved_contexts"],
                reference=sample["reference"],
            )
        )
        await asyncio.sleep(3)
        graded_rows.append({
            "user_input": sample["user_input"],
            "response": sample["response"],
            "reference": sample["reference"],
            "faithfulness": faithfulness_result.value,
            "context_precision": context_precision_result.value,
        })
    return graded_rows

graded_rows = asyncio.run(grade_all(data_samples))
result_df = pd.DataFrame(graded_rows)

print("\n--- FINAL REPORT CARD ---")
print(result_df)

client.close()