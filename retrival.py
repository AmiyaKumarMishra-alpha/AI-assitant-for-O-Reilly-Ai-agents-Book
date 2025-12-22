from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama

VECTOR_DB_PATH = "./vector_db"
top_k = 20

# Confidence calibration (tunable)
CONF_WEIGHT_TOP = 0.8      # weight for the top similarity in combined score
CONF_WEIGHT_DOM = 0.2      # weight for dominance (top / sum(scores))
CONF_TOP_HIGH = 0.5        # top >= this -> High
CONF_TOP_MED = 0.25        # top >= this -> Medium
CONF_DOM_HIGH = 0.5        # dominance >= this -> High
CONF_DOM_MED = 0.25        # dominance >= this -> Medium

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

#load faiss vector store
vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieve_similar_chunks(query):
    # prefer the API that returns (Document, score) tuples
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception:
        # fallback to the older similarity_search which returns Documents only
        docs = vector_store.similarity_search(query, k=top_k)
        results = [(doc, None) for doc in docs]

    # First, collect docs and raw scores so we can normalize as a group
    items = []
    raw_scores = []
    for item in results:
        if isinstance(item, tuple) and len(item) >= 2:
            doc, raw_score = item[0], item[1]
        else:
            doc, raw_score = item, None
        items.append((doc, raw_score))
        if raw_score is not None:
            try:
                raw_scores.append(float(raw_score))
            except Exception:
                raw_scores.append(None)

    # Compute normalized similarities from raw_scores in a robust way
    norm_values = []
    if raw_scores:
        import math
        # Filter out any None values
        filtered = [s for s in raw_scores if s is not None]
        if filtered:
            max_abs = max(abs(s) for s in filtered)
            # Case 1: positive scores with max > 1 -> treat as unnormalized similarities and scale by max
            if all(s >= 0 for s in filtered) and max(filtered) > 1:
                max_s = max(filtered)
                norm_values = [s / max_s for s in filtered]
            # Case 2: distances (negative or large magnitude) -> convert via exp(-|s|/scale)
            elif any(abs(s) > 1 for s in filtered) or any(s < 0 for s in filtered):
                scale = max_abs or 1.0
                norm_values = [math.exp(-abs(s) / scale) for s in filtered]
            # Case 3: already in [0,1]
            else:
                norm_values = [float(s) for s in filtered]

    # Build the final ranked chunks, assigning normalized scores back to items (None preserved)
    ranked_chunks = []
    idx = 0
    for doc, raw_score in items:
        norm_score = None
        if raw_score is not None:
            # safe-guard: if normalization failed, leave None
            if idx < len(norm_values):
                norm_score = norm_values[idx]
            idx += 1

        ranked_chunks.append({
            "content": getattr(doc, "page_content", str(doc)),
            "metadata": getattr(doc, "metadata", {}),
            "score": norm_score,
        })
    return ranked_chunks

def getFinalPrompt(user_query):
    system_prompt = """You are a helpful AI assistant. Use ONLY the provided retrieved chunks (labeled [chunk-N]) and the user's query exactly as given: {user_query}.
                    Do NOT use any external knowledge or make things up. Provide a concise, factual answer grounded only in the provided chunks.
                    Return ONLY the final answer text — no chain-of-thought, analysis, or extra commentary. Do NOT mention the book, the author, or apologize.
                    If the chunks do not contain relevant information, reply exactly: "I could not find any information on that topic in the provided context."
                    If you include citations, cite at most two chunks in the format [source: chunk-N]. Keep the answer under 200 words. Avoid profanity and harmful content."""
    return system_prompt.format(user_query=user_query)

#load llm model and generate answer
llm = Ollama(
    model="llama3.1:8b",    temperature=0.8
)


def generate_answer(prompt, chunks):
    # Build a structured input that labels chunks so the model can cite them
    chunk_texts = []
    for i, c in enumerate(chunks, start=1):
        score_part = f" (score: {c['score']:.4f})" if c.get('score') is not None else ""
        chunk_texts.append(f"[chunk-{i}]{score_part}: {c['content']}")

    combined_input = prompt + "\n\n" + "\n\n".join(chunk_texts)

    # Call the LLM
    output = llm(combined_input)

    # Confidence calculation: combine absolute top-similarity with relative dominance
    # Rationale: softmax-only can yield low probabilities when many similar chunks exist.
    import math
    valid_scores = [c['score'] for c in chunks if c.get('score') is not None]
    confidence_score = 0.0
    confidence = "Low"

    if valid_scores:
        top = max(valid_scores)
        ssum = sum(valid_scores)
        dominance = top / (ssum if ssum > 0 else 1e-12)

        # Combined metric (weights are tunable): favor absolute similarity but consider dominance
        confidence_score = CONF_WEIGHT_TOP * top + CONF_WEIGHT_DOM * dominance

        # Map to labels using configurable thresholds
        if top >= CONF_TOP_HIGH or dominance >= CONF_DOM_HIGH or confidence_score >= 0.6:
            confidence = "High"
        elif top >= CONF_TOP_MED or dominance >= CONF_DOM_MED or confidence_score >= 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"

    print(f"Confidence Level: {confidence} (score={confidence_score:.3f}, top={top if valid_scores else 'N/A'}, dominance={dominance if valid_scores else 'N/A'})")

    # if confidence is low, refused to give a output and follow a safe fallback
    if confidence == "Low":
        output  = "Disclaimer: The confidence level of this answer is low as the retrieved information may not be sufficient or relevant. Please verify the information from authoritative sources."
    return output

if __name__ == "__main__":
    query = input("Enter your question: ")
    chunks = retrieve_similar_chunks(query)

    if not chunks:
        print("No relevant chunks found.")
    else:
        final_prompt = getFinalPrompt(query)
        # only pass  top 5 results from chunks out of 20
        chunks = chunks[:5]
        answer = generate_answer(final_prompt, chunks)
        print("Answer:", answer)