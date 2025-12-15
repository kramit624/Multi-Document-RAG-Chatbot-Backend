import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- Be CONCISE and DIRECT - give short, to-the-point answers
- Do NOT explain where you found the information unless asked
- Do NOT quote large sections of text
- If the answer is not in the context, say "This information is not present in the document."
- Do not make up information

Answer:"""
)

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that creates clear, medium-length summaries.

Context:
{context}

Question: {question}

Instructions:
- Create a CONCISE but INFORMATIVE summary (4-6 paragraphs maximum)
- Focus on the MAIN themes and key points only
- Write in a natural, flowing paragraph style - NO bullet points or lists
- Do NOT include section headers or formatting
- Cover the most important ideas without excessive detail
- Do not include references or citations
- Do not make up information not present in the context

Summary:"""
)

def answer_question(question: str):
    if not os.path.exists(VECTORSTORE_DIR):
        return "No documents uploaded yet. Please upload a PDF first."

    try:
        # Check if user is asking for a summary
        summary_keywords = ['summarize', 'summarise', 'summary', 'overview', 'brief', 'gist', 'main points']
        is_summary_request = any(keyword in question.lower() for keyword in summary_keywords)
        
        embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=256
        )

        print("📂 Loading vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # For summaries, get MORE chunks to cover the whole document
        if is_summary_request:
            print("📝 Summary request detected - fetching more chunks...")
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 12}  # Get more chunks for better summary
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )

        print(f"🔍 Searching for: {question}")
        docs = retriever.invoke(question)

        print(f"📋 Found {len(docs)} relevant chunks")

        if not docs:
            return "This information is not present in the document."

        # Debug: print what we found
        for i, doc in enumerate(docs):
            print(f"\n--- Chunk {i+1} (score/relevance) ---")
            print(doc.page_content[:200] + "...")

        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # Choose the right prompt based on request type
        if is_summary_request:
            final_prompt = SUMMARY_PROMPT.format(
                context=context,
                question=question
            )
            max_tokens = 800  # Medium length for summaries
            system_msg = "You are a helpful assistant that creates clear, medium-length summaries in natural paragraph form without bullet points or excessive formatting."
        else:
            final_prompt = PROMPT.format(
                context=context,
                question=question
            )
            max_tokens = 500
            system_msg = "You are a concise assistant. Give short, direct answers without explanations unless specifically asked."

        print(f"🤖 Sending to Groq... (Summary mode: {is_summary_request})")
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens
        )

        answer = response.choices[0].message.content.strip()
        print(f"✅ Answer: {answer[:100]}...")
        
        return answer

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return f"Error processing question: {str(e)}"
