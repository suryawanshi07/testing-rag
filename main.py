import weaviate
from weaviate.auth import AuthApiKey
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import embed_anything
from embed_anything import EmbeddingModel, WhichModel
from typing import Dict, List, Union, Any
from dotenv import load_dotenv
import os
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
import uvicorn
import time

load_dotenv()

app = FastAPI()

# Initialize Weaviate client
client = weaviate.Client(
    url="https://totxgehfsvinwfz33rkxew.c0.australia-southeast1.gcp.weaviate.cloud",
    auth_client_secret=AuthApiKey("8jGoywAvKvRLlpBYHgGs0bR4AE6A0Z5CklJB"),
)

# Initialize the Jina embedding model
jina_model = EmbeddingModel.from_pretrained_hf(
    WhichModel.Jina, model_id="jinaai/jina-embeddings-v2-small-en"
)

class JinaEmbeddingWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text):
        return embed_anything.embed_query([text], self.model)[0].embedding

jina_embeddings = JinaEmbeddingWrapper(jina_model)

def recursive_chunks(content: str) -> List[str]:
    """Split content into chunks using LangChain's recursive splitter."""
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=4000,
        chunk_overlap=200
    )
    
    documents = md_splitter.create_documents([content])
    chunks = [doc.page_content for doc in documents]
    
    return chunks

def store_chunks_in_weaviate(chunks: List[str]) -> Dict[str, Any]:
    """Store chunks in Weaviate with embeddings."""
    stored_count = 0
    batch_size = 50  # Reduced batch size
    retry_count = 3
    
    try:
        # Configure batch settings
        client.batch.configure(
            batch_size=batch_size,
            dynamic=True,
            timeout_retries=3,
            connection_error_retries=3
        )
        
        print(f"\nStarting to process {len(chunks)} chunks...")
        
        # Process chunks in smaller batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            retry = 0
            print(f"\nProcessing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}")
            
            while retry < retry_count:
                try:
                    with client.batch as batch:
                        for chunk_num, chunk in enumerate(batch_chunks, 1):
                            if not chunk.strip():
                                continue
                                
                            print(f"Processing chunk {i + chunk_num}/{len(chunks)}")
                            print(f"Chunk content preview: {chunk[:100]}...")
                                
                            # Get embedding for the chunk
                            embedding = embed_anything.embed_query([chunk], jina_model)[0].embedding
                            
                            # Store in Weaviate
                            properties = {
                                "text": chunk
                            }
                            
                            batch.add_data_object(
                                data_object=properties,
                                class_name="Nishat",
                                vector=embedding
                            )
                            stored_count += 1
                            print(f"Successfully stored chunk {i + chunk_num}")
                    
                    print(f"Successfully completed batch {i//batch_size + 1}")
                    break  # If successful, break retry loop
                    
                except Exception as e:
                    retry += 1
                    print(f"Batch failed (attempt {retry}/{retry_count}): {str(e)}")
                    if retry == retry_count:
                        raise Exception(f"Failed to store batch after {retry_count} retries: {str(e)}")
                    time.sleep(1)  # Wait before retrying
    
        print(f"\nCompleted processing. Successfully stored {stored_count} chunks.")
        return {
            "total_chunks": stored_count,
            "status": "success"
        }
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return {
            "total_chunks": stored_count,
            "status": "partial_failure",
            "error": str(e)
        }

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
async def search_documents(request: QueryRequest) -> Dict[str, Union[str, List[dict]]]:
    try:
        query_vector = jina_embeddings.embed_query(request.query)
        
        # Get more initial results for better coverage
        result = (
            client.query.get(
                "Nishat",
                ["text"]
            )
            .with_hybrid(
                query=request.query,
                vector=query_vector,
                alpha=0.5,
                properties=["text"]
            )
            .with_additional(["score"])
            .with_limit(5)
            .do()
        )
        
        if not result["data"]["Get"]["Nishat"]:
            return {"message": "No relevant information found for your query"}
        
        # Fixed: Convert score to float before comparison
        filtered_results = [
            {
                "text": item["text"],
                "score": item.get("_additional", {}).get("score", 0)
            }
            for item in result["data"]["Get"]["Nishat"]
            if float(item.get("_additional", {}).get("score", 0)) > 0.7
        ]
        
        if not filtered_results:
            return {"message": "No relevant information found for your query"}
            
        # Sort results by score in descending order
        filtered_results.sort(key=lambda x: float(x["score"]), reverse=True)
        return {"results": filtered_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"\nProcessing file: {file.filename}")
        
        # Save uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            print(f"Saved temporary file: {temp_file_path}")

            # Convert PDF to markdown
            print("Converting PDF to markdown...")
            md_text = pymupdf4llm.to_markdown(temp_file_path)
            print(f"Extracted text length: {len(md_text)} characters")
            
            if len(md_text.strip()) == 0:
                raise ValueError("No text could be extracted from the PDF")

            # Split into chunks
            print("Splitting text into chunks...")
            chunks = recursive_chunks(md_text)
            print(f"Created {len(chunks)} chunks")
            
            if not chunks:
                raise ValueError("No chunks were created from the document")

            # Store chunks in Weaviate with improved error handling
            print("Storing chunks in Weaviate...")
            result = store_chunks_in_weaviate(chunks)
            
            if result["status"] == "success":
                return {
                    "message": "Document processed successfully",
                    "chunks_stored": result["total_chunks"]
                }
            else:
                return {
                    "message": "Document processed with some errors",
                    "chunks_stored": result["total_chunks"],
                    "error": result.get("error", "Unknown error occurred")
                }

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                print(f"Cleaning up temporary file: {temp_file_path}")
                os.remove(temp_file_path)

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 