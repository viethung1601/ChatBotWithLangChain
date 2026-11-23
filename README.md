# Chatbot-with-LangChain

A Chatbot is built with RAG Pipeline using LangChain framework. 

## RAG Pipeline
- PDF papers/files ------------------------------> Chunks ---------------------------------> Vectors ----------> Vector Database 
                  (RecursiveCharacterTextSplitter)        (Embedding with BAAI/bge-m3 Model)
                  or
                  (SematicChunker)
- User's question ------------------------> Vector ------------------------> Retrieve top k vectors ----------------> LLM  
                  (Embedding with                  (Compare with all chunks                         (include them
                  the same Embedding Model          Vector in Vector DB using                        additionally in
                  with Chunks)                      COSINE Similarity)                               context)
## Explaination
1. Chatbot's Model: Claude Model using Claude API
2. Embedding Model: BAAI/bge-m3 running local
3. Text from PDF splitted to chunks: RecursiveCharacterTextSplitter or SemanticChunker
  a) RecursiveCharacterTextSplitter: Split text based on Markdown seperators
  b) SemanticChunker: split text based on the Meaning
