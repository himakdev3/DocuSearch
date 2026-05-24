# ❓ Frequently Asked Questions

## General

### Q: What is this app for?
**A:** It is a local document search app for PDF, Word, Text, and PowerPoint (.pptx) files. You upload documents, index them, ask a question, and review matching passages with citations.

### Q: Do I need OpenAI or Anthropic API keys?
**A: No!** Those keys are optional placeholders for future features. Leave them blank. The app works perfectly without them.

### Q: What about the embedding model?
**A:** SentenceTransformers downloads once (~90MB) and then runs locally.

----

## Features & Functionality

### Q: What can this app do?
**A:** 
- ✅ Upload and process PDF, Word, Text, and PowerPoint (.pptx) documents
- ✅ Semantic search across your documents
- ✅ Show relevant passages with citations
- ✅ OCR text from supported embedded images
- ✅ Export results as CSV or JSON

### Q: Can it generate answers like ChatGPT?
**A: Not currently.** It focuses on retrieving the most relevant passages from your documents with citations.

### Q: Does it work offline?
**A: Yes!** After the initial model download, everything runs offline.

----

## Technical Questions

### Q: What happens when I upload a PDF?
**A:**
1. Text is extracted locally (PyPDF2)
2. Images can be OCR processed when supported
3. Content is split into chunks with overlap
4. Each chunk gets an embedding (local SentenceTransformer)
5. Embeddings are stored in a FAISS index (local)
6. All data stays on your computer

### Q: How does search work?
**A:**
1. Your query gets an embedding (local)
2. FAISS finds similar document chunks (local)
3. Results ranked by similarity score
4. Citations show source document and page

### Q: Can I use my own documents?
**A: Absolutely!** Upload your own PDFs, Word files, text files, and PowerPoint (.pptx) files. They stay on your machine, never sent anywhere.

----

## Performance

### Q: How fast is it?
**A:**
- Document processing: ~10 pages/second
- Search: <100ms for 10,000 chunks
- Depends on your CPU

### Q: How many documents can I process?
**A:** Thousands! Limited only by your RAM. Each chunk uses ~1KB in memory.

### Q: Can I use GPU?
**A: Yes!** Set `USE_GPU=true` in .env file (requires CUDA). Makes embeddings much faster.

----

## Setup & Configuration

### Q: What do I need to install?
**A:** Just Python packages:
```bash
pip install -r requirements.txt
```

### Q: Do I need to configure anything?
**A: No!** Defaults work well. Optional: edit `.env` to customize chunk size and retrieval settings.

### Q: Where are my documents stored?
**A:** In memory during use. Upload each time you start the app, or add PDFs to `data/pdfs/` folder.

----

## Future Enhancements

### Q: Can I add answer generation?
**A: Yes!** Two options:
1. **Local**: Use local LLMs (Ollama, LlamaCPP, etc.)
2. **Paid**: Add OpenAI/Anthropic API keys (optional)

### Q: Can I deploy this online?
**A: Yes!** Deploy to Streamlit Cloud, Heroku, AWS, Azure, etc. Still runs locally on the server.
