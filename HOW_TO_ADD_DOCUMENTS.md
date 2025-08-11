# 📚 How to Add New Documents to LightRAG

## 🚀 **Quick Start - 3 Simple Steps**

### **Step 1: Add PDF Files**
```bash
# Copy your PDF files to the papers directory
cp your_research_paper.pdf ./papers/
cp another_document.pdf ./papers/
```

### **Step 2: Update Knowledge Base**
```bash
# Run the update script
python update_lightrag_knowledge.py

# OR use the simple shell script
./update_knowledge.sh
```

### **Step 3: Query Your Documents**
```bash
# Test your updated knowledge base
python demo_lightrag_working.py
```

## 📋 **Detailed Instructions**

### **What Files Can You Add?**
- ✅ **PDF Files** (Recommended) - Research papers, documents, reports
- ⚠️ **Text Files** - Currently only PDFs are processed by the system

### **Where to Put Files?**
- **Directory**: `./papers/`
- **Example**: 
  ```
  papers/
  ├── metabolomics_research_2024.pdf
  ├── clinical_study_results.pdf
  └── biomarker_analysis.pdf
  ```

### **What Happens When You Run the Update Script?**

1. **🔍 Scans** the `./papers/` directory for new files
2. **📖 Extracts** text content from PDF files
3. **🧠 Processes** content using AI to understand concepts
4. **🗄️ Stores** knowledge in searchable format
5. **✅ Confirms** successful processing

### **Example Output:**
```
🔄 Updating LightRAG Knowledge Base
============================================================
1️⃣ Initializing LightRAG system...
✅ System initialized

2️⃣ Found 3 files to process:
   📄 research_paper_1.pdf: 1,234,567 bytes
   📄 clinical_study.pdf: 987,654 bytes
   📄 analysis_report.pdf: 456,789 bytes

3️⃣ Processing 3 documents...

4️⃣ Processing Results:
   ✅ Successfully processed: 3 files
   ❌ Failed to process: 0 files
   ⏱️  Processing time: 15.23 seconds

🎉 Knowledge base updated successfully!
```

## 🔄 **Workflow for Regular Updates**

### **Daily/Weekly Routine:**
1. **Add new PDFs** to `./papers/` directory
2. **Run update script**: `python update_lightrag_knowledge.py`
3. **Test with queries** to verify new content is available

### **Batch Processing:**
```bash
# Add multiple files at once
cp /path/to/research/*.pdf ./papers/

# Update knowledge base
python update_lightrag_knowledge.py

# Verify with a test query
python -c "
import asyncio
from src.lightrag_integration.component import LightRAGComponent
from src.lightrag_integration.config.settings import LightRAGConfig

async def test_query():
    config = LightRAGConfig.from_env()
    component = LightRAGComponent(config)
    await component.initialize()
    result = await component.query('What new topics are covered?')
    print(f'Answer: {result[\"answer\"]}')
    await component.cleanup()

asyncio.run(test_query())
"
```

## 🛠️ **Troubleshooting**

### **Common Issues:**

**❌ "No PDF files found"**
- **Solution**: Ensure files are in `./papers/` directory and have `.pdf` extension

**❌ "Failed to process PDF"**
- **Solution**: Check if PDF is corrupted or password-protected
- **Alternative**: Try converting to a different PDF format

**❌ "API key errors"**
- **Solution**: Verify your Groq/OpenAI API keys are configured correctly

### **File Requirements:**
- ✅ **Valid PDF format**
- ✅ **Not password-protected**
- ✅ **Contains extractable text** (not just images)
- ✅ **Under 50MB** (configurable limit)

## 📊 **Monitoring Your Knowledge Base**

### **Check Current Status:**
```bash
# See what's currently in your knowledge base
python trace_lightrag_data_sources.py
```

### **View Processing Logs:**
```bash
# Check detailed logs
tail -f data/lightrag_cache/lightrag.log
```

### **Statistics:**
The update script shows:
- Number of documents processed
- Processing time
- Success/failure rates
- Current knowledge base size

## 🎯 **Best Practices**

### **File Organization:**
```
papers/
├── 2024/
│   ├── metabolomics_jan_2024.pdf
│   └── clinical_study_feb_2024.pdf
├── 2023/
│   └── historical_research.pdf
└── reference/
    └── methodology_guide.pdf
```

### **Naming Conventions:**
- Use descriptive filenames
- Include dates when relevant
- Avoid special characters
- Example: `metabolomics_biomarkers_study_2024.pdf`

### **Regular Maintenance:**
- **Weekly**: Add new research papers
- **Monthly**: Review and organize files
- **Quarterly**: Clean up outdated documents

## 🚀 **Advanced Usage**

### **Programmatic Updates:**
```python
import asyncio
from pathlib import Path
from src.lightrag_integration.component import LightRAGComponent
from src.lightrag_integration.config.settings import LightRAGConfig

async def update_knowledge_base():
    config = LightRAGConfig.from_env()
    component = LightRAGComponent(config)
    await component.initialize()
    
    # Get all PDF files
    pdf_files = list(Path("./papers").glob("*.pdf"))
    file_paths = [str(f) for f in pdf_files]
    
    # Process documents
    result = await component.ingest_documents(file_paths)
    print(f"Processed {result['successful']} documents")
    
    await component.cleanup()

# Run the update
asyncio.run(update_knowledge_base())
```

### **Automated Monitoring:**
```bash
# Set up a cron job to automatically process new files
# Add to crontab: 0 9 * * * cd /path/to/project && python update_lightrag_knowledge.py
```

---

## 🎉 **Summary**

**To add new information to LightRAG:**

1. **📁 Add PDF files** to `./papers/` directory
2. **🔄 Run**: `python update_lightrag_knowledge.py`
3. **✅ Done!** Your knowledge base is updated

**That's it!** LightRAG will automatically process your documents and make them searchable through intelligent queries. 🚀