# 🚀 Auto-ETL System

An intelligent ETL pipeline that automatically ingests, analyzes, and discovers insights from multi-source data using PostgreSQL with pgvector and Google Gemini.

## ✨ Key Features

- **🔄 Auto Everything**: Point to files → Get insights
- **🧠 AI-Powered**: Gemini for analysis + embeddings
- **🔗 Smart Relationships**: Detects foreign keys using semantic similarity
- **📊 Auto KPIs**: Domain-specific metrics calculated automatically
- **🔍 Semantic Search**: Find similar data using vector embeddings

## 🎯 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Python 3.8+
- [Gemini API Key](https://makersuite.google.com/app/apikey)

### Setup

1. **Configure API Key**
   ```bash
   cd backend
   # Create .env file and add:
   # GEMINI_API_KEY=your_key_here
   ```

2. **Start Database**
   ```bash
   docker-compose up -d
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Tables & Test**
   ```bash
   python init_database.py
   python test_etl_pipeline.py
   ```

## 📁 How It Works

```python
from src.graph.etl_workflow import ETLWorkflow

# Just point to your files
etl = ETLWorkflow(company_name="YourCompany")
result = etl.run_etl([
    "sales.csv",
    "customers.xlsx",
    "products.json"
])

# Automatically:
# → Detects domains (Finance, Marketing, etc.)
# → Finds relationships between tables
# → Cleans data intelligently
# → Calculates KPIs
# → Generates insights
```

## 🔄 ETL Pipeline

1. **Ingestion** → Load any format (CSV, Excel, JSON, Parquet)
2. **Semantic Analysis** → Understand business context with LLM
3. **Relationship Detection** → Find foreign keys using embeddings
4. **Smart Cleaning** → Context-aware with FK preservation
5. **KPI Calculation** → Domain-specific metrics
6. **Storage** → PostgreSQL with vector embeddings

## 🐳 Database Management

```bash
docker-compose up -d      # Start PostgreSQL with pgvector
docker-compose stop       # Stop database
docker-compose down -v    # Remove database and all data
docker-compose logs       # View logs
```

## 📊 What You Get

### Automatic Insights
- Revenue trends and growth rates
- Customer segmentation patterns
- Cross-table correlations
- Anomaly detection
- Business KPIs by domain

### Example Output
```
✅ Detected: customer_id → customers.customer_id (95% confidence)
✅ Domain: Finance (Revenue data detected)
✅ KPIs: Profit Margin: 28.5%, Revenue Growth: 15.2%
✅ Insight: "Tech industry customers generate 2.3x higher revenue"
```

## 🛠️ Advanced Usage

### Multi-Table Discovery
```python
from src.graph.multi_table_discovery import MultiTableDiscovery

discovery = MultiTableDiscovery()
insights = discovery.run_discovery(
    company_id="company_id",
    dataset_ids=["dataset1", "dataset2"],
    analysis_name="Q4 Analysis"
)
```

### Semantic Search
```python
# Find similar datasets
similar = SimilarityRepository.find_similar_datasets(
    session, embedding, threshold=0.75
)
```

## 📚 Documentation

- [ETL Pipeline Details](ETL_README.md)
- [Docker Setup Guide](DOCKER_SETUP.md)
- [API Reference](src/README.md)

## 🤝 Contributing

Contributions welcome! The codebase is modular and well-documented.

## 📄 License

[Your License]

---

**Built with**: PostgreSQL + pgvector | Google Gemini | LangGraph | Python