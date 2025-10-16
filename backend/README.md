# ETL to Insights AI Agent

A production-ready LangGraph-based AI agent that transforms business data into actionable insights through intelligent two-phase analysis.

## 🎯 Overview

This agent helps businesses identify and solve their biggest challenges through data-driven analysis:

- **Phase 1**: Collects business context, identifies challenges, and prioritizes them based on impact
- **Phase 2**: Performs ETL on your data, conducts statistical analysis, generates visualizations, and produces professional reports

## ✨ Key Features

### Phase 1: Problem Identification
- 🎤 Interactive business context collection
- 🧠 AI-powered challenge identification from pain points and objectives
- 📊 Intelligent prioritization (0-100 scoring with 4-tier system)
- 💾 ChromaDB integration for persistent context storage

### Phase 2: Analysis & Reporting
- 📥 **ETL Pipeline**: Automated extraction from PDF, Excel, CSV
- 🧪 **Statistical Analysis**: T-tests, ANOVA, regression, correlation, causality
- 📈 **Visualizations**: Distribution plots, heatmaps, time series, scatter plots
- 📝 **Reports**: Professional analytical & business insight reports
- 📊 **Interactive Dashboard**: Plotly-powered HTML dashboard with charts, filters, and KPIs

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv 
```

```
.\.venv\Scripts\activate
```


```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the Agent

**Demo mode (quick test):**
```bash
python main.py --demo
```

**Interactive mode (your data):**
```bash
python main.py
```

### 4. Run Tests

```bash
# Complete test (Phase 1 + analyze top 3 challenges)
python run_test.py 1

# Incremental test (Phase 1 once + Phase 2 once)
python run_test.py 2

# Quick test (Phase 1 only)
python run_test.py 3
```

---

## 📁 Project Structure

```
clue/
├── src/
│   ├── models/           # Pydantic data models
│   │   ├── business_context.py
│   │   ├── challenge.py
│   │   ├── analysis_result.py
│   │   └── report.py
│   ├── phase1/           # Problem identification
│   │   ├── context_collector.py
│   │   └── challenge_prioritizer.py
│   ├── phase2/           # Analysis and reporting
│   │   ├── etl_pipeline.py
│   │   ├── statistical_analyzer.py
│   │   ├── visualization_engine.py
│   │   ├── report_generator.py
│   │   └── dashboard_generator.py
│   ├── graph/            # LangGraph workflow
│   │   ├── state.py
│   │   └── workflow.py
│   └── utils/            # Shared utilities
│       ├── llm_client.py
│       └── chroma_manager.py
├── data/
│   ├── uploads/          # Input data by department
│   │   ├── Sales/
│   │   ├── Marketing/
│   │   ├── Customer_Success/
│   │   └── Product/
│   └── outputs/          # Generated artifacts
│       ├── reports/
│       ├── visualizations/
│       ├── dashboards/
│       └── processed/
├── test_data/            # Test data files
├── chroma_db/            # Vector database storage
├── main.py               # Entry point
├── run_test.py           # Test suite runner
└── create_test_data.py   # Test data generator
```

---

## 📖 How It Works

### Workflow Architecture

```
┌─────────────────────────────────────────────┐
│         LangGraph Workflow                  │
├─────────────────────────────────────────────┤
│  PHASE 1: Problem Identification            │
│  ├─ Collect Context                         │
│  ├─ Identify Challenges (LLM)               │
│  ├─ Score & Prioritize                      │
│  └─ Store in ChromaDB                       │
│                                              │
│  PHASE 2: Analysis & Reporting              │
│  For each challenge (by priority):          │
│  ├─ ETL Pipeline                            │
│  │  ├─ Extract (CSV, Excel, PDF)           │
│  │  ├─ Transform (Clean, Standardize)      │
│  │  └─ Load (Memory + Disk)                │
│  ├─ Statistical Analysis                    │
│  │  ├─ Descriptive Statistics              │
│  │  ├─ Correlation Analysis                │
│  │  ├─ Hypothesis Testing                  │
│  │  ├─ Causality Analysis                  │
│  │  └─ Time Series (if applicable)         │
│  ├─ Visualization Generation                │
│  │  ├─ Distributions                       │
│  │  ├─ Heatmaps                            │
│  │  ├─ Time Series                         │
│  │  └─ Scatter Plots                       │
│  └─ Save Analysis Result                    │
│                                              │
│  Generate Final Reports:                    │
│  ├─ Analytical Report (Technical)           │
│  ├─ Business Insight Report (Executive)     │
│  └─ Interactive Dashboard (Visual)          │
└─────────────────────────────────────────────┘
```

### Challenge Prioritization Algorithm

Each challenge receives a score (0-100) based on:
- **Business Goal Impact**: 0-30 points
- **Departments Affected**: 0-20 points
- **Success Metric Alignment**: 0-25 points
- **Pain Point Urgency**: 0-25 points

Priority Levels:
- **CRITICAL** (80-100): Immediate action required
- **HIGH** (60-79): High priority, near-term focus
- **MEDIUM** (40-59): Important, plan for resolution
- **LOW** (0-39): Monitor, address when capacity allows

---

## 📊 Data Requirements

### Input Data Format

Place your data files in department-specific folders:

```
data/uploads/
├── Sales/
│   ├── sales_data.csv
│   └── pipeline_report.xlsx
├── Marketing/
│   └── campaign_metrics.csv
└── Customer_Success/
    └── customer_health.xlsx
```

**Supported Formats:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- PDF (`.pdf`) - text extraction

---

## 📈 Output Reports

### 1. Analytical Report
Technical report for data analysts:
- Executive summary
- Methodology
- Detailed analysis per challenge
- Statistical test results
- Key findings & limitations

### 2. Business Insight Report
Executive report for leadership:
- Strategic insights
- Opportunities identified
- Prioritized action items
- Expected business impact

### 3. Interactive Dashboard
Visual HTML dashboard with:
- Challenge overview with test counts
- Statistical significance gauge
- Key findings comparison charts
- Correlation heatmaps
- Data sources breakdown
- Causality insights tracker
- **Fully interactive**: hover, zoom, filter
- **Portable**: works in any browser, no server needed

**Locations**:
- Reports: `data/outputs/reports/`
- Dashboards: `data/outputs/dashboards/`

---

## 🛠️ Technology Stack

- **LangGraph**: Stateful workflow orchestration
- **LangChain**: LLM integration and chains
- **Google Gemini**: Language model for analysis
- **ChromaDB**: Vector database for context
- **Pandas**: Data manipulation
- **NumPy/SciPy**: Numerical computing
- **Statsmodels**: Statistical analysis
- **Matplotlib/Seaborn/Plotly**: Visualization
- **Pydantic**: Data validation

---

## 🧪 Testing

### Test Data

The test suite includes realistic business data across 4 departments:

**Sales** (2 files)
- 180 days of sales performance data (CSV)
- Q1 2024 business report (PDF)

**Marketing** (1 file)
- Campaign performance with 2 sheets (Excel)

**Customer Success** (2 files)
- 300 customer health metrics with 2 sheets (Excel)
- Customer feedback summary (PDF)

**Operations** (1 file)
- 24 weeks of operational metrics (CSV)

### Generate Test Data

```bash
python create_test_data.py
```

Note: Requires `reportlab` for PDF generation:
```bash
pip install reportlab
```

### Run Tests

**Complete Test:**
```bash
python run_test.py 1
```
- Runs Phase 1 (identifies all challenges)
- Analyzes top 3 challenges in Phase 2
- Generates comprehensive reports
- Takes ~10-15 minutes

**Incremental Test:**
```bash
python run_test.py 2
```
- Phase 1 once, then Phase 2 once
- Demonstrates independent phase execution
- Analyzes highest priority challenge only

**Quick Test (Phase 1 Only):**
```bash
python run_test.py 3
```
- Only runs challenge identification
- Fast (~2-3 minutes)

### Expected Test Outcomes

The agent should identify challenges like:
- 🔴 **CRITICAL**: "High Customer Churn and Suboptimal Retention"
- 🟠 **HIGH**: "Ineffective Lead-to-Opportunity Management"
- 🟠 **HIGH**: "Inconsistent Customer Experience"
- 🟡 **MEDIUM**: "Marketing Attribution Gaps"

---

## ⚙️ Design Principles

✅ **Clean Code**: Modular architecture, single responsibility
✅ **No Hardcoding**: Real implementations, no fallback/fake code
✅ **Scalable**: LangGraph for structured, extensible workflows
✅ **Production-Ready**: Error handling, logging, validation
✅ **AI-Guided**: LLM selects appropriate analysis methods

---

## 🎯 Example Use Cases

1. **Sales Performance**: Analyze conversion rates, identify bottlenecks
2. **Customer Churn**: Predict at-risk customers, retention strategies
3. **Product Analytics**: Feature adoption, usage patterns
4. **Marketing ROI**: Campaign effectiveness, channel attribution

---

## 📚 Key Components

### Phase 1 Modules

**ContextCollector** ([src/phase1/context_collector.py](src/phase1/context_collector.py))
- Collects and validates business context
- LLM-powered validation and gap identification
- ChromaDB integration for persistence

**ChallengePrioritizer** ([src/phase1/challenge_prioritizer.py](src/phase1/challenge_prioritizer.py))
- Identifies challenges from pain points
- Multi-factor scoring algorithm (0-100)
- Four-tier priority system

### Phase 2 Modules

**ETLPipeline** ([src/phase2/etl_pipeline.py](src/phase2/etl_pipeline.py))
- Extracts from CSV, Excel, PDF
- Cleans and transforms data
- Handles missing values, outliers, duplicates

**StatisticalAnalyzer** ([src/phase2/statistical_analyzer.py](src/phase2/statistical_analyzer.py))
- LLM-guided test selection
- T-tests, ANOVA, regression, correlation
- Generic threshold detection (no hardcoding)
- Sequential column exclusion

**VisualizationEngine** ([src/phase2/visualization_engine.py](src/phase2/visualization_engine.py))
- Distributions, heatmaps, time series
- Professional styling
- High-resolution PNG output

**ReportGenerator** ([src/phase2/report_generator.py](src/phase2/report_generator.py))
- Analytical report (technical)
- Business insight report (executive)
- Markdown formatting

**DashboardGenerator** ([src/phase2/dashboard_generator.py](src/phase2/dashboard_generator.py))
- Interactive Plotly visualizations
- Challenge overview & KPIs
- Statistical significance tracking
- Correlation analysis charts
- Standalone HTML export

---

## 🔧 Advanced Usage

### Run Specific Phases

**Phase 1 Only:**
```bash
python main.py --phase1
```

**Phase 2 Only:**
```bash
python main.py --phase2
```

**Check Status:**
```bash
python main.py --status
```

### Clear Database

```bash
# Remove ChromaDB to start fresh
rm -rf chroma_db
```

---

## 🚨 Troubleshooting

### "No data found"
- Run `python create_test_data.py` first
- Or place your files in `data/uploads/[Department]/`

### "API rate limit"
- Gemini API may have rate limits
- Add delays between requests if needed

### "PDF generation failed"
- Install reportlab: `pip install reportlab`
- Or skip PDFs (CSV/Excel tests still work)

### "No challenges identified"
- Check GEMINI_API_KEY in .env
- Verify LLM is responding
- Check ChromaDB permissions

---

## 📦 Dependencies

Everything needed is in `requirements.txt`:

```
langchain>=0.1.0
langchain-google-genai>=0.0.5
langgraph>=0.0.20
chromadb>=0.4.18
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
openpyxl>=3.1.0
PyPDF2>=3.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
reportlab>=4.0.0  # For test data generation only
```

---

## 🤝 Contributing

This is a production-ready implementation. For issues or enhancements, ensure:
- Code follows existing architecture
- All implementations are real (no mocks/hardcodes)
- Tests are included for new features

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with Claude Code** 🤖
