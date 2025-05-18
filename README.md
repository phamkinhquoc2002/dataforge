<div align="center"><img src="https://github.com/phamkinhquoc2002/dataforge/blob/main/truelogo.png" alt="My Image" width="900"/></div>

# ⚡️ SynForge
## The Ultimate Synthetic Data Generation Framework
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python) ![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative)

Synforge is a lightweight Python framework designed to generate high-quality synthetic datasets tailored to your specific tasks. With a powerful PDF parser and intelligent knowledge localization, DataForge enhances your data generation workflows by integrating custom domain knowledge—offering a smarter alternative to traditional knowledge distillation techniques. 

## Features
* **Multi-Agent Orchestration**: Compose complex synthetic data pipelines with agentic collaboration.
* **LLM Powerhouse**: Seamless integration with Google Gemini, OpenAI GPT, and **more to come**...
* **Human-in-the-Loop**: Pause, review, and steer data generation interactively—no more black-box outputs!
* **User-Friendly Error Handling**: Clear and beautifully formatted error logs.
* **Python-centric design**: Easily integrates with frameworks like `Langchain`.
* **Document-Aware Generation**: Automatically retrieve the necessary localized knowledge, chunk, and leverage it as context for synthetic data generation.
## 📦 Installation
The most simple way to use dataforge is to install it through pip. It is recommended to use Python **3.11** or **3.12**.
```
pip install synforge
```
## 🧪 Example
### ⚠️ Environment Variables
Make sure to define the following environment variables in a `.env` file before using SynForge:
```
H_TOKEN="...."
GEMINI_API_KEY="...."
```
* `HF_TOKEN`: Required to download the lightweight vision model Docling, used for PDF parsing.

* `GEMINI_API_KEY`: Required to access the LLM for synthetic data generation. (Gemini in this case!)
### 🔧Sample Code
1. LLM Engine initialization and parse context sources:
```
from synforge.utils import pdf_parser
from synforge.llm_providers import GoogleAIModel

# Load environment variables
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize the LLM engine
llm_engine = GoogleAIModel(
          model="gemini-2.0-flash",
          api_key=gemini_api_key,
          max_tokens= 4096,
          temperature= 0.6,
          top_p= 0.5,
          api_key=gemini_api_key
)

# Parse knowledge from PDF
pdf_content = pdf_parser("input.pdf")
```
2. Define entities in the system: task (user-defined task), and agent (the multi-agentic system used for data generation), retriever (for context localization).
```
from src.data_forge.tasks import Task
from langchain_community.retrievers import BM25Retriever
from src.data_forge.multi_agent import SyntheticDataGenerator

# Define retriever for knowledge localization
retriever = BM25Retriever.from_documents(pdf_content)

# Define task
task = Task(
    task_name="SFT", #Can be SFT, DPO, Multi-Dialogue or Custom
    localization="Data Engineering, the ETL process, Retrieval-Augmented Generation", #Used to localize the parts that you want to use to for synthetic data generation.
    task_description="Help me generate a synthetic dataset to train a smaller model to reason.", #Description of the task
    rows_per_batch=5, #The number of data rows in every iteration (every LLM call)
    language="japanese" #Language of the data output
)

# Initialize the synthetic data agent
agent = SyntheticDataGenerator(
    llm =llm_engine,
    retriever=retriever,
    output_path="./output/",
    thread_id={"configurable": {"thread_id": "1"}} # Used for Human-in-the-loop 
)
```
## 🚧 UPCOMING
SynForge is still actively in development! We welcome contributors and collaborators to help us shape the future of synthetic data generation for AI engineers.

* 📬 Want to contribute? -> Reach out via email to get involved!

* Planned features:
 - Advanced retrieval techniques
 - Enhanced agent customization
 - More Task-specific prompt templates

## 📚 Citation
@software{synforge,
  author = {Quoc Pham},
  title = {SynForge: The Ultimate Synthetic Data Generation Framework},
  url = {https://github.com/phamkinhquoc2002/dataforge},
  year = {2025}
}




