# Project : RAG (Retrieval-Augmented Generation)
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge 
base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions 
of parameters to generate original output for tasks like answering questions,completing sentences etc.AG extends the already powerful capabilities of LLMs 
to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM 
output so it remains relevant, accurate, and useful in various contexts.
# How to run?
### Step 01 :
Clone the repository

```bash
git clone https://github.com/MayukhBaruah19/RAG-application.git
```
### STEP 02 :
Create a conda environment after opening the repository : 
```bash
conda create -p venv python=3.13.5 -y
```
activate the envirnment
```bash
conda activate venv/
```
### step 03 : 
install the requirements
```bash
pip install -r requirements.txt
```
### step 04 :
```
streamlit run app.py
```
