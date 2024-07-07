import torch
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "juierror/flan-t5-text2sql-with-schema-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

def prepare_input(question: str, tables: Dict[str, List[str]]):
    tables_str = ", ".join(f"{table_name}({','.join(columns)})" for table_name, columns in tables.items())
    prompt = get_prompt(tables_str, question)
    input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").input_ids
    return input_ids.to(device)

def inference(question: str, tables: Dict[str, List[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

class QueryRequest(BaseModel):
    question: str
    tables: Dict[str, List[str]]

@app.post("/generate_sql")
async def generate_sql(request: QueryRequest):
    try:
        sql_query = inference(request.question, request.tables)
        return {"sql_query": sql_query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Text2SQL API is running. Use POST /generate_sql to generate SQL queries."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)