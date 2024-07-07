import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Diagnostic information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Force CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
    model = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2").to(device)
    
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")

    def get_prompt(tables, question):
        prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
        return prompt

    def prepare_input(question: str, tables: dict[str, List[str]]):
        tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
        tables = ", ".join(tables)
        prompt = get_prompt(tables, question)
        input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").input_ids
        return input_ids.to(device)

    def inference(question: str, tables: dict[str, List[str]]) -> str:
        input_data = prepare_input(question=question, tables=tables)
        outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
        result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
        return result

    print(inference("how many people with name jui and age less than 25", {
        "people_name": ["id", "name"],
        "people_age": ["people_id", "age"]
    }))

    print(inference("what is id with name jui and age less than 25", {
        "people_name": ["id", "name", "age"]
    }))

except RuntimeError as e:
    print(f"RuntimeError: {e}")
    if "CUDA out of memory" in str(e):
        print("You may need to reduce batch size or model size.")
    elif "CUDA driver version is insufficient" in str(e):
        print("You may need to update your CUDA drivers.")
    else:
        print("An unexpected error occurred. Please check your CUDA installation and GPU compatibility.")
except Exception as e:
    print(f"An error occurred: {e}")