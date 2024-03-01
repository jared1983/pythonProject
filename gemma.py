from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", token="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")

input_text = "为什么中国人要在过年的时候包饺子吃，请用中文回答"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=2000)
print(tokenizer.decode(outputs[0]))
