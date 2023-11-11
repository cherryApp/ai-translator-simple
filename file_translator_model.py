from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "I wish I hadn't seen such a horrible film.",
    "She's at school."
]

model_name = "pytorch-models/opus-mt-tc-big-en-hu"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

for t in translated:
    print( tokenizer.decode(t, skip_special_tokens=True) )

# expected output:
#     Bárcsak ne láttam volna ilyen szörnyű filmet.
#     Iskolában van.
