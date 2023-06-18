import glob
import json

data_files = list(glob.glob('./output/output*.json'))
data_files.sort()

final_list = []

for file in data_files:
    with open(file=file, mode='r', encoding='utf8') as source:
        items = json.loads(source.read())
        final_list += items

output_file_name = './output/translated_alpaca_data_cleaned.json'
with open(output_file_name, "w", encoding="utf-8") as temp:
        temp.write(json.dumps(final_list, ensure_ascii=False))

print('{} record has been written into the output file.'.format(len(final_list)))
