import json
import random


new_data = []
with open("C:\\Users\\duyanb.XINAO\\Desktop\\v7_badcase_0703_resutl-from_server_to_json.json", 'r', encoding='utf-8') as f:
    data_list = json.load(f)
    random.shuffle(data_list)
    for data in data_list:
        contents = data['x'].split("\n")
        if len(contents) > 1:
            last_content = contents[-1]
        else:
            last_content = data['x']

        new_data.append({'id': data['id'], 'contents': data['x'], "last_content": last_content, 'y': data['y'], 'pred1': data['pred1']})


with open('C:\\Users\\duyanb.XINAO\\Desktop\\v7.json', 'w', encoding='utf-8') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False)