import json
with open(r'C:\Users\ahmad\.gemini\antigravity-ide\brain\43c7198f-f2b7-4e72-bcff-e1d63e84114e\.system_generated\logs\transcript.jsonl', encoding='utf-8') as f:
    lines = f.readlines()[-60:]
    for line in lines:
        data = json.loads(line)
        if data.get('type') in ['USER_INPUT', 'PLANNER_RESPONSE']:
            print(f"{data.get('source')}: {data.get('content', '')[:300]}")
