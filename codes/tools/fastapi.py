# -*- coding: utf-8 -*-
import os
import json
import uvicorn
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile


app = FastAPI()

DATA_FOLDER = "../../"


@app.post("/api/submit")
async def submit(task_type: str, task_json: str, data_file: UploadFile):
    accepted, message, task_id = False, "", ""

    try:
        task_config = json.loads(task_json)
    except json.JSONDecodeError:
        pass

    # deal with file
    file_name = str(uuid4()) + ".json"
    with open(os.path.join(DATA_FOLDER, file_name), "wb") as f:
        shutil.copyfileobj(data_file.file, f)

    return {"accepted": accepted, "message": message, "task_id": task_id}


@app.get("/api/status")
async def status(task_id: str):
    task_status = ""

    return {"task_status": task_status}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)