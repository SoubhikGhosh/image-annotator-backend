# main.py
# To run this app:
# 1. pip install -r requirements.txt
# 2. uvicorn main:app --reload

import asyncio
import io
import os
import shutil
import zipfile
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image

# --- App Initialization ---
app = FastAPI(
    title="Image Labeling Tool API",
    description="API for a simple image labeling tool with file-based persistence.",
    version="1.2.0"
)

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory and Data File Setup ---
DATA_DIR = "data"
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)

# Define paths for our CSV "database" files
DB_PATHS = {
    "tasks": os.path.join(DATA_DIR, "tasks.csv"),
    "images": os.path.join(DATA_DIR, "images.csv"),
    "labels": os.path.join(DATA_DIR, "labels.csv"),
    "annotations": os.path.join(DATA_DIR, "annotations.csv"),
}

# --- Pydantic Models ---
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class AnnotationIn(BaseModel):
    label_id: int
    bounding_box: BoundingBox

class AnnotationOut(BaseModel):
    id: int
    image_id: int
    label_id: int
    bounding_box: BoundingBox

class ImageOut(BaseModel):
    id: int
    task_id: int
    original_filename: str
    storage_path: str
    status: str
    width: int
    height: int

class TaskOut(BaseModel):
    id: int
    name: str
    status: str
    created_at: datetime

class LabelOut(BaseModel):
    id: int
    name: str

class LabelIn(BaseModel):
    name: str

# --- CSV-based Database Simulation ---
db: Dict[str, pd.DataFrame] = {}

def load_or_initialize_db():
    """
    Loads data from CSV files into pandas DataFrames.
    If a file doesn't exist, it creates an empty DataFrame with the correct columns.
    """
    global db
    
    schemas = {
        "tasks": {"id": pd.Int64Dtype(), "name": object, "status": object, "created_at": object},
        "images": {"id": pd.Int64Dtype(), "task_id": pd.Int64Dtype(), "original_filename": object, "storage_path": object, "status": object, "width": pd.Int64Dtype(), "height": pd.Int64Dtype()},
        "labels": {"id": pd.Int64Dtype(), "name": object},
        "annotations": {"id": pd.Int64Dtype(), "image_id": pd.Int64Dtype(), "label_id": pd.Int64Dtype(), "bounding_box": object},
    }
    
    for table, path in DB_PATHS.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            db[table] = pd.read_csv(path)
        else:
            db[table] = pd.DataFrame(columns=list(schemas[table].keys())).astype(schemas[table])
            if table == "labels" and db[table].empty:
                default_labels = [{"id": 1, "name": "Cat"}, {"id": 2, "name": "Dog"}]
                db["labels"] = pd.DataFrame(default_labels)
                save_table("labels")
    
    if 'bounding_box' in db['annotations'].columns:
        db['annotations']['bounding_box'] = db['annotations']['bounding_box'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

@app.on_event("startup")
async def startup_event():
    print("Loading database from CSV files...")
    load_or_initialize_db()
    print("Database loaded.")

def save_table(table_name: str):
    if table_name in db:
        db[table_name].to_csv(DB_PATHS[table_name], index=False)

# --- Background Task for ZIP Processing ---
async def process_zip_file(task_id: int, zip_path: str):
    await asyncio.sleep(2)
    task_dir = os.path.join(TASKS_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

    try:
        current_max_id = db["images"]['id'].max() if not db["images"].empty else 0
        image_id_counter = int(current_max_id) + 1
        new_images = []

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.startswith('__MACOSX/') or os.path.basename(filename).startswith('.'): continue
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    sanitized_filename = os.path.basename(filename)
                    if not sanitized_filename: continue
                    target_path = os.path.join(task_dir, sanitized_filename)
                    with zip_ref.open(filename) as source, open(target_path, "wb") as f: f.write(source.read())

                    try:
                        with Image.open(target_path) as img: width, height = img.size
                    except Exception: width, height = 0, 0
                    
                    new_images.append({
                        "id": image_id_counter, "task_id": task_id, "original_filename": sanitized_filename,
                        "storage_path": target_path, "status": "unlabeled", "width": width, "height": height
                    })
                    image_id_counter += 1
        
        if new_images:
            db["images"] = pd.concat([db["images"], pd.DataFrame(new_images)], ignore_index=True)
        
        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty: db["tasks"].loc[task_idx, "status"] = "ready"
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty: db["tasks"].loc[task_idx, "status"] = "failed"
    finally:
        save_table("images")
        save_table("tasks")
        if os.path.exists(zip_path): os.remove(zip_path)

def get_next_id(table_name: str) -> int:
    table = db.get(table_name)
    if table is None or table.empty: return 1
    return int(table["id"].max()) + 1

# --- API Endpoints ---
@app.get("/api/tasks", response_model=List[TaskOut], tags=["Tasks"])
async def get_tasks():
    if db["tasks"].empty: return []
    tasks_df = db["tasks"].copy()
    tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
    return tasks_df.sort_values(by="created_at", ascending=False).to_dict('records')

@app.post("/api/tasks/upload", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def create_upload_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)

    task_id = get_next_id("tasks")
    new_task = {"id": task_id, "name": file.filename, "status": "processing", "created_at": datetime.now()}
    db["tasks"] = pd.concat([db["tasks"], pd.DataFrame([new_task])], ignore_index=True)
    save_table("tasks")
    
    background_tasks.add_task(process_zip_file, task_id, file_path)
    return new_task

@app.get("/api/tasks/{task_id}/images", response_model=List[ImageOut], tags=["Images"])
async def get_task_images(task_id: int):
    images = db["images"][db["images"]["task_id"] == task_id]
    return images.to_dict('records')

@app.get("/api/images/{image_id}", tags=["Images"])
async def get_image_file(image_id: int):
    image_record = db["images"][db["images"]["id"] == image_id]
    if image_record.empty: raise HTTPException(status_code=404, detail="Image not found")
    path = image_record.iloc[0]["storage_path"]
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="Image file not found on disk")
    return FileResponse(path)

@app.get("/api/labels", response_model=List[LabelOut], tags=["Labels"])
async def get_labels():
    return db["labels"].sort_values(by="name").to_dict('records')

@app.post("/api/labels", response_model=LabelOut, status_code=status.HTTP_201_CREATED, tags=["Labels"])
async def create_label(label: LabelIn):
    if db["labels"]["name"].str.lower().eq(label.name.lower()).any():
        raise HTTPException(status_code=409, detail="Label with this name already exists")
    
    new_label = {"id": get_next_id("labels"), "name": label.name}
    db["labels"] = pd.concat([db["labels"], pd.DataFrame([new_label])], ignore_index=True)
    save_table("labels")
    return new_label

@app.delete("/api/labels/{label_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Labels"])
async def delete_label(label_id: int):
    if not (db["labels"]["id"] == label_id).any():
        raise HTTPException(status_code=404, detail="Label not found")
    if not db["annotations"].empty and (db["annotations"]["label_id"] == label_id).any():
        raise HTTPException(status_code=409, detail="Cannot delete label. It is used in annotations.")
    db["labels"] = db["labels"][db["labels"]["id"] != label_id]
    save_table("labels")
    return

@app.post("/api/images/{image_id}/annotations", response_model=AnnotationOut, status_code=status.HTTP_201_CREATED, tags=["Annotations"])
async def create_annotation(image_id: int, annotation: AnnotationIn):
    if not (db["images"]["id"] == image_id).any():
        raise HTTPException(status_code=404, detail="Image not found")

    ann_id = get_next_id("annotations")
    new_annotation = {"id": ann_id, "image_id": image_id, "label_id": annotation.label_id, "bounding_box": annotation.bounding_box.dict()}
    db["annotations"] = pd.concat([db["annotations"], pd.DataFrame([new_annotation])], ignore_index=True)
    update_statuses(image_id=image_id)
    save_table("annotations"); save_table("images"); save_table("tasks")
    return new_annotation

@app.delete("/api/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Annotations"])
async def delete_annotation(annotation_id: int):
    ann_df = db["annotations"]
    if ann_df[ann_df["id"] == annotation_id].empty:
        raise HTTPException(status_code=404, detail="Annotation not found")
    image_id = ann_df[ann_df["id"] == annotation_id].iloc[0]["image_id"]
    db["annotations"] = ann_df[ann_df["id"] != annotation_id]
    update_statuses(image_id=int(image_id))
    save_table("annotations"); save_table("images"); save_table("tasks")
    return

def update_statuses(image_id: int):
    img_record = db["images"][db["images"]["id"] == image_id]
    if img_record.empty: return
    task_id = img_record.iloc[0]["task_id"]
    image_annotations = db["annotations"][db["annotations"]["image_id"] == image_id]
    new_image_status = "labeled" if not image_annotations.empty else "unlabeled"
    db["images"].loc[db["images"]["id"] == image_id, "status"] = new_image_status
    task_images = db["images"][db["images"]["task_id"] == task_id]
    if not task_images.empty:
        if (task_images["status"] == "labeled").all(): new_task_status = "completed"
        elif (task_images["status"] == "labeled").any(): new_task_status = "in_progress"
        else: new_task_status = "ready"
        db["tasks"].loc[db["tasks"]["id"] == task_id, "status"] = new_task_status

@app.get("/api/tasks/{task_id}/annotations", response_model=List[AnnotationOut], tags=["Annotations"])
async def get_task_annotations(task_id: int):
    task_image_ids = set(db["images"][db["images"]["task_id"] == task_id]["id"])
    if not task_image_ids: return []
    task_annotations = db["annotations"][db["annotations"]["image_id"].isin(task_image_ids)]
    return task_annotations.to_dict('records')

@app.get("/api/tasks/{task_id}/export", tags=["Export"])
async def export_task_annotations_to_excel(task_id: int):
    """Exports all annotations for a task to an Excel file. (FIXED)"""
    task_df = db["tasks"]
    task_record = task_df[task_df["id"] == task_id]
    if task_record.empty: raise HTTPException(status_code=404, detail="Task not found")
    
    images_df = db["images"][db["images"]["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images found for this task.")
    
    image_ids = set(images_df["id"])
    annotations_df = db["annotations"][db["annotations"]["image_id"].isin(image_ids)]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations to export for this task.")
    
    labels_df = db["labels"]
    
    # Build the data list manually to avoid complex merge issues
    export_data = []
    for _, ann in annotations_df.iterrows():
        image_info = images_df[images_df["id"] == ann["image_id"]].iloc[0]
        label_info = labels_df[labels_df["id"] == ann["label_id"]].iloc[0]
        bbox = ann["bounding_box"]
        
        export_data.append({
            "task_id": task_id,
            "task_name": task_record.iloc[0]["name"],
            "image_id": ann["image_id"],
            "image_filename": image_info["original_filename"],
            "image_width": image_info["width"],
            "image_height": image_info["height"],
            "annotation_id": ann["id"],
            "label_name": label_info["name"],
            "bbox_x": bbox["x"],
            "bbox_y": bbox["y"],
            "bbox_width": bbox["width"],
            "bbox_height": bbox["height"],
        })

    df = pd.DataFrame(export_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Annotations')
    output.seek(0)
    
    headers = {'Content-Disposition': f'attachment; filename="task_{task_id}_annotations.xlsx"'}
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
