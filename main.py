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
from typing import List, Dict

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

# --- App Initialization ---
app = FastAPI(
    title="Image Labeling Tool API",
    description="API for a simple image labeling tool using FastAPI.",
    version="1.0.0"
)

# --- CORS Configuration ---
# Allows the React frontend (running on http://localhost:3000) to communicate with this backend.
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


# --- Directory Setup ---
# Create directories to store uploaded zips and extracted task images.
# In a production environment, you would use a more robust solution like S3.
UPLOADS_DIR = "data/uploads"
TASKS_DIR = "data/tasks"
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(TASKS_DIR, exist_ok=True)


# --- Pydantic Models (Data Schemas) ---
# These models define the shape of the data for requests and responses.

class BoundingBox(BaseModel):
    # *** CHANGE: Updated from int to float to allow for precise coordinates ***
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
    status: str  # 'unlabeled' or 'labeled'
    width: int
    height: int

class TaskOut(BaseModel):
    id: int
    name: str
    status: str  # 'processing', 'ready', 'in_progress', 'completed'
    created_at: datetime

class LabelOut(BaseModel):
    id: int
    name: str

class LabelIn(BaseModel):
    name: str

# --- In-Memory Database Simulation ---
# Using dictionaries to simulate database tables for rapid prototyping.
# In a real application, this would be replaced with a PostgreSQL database and SQLAlchemy ORM.
db: Dict[str, list] = {
    "tasks": [],
    "images": [],
    "labels": [],
    "annotations": [],
}

# Pre-populate with some data for demonstration
db["labels"] = [
    {"id": 1, "name": "Cat"},
    {"id": 2, "name": "Dog"},
]
NEXT_LABEL_ID = 3


# --- Background Task for ZIP Processing ---

async def process_zip_file(task_id: int, zip_path: str):
    """
    Simulates a background worker processing the uploaded ZIP file.
    - Unzips the file.
    - Filters for valid image types.
    - Creates image records in the 'database'.
    - Updates the task status.
    """
    await asyncio.sleep(2) # Simulate processing time

    task_dir = os.path.join(TASKS_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
    image_id_counter = len(db["images"]) + 1

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                # *** FIX: Ignore __MACOSX directory and other hidden files (like .DS_Store) ***
                if filename.startswith('__MACOSX/') or os.path.basename(filename).startswith('.'):
                    continue

                if os.path.splitext(filename)[1].lower() in image_extensions:
                    # Extract file and save it
                    source = zip_ref.open(filename)
                    # Ensure we only use the filename and not any directory structure from the zip
                    sanitized_filename = os.path.basename(filename)
                    if not sanitized_filename: continue # Skip if it's just a directory entry

                    target_path = os.path.join(task_dir, sanitized_filename)
                    with open(target_path, "wb") as f:
                        f.write(source.read())

                    # Get image dimensions
                    try:
                        with Image.open(target_path) as img:
                            width, height = img.size
                    except Exception:
                        width, height = 0, 0 # Could not read image

                    # Create image record
                    new_image = {
                        "id": image_id_counter,
                        "task_id": task_id,
                        "original_filename": sanitized_filename,
                        "storage_path": target_path,
                        "status": "unlabeled",
                        "width": width,
                        "height": height
                    }
                    db["images"].append(new_image)
                    image_id_counter += 1

        # Update task status upon successful processing
        for task in db["tasks"]:
            if task["id"] == task_id:
                task["status"] = "ready"
                break
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        # Update task status to reflect failure
        for task in db["tasks"]:
            if task["id"] == task_id:
                task["status"] = "failed"
                break
    finally:
        # Clean up the original zip file
        os.remove(zip_path)


# --- API Endpoints ---

@app.get("/api/tasks", response_model=List[TaskOut], tags=["Tasks"])
async def get_tasks():
    """Retrieve all tasks."""
    # Return tasks in reverse chronological order
    return sorted(db["tasks"], key=lambda x: x["created_at"], reverse=True)


@app.post("/api/tasks/upload", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def create_upload_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a ZIP file, creates a task, and starts background processing."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a ZIP file.")

    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create a new task in the 'database'
    task_id = len(db["tasks"]) + 1
    new_task = {
        "id": task_id,
        "name": file.filename,
        "status": "processing",
        "created_at": datetime.now()
    }
    db["tasks"].append(new_task)

    # Add the ZIP processing to a background task
    background_tasks.add_task(process_zip_file, task_id, file_path)

    return new_task

@app.get("/api/tasks/{task_id}/images", response_model=List[ImageOut], tags=["Images"])
async def get_task_images(task_id: int):
    """Get all images associated with a specific task."""
    images = [img for img in db["images"] if img["task_id"] == task_id]
    return images

@app.get("/api/images/{image_id}", tags=["Images"])
async def get_image_file(image_id: int):
    """Serves a single image file."""
    image_record = next((img for img in db["images"] if img["id"] == image_id), None)
    if not image_record or not os.path.exists(image_record["storage_path"]):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_record["storage_path"])

@app.get("/api/labels", response_model=List[LabelOut], tags=["Labels"])
async def get_labels():
    """Retrieve all available labels."""
    return db["labels"]

@app.post("/api/labels", response_model=LabelOut, status_code=status.HTTP_201_CREATED, tags=["Labels"])
async def create_label(label: LabelIn):
    """Creates a new label."""
    global NEXT_LABEL_ID
    # Check for duplicates
    if any(l["name"].lower() == label.name.lower() for l in db["labels"]):
        raise HTTPException(status_code=409, detail="Label with this name already exists")
    
    new_label = {"id": NEXT_LABEL_ID, "name": label.name}
    db["labels"].append(new_label)
    NEXT_LABEL_ID += 1
    return new_label

@app.post("/api/images/{image_id}/annotations", response_model=AnnotationOut, status_code=status.HTTP_201_CREATED, tags=["Annotations"])
async def create_annotation(image_id: int, annotation: AnnotationIn):
    """Creates a new annotation for a specific image."""
    # Ensure image exists
    if not any(img["id"] == image_id for img in db["images"]):
        raise HTTPException(status_code=404, detail="Image not found")

    # Create annotation record
    ann_id = len(db["annotations"]) + 1
    new_annotation = {
        "id": ann_id,
        "image_id": image_id,
        "label_id": annotation.label_id,
        "bounding_box": annotation.bounding_box.dict()
    }
    db["annotations"].append(new_annotation)

    # Update image and task status
    for img in db["images"]:
        if img["id"] == image_id:
            img["status"] = "labeled"
            # Check if all images in the task are now labeled
            task_images = [i for i in db["images"] if i["task_id"] == img["task_id"]]
            if all(i["status"] == "labeled" for i in task_images):
                for task in db["tasks"]:
                    if task["id"] == img["task_id"]:
                        task["status"] = "completed"
            else:
                 for task in db["tasks"]:
                    if task["id"] == img["task_id"] and task["status"] != "completed":
                        task["status"] = "in_progress"

            break

    return new_annotation


@app.get("/api/tasks/{task_id}/annotations", response_model=List[AnnotationOut], tags=["Annotations"])
async def get_task_annotations(task_id: int):
    """Gets all annotations for a given task."""
    task_image_ids = {img["id"] for img in db["images"] if img["task_id"] == task_id}
    return [ann for ann in db["annotations"] if ann["image_id"] in task_image_ids]


@app.get("/api/tasks/{task_id}/export", tags=["Export"])
async def export_task_annotations_to_excel(task_id: int):
    """Exports all annotations for a task to an Excel file."""
    task = next((t for t in db["tasks"] if t["id"] == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Gather data
    data = []
    task_image_ids = {img["id"]: img for img in db["images"] if img["task_id"] == task_id}
    labels_map = {lbl["id"]: lbl["name"] for lbl in db["labels"]}

    for ann in db["annotations"]:
        if ann["image_id"] in task_image_ids:
            image = task_image_ids[ann["image_id"]]
            data.append({
                "task_id": task_id,
                "task_name": task["name"],
                "image_id": image["id"],
                "image_filename": image["original_filename"],
                "image_width": image["width"],
                "image_height": image["height"],
                "annotation_id": ann["id"],
                "label_name": labels_map.get(ann["label_id"], "Unknown"),
                "bbox_x": ann["bounding_box"]["x"],
                "bbox_y": ann["bounding_box"]["y"],
                "bbox_width": ann["bounding_box"]["width"],
                "bbox_height": ann["bounding_box"]["height"],
            })
    
    # *** FIX: Check if data list is empty. If so, raise an error. ***
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="No annotations found to export for this task."
        )

    # Create Excel file in memory
    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Annotations')
    output.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="task_{task_id}_annotations.xlsx"'
    }

    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
