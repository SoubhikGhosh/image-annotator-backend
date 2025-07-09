import asyncio
import io
import os
import zipfile
import base64
import shutil
import yaml
from datetime import datetime
from typing import List, Dict, Optional, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageFilter

# --- App Initialization ---
app = FastAPI(
    title="Image Labeling Tool API",
    description="API for a simple image labeling tool with file-based persistence and image processing.",
    version="1.5.0"
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
    expose_headers=["Content-Disposition"]
)

# --- Directory and Data File Setup ---
DATA_DIR = "data"
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
os.makedirs(TASKS_DIR, exist_ok=True)

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
    status: str
    width: int
    height: int
    data: str # Base64 encoded image data

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

class TopLabelStat(BaseModel):
    name: str
    count: int

class DashboardStats(BaseModel):
    total_tasks: int
    total_images: int
    total_labels: int
    total_annotations: int
    task_status_counts: Dict[str, int]
    image_status_counts: Dict[str, int]
    top_labels: List[TopLabelStat]
    recent_tasks: List[TaskOut]

class ProcessRequest(BaseModel):
    action: str
    label_ids: List[int]
    blur_radius: Optional[int] = 15

# --- CSV-based Database Simulation ---
db: Dict[str, pd.DataFrame] = {}

def load_or_initialize_db():
    global db
    schemas = {
        "tasks": {"id": pd.Int64Dtype(), "name": object, "status": object, "created_at": object},
        "images": {"id": pd.Int64Dtype(), "task_id": pd.Int64Dtype(), "original_filename": object, "data": object, "status": object, "width": pd.Int64Dtype(), "height": pd.Int64Dtype()},
        "labels": {"id": pd.Int64Dtype(), "name": object},
        "annotations": {"id": pd.Int64Dtype(), "image_id": pd.Int64Dtype(), "label_id": pd.Int64Dtype(), "bounding_box": object},
    }
    
    for table, path in DB_PATHS.items():
        if os.path.exists(path) and os.path.getsize(path) > 0:
            db[table] = pd.read_csv(path)
        else:
            db[table] = pd.DataFrame(columns=list(schemas[table].keys())).astype(schemas[table])
            if table == "labels" and db[table].empty:
                db["labels"] = pd.DataFrame([{"id": 1, "name": "Cat"}, {"id": 2, "name": "Dog"}])
                save_table("labels")
    
    # Ensure 'bounding_box' is correctly parsed after loading from CSV
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

# --- Background & Helper Functions ---
async def process_zip_file(task_id: int, zip_content: bytes):
    """
    Processes an uploaded zip file in the background.
    - Extracts valid image files (.jpg, .png, etc.) and single-page PDFs.
    - Flattens the directory structure, renaming files with their path (e.g., folder_subfolder_file.png).
    - Converts all extracted files to PNG format, encodes them in Base64, and adds them to the database.
    - Updates the task status to 'ready' on success or 'failed' on error.
    """
    await asyncio.sleep(2)  # Simulate initial processing time

    task_dir = os.path.join(TASKS_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)

    # MODIFICATION: Added '.pdf' to the set of supported extensions.
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}
    output_format = "PNG"

    try:
        current_max_id = db["images"]['id'].max() if not db["images"].empty else 0
        image_id_counter = int(current_max_id) + 1
        new_images = []

        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for filename in zip_ref.namelist():
                # Skip common junk files, hidden files, and directory entries
                if filename.startswith('__MACOSX/') or filename.endswith('/') or filename.split('/')[-1].startswith('.'):
                    continue

                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in image_extensions:
                    # MODIFICATION: Flatten folder structure into the filename.
                    # e.g., "level1/level2/image.jpeg" becomes "level1_level2_image.png"
                    base_name, _ = os.path.splitext(filename)
                    sanitized_filename = f"{base_name.replace('/', '_')}.{output_format.lower()}"

                    if not sanitized_filename:
                        continue

                    try:
                        with zip_ref.open(filename) as source:
                            content_bytes = source.read()
                            img_bytes = None

                            # MODIFICATION: Add block to handle PDF to image conversion.
                            if file_ext == '.pdf':
                                try:
                                    pdf_doc = fitz.open(stream=content_bytes, filetype="pdf")
                                    # Process only the first page of the PDF
                                    if pdf_doc.page_count > 0:
                                        page = pdf_doc.load_page(0)
                                        pix = page.get_pixmap()
                                        img_bytes = pix.tobytes(output_format.lower())
                                    else:
                                        print(f"Skipping empty or invalid PDF: {filename}")
                                        continue
                                    pdf_doc.close()
                                except Exception as e:
                                    print(f"Skipping {filename} due to PDF processing error: {e}")
                                    continue
                            else:
                                # This is the original path for standard image files
                                img_bytes = content_bytes

                            if img_bytes is None:
                                continue

                            # Process the image bytes (from either PDF or image) using PIL
                            with Image.open(io.BytesIO(img_bytes)) as img:
                                width, height = img.size
                                img_buffer = io.BytesIO()

                                # Ensure consistency by converting to RGB if necessary
                                if img.mode not in ("RGB", "RGBA"):
                                    img = img.convert("RGB")

                                img.save(img_buffer, format=output_format)
                                img_buffer.seek(0)
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

                            new_images.append({
                                "id": image_id_counter,
                                "task_id": task_id,
                                "original_filename": sanitized_filename,
                                "data": img_base64,
                                "status": "unlabeled",
                                "width": width,
                                "height": height
                            })
                            image_id_counter += 1
                    except Exception as e:
                        print(f"Skipping file {filename} due to a general processing error: {e}")
                        continue

        if new_images:
            db["images"] = pd.concat([db["images"], pd.DataFrame(new_images)], ignore_index=True)

        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty:
            db["tasks"].loc[task_idx, "status"] = "ready"
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty:
            db["tasks"].loc[task_idx, "status"] = "failed"
    finally:
        save_table("images")
        save_table("tasks")

def get_next_id(table_name: str) -> int:
    table = db.get(table_name)
    if table is None or table.empty: return 1
    return int(table["id"].max()) + 1

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

# --- API Endpoints ---

@app.get("/api/dashboard-summary", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_summary():
    tasks_df, images_df, labels_df, annotations_df = db["tasks"], db["images"], db["labels"], db["annotations"]
    task_status_counts = tasks_df['status'].value_counts().to_dict() if not tasks_df.empty else {}
    top_labels = []
    if not annotations_df.empty and not labels_df.empty:
        label_counts = annotations_df['label_id'].value_counts().reset_index(); label_counts.columns = ['label_id', 'count']
        merged = pd.merge(label_counts, labels_df, left_on='label_id', right_on='id')
        top_labels = merged[['name', 'count']].sort_values('count', ascending=False).head(5).to_dict('records')
    recent_tasks = []
    if not tasks_df.empty:
        tasks_df_copy = tasks_df.copy(); tasks_df_copy['created_at'] = pd.to_datetime(tasks_df_copy['created_at'])
        recent_tasks = tasks_df_copy.sort_values(by="created_at", ascending=False).head(5).to_dict('records')
    return {"total_tasks": len(tasks_df), "total_images": len(images_df), "total_labels": len(labels_df),
        "total_annotations": len(annotations_df), "task_status_counts": task_status_counts,
        "image_status_counts": images_df['status'].value_counts().to_dict() if not images_df.empty else {},
        "top_labels": top_labels, "recent_tasks": recent_tasks}

@app.post("/api/tasks/{task_id}/process", tags=["Export"])
async def process_task_images(task_id: int, request: ProcessRequest):
    if not request.label_ids: raise HTTPException(status_code=400, detail="No labels selected.")
    images_df = db["images"][db["images"]["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images in task.")
    image_ids = set(images_df["id"])
    annotations_df = db["annotations"][(db["annotations"]["image_id"].isin(image_ids)) & (db["annotations"]["label_id"].isin(request.label_ids))]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations for selected labels.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if request.action == "crop":
            for _, ann in annotations_df.iterrows():
                img_rec = images_df[images_df["id"] == ann["image_id"]].iloc[0]
                img_data_b64 = img_rec["data"]
                
                try:
                    img_bytes = base64.b64decode(img_data_b64)
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        bbox = ann["bounding_box"]
                        box_tuple = (int(bbox['x']), int(bbox['y']), int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height']))
                        cropped = img.crop(box_tuple)
                        img_byte_arr = io.BytesIO(); cropped.save(img_byte_arr, format='PNG'); img_byte_arr.seek(0)
                        base, _ = os.path.splitext(img_rec["original_filename"])
                        zip_file.writestr(f"{base}_crop_{ann['id']}.png", img_byte_arr.getvalue())
                except Exception as e:
                    print(f"Error processing crop for image {img_rec['original_filename']}: {e}")
                    continue
        else:
            imgs_to_process = images_df[images_df['id'].isin(set(annotations_df['image_id']))]
            for _, img_rec in imgs_to_process.iterrows():
                img_data_b64 = img_rec["data"]
                
                try:
                    img_bytes = base64.b64decode(img_data_b64)
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        img = img.convert("RGBA")
                        anns_for_img = annotations_df[annotations_df["image_id"] == img_rec["id"]]
                        for _, ann in anns_for_img.iterrows():
                            bbox = ann["bounding_box"]
                            box = (int(bbox['x']), int(bbox['y']), int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height']))
                            
                            if request.action == "blacken":
                                region = Image.new('RGBA', (box[2]-box[0], box[3]-box[1]), (0, 0, 0, 255))
                            elif request.action == "blur":
                                region = img.crop(box)
                                radius = max(1, min(request.blur_radius, 50))
                                region = region.filter(ImageFilter.GaussianBlur(radius=radius))
                            
                            img.paste(region, box)
                        
                        img_byte_arr = io.BytesIO(); img.save(img_byte_arr, format='PNG'); img_byte_arr.seek(0)
                        zip_file.writestr(img_rec["original_filename"], img_byte_arr.getvalue())
                except Exception as e:
                    print(f"Error processing {request.action} for image {img_rec['original_filename']}: {e}")
                    continue

    zip_buffer.seek(0)
    filename = f"task_{task_id}_{request.action}.zip"
    return StreamingResponse(zip_buffer, headers={'Content-Disposition': f'attachment; filename="{filename}"'}, media_type="application/zip")


# --- Standard CRUD Endpoints ---
@app.get("/api/tasks", response_model=List[TaskOut], tags=["Tasks"])
async def get_tasks():
    if db["tasks"].empty: return []
    tasks_df = db["tasks"].copy(); tasks_df['created_at'] = pd.to_datetime(tasks_df['created_at'])
    return tasks_df.sort_values(by="created_at", ascending=False).to_dict('records')

@app.post("/api/tasks/upload", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def create_upload_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"): raise HTTPException(status_code=400, detail="Invalid file type.")
    
    zip_content = await file.read()
    
    task_id = get_next_id("tasks")
    new_task = {"id": task_id, "name": file.filename, "status": "processing", "created_at": datetime.now()}
    db["tasks"] = pd.concat([db["tasks"], pd.DataFrame([new_task])], ignore_index=True)
    save_table("tasks")
    
    background_tasks.add_task(process_zip_file, task_id, zip_content)
    return new_task

@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
async def delete_task(task_id: int):
    """
    Deletes a task and all associated data including images, annotations,
    and the task directory.
    """
    global db
    # 1. Check if task exists
    task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
    if task_idx.empty:
        raise HTTPException(status_code=404, detail="Task not found")

    # 2. Find associated image IDs
    image_ids_to_delete = set(db["images"][db["images"]["task_id"] == task_id]["id"])

    # 3. Delete annotations for those images
    if image_ids_to_delete:
        db["annotations"] = db["annotations"][~db["annotations"]["image_id"].isin(image_ids_to_delete)]

    # 4. Delete the images
    db["images"] = db["images"][db["images"]["task_id"] != task_id]

    # 5. Delete the task
    db["tasks"] = db["tasks"][db["tasks"]["id"] != task_id]
    
    # 6. Delete the physical task directory
    task_dir = os.path.join(TASKS_DIR, str(task_id))
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    # 7. Save changes to all tables
    save_table("annotations")
    save_table("images")
    save_table("tasks")

@app.get("/api/tasks/{task_id}/images", response_model=List[ImageOut], tags=["Images"])
async def get_task_images(task_id: int):
    return db["images"][db["images"]["task_id"] == task_id].to_dict('records')

@app.get("/api/labels", response_model=List[LabelOut], tags=["Labels"])
async def get_labels():
    return db["labels"].sort_values(by="name").to_dict('records')

@app.post("/api/labels", response_model=LabelOut, status_code=status.HTTP_201_CREATED, tags=["Labels"])
async def create_label(label: LabelIn):
    if db["labels"]["name"].str.lower().eq(label.name.lower()).any():
        raise HTTPException(status_code=409, detail="Label already exists")
    new_label = {"id": get_next_id("labels"), "name": label.name}
    db["labels"] = pd.concat([db["labels"], pd.DataFrame([new_label])], ignore_index=True)
    save_table("labels")
    return new_label

@app.delete("/api/labels/{label_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Labels"])
async def delete_label(label_id: int):
    if not (db["labels"]["id"] == label_id).any(): raise HTTPException(status_code=404, detail="Label not found")
    if not db["annotations"].empty and (db["annotations"]["label_id"] == label_id).any():
        raise HTTPException(status_code=409, detail="Label in use.")
    db["labels"] = db["labels"][db["labels"]["id"] != label_id]
    save_table("labels")

@app.post("/api/images/{image_id}/annotations", response_model=AnnotationOut, status_code=status.HTTP_201_CREATED, tags=["Annotations"])
async def create_annotation(image_id: int, annotation: AnnotationIn):
    if not (db["images"]["id"] == image_id).any(): raise HTTPException(status_code=404, detail="Image not found")
    ann_id = get_next_id("annotations")
    new_ann = {"id": ann_id, "image_id": image_id, "label_id": annotation.label_id, "bounding_box": annotation.bounding_box.dict()}
    db["annotations"] = pd.concat([db["annotations"], pd.DataFrame([new_ann])], ignore_index=True)
    update_statuses(image_id=image_id)
    save_table("annotations"); save_table("images"); save_table("tasks")
    return new_ann

@app.delete("/api/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Annotations"])
async def delete_annotation(annotation_id: int):
    ann_df = db["annotations"]
    if ann_df[ann_df["id"] == annotation_id].empty: raise HTTPException(status_code=404, detail="Annotation not found")
    image_id = ann_df[ann_df["id"] == annotation_id].iloc[0]["image_id"]
    db["annotations"] = ann_df[ann_df["id"] != annotation_id]
    update_statuses(image_id=int(image_id))
    save_table("annotations"); save_table("images"); save_table("tasks")

@app.get("/api/tasks/{task_id}/annotations", response_model=List[AnnotationOut], tags=["Annotations"])
async def get_task_annotations(task_id: int):
    task_image_ids = set(db["images"][db["images"]["task_id"] == task_id]["id"])
    if not task_image_ids: return []
    return db["annotations"][db["annotations"]["image_id"].isin(task_image_ids)].to_dict('records')

@app.get("/api/tasks/{task_id}/export", tags=["Export"])
async def export_task_annotations_to_excel(task_id: int):
    task_df, images_df, annotations_df, labels_df = db["tasks"], db["images"], db["annotations"], db["labels"]
    task_rec = task_df[task_df["id"] == task_id]
    if task_rec.empty: raise HTTPException(status_code=404, detail="Task not found")
    images_df = images_df[images_df["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images in task.")
    annotations_df = annotations_df[annotations_df["image_id"].isin(set(images_df["id"]))]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations to export.")
    
    export_data = []
    for _, ann in annotations_df.iterrows():
        img_info = images_df[images_df["id"] == ann["image_id"]].iloc[0]
        label_info = labels_df[labels_df["id"] == ann["label_id"]].iloc[0]
        bbox = ann["bounding_box"]
        export_data.append({
            "task_id": task_id, "task_name": task_rec.iloc[0]["name"], "image_id": ann["image_id"],
            "image_filename": img_info["original_filename"], "image_width": img_info["width"], "image_height": img_info["height"],
            "annotation_id": ann["id"], "label_name": label_info["name"], "bbox_x": bbox["x"], "bbox_y": bbox["y"],
            "bbox_width": bbox["width"], "bbox_height": bbox["height"],
        })
    df = pd.DataFrame(export_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Annotations')
    output.seek(0)
    headers = {'Content-Disposition': f'attachment; filename="task_{task_id}_annotations.xlsx"'}
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.get("/api/tasks/{task_id}/export-yolo", tags=["Export"])
async def export_task_annotations_to_yolo(task_id: int):
    # Fetch data
    task = db["tasks"][db["tasks"]["id"] == task_id]
    if task.empty: raise HTTPException(status_code=404, detail="Task not found")
    
    images_df = db["images"][db["images"]["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images in task.")

    image_ids = set(images_df["id"])
    annotations_df = db["annotations"][db["annotations"]["image_id"].isin(image_ids)]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations to export.")

    # Create label mapping for this task's specific labels
    label_ids_in_task = sorted(list(annotations_df["label_id"].unique()))
    all_labels_df = db["labels"]
    labels_in_task_df = all_labels_df[all_labels_df["id"].isin(label_ids_in_task)]
    
    label_map = {label_id: i for i, label_id in enumerate(labels_in_task_df["id"])}
    class_names = list(labels_in_task_df["name"])

    # Prepare for zipping
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create dataset.yaml
        dataset_yaml = {
            'path': f'../{task.iloc[0]["name"].replace(".zip", "")}',
            'train': 'images',
            'val': 'images',
            'names': {i: name for i, name in enumerate(class_names)}
        }
        zip_file.writestr("dataset.yaml", yaml.dump(dataset_yaml, sort_keys=False))

        # Process each image with annotations
        images_with_anns = images_df[images_df['id'].isin(set(annotations_df['image_id']))]
        for _, img_rec in images_with_anns.iterrows():
            # Write image file
            try:
                img_bytes = base64.b64decode(img_rec["data"])
                image_path = f"images/{img_rec['original_filename']}"
                zip_file.writestr(image_path, img_bytes)
            except Exception:
                continue # Skip if image data is corrupt

            # Write YOLO label file
            yolo_lines = []
            anns_for_img = annotations_df[annotations_df["image_id"] == img_rec["id"]]
            img_w, img_h = img_rec["width"], img_rec["height"]

            for _, ann in anns_for_img.iterrows():
                bbox = ann["bounding_box"]
                label_idx = label_map.get(ann["label_id"])
                if label_idx is None: continue

                x_center = (bbox['x'] + bbox['width'] / 2) / img_w
                y_center = (bbox['y'] + bbox['height'] / 2) / img_h
                width_norm = bbox['width'] / img_w
                height_norm = bbox['height'] / img_h
                yolo_lines.append(f"{label_idx} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
            
            if yolo_lines:
                base, _ = os.path.splitext(img_rec["original_filename"])
                label_path = f"labels/{base}.txt"
                zip_file.writestr(label_path, "\n".join(yolo_lines))

    zip_buffer.seek(0)
    filename = f"task_{task_id}_yolo.zip"
    return StreamingResponse(zip_buffer, headers={'Content-Disposition': f'attachment; filename="{filename}"'}, media_type="application/zip")
