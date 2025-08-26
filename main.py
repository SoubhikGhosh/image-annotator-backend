import asyncio
import io
import os
import zipfile
import base64
import shutil
import yaml
import fitz # PyMuPDF
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from PIL import Image, ImageFilter

# NEW: Authentication imports
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- App Initialization & Auth Config (NEW) ---
app = FastAPI(
    title="Image Labeling Tool API",
    description="API for a multi-user image labeling tool.",
    version="2.0.0"
)

# NEW: Security configuration
SECRET_KEY = "a_very_secret_key_change_me" # Use a strong, secret key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")


# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
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
    "users": os.path.join(DATA_DIR, "users.csv"), # NEW
    "tasks": os.path.join(DATA_DIR, "tasks.csv"),
    "images": os.path.join(DATA_DIR, "images.csv"),
    "labels": os.path.join(DATA_DIR, "labels.csv"),
    "annotations": os.path.join(DATA_DIR, "annotations.csv"),
}

# --- Pydantic Models ---
# NEW: User models
class User(BaseModel):
    id: int
    username: str

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

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

class AnnotationSummary(BaseModel):
    label_name: str
    count: int

class TaskOut(BaseModel):
    id: int
    name: str
    status: str
    created_at: datetime
    total_images: int
    labeled_images: int
    annotation_summary: List[AnnotationSummary]

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
        "users": {"id": pd.Int64Dtype(), "username": object, "hashed_password": object}, # NEW
        "tasks": {"id": pd.Int64Dtype(), "user_id": pd.Int64Dtype(), "name": object, "status": object, "created_at": object}, # MODIFIED
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
                db["labels"] = pd.DataFrame([{"id": 1, "name": "Sample Label 1"}, {"id": 2, "name": "Sample Label 2"}])
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


# --- NEW: Authentication Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user_df = db["users"]
    user_record = user_df[user_df["username"] == username]
    if user_record.empty:
        raise credentials_exception
    
    user_dict = user_record.iloc[0].to_dict()
    return User(**user_dict)


# --- Background & Helper Functions ---
async def process_zip_file(task_id: int, zip_content: bytes):
    # This function remains largely the same, no changes needed.
    await asyncio.sleep(2)
    task_dir = os.path.join(TASKS_DIR, str(task_id))
    os.makedirs(task_dir, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".pdf"}
    output_format = "PNG"

    try:
        current_max_id = db["images"]['id'].max() if not db["images"].empty else 0
        image_id_counter = int(current_max_id) + 1
        new_images = []

        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.startswith('__MACOSX/') or filename.endswith('/') or filename.split('/')[-1].startswith('.'):
                    continue
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in image_extensions:
                    base_name, _ = os.path.splitext(filename)
                    sanitized_filename = f"{base_name.replace('/', '_')}.{output_format.lower()}"
                    if not sanitized_filename: continue
                    try:
                        with zip_ref.open(filename) as source:
                            content_bytes = source.read()
                            img_bytes = None
                            if file_ext == '.pdf':
                                try:
                                    pdf_doc = fitz.open(stream=content_bytes, filetype="pdf")
                                    if pdf_doc.page_count > 0:
                                        page = pdf_doc.load_page(0)
                                        pix = page.get_pixmap()
                                        img_bytes = pix.tobytes(output_format.lower())
                                    pdf_doc.close()
                                except Exception as e:
                                    print(f"Skipping {filename} due to PDF processing error: {e}")
                                    continue
                            else:
                                img_bytes = content_bytes
                            if img_bytes is None: continue
                            with Image.open(io.BytesIO(img_bytes)) as img:
                                width, height = img.size
                                img_buffer = io.BytesIO()
                                if img.mode not in ("RGB", "RGBA"): img = img.convert("RGB")
                                img.save(img_buffer, format=output_format)
                                img_buffer.seek(0)
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                            new_images.append({"id": image_id_counter, "task_id": task_id, "original_filename": sanitized_filename, "data": img_base64, "status": "unlabeled", "width": width, "height": height})
                            image_id_counter += 1
                    except Exception as e:
                        print(f"Skipping file {filename} due to processing error: {e}")
                        continue
        if new_images:
            db["images"] = pd.concat([db["images"], pd.DataFrame(new_images)], ignore_index=True)
        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty:
            db["tasks"].loc[task_idx, "status"] = "ready"
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        task_idx = db["tasks"][db["tasks"]["id"] == task_id].index
        if not task_idx.empty: db["tasks"].loc[task_idx, "status"] = "failed"
    finally:
        save_table("images")
        save_table("tasks")


def get_next_id(table_name: str) -> int:
    table = db.get(table_name)
    if table is None or table.empty: return 1
    return int(table["id"].max()) + 1

def update_statuses(image_id: int):
    # This function remains the same
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

# --- NEW: Authentication Endpoints ---
@app.post("/api/users/register", response_model=User, tags=["Users"])
async def register_user(user: UserCreate):
    users_df = db["users"]
    if not users_df[users_df["username"] == user.username].empty:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user_id = get_next_id("users")
    new_user_data = {"id": new_user_id, "username": user.username, "hashed_password": hashed_password}
    
    db["users"] = pd.concat([db["users"], pd.DataFrame([new_user_data])], ignore_index=True)
    save_table("users")
    return {"id": new_user_id, "username": user.username}

@app.post("/api/token", response_model=Token, tags=["Users"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    users_df = db["users"]
    user_record = users_df[users_df["username"] == form_data.username]
    if user_record.empty:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    user = user_record.iloc[0]
    if not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


# --- MODIFIED: Endpoints now require authentication and are user-scoped ---

@app.get("/api/dashboard-summary", response_model=DashboardStats, tags=["Dashboard"])
async def get_dashboard_summary(current_user: User = Depends(get_current_user)):
    # Filter all dataframes to only include data for the current user's tasks
    user_tasks_df = db["tasks"][db["tasks"]["user_id"] == current_user.id]
    if user_tasks_df.empty:
        return {"total_tasks": 0, "total_images": 0, "total_labels": len(db["labels"]), "total_annotations": 0, "task_status_counts": {}, "top_labels": [], "recent_tasks": []}

    user_task_ids = set(user_tasks_df["id"])
    user_images_df = db["images"][db["images"]["task_id"].isin(user_task_ids)]
    user_image_ids = set(user_images_df["id"])
    user_annotations_df = db["annotations"][db["annotations"]["image_id"].isin(user_image_ids)]
    
    # Perform calculations on the filtered data
    task_status_counts = user_tasks_df['status'].value_counts().to_dict()
    
    top_labels = []
    if not user_annotations_df.empty:
        label_counts = user_annotations_df['label_id'].value_counts().reset_index()
        label_counts.columns = ['label_id', 'count']
        merged = pd.merge(label_counts, db["labels"], left_on='label_id', right_on='id')
        top_labels = merged[['name', 'count']].sort_values('count', ascending=False).head(5).to_dict('records')
    
    recent_tasks = []
    if not user_tasks_df.empty:
        user_tasks_df_copy = user_tasks_df.copy()
        user_tasks_df_copy['created_at'] = pd.to_datetime(user_tasks_df_copy['created_at'])
        sorted_tasks = user_tasks_df_copy.sort_values(by="created_at", ascending=False).head(5)
        
        for _, task in sorted_tasks.iterrows():
            task_images = user_images_df[user_images_df['task_id'] == task['id']]
            task_annotations = user_annotations_df[user_annotations_df['image_id'].isin(set(task_images['id']))]
            
            annotation_summary = []
            if not task_annotations.empty:
                label_counts = task_annotations['label_id'].value_counts().reset_index()
                label_counts.columns = ['label_id', 'count']
                summary_df = pd.merge(label_counts, db["labels"], left_on='label_id', right_on='id')
                annotation_summary = [{"label_name": row['name'], "count": row['count']} for _, row in summary_df.iterrows()]
            
            task_data = task.to_dict()
            task_data['total_images'] = len(task_images)
            task_data['labeled_images'] = len(task_images[task_images['status'] == 'labeled'])
            task_data['annotation_summary'] = annotation_summary
            recent_tasks.append(task_data)
            
    return {"total_tasks": len(user_tasks_df), "total_images": len(user_images_df), "total_labels": len(db["labels"]), "total_annotations": len(user_annotations_df), "task_status_counts": task_status_counts, "top_labels": top_labels, "recent_tasks": recent_tasks}


@app.get("/api/tasks", response_model=List[TaskOut], tags=["Tasks"])
async def get_tasks(current_user: User = Depends(get_current_user)):
    # MODIFIED: Filter tasks by current user's ID
    user_tasks_df = db["tasks"][db["tasks"]["user_id"] == current_user.id]

    if user_tasks_df.empty: return []

    user_tasks_df_copy = user_tasks_df.copy()
    user_tasks_df_copy['created_at'] = pd.to_datetime(user_tasks_df_copy['created_at'])
    sorted_tasks = user_tasks_df_copy.sort_values(by="created_at", ascending=False)
    
    tasks_with_details = []
    for _, task in sorted_tasks.iterrows():
        task_id = task['id']
        task_images = db["images"][db["images"]['task_id'] == task_id]
        total_images = len(task_images)
        labeled_images = len(task_images[task_images['status'] == 'labeled'])
        
        annotation_summary = []
        if not task_images.empty:
            task_annotations = db["annotations"][db["annotations"]['image_id'].isin(set(task_images['id']))]
            if not task_annotations.empty:
                label_counts = task_annotations['label_id'].value_counts().reset_index()
                label_counts.columns = ['label_id', 'count']
                summary_df = pd.merge(label_counts, db["labels"], left_on='label_id', right_on='id')
                annotation_summary = [{"label_name": row['name'], "count": row['count']} for _, row in summary_df.iterrows()]
        
        task_data = task.to_dict()
        task_data['total_images'] = total_images
        task_data['labeled_images'] = labeled_images
        task_data['annotation_summary'] = annotation_summary
        tasks_with_details.append(task_data)

    return tasks_with_details


@app.post("/api/tasks/upload", response_model=TaskOut, status_code=status.HTTP_202_ACCEPTED, tags=["Tasks"])
async def create_upload_task(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user) # MODIFIED
):
    if not file.filename.endswith(".zip"): raise HTTPException(status_code=400, detail="Invalid file type.")
    
    zip_content = await file.read()
    
    task_id = get_next_id("tasks")
    new_task_data = {
        "id": task_id, 
        "user_id": current_user.id, # MODIFIED: Assign ownership
        "name": file.filename, 
        "status": "processing", 
        "created_at": datetime.now()
    }
    db["tasks"] = pd.concat([db["tasks"], pd.DataFrame([new_task_data])], ignore_index=True)
    save_table("tasks")
    
    background_tasks.add_task(process_zip_file, task_id, zip_content)
    
    response_task = new_task_data.copy()
    response_task['total_images'] = 0
    response_task['labeled_images'] = 0
    response_task['annotation_summary'] = []
    
    return response_task


@app.delete("/api/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Tasks"])
async def delete_task(task_id: int, current_user: User = Depends(get_current_user)): # MODIFIED
    task_record = db["tasks"][(db["tasks"]["id"] == task_id) & (db["tasks"]["user_id"] == current_user.id)]
    if task_record.empty:
        raise HTTPException(status_code=404, detail="Task not found or you don't have permission.")

    image_ids_to_delete = set(db["images"][db["images"]["task_id"] == task_id]["id"])
    if image_ids_to_delete:
        db["annotations"] = db["annotations"][~db["annotations"]["image_id"].isin(image_ids_to_delete)]
    db["images"] = db["images"][db["images"]["task_id"] != task_id]
    db["tasks"] = db["tasks"][db["tasks"]["id"] != task_id]
    
    task_dir = os.path.join(TASKS_DIR, str(task_id))
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    save_table("annotations"); save_table("images"); save_table("tasks")


# The following functions need to be secured as well.
# This helper function will be used to verify task ownership.
async def get_task_for_user(task_id: int, current_user: User = Depends(get_current_user)):
    task_record = db["tasks"][(db["tasks"]["id"] == task_id) & (db["tasks"]["user_id"] == current_user.id)]
    if task_record.empty:
        raise HTTPException(status_code=404, detail="Task not found or you don't have permission.")
    return task_record.iloc[0]


@app.get("/api/tasks/{task_id}/images", response_model=List[ImageOut], tags=["Images"])
async def get_task_images(task_id: int, task: pd.Series = Depends(get_task_for_user)):
    return db["images"][db["images"]["task_id"] == task_id].to_dict('records')


@app.get("/api/tasks/{task_id}/annotations", response_model=List[AnnotationOut], tags=["Annotations"])
async def get_task_annotations(task_id: int, task: pd.Series = Depends(get_task_for_user)):
    task_image_ids = set(db["images"][db["images"]["task_id"] == task_id]["id"])
    if not task_image_ids: return []
    return db["annotations"][db["annotations"]["image_id"].isin(task_image_ids)].to_dict('records')


@app.post("/api/images/{image_id}/annotations", response_model=AnnotationOut, status_code=status.HTTP_201_CREATED, tags=["Annotations"])
async def create_annotation(image_id: int, annotation: AnnotationIn, current_user: User = Depends(get_current_user)):
    # Verify the user owns the task this image belongs to
    image_record = db["images"][db["images"]["id"] == image_id]
    if image_record.empty: raise HTTPException(status_code=404, detail="Image not found")
    task_id = image_record.iloc[0]['task_id']
    await get_task_for_user(task_id, current_user) # Re-use dependency logic for check
    
    ann_id = get_next_id("annotations")
    new_ann = {"id": ann_id, "image_id": image_id, "label_id": annotation.label_id, "bounding_box": annotation.bounding_box.dict()}
    db["annotations"] = pd.concat([db["annotations"], pd.DataFrame([new_ann])], ignore_index=True)
    update_statuses(image_id=image_id)
    save_table("annotations"); save_table("images"); save_table("tasks")
    return new_ann

@app.delete("/api/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Annotations"])
async def delete_annotation(annotation_id: int, current_user: User = Depends(get_current_user)):
    ann_record = db["annotations"][db["annotations"]["id"] == annotation_id]
    if ann_record.empty: raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Verify ownership by checking the parent image and task
    image_id = int(ann_record.iloc[0]["image_id"])
    image_record = db["images"][db["images"]["id"] == image_id]
    task_id = image_record.iloc[0]['task_id']
    await get_task_for_user(task_id, current_user)

    db["annotations"] = db["annotations"][db["annotations"]["id"] != annotation_id]
    update_statuses(image_id=image_id)
    save_table("annotations"); save_table("images"); save_table("tasks")


# Labels are global and not user-specific in this design, so they remain largely unchanged.
@app.get("/api/labels", response_model=List[LabelOut], tags=["Labels"])
async def get_labels():
    return db["labels"].sort_values(by="name").to_dict('records')

@app.post("/api/labels", response_model=LabelOut, status_code=status.HTTP_201_CREATED, tags=["Labels"])
async def create_label(label: LabelIn, current_user: User = Depends(get_current_user)): # Secured
    if db["labels"]["name"].str.lower().eq(label.name.lower()).any():
        raise HTTPException(status_code=409, detail="Label already exists")
    new_label = {"id": get_next_id("labels"), "name": label.name}
    db["labels"] = pd.concat([db["labels"], pd.DataFrame([new_label])], ignore_index=True)
    save_table("labels")
    return new_label

@app.delete("/api/labels/{label_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Labels"])
async def delete_label(label_id: int, current_user: User = Depends(get_current_user)): # Secured
    if not (db["labels"]["id"] == label_id).any(): raise HTTPException(status_code=404, detail="Label not found")
    if not db["annotations"].empty and (db["annotations"]["label_id"] == label_id).any():
        raise HTTPException(status_code=409, detail="Cannot delete label: it is currently in use by one or more annotations.")
    db["labels"] = db["labels"][db["labels"]["id"] != label_id]
    save_table("labels")

# Export endpoints also need to be secured
@app.get("/api/tasks/{task_id}/export", tags=["Export"])
async def export_task_annotations_to_excel(task_id: int, task: pd.Series = Depends(get_task_for_user)):
    # ... (rest of the function is the same, as `get_task_for_user` has already verified ownership)
    images_df = db["images"][db["images"]["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images in task.")
    annotations_df = db["annotations"][db["annotations"]["image_id"].isin(set(images_df["id"]))]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations to export.")
    
    export_data = []
    for _, ann in annotations_df.iterrows():
        img_info = images_df[images_df["id"] == ann["image_id"]].iloc[0]
        label_info = db["labels"][db["labels"]["id"] == ann["label_id"]].iloc[0]
        bbox = ann["bounding_box"]
        export_data.append({
            "task_id": task_id, "task_name": task["name"], "image_id": ann["image_id"],
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
async def export_task_annotations_to_yolo(task_id: int, task: pd.Series = Depends(get_task_for_user)):
    # ... (rest of the function is the same)
    images_df = db["images"][db["images"]["task_id"] == task_id]
    if images_df.empty: raise HTTPException(status_code=404, detail="No images in task.")
    image_ids = set(images_df["id"])
    annotations_df = db["annotations"][db["annotations"]["image_id"].isin(image_ids)]
    if annotations_df.empty: raise HTTPException(status_code=404, detail="No annotations to export.")
    label_ids_in_task = sorted(list(annotations_df["label_id"].unique()))
    labels_in_task_df = db["labels"][db["labels"]["id"].isin(label_ids_in_task)]
    label_map = {label_id: i for i, label_id in enumerate(labels_in_task_df["id"])}
    class_names = list(labels_in_task_df["name"])

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        dataset_yaml = {'path': f'../{task["name"].replace(".zip", "")}', 'train': 'images', 'val': 'images', 'names': {i: name for i, name in enumerate(class_names)}}
        zip_file.writestr("dataset.yaml", yaml.dump(dataset_yaml, sort_keys=False))
        images_with_anns = images_df[images_df['id'].isin(set(annotations_df['image_id']))]
        for _, img_rec in images_with_anns.iterrows():
            try:
                img_bytes = base64.b64decode(img_rec["data"])
                zip_file.writestr(f"images/{img_rec['original_filename']}", img_bytes)
            except Exception: continue
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
                base, _ = os.path.splitext(img_rec['original_filename'])
                zip_file.writestr(f"labels/{base}.txt", "\n".join(yolo_lines))
    zip_buffer.seek(0)
    filename = f"task_{task_id}_yolo.zip"
    return StreamingResponse(zip_buffer, headers={'Content-Disposition': f'attachment; filename="{filename}"'}, media_type="application/zip")

@app.post("/api/tasks/{task_id}/process", tags=["Export"])
async def process_task_images(task_id: int, request: ProcessRequest, task: pd.Series = Depends(get_task_for_user)):
    # ... (rest of the function is the same)
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
                                region = img.crop(box).filter(ImageFilter.GaussianBlur(radius=max(1, min(request.blur_radius, 50))))
                            img.paste(region, box)
                        img_byte_arr = io.BytesIO(); img.save(img_byte_arr, format='PNG'); img_byte_arr.seek(0)
                        zip_file.writestr(img_rec["original_filename"], img_byte_arr.getvalue())
                except Exception as e:
                    print(f"Error processing {request.action} for image {img_rec['original_filename']}: {e}")
                    continue
    zip_buffer.seek(0)
    filename = f"task_{task_id}_{request.action}.zip"
    return StreamingResponse(zip_buffer, headers={'Content-Disposition': f'attachment; filename="{filename}"'}, media_type="application/zip")