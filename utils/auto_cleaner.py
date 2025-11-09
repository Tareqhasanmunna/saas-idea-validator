#utils/auto_cleaner.py
import os, time, shutil
from datetime import datetime, timedelta

def delete_old_files(folder_path, days=60):
    now = time.time()
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                if os.stat(file_path).st_mtime < now - days * 86400:
                    os.remove(file_path)
            except Exception:
                pass
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                if os.stat(dir_path).st_mtime < now - days * 86400:
                    shutil.rmtree(dir_path)
            except Exception:
                pass

def clean_everything(project_root="E:/saas-idea-validator"):
    # adjust these paths as needed
    paths = [
        os.path.join(project_root, "models"),
        os.path.join(project_root, "reports"),
        os.path.join(project_root, "data", "raw"),
        os.path.join(project_root, "data", "processed")
    ]
    for p in paths:
        delete_old_files(p, days=60)
