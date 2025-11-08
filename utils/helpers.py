import os
import yaml
import logging
from datetime import datetime
from pathlib import Path

def setup_rl_logger(log_file, logger_name="RLSystem"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def cleanup_old_files(directory, retention_days, file_pattern="*", exclude_dirs=None, logger=None):
    if not os.path.exists(directory):
        return {"deleted_count": 0, "kept_count": 0}
    
    deleted = 0
    kept = 0
    cutoff_time = datetime.now().timestamp() - (retention_days * 86400)
    
    for file in Path(directory).glob(file_pattern):
        if file.is_file():
            if file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    deleted += 1
                    if logger:
                        logger.info(f"Deleted: {file}")
                except:
                    pass
            else:
                kept += 1
    
    return {"deleted_count": deleted, "kept_count": kept}

def cleanup_directory_by_count(directory, keep_count, file_pattern="*", logger=None):
    if not os.path.exists(directory):
        return {"deleted_count": 0, "kept_count": 0}
    
    files = sorted(Path(directory).glob(file_pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    deleted = 0
    
    for file in files[keep_count:]:
        if file.is_file():
            try:
                file.unlink()
                deleted += 1
                if logger:
                    logger.info(f"Deleted old model: {file.name}")
            except:
                pass
    
    return {"deleted_count": deleted, "kept_count": min(len(files), keep_count)}

def archive_files(source_dir, archive_dir, file_pattern, logger=None):
    import shutil
    if not os.path.exists(source_dir):
        return 0
    
    os.makedirs(archive_dir, exist_ok=True)
    archived = 0
    
    for file in Path(source_dir).glob(file_pattern):
        if file.is_file() and (datetime.now().timestamp() - file.stat().st_mtime) > 2592000:
            try:
                shutil.move(str(file), os.path.join(archive_dir, file.name))
                archived += 1
                if logger:
                    logger.info(f"Archived: {file.name}")
            except:
                pass
    
    return archived

def get_best_model_path(model_dir, extension=".joblib"):
    if not os.path.exists(model_dir):
        return None
    
    files = list(Path(model_dir).glob(f"*{extension}"))
    if not files:
        return None
    
    return str(max(files, key=lambda x: x.stat().st_mtime))

def get_directory_size_mb(directory):
    if not os.path.exists(directory):
        return 0
    
    total = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
    return total / (1024 * 1024)

def create_backup(source_path, backup_dir, logger=None):
    import shutil
    if not os.path.exists(source_path):
        return None
    
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"{Path(source_path).stem}_{timestamp}{Path(source_path).suffix}")
    
    try:
        shutil.copy2(source_path, backup_path)
        if logger:
            logger.info(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        if logger:
            logger.error(f"Backup failed: {str(e)}")
        return None

def get_dataframe_memory_usage_mb(df):
    return df.memory_usage(deep=True).sum() / (1024 * 1024)