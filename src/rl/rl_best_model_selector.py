import os
import pandas as pd
import joblib
import logging
from datetime import datetime
import json
import yaml

logger = logging.getLogger(__name__)

class RLModelSelector:
    def __init__(self, rl_models_dir='models/rl_models', best_rl_model_dir='models/best_rl_model',
                 config_file='config.yaml', logger_obj=None):
        self.rl_models_dir = rl_models_dir
        self.best_rl_model_dir = best_rl_model_dir
        self.logger = logger_obj or logger
        
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.retention_policy = self.config.get('retention', {})
        self.rl_config = self.config.get('rl', {})
        
        os.makedirs(rl_models_dir, exist_ok=True)
        os.makedirs(best_rl_model_dir, exist_ok=True)
        
        self.performance_log = os.path.join(rl_models_dir, '_performance_log.json')
        self.performance_history = self._load_performance_history()
        
        self.logger.info(f"[RL SELECTOR] Initialized")
    
    def _load_performance_history(self):
        if os.path.exists(self.performance_log):
            try:
                with open(self.performance_log, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_performance_history(self):
        try:
            with open(self.performance_log, "w") as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"[RL SELECTOR] Error saving: {str(e)}")
    
    def record_model_performance(self, model_name, metrics, episode_num=None):
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'episode': episode_num
        }
        
        self.performance_history[model_name].append(record)
        self._save_performance_history()
        
        self.logger.info(f"[RL SELECTOR] Recorded: {model_name} - accuracy={metrics.get('accuracy', 0):.4f}")
    
    def get_model_best_metric(self, model_name, metric='accuracy'):
        if model_name not in self.performance_history:
            return 0
        
        records = self.performance_history[model_name]
        values = [r['metrics'].get(metric, 0) for r in records]
        
        return max(values) if values else 0
    
    def select_best_model(self, metric='accuracy'):
        self.logger.info(f"[RL SELECTOR] Selecting best model (metric: {metric})")
        
        if not self.performance_history:
            self.logger.warning("[RL SELECTOR] No performance history")
            return None
        
        best_model = None
        best_value = -1
        
        for model_name, records in self.performance_history.items():
            if not records:
                continue
            
            latest_record = records[-1]
            value = latest_record['metrics'].get(metric, 0)
            
            if value > best_value:
                best_value = value
                best_model = model_name
        
        if best_model:
            model_path = os.path.join(self.rl_models_dir, best_model)
            self.logger.info(f"[RL SELECTOR] ✓ Best model: {best_model} ({metric}={best_value:.4f})")
            
            return {
                'model_name': best_model,
                'model_path': model_path,
                'metric_value': best_value,
                'metric': metric
            }
        
        return None
    
    def copy_best_to_serving(self, best_model_info):
        if not best_model_info:
            return False
        
        try:
            import shutil
            
            source = best_model_info['model_path']
            dest = os.path.join(self.best_rl_model_dir, best_model_info['model_name'])
            
            if os.path.exists(source):
                shutil.copy2(source, dest)
                self.logger.info(f"[RL SELECTOR] ✓ Copied to serving: {dest}")
                return True
            else:
                self.logger.error(f"[RL SELECTOR] Source not found: {source}")
                return False
        
        except Exception as e:
            self.logger.error(f"[RL SELECTOR] Copy error: {str(e)}")
            return False
    
    def cleanup_old_models(self):
        try:
            keep_count = self.rl_config.get('rl_model_retention_count', 10)
            
            model_files = []
            for f in os.listdir(self.rl_models_dir):
                if f.endswith('.pkl') and not f.startswith('_'):
                    full_path = os.path.join(self.rl_models_dir, f)
                    mtime = os.path.getmtime(full_path)
                    model_files.append((f, full_path, mtime))
            
            model_files.sort(key=lambda x: x[2], reverse=True)
            
            if len(model_files) > keep_count:
                to_delete = model_files[keep_count:]
                
                for model_name, model_path, _ in to_delete:
                    try:
                        os.remove(model_path)
                        self.logger.info(f"[RL SELECTOR] Deleted: {model_name}")
                        if model_name in self.performance_history:
                            del self.performance_history[model_name]
                    except:
                        pass
                
                self._save_performance_history()
            
            kept = min(len(model_files), keep_count)
            self.logger.info(f"[RL SELECTOR] Cleanup: kept {kept}, deleted {len(model_files) - kept}")
        
        except Exception as e:
            self.logger.error(f"[RL SELECTOR] Cleanup error: {str(e)}")
    
    def get_model_rankings(self, metric='accuracy', top_n=10):
        rankings = []
        
        for model_name, records in self.performance_history.items():
            if records:
                latest = records[-1]
                value = latest['metrics'].get(metric, 0)
                rankings.append((model_name, value))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings[:top_n]
    
    def generate_report(self, output_file=None):
        report = []
        report.append("=" * 80)
        report.append("RL MODEL REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        best = self.select_best_model()
        if best:
            report.append(f"\n[BEST]")
            report.append(f"  Name: {best['model_name']}")
            report.append(f"  Accuracy: {best['metric_value']:.4f}")
        
        report.append(f"\n[TOP MODELS]")
        rankings = self.get_model_rankings(top_n=10)
        for i, (name, value) in enumerate(rankings, 1):
            report.append(f"  {i}. {name}: {value:.4f}")
        
        report_str = "\n".join(report)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_str)
            self.logger.info(f"[RL SELECTOR] Report saved: {output_file}")
        
        return report_str