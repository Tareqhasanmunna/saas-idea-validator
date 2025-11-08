import os
import time
import json
import yaml
import logging
from datetime import datetime
import signal

from utils.helpers import setup_rl_logger, cleanup_old_files, cleanup_directory_by_count, archive_files, get_directory_size_mb
from raw_data_merge import merge_raw_batches
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.training.sl_training.main_sl import train_sl_model
from src.training.sl_training.best_model_finder import find_and_copy_best_model

config_path = 'config.yaml'
with open(config_path, "r", encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

class RLBrain:
    def __init__(self):
        self.config = CONFIG
        self.state = "initialized"
        self.paused = False
        self.stop_flag = False
        self.episodes_completed = 0
        self.cycle_count = 0
        self.last_cycle_time = None
        self.error_count = 0
        
        log_file = self.config['logging'].get('rl_log_file', 'src/rl/logs/rl_controller.log')
        self.logger = setup_rl_logger(log_file, logger_name="RLBrain")
        
        self.state_file = "src/rl/.controller_state"
        self.command_file = "src/rl/.controller_command"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(self.config['paths'].get('rl_models_dir', 'models/rl_models'), exist_ok=True)
        os.makedirs(self.config['paths'].get('best_rl_model_dir', 'models/best_rl_model'), exist_ok=True)
        
        self.preprocessor = PreprocessingPipeline(logger_obj=self.logger)
        
        self.logger.info("=" * 80)
        self.logger.info("[RL BRAIN] Initialized")
        self.logger.info("=" * 80)
    
    def _save_state(self):
        state = {
            "status": self.state,
            "paused": self.paused,
            "episodes_completed": self.episodes_completed,
            "cycle_count": self.cycle_count,
            "last_cycle": self.last_cycle_time,
            "timestamp": datetime.now().isoformat(),
            "error_count": self.error_count
        }
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _read_commands(self):
        if not os.path.exists(self.command_file):
            return None
        try:
            with open(self.command_file, "r") as f:
                cmd_data = json.load(f)
            os.remove(self.command_file)
            return cmd_data.get("command")
        except:
            return None
    
    def _handle_command(self, command):
        if command == "pause":
            self.logger.info("[COMMAND] Pause")
            self.paused = True
            self.state = "paused"
        elif command == "run":
            self.logger.info("[COMMAND] Run")
            self.paused = False
            self.state = "running"
        elif command == "stop":
            self.logger.info("[COMMAND] Stop")
            self.stop_flag = True
            self.state = "stopping"
    
    def _run_cycle(self):
        cycle_start = time.time()
        self.cycle_count += 1
        self.logger.info("=" * 80)
        self.logger.info(f"[CYCLE #{self.cycle_count}] Starting")
        self.logger.info("=" * 80)
        
        self.state = "running"
        self.error_count = 0
        
        try:
            merged = merge_raw_batches()
            if not merged['success']:
                self.error_count += 1
                raise Exception("Merge failed")
            
            result = train_sl_model(logger_obj=self.logger)
            if not result['success']:
                self.error_count += 1
                raise Exception("SL training failed")
            
            result = find_and_copy_best_model(logger_obj=self.logger)
            if not result['success']:
                self.error_count += 1
                raise Exception("Model selection failed")
            
            cycle_time = time.time() - cycle_start
            self.episodes_completed += 1
            self.last_cycle_time = datetime.now().isoformat()
            self.state = "idle"
            
            self.logger.info("=" * 80)
            self.logger.info(f"[CYCLE #{self.cycle_count}] ✓ COMPLETE in {cycle_time:.1f}s")
            self.logger.info("=" * 80)
            
            return True
        
        except Exception as e:
            self.logger.error(f"[CYCLE #{self.cycle_count}] ✗ FAILED: {str(e)}")
            self.state = "error"
            self.error_count += 1
            return False
    
    def run(self):
        self.logger.info("[RL BRAIN] Starting main loop...")
        
        def signal_handler(sig, frame):
            self.logger.info("[RL BRAIN] Received shutdown signal")
            self.stop_flag = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while not self.stop_flag:
            cmd = self._read_commands()
            if cmd:
                self._handle_command(cmd)
            
            self._save_state()
            
            if self.paused:
                self.logger.debug("[RL BRAIN] Paused")
                time.sleep(5)
                continue
            
            if self.state == "initialized" or self.state == "idle":
                if self.config['rl'].get('run_on_startup', True) and self.cycle_count == 0:
                    self.logger.info("[RL BRAIN] Running on startup...")
                    self._run_cycle()
                else:
                    mode = self.config['rl'].get('mode', 'test')
                    delay = self.config['rl'].get(f'delay_seconds_{mode}', 60)
                    
                    self.logger.info(f"[RL BRAIN] Waiting {delay}s until next cycle...")
                    
                    for _ in range(int(delay / 5)):
                        time.sleep(5)
                        cmd = self._read_commands()
                        if cmd:
                            self._handle_command(cmd)
                            break
                        if self.stop_flag:
                            break
                    
                    if not self.stop_flag:
                        self._run_cycle()
        
        self.logger.info("[RL BRAIN] Shutting down...")
        if self.config['rl'].get('pause_on_shutdown', True):
            self.paused = True
            self._save_state()
        
        self.logger.info("[RL BRAIN] Shutdown complete")

def main():
    brain = RLBrain()
    brain.run()

if __name__ == "__main__":
    main()