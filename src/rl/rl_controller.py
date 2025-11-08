import os
import json
import time
from datetime import datetime
import logging

class RLController:
    def __init__(self, state_file="src/rl/.controller_state", command_file="src/rl/.controller_command"):
        self.state_file = state_file
        self.command_file = command_file
        self.logger = logging.getLogger(__name__)
        os.makedirs(os.path.dirname(command_file), exist_ok=True)
    
    def send_command(self, command):
        if command not in ['run', 'pause', 'stop']:
            return {'success': False, 'message': f"Invalid command: {command}"}
        
        try:
            cmd_data = {"command": command, "timestamp": datetime.now().isoformat()}
            with open(self.command_file, "w") as f:
                json.dump(cmd_data, f, indent=2)
            self.logger.info(f"[CONTROLLER] Command sent: {command}")
            return {'success': True, 'message': f"Command '{command}' sent"}
        except Exception as e:
            return {'success': False, 'message': f"Error: {str(e)}"}
    
    def get_status(self):
        try:
            if not os.path.exists(self.state_file):
                return None
            with open(self.state_file, "r") as f:
                return json.load(f)
        except:
            return None
    
    def is_running(self):
        status = self.get_status()
        return status is not None and status.get('status') not in ['initialized', 'stopped']
    
    def is_paused(self):
        status = self.get_status()
        return status is not None and status.get('paused', False)
    
    def wait_for_command_processed(self, timeout=5):
        start = time.time()
        while time.time() - start < timeout:
            if not os.path.exists(self.command_file):
                return True
            time.sleep(0.5)
        return False
    
    def run(self):
        return self.send_command('run')
    
    def pause(self):
        return self.send_command('pause')
    
    def stop(self):
        return self.send_command('stop')
    
    def status_string(self):
        status = self.get_status()
        if status is None:
            return "RL brain not running"
        lines = [
            f"Status: {status.get('status', 'unknown')}",
            f"Paused: {status.get('paused', False)}",
            f"Cycles: {status.get('cycle_count', 0)}",
            f"Episodes: {status.get('episodes_completed', 0)}",
            f"Last cycle: {status.get('last_cycle', 'Never')}",
            f"Errors: {status.get('error_count', 0)}"
        ]
        return "\n".join(lines)

def main():
    import sys
    logging.basicConfig(level=logging.INFO)
    controller = RLController()
    
    if len(sys.argv) < 2:
        print("Usage: python rl_controller.py [run|pause|stop|status]")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'status':
        status = controller.get_status()
        if status:
            print("\n[RL STATUS]")
            print(controller.status_string())
        else:
            print("RL brain not running")
    elif command in ['run', 'pause', 'stop']:
        result = controller.send_command(command)
        print(result['message'])
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()