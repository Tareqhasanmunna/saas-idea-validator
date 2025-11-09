import os
import json
import threading
import time

class RLController:
    def __init__(self):
        # Controller ফাইলের পথ
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.command_file = os.path.join(self.project_root, "src/rl/.controller_command")
        self.state_file = os.path.join(self.project_root, "src/rl/.controller_state")
        self.lock = threading.Lock()

    def _write_command(self, cmd):
        with self.lock:
            try:
                with open(self.command_file, "w", encoding="utf-8") as f:
                    json.dump({"command": cmd}, f)
                return {"success": True, "message": f"Command '{cmd}' sent."}
            except Exception as e:
                return {"success": False, "message": f"Failed to send command: {str(e)}"}

    def run(self):
        return self._write_command("run")

    def pause(self):
        return self._write_command("pause")

    def stop(self):
        return self._write_command("stop")

    def get_status(self):
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state
        except FileNotFoundError:
            return None

    def status_string(self):
        state = self.get_status()
        if state is None:
            return "No RL state available."
        s = f"Status: {state.get('status')}\n"
        s += f"Paused: {state.get('paused')}\n"
        s += f"Episodes Completed: {state.get('episodes_completed')}\n"
        s += f"Cycle Count: {state.get('cycle_count')}\n"
        s += f"Last Cycle Time: {state.get('last_cycle')}\n"
        s += f"Error Count: {state.get('error_count')}\n"
        return s
