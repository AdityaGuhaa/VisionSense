import subprocess
import cv2
import threading
import base64
import json
import urllib.request
import atexit
import time

class VisionLLM:
    def __init__(self):
        print("Loading Vision LLM Server...")
        self.cli_path = "/home/adityaguha/llama.cpp/build/bin/llama-server"
        self.model_path = "models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
        self.mmproj_path = "models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
        
        self.server_process = subprocess.Popen(
            [
                self.cli_path,
                "-m", self.model_path,
                "--mmproj", self.mmproj_path,
                "-ngl", "35",
                "-c", "2048",
                "--port", "8080"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        atexit.register(self.server_process.kill)
        
        # Give the server a few seconds to load the model into VRAM
        time.sleep(3)
        
        self.scene_text = "Analyzing scene..."
        self.busy = False
        print("Vision LLM loaded successfully")

    def _run_vlm(self, frame, labels):
        self.busy = True
        try:
            # Encode frame to jpeg base64
            _, buffer = cv2.imencode('.jpg', frame)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            prompt = f"Objects detected: {', '.join(labels)}. Describe the scene briefly in one short sentence."
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
                        ]
                    }
                ]
            }
            
            req = urllib.request.Request(
                "http://localhost:8080/v1/chat/completions",
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                text = res_data["choices"][0]["message"]["content"]
                if text:
                    self.scene_text = text.strip()
                    
        except Exception as e:
            print("VLM ERROR:", e)
        
        self.busy = False

    def describe_scene(self, frame, labels):
        if not self.busy:
            thread = threading.Thread(
                target=self._run_vlm,
                args=(frame.copy(), labels.copy())
            )
            thread.start()
        return self.scene_text