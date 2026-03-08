import subprocess
import tempfile
import cv2
import threading


class VisionLLM:

    def __init__(self):

        print("Loading Vision LLM...")

        self.cli_path = "/home/adityaguha/llama.cpp/build/bin/llama-mtmd-cli"
        self.model_path = "models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
        self.mmproj_path = "models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

        self.scene_text = "Analyzing scene..."
        self.busy = False

        print("Vision LLM loaded successfully")

    def _run_vlm(self, frame, labels):

        self.busy = True

        try:

            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:

                cv2.imwrite(tmp.name, frame)

                prompt = f"Objects detected: {', '.join(labels)}. Describe the scene briefly."

                result = subprocess.run(
                    [
                        self.cli_path,
                        "-m", self.model_path,
                        "--mmproj", self.mmproj_path,
                        "--image", tmp.name,
                        "-ngl", "35",
                        "-c", "2048",
                        "-p", prompt
                    ],
                    capture_output=True,
                    text=True
                )

            output = result.stdout.split("\n")

            filtered = []

            for line in output:

                if (
                    line.startswith("[New LWP")
                    or "ggml_" in line
                    or "load_" in line
                    or "decoded" in line
                    or "main:" in line
                    or line.strip() == ""
                ):
                    continue

                filtered.append(line.strip())

            text = " ".join(filtered)

            if text:
                self.scene_text = text

        except Exception as e:

            self.scene_text = "VLM error"

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