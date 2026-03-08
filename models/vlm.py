import subprocess
import tempfile
import cv2


class VisionLLM:

    def __init__(self):

        print("Loading Vision LLM...")

        # path to llama.cpp executable
        self.cli_path = "/home/adityaguha/llama.cpp/build/bin/llama-mtmd-cli"

        # model paths
        self.model_path = "models/qwen_vl/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
        self.mmproj_path = "models/qwen_vl/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"

        print("Vision LLM loaded successfully")

    def describe_scene(self, frame):

        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:

            cv2.imwrite(tmp.name, frame)

            process = subprocess.run(
                [
                    self.cli_path,
                    "-m", self.model_path,
                    "--mmproj", self.mmproj_path,
                    "--image", tmp.name,
                    "-ngl", "20",
                    "-c", "2048",
                    "-p", "Describe what is happening in this scene."
                ],
                capture_output=True,
                text=True
            )

        raw_output = process.stdout

        # Clean unwanted logs
        lines = raw_output.split("\n")

        filtered = []
        for line in lines:

            if (
                line.startswith("[New LWP") or
                "ggml_" in line or
                "load_" in line or
                "image slice" in line or
                "decoded" in line or
                "main:" in line or
                "WARN" in line or
                line.strip() == ""
            ):
                continue

            filtered.append(line.strip())

        cleaned_output = " ".join(filtered)

        # Fallback if nothing detected
        if cleaned_output == "":
            cleaned_output = "Analyzing scene..."

        return cleaned_output