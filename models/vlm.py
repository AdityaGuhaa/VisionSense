class VisionLLM:

    def __init__(self):
        pass

    def describe_scene(self, objects):

        if len(objects) == 0:
            return "No objects detected"

        unique_objects = list(set(objects))

        description = "I see: " + ", ".join(unique_objects)

        return description