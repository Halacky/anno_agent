from langchain.prompts import PromptTemplate

ANNOTATION_SYSTEM_PROMPT = """You are an expert annotation AI assistant. Your task is to analyze images and provide detailed, structured annotations based on the given requirements.

Always provide your annotations in valid JSON format with the following structure:
{
  "annotations": [
    {
      "object": "object_name",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": 0.95,
      "attributes": {}
    }
  ],
  "metadata": {
    "image_analysis": "brief description"
  }
}

Be precise, consistent, and follow the annotation guidelines provided."""

VIDEO_ANNOTATION_PROMPT = PromptTemplate(
    input_variables=["task_description", "frame_number", "total_frames"],
    template="""Analyze this video frame and provide annotations.

Task: {task_description}

Frame: {frame_number} of {total_frames}

Provide detailed annotations in JSON format as specified."""
)

VIDEO_WITH_KEYFRAMES_PROMPT = PromptTemplate(
    input_variables=["task_description", "frame_number", "keyframe_annotations"],
    template="""Analyze this video frame and provide annotations based on the reference keyframes.

Task: {task_description}

Frame number: {frame_number}

Reference keyframe annotations:
{keyframe_annotations}

Use the reference annotations to maintain consistency. Provide annotations in JSON format."""
)

IMAGE_ANNOTATION_PROMPT = PromptTemplate(
    input_variables=["task_description", "image_name"],
    template="""Analyze this image and provide annotations.

Task: {task_description}

Image: {image_name}

Provide detailed annotations in JSON format as specified."""
)

IMAGE_WITH_EXAMPLES_PROMPT = PromptTemplate(
    input_variables=["task_description", "image_name", "example_annotations"],
    template="""Analyze this image and provide annotations based on the provided examples.

Task: {task_description}

Image: {image_name}

Example annotations from reference images:
{example_annotations}

Follow the same annotation style and structure. Provide annotations in JSON format."""
)