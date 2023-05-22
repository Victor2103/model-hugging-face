from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import gradio as gr
import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


def predict(x):
    inputs = image_processor(images=x, return_tensors="pt")
    outputs = model(**inputs)
    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # print results
    target_sizes = torch.tensor([x.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes)[0]
    returning = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        returning += f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box} \n"
    return (returning)


with gr.Blocks() as demo:
    gr.Markdown("# Detect object from a given image ! ")
    image = gr.Image(label="Input Image", type="pil")
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=image, outputs=output)
    gr.Markdown("## Examples")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    example=Image.open(requests.get(url, stream=True).raw)
    gr.Examples(examples=["example.jpg"],
                cache_examples=True,
                inputs=image,
                outputs=output,
                fn=predict)


demo.launch(server_name="0.0.0.0")
