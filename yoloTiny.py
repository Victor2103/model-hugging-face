from transformers import YolosImageProcessor, YolosForObjectDetection, DetrImageProcessor, DetrForObjectDetection, pipeline
import gradio as gr
# import requests
from PIL import ImageDraw
import torch

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)


models = ['hustvl/yolos-small', 'hustvl/yolos-tiny', 'facebook/detr-resnet-50']


def predict(im, option_model):
    if option_model == 'facebook/detr-resnet-50':
        image_processor = DetrImageProcessor.from_pretrained(option_model)
        model = DetrForObjectDetection.from_pretrained(option_model)
    else:
        model = YolosForObjectDetection.from_pretrained(option_model)
        image_processor = YolosImageProcessor.from_pretrained(option_model)
    with torch.no_grad():
        inputs = image_processor(images=im, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes)[0]
    draw = ImageDraw.Draw(im)
    text = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")
        text += f"Detected {model.config.id2label[label.item()]} with confidence "
        text += f"{round(score.item(), 3)} at location {box} \n"
    return ([text, im])


title = "Object Detection"

with gr.Blocks(title="Object-detection") as demo:
    gr.Markdown("# Detect object from a given image ! ")
    with gr.Column():
        options = gr.Dropdown(
            choices=models, label='Select Object Detection Model', show_label=True)
    with gr.Row():
        image = gr.Image(label="Input Image", type="pil")
        output_text = gr.Textbox(label="Output Text")
        output_image = gr.Image(label="Output Image", type="pil")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=[image, options], outputs=[
                       output_text, output_image])
    gr.Markdown("## Examples")
    gr.Examples(examples=["examples/example_1.jpg", "examples/example_2.jpg", "examples/example_3.jpg", "examples/example_4.jpg"],
                cache_examples=True,
                inputs=image,
                outputs=[output_text, output_image],
                fn=predict)

demo.launch(server_name="0.0.0.0")
