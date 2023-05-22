from transformers import YolosImageProcessor, YolosForObjectDetection, pipeline
import gradio as gr

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


def predict(x):
    obj_detector = pipeline("object-detection", model=model,
                            feature_extractor=image_processor)
    return (str(obj_detector(x)))


with gr.Blocks() as demo:
    gr.Markdown("# Detect object from a given image ! ")
    image = gr.Image(label="Input Image", type="pil")
    output = gr.Textbox(label="Output Text")
    generate_btn = gr.Button("Generate")
    generate_btn.click(fn=predict, inputs=image, outputs=output)
    gr.Markdown("## Examples")
    gr.Examples(examples=["example.jpg"],
                cache_examples=True,
                inputs=image,
                outputs=output,
                fn=predict)


demo.launch(server_name="0.0.0.0")

"""
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(test(image))
"""
