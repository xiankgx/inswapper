import glob
import gradio as gr
import os
from itertools import product

from PIL import Image
from restoration import *
from swapper import *

SWAP_MODEL_PATH = "./checkpoints/inswapper_128.onnx"
RESTORE_MODEL_PATH = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"


class Inswapper():

    def __init__(
        self,
        swap_ckpt_path=SWAP_MODEL_PATH,
        restore_ckpt_path=RESTORE_MODEL_PATH,
        device=None,
    ):
        self.swap_ckpt_path = swap_ckpt_path
        self.restore_ckpt_path = restore_ckpt_path
        
        check_ckpts()
        self.upsampler = set_realesrgan()

        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.device = device

        # Load CodeFormer for face restoration
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        checkpoint = torch.load(self.restore_ckpt_path)["params_ema"]
        self.codeformer_net.load_state_dict(checkpoint)
        self.codeformer_net.eval()

    def swap_face(
        self,
        source_img: Image.Image,
        target_img: Image.Image,
    ) -> Image.Image:
        # Swap face
        result_image = process(
            source_img=[source_img],
            target_img=target_img,
            source_indexes='-1',
            target_indexes='-1',
            model=self.swap_ckpt_path
        )
        return result_image

    def restore_face(
        self, 
        result_image: Image.Image, 
        enhance_bg: bool=True, 
        upsample_face: bool=True, 
        upscale: int=1, 
        codeformer_fidelity: float=0.5
    ) -> Image.Image:
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(
            result_image, 
            enhance_bg, 
            upsample_face, 
            upscale, 
            codeformer_fidelity,
            self.upsampler,
            self.codeformer_net,
            self.device)
        # print(f"result_image: {result_image}")
        result_image = Image.fromarray(result_image)
        return result_image
        
    def predict(
        self,         
        source_img: Image.Image,
        target_img: Image.Image,
        restore_face: bool,  
        enhance_bg: bool = True,
        upsample_face: bool = True,
        upscale: int = 1,
        codeformer_fidelity: float = 0.5,    
    ):
        img = self.swap_face(source_img, target_img)
        if restore_face:
            img = self.restore_face(
                img, 
                enhance_bg=enhance_bg, 
                upsample_face=upsample_face, 
                upscale=upscale, 
                codeformer_fidelity=codeformer_fidelity
            )
        return img


if __name__ == "__main__":
    predictor = Inswapper()

    test_images = glob.glob("test_images/**/*.*", recursive=True)
    test_images = list(filter(lambda p: any(os.path.splitext(p)[-1] in ext for ext in [".jpg", ".jpeg", ".png", ".webp"]), test_images))
    print(f"test_image: {len(test_images)}")

    samples = [list(t) for t in list(product(test_images, test_images)) if t[0] != t[1]]
    
    gr.close_all()
    with gr.Blocks() as demo:
        gr.Markdown("""Inswapper Demo""")

        with gr.Tab(f"Generate"):

            with gr.Row():

                with gr.Column():
                    src_image = gr.Image(
                        source="upload", type="pil", label="Source image"
                    )
                    tgt_image = gr.Image(
                        source="upload", type="pil", label="Target image"
                    )

                    restore_face = gr.Checkbox(label="Restore face", value=True)
                    upsample_face = gr.Checkbox(label="Upsample face", value=True)
                    background_enhance = gr.Checkbox(label="Enhance background", value=True)
                    upscale = gr.Slider(
                        label="Upscale", 
                        value=1,
                        minimum=1,
                        maximum=4,
                        step=1
                    )
                    codeformer_fidelity = gr.Slider(
                        label="CodeFormer fidelity", 
                        value=0.5,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1
                    )
                    
                    submit = gr.Button("Submit")
                   
                with gr.Column():
                    generated = gr.Image(label="Output", type="pil")

            examples = gr.Examples(
                samples,
                [src_image, tgt_image],
            )

            submit.click(
                predictor.predict,
                inputs=[
                    src_image,
                    tgt_image,
                    restore_face,   
                    upsample_face,
                    background_enhance, 
                    upscale,
                    codeformer_fidelity,
                ],
                outputs=[
                    generated
                ],
            )

    demo.launch(server_name="0.0.0.0", server_port=5001)