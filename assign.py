from pathlib import Path
import cv2
from matplotlib import pyplot as plt
import cv2, os, sys
import numpy as np
from PIL import Image, ImageOps
from diffusers import StableDiffusionXLControlNetInpaintPipeline, DPMSolverMultistepScheduler, ControlNetModel, StableDiffusionXLInpaintPipeline
import torch
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from typing import Optional, Union, List, Tuple


class Preprocessor:
    segment_model_path = './sam_vit_h_4b8939.pth'
    detect_model_path = './yolov8x.pt'

    valid_products = [
        'suitcase',
        'handbag',
        'bottle',
        'car',
        'chair',
        'couch'
    ]

    def __init__(self, device='cuda'):
        self._device = device
        self._detect_model = YOLO(self.detect_model_path)
        self._detect_model_classes = self._detect_model.names

        sam = sam_model_registry["vit_h"](checkpoint=self.segment_model_path)
        sam.to(device=self._device)
        self._segment_model = SamPredictor(sam)

    @staticmethod
    def __resize(image: Image.Image, height: int, width: int) -> Image.Image:
        image = image.resize((width, height))
        return image

    def _detect(self, image):
        results = self._detect_model(image)
        classes = list(
            map(self._detect_model_classes.__getitem__, list(results[0].boxes.cls.cpu().numpy().astype(int))))
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

        if not len(boxes) > 0:
            return None, None

        area_covered = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 0])  # area of boxes
        box_no = np.argmax(area_covered)  # box index with max area
        box = boxes[box_no]
        class_ = classes[box_no]

        if class_ not in self.valid_products:
            return None, None

        box_width = boxes[box_no, 2] - boxes[box_no, 0]
        box_height = boxes[box_no, 3] - boxes[box_no, 1]
        image_height, image_width, _ = image.shape

        if box_width / image_width > 0.5 or box_height / image_height > 0.5:
            # ordering of the box coordinates are changed
            box_mod = [
                max(0, box[1] - int(image_height * 0.05)),
                max(0, box[0] - int(image_width * 0.05)),
                min(image_height - 1, box[3] + int(image_height * 0.05)),
                min(image_width - 1, box[2] + int(image_width * 0.05))
            ]
            return box_mod, class_

    def _segment(self, image, bbox):
        self._segment_model.set_image(image)
        x1, y1, x2, y2 = bbox  # [40, 640, 1920, 1210]
        input_points, input_labels, input_box = (None, None, np.array([[y1, x1, y2, x2]]))
        masks, scores, _ = self._segment_model.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box,
            multimask_output=False,
        )

        return masks[0].astype(np.uint8) * 255

    def __extract_subject(self, image: Union[Image.Image, np.ndarray]) \
            -> Tuple[Image.Image, List[int], Image.Image, str]:
        """
        Removes background from the image and generates the corresponding mask
        Args:
            image: image
            mask_blur: feather edges of mask by this amount

        Returns: Image with background removed, mask, class label

        """

        image = np.array(image)
        bbox, class_ = self._detect(image)
        if bbox is None:
            return None, None, None, None
        mask = self._segment(image, bbox)

        mask = np.array(mask)

        mask1 = np.expand_dims(mask / 255, -1)
        image = image * mask1

        # mask = (1 - mask[:, :, 0]) * 255  # remove the last axis, invert the mask

        image = Image.fromarray(image.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = ImageOps.invert(mask)

        return image, bbox, mask, class_

    @staticmethod
    def __handle_wrong_dimensions(d: int):
        # makes the dimension divisible by 8 by reducing it slightly
        return d - (d % 8)

    @staticmethod
    def __find_image_placeholder(canvas_height: int, canvas_width: int, image: Image.Image):
        ar = image.width / image.height

        if ar <= 1:
            leave_percent_left = 1 / 10
            leave_percent_right = 1 / 10
            leave_percent_top = 1 / 10
            leave_percent_bottom = 1 / 10
        else:
            leave_percent_left = 1 / 10
            leave_percent_right = 1 / 10
            leave_percent_top = 1 / 10
            leave_percent_bottom = 1 / 10

        return leave_percent_left, leave_percent_right, leave_percent_top, leave_percent_bottom

    @staticmethod
    def __get_edge(image: Image.Image) -> Image.Image:
        return Image.fromarray(cv2.Canny(np.array(image), 100, 200))

    def preprocess(self,
                   image: Image.Image,
                   height: int,
                   width: int
                   ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, str]:
        """
        Preprocess the image
        If ar is to be maintained and required height and width doesn't match with ar, padding is performed with `pad_with`

        Args:
        image: A PIL image object
        height: required height
        width: required width
        mask_blur: Feather the mask edges by this amount

        Returns:
        Resized image with background, image without background, mask, mask edge, class label
        """

        # TODO: check if image has 4 channels, then discard the last layer
        # TODO: Handle 8 divisibility in a better way without distorting ar
        # TODO: Handle maintain_ar
        # TODO: prevent exif tags from being read by PIL (try the image in 10)

        image = ImageOps.exif_transpose(image).convert("RGB")

        # check and fix 8 divisibility
        width = self.__handle_wrong_dimensions(width)
        height = self.__handle_wrong_dimensions(height)

        # remove background and generate mask
        image_without_bg, bbox, mask, class_label = self.__extract_subject(image=image)  # remove background
        if image_without_bg is None:
            return None, None, None, None, None

        x1, y1, x2, y2 = bbox  # self.__find_bounding_box(image=image_without_bg, mask=mask)  # find bounding box
        image_without_bg = image_without_bg.crop((y1, x1, y2, x2))  # crop image along the bounding box
        image_with_bg = image.crop((y1, x1, y2, x2))  # crop image along the bounding box
        mask = mask.crop((y1, x1, y2, x2))  # crop mask along the bounding box
        # edge = self.__get_edge(mask)

        # find image placeholder on canvas
        leave_percent_left, leave_percent_right, leave_percent_top, leave_percent_bottom = \
            self.__find_image_placeholder(canvas_height=height, canvas_width=width, image=image)

        # place image in image placeholder
        req_w, req_h = width, height  # required width and height of the canvas
        w, h = image_without_bg.size  # width and height of the given image
        a = req_w * (1 - leave_percent_right - leave_percent_left)
        b = req_h * (1 - leave_percent_top - leave_percent_bottom)

        resize_by = min(
            np.sqrt(a * b / w / h),
            a / w,
            b / h
        )

        image_without_bg = self.__resize(
            image_without_bg, height=int(image_without_bg.height * resize_by),
            width=int(image_without_bg.width * resize_by)
        )
        image_with_bg = self.__resize(
            image_with_bg, height=int(image_with_bg.height * resize_by), width=int(image_with_bg.width * resize_by)
        )
        mask = self.__resize(
            mask, height=int(mask.height * resize_by), width=int(mask.width * resize_by)
        )

        # mid point of the image after it will be placed on the canvas
        m = (req_h * leave_percent_top + b / 2, req_w * leave_percent_left + a / 2)
        place_image_at = [int(m[0] - image_without_bg.height / 2), int(m[1] - image_without_bg.width / 2)]

        canvas_image_without_bg = Image.fromarray(np.ones((req_h, req_w, 3), dtype=np.uint8))
        canvas_image_with_bg = Image.fromarray(np.ones((req_h, req_w, 3), dtype=np.uint8))
        canvas_mask = Image.fromarray(np.ones((req_h, req_w), dtype=np.uint8) * 255)
        canvas_edge = Image.fromarray(np.ones((req_h, req_w), dtype=np.uint8) * 255)

        canvas_image_without_bg.paste(image_without_bg, (place_image_at[1], place_image_at[0]))
        canvas_image_with_bg.paste(image_with_bg, (place_image_at[1], place_image_at[0]))
        canvas_mask.paste(mask, (place_image_at[1], place_image_at[0]))
        canvas_edge = self.__get_edge(canvas_mask)
        # canvas_edge.paste(edge, (place_image_at[1], place_image_at[0]))

        return canvas_image_with_bg, canvas_image_without_bg, canvas_mask, canvas_edge, class_label


class BGGenerator:
    def __init__(self, device='cuda'):
        self._device = device
        self._preprocessor = Preprocessor(self._device)
        self._control_nets = [
            self.__load_canny_cn(),
        ]
        #         self._vae = load_vae()

        # Stable Diffusion 1.5 StableDiffusionControlNetInpaintPipeline or StableDiffusionControlNetImg2ImgPipeline
        self._gen_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            controlnet=self._control_nets,
            torch_dtype=torch.float16,
            #             vae=self._vae,
        )

        self._gen_pipe.scheduler = self.__load_scheduler(self._gen_pipe.scheduler.config)
        self._gen_pipe.enable_model_cpu_offload(device=self._device)

        self._restore_pipe = StableDiffusionXLInpaintPipeline(
            vae=self._gen_pipe.vae,
            text_encoder=self._gen_pipe.text_encoder,
            text_encoder_2=self._gen_pipe.text_encoder_2,
            tokenizer=self._gen_pipe.tokenizer,
            tokenizer_2=self._gen_pipe.tokenizer_2,
            unet=self._gen_pipe.unet,
            scheduler=self._gen_pipe.scheduler
        )

        self._restore_pipe.enable_model_cpu_offload(device=self._device)

    def __load_canny_cn(self):
        model = ControlNetModel.from_pretrained(
            'diffusers/controlnet-canny-sdxl-1.0', torch_dtype=torch.float16
        ).to(self._device)
        return model

    @staticmethod
    def __load_scheduler(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

    def generate(
            self,
            image: Image.Image,
            prompt,
            height: Optional[int] = 1024,
            width: Optional[int] = 1024,
            neg_prompt: Optional[str] = None,
            steps: Optional[int] = 20,
            num_images: Optional[int] = 1,
            controlnet_weight: Optional[Union[float, List[float]]] = 0.3,
            cfg: Optional[float] = 7.0,
            denoise_strength: Optional[float] = 1.0
    ):

        """
        Generate a background for a given image

        Args:
            denoise_strength:
            cfg:
            image: PIL object, url, filepath
            prompt: Prompt describing the background
            height: height of the output image
            width: width of the output image
            neg_prompt: negative prompt
            steps: number of inference steps
            num_images: Number of images to generate
            controlnet_weight: Weights of controlnet

        Returns:

        """
        # TODO: add more customisation while calling

        image_with_bg, image_without_bg, image_mask, image_edge, class_label = self._preprocessor.preprocess(
            image,
            height=height,
            width=width
        )

        if image_with_bg is None:
            return [None] * 7

        control_images = [
            image_edge
        ]

        prompt_mod = class_label + ', ' + prompt

        print([prompt_mod, neg_prompt, steps, controlnet_weight, height, width, num_images, cfg, denoise_strength])
        generated_images = self._gen_pipe(
            prompt=prompt_mod,
            image=image_without_bg,
            control_image=control_images,
            mask_image=[image_mask],
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            controlnet_conditioning_scale=controlnet_weight,
            height=height,
            width=width,
            num_images_per_prompt=num_images,
            guidance_scale=cfg,
            strength=denoise_strength,
            #             generator=generator
        ).images

        generated_images1 = self.__post_process(
            prompt=prompt,
            generated_images=generated_images,
            image_with_bg=image_with_bg,
            image_mask=image_mask,
            height=height,
            width=width
        )

        return (
            generated_images1,  # generated image modified
            #             df,
            generated_images,  # generated image actual/distorted
            image,  # original input image
            image_with_bg,  # image with background used for generation
            image_without_bg,  # image without background used for generation
            image_mask,  # image mask used for generation
            image_edge  # image mask used for generation
        )

    def __post_process(self, prompt: str,
                       generated_images: Union[np.ndarray, Image.Image, List[np.ndarray], List[Image.Image]],
                       image_with_bg: Image.Image, image_mask: Image.Image, height: int, width: int):
        # paste the original image on top of generated image
        if not isinstance(generated_images, list):
            generated_images = [generated_images]

        b = []
        #         b1 = []

        #         mask = np.expand_dims(mask, -1)

        for img in generated_images:
            #             a, a1 = self.__fix_halos_n_edges(
            #                 prompt=prompt,
            #                 neg_prompt=None,
            #                 height=height,
            #                 width=width,
            #                 image=img,
            #                 mask=image_mask
            #             )
            image_mask = np.expand_dims(np.array(image_mask), axis=-1)
            a = np.array(img) * (image_mask / 255.0) + np.array(image_with_bg) * (1 - image_mask / 255.0)
            a = Image.fromarray(a.astype(np.uint8))
            b.append(a)
        #             b1.append(a1)

        return b  # , b1

    def __fix_halos_n_edges(self, prompt: str, neg_prompt: str, height: int, width: int,
                            image: Union[np.ndarray, Image.Image, List[np.ndarray], List[Image.Image]],
                            mask: Image.Image):
        # 1. blur the mask in two ways - low and heavy
        mask = 255 - np.array(mask)
        image = np.array(image)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_blurred_low = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_blurred_high = cv2.GaussianBlur(mask, (55, 55), 0)

        # 2. dilate and erode the blurred mask
        kernel_low_blur = np.ones((5, 5), np.uint8)
        kernel_high_blur = np.ones((55, 55), np.uint8)
        mask_eroded_low_blur = cv2.erode(mask_blurred_low, kernel_low_blur, iterations=1)
        mask_dilated_high_blur = cv2.dilate(mask_blurred_high, kernel_high_blur,
                                            iterations=2)  # dilate the image more than it is eroded

        h, w = height, width
        #         if h > w: # keep h at 1024
        #             h_new, w_new = (1024, int(w / h * 1024//8*8))
        #         else: # keep w at 1024
        #             h_new, w_new = (int(h / w * 1024//8*8), 1024)

        #         mask_image = cv2.resize(mask_dilated_high_blur, (w_new, h_new))  # RGB array (3 channels)
        mask_image = mask_dilated_high_blur
        # replace the masked area of the input image with the average color
        image_mean = image.reshape((h * w, 3)).mean(axis=0).astype(np.uint8)
        #         image_resized = cv2.resize(image, (w_new, h_new))
        image_resized = image
        in_image = np.where(mask_image < (127, 127, 127), image_resized, image_mean).astype(
            np.uint8)  # RGB array (3 channels)
        #         mask_eroded_low_blur = cv2.resize(mask_eroded_low_blur, (w_new, h_new))

        in_image = Image.fromarray(in_image)
        mask_image = Image.fromarray(mask_image)

        #         prompt = prompt if prompt is not None else ''
        #         neg_prompt = 'person, man, woman, human, people'

        gen_image = self._restore_pipe(
            prompt=prompt,
            neg_prompt=neg_prompt,
            image=in_image,
            height=h,
            width=w,
            mask_image=mask_image,
            guidance_scale=7.0,
            num_inference_steps=20,
            strength=1.0,
        ).images[0]

        gen_image_mod = (
                image_resized * (mask_eroded_low_blur / 255) + gen_image * (1 - mask_eroded_low_blur / 255)).astype(
            np.uint8)
        #         gen_image_mod = cv2.resize(gen_image_mod, (w, h))

        return Image.fromarray(gen_image_mod), gen_image