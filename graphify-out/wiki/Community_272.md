# Community 272

> 24 nodes

## Key Concepts

- **mm_utils.py** (16 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **process_anyres_image()** (8 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **._process_single_image_task()** (7 connections) — `python/sglang/srt/multimodal/processors/llava.py`
- **get_anyres_image_grid_shape()** (5 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **ensure_numpy()** (4 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **select_best_resolution()** (4 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **expand2square()** (4 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **process_images()** (4 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **resize_and_pad_image()** (3 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **divide_to_patches()** (3 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **unpad_image()** (3 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **unpad_image_shape()** (3 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **get_dp_encoder_lb_assignment()** (3 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **.image_processor()** (3 connections) — `python/sglang/srt/tokenizer/tiktoken_tokenizer.py`
- **load_image_from_base64()** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Convert torch.Tensor to numpy array if needed (v5 compat).** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Selects the best resolution from a list of possible resolutions based on the ori** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Resize and pad an image to a target resolution while maintaining aspect ratio.** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Divides an image into patches of a specified size.      Args:         image (PIL** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Calculate the shape of the image patch grid after the preprocessing for images o** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Process an image with variable resolutions.      Args:         image (PIL.Image.** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Unpads a PyTorch tensor of a padded and resized image.      Args:     tensor (to** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Unpads a PyTorch tensor of a padded and resized image     and returns the new sh** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`
- **Generate load balancing assignment and metadata     for distributing data across** (1 connections) — `python/sglang/srt/multimodal/mm_utils.py`

## Relationships

- [[Vision-Language Model Configs]] (7 shared connections)
- [[Community 59]] (3 shared connections)
- [[Community 167]] (1 shared connections)
- [[Anthropic/OpenAI API Entrypoints]] (1 shared connections)
- [[Community 106]] (1 shared connections)
- [[Community 415]] (1 shared connections)

## Source Files

- `python/sglang/srt/multimodal/mm_utils.py`
- `python/sglang/srt/multimodal/processors/llava.py`
- `python/sglang/srt/tokenizer/tiktoken_tokenizer.py`

## Audit Trail

- EXTRACTED: 65 (81%)
- INFERRED: 15 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*