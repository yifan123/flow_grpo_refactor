# Flow-GRPO Reward Module

Clean, minimal interface for reward evaluation. All training uses `multi_score`.

## Interface

```python
from flow_grpo.reward import multi_score
```

### `multi_score(device, score_dict)`

**Parameters:**
- `device`: GPU device (e.g., "cuda")  
- `score_dict`: Dictionary of reward weights (e.g., `{"pickscore": 0.5, "aesthetic": 0.3}`)

**Supported rewards:**
- `"aesthetic"` - CLIP-based aesthetic quality
- `"clipscore"` - CLIP text-image similarity  
- `"pickscore"` - Human preference scoring
- `"imagereward"` - T2I alignment scoring
- `"ocr"` - OCR-based text evaluation
- `"qwenvl"` - Multimodal LLM scoring
- `"geneval"` - Compositional generation evaluation
- `"deqa"` - Image quality assessment
- `"unifiedreward"` - State-of-the-art multimodal reward
- `"jpeg_compressibility"` - Compression-based quality
- `"image_similarity"` - Image-to-image similarity

## Usage

```python
from flow_grpo.reward import multi_score

# Multi-reward
reward_fn = multi_score("cuda", {"pickscore": 0.7, "aesthetic": 0.3})

# Single reward  
reward_fn = multi_score("cuda", {"aesthetic": 1.0})

# Use in training
scores, metadata = reward_fn(images, prompts, metadata)
```

## Remote Reward Services

Some reward models run as remote services:

- **GenEval**: Requires reward-server setup (port 18085)
- **DeQA**: Remote evaluation service (port 18086)
- **UnifiedReward**: SGLang server deployment (port 17140)

See the main README for setup instructions for these services.

## Dependencies

Different reward models have different dependencies:
- **PickScore**: No additional dependencies
- **OCR**: PaddleOCR (`pip install paddlepaddle-gpu paddleocr`)
- **ImageReward**: ImageReward package (`pip install image-reward`)
- **UnifiedReward**: SGLang (`pip install "sglang[all]"`)

Install only the dependencies for reward models you plan to use to avoid version conflicts.