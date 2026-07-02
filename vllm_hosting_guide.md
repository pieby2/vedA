# Hosting the Fine-Tuned Vedaz Astrologer Model on a VPS Using vLLM

This document covers the full process of deploying our fine-tuned Qwen2.5 model on a Virtual Private Server using vLLM as the inference engine.

## Why vLLM?

vLLM is an open-source LLM serving framework that uses PagedAttention for efficient memory management. In practice, this means:

- **2-4x higher throughput** compared to naive HuggingFace generation
- **Continuous batching** — handles multiple users at once without blocking
- **OpenAI-compatible API** — drop-in replacement, our app code barely changes
- **LoRA hot-swapping** — can serve multiple LoRA adapters from one base model

For a production astrology chatbot serving Indian users, vLLM is a solid choice because we need low latency (users expect quick replies) and the ability to handle concurrent chats.

---

## Step 1: Choosing a VPS

For running a 1.5B-3B parameter model, we need a GPU VPS. Some options:

| Provider | GPU | VRAM | Monthly Cost (approx) |
|----------|-----|------|-----------------------|
| RunPod | RTX 4090 | 24GB | ~$0.44/hr |
| Vast.ai | RTX 3090 | 24GB | ~$0.20/hr |
| Lambda Labs | A10 | 24GB | ~$0.60/hr |
| AWS | g5.xlarge (A10G) | 24GB | ~$1.00/hr |
| Hetzner | (CPU only) | — | ~€40/mo (not ideal) |

For our Qwen2.5-1.5B model, even an RTX 3090 with 24GB VRAM is more than enough. If running the 3B model in fp16, we'd need about 6-7GB VRAM, so basically any GPU VPS works.

**My recommendation**: Start with Vast.ai or RunPod for development (pay-per-hour), then move to a reserved instance once we're happy with the setup.

---

## Step 2: Preparing the Model

After fine-tuning (see `finetune_qwen.py`), we have two options:

### Option A: Merge LoRA weights into base model (simpler)
```python
# run this after training
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("./vedaz-qwen-lora")
model.save_pretrained_merged(
    "./vedaz-qwen-merged",
    tokenizer,
    save_method="merged_16bit"
)
```

Then upload `vedaz-qwen-merged/` to the VPS.

### Option B: Keep LoRA separate (more flexible)
Upload just the LoRA adapter (~50MB) and let vLLM load it on top of the base model. This is better if we're experimenting with multiple adapters.

---

## Step 3: Setting Up the VPS

Once you SSH into your GPU VPS:

```bash
# basic setup
sudo apt update && sudo apt install -y python3-pip git

# create a virtualenv (keeps things clean)
python3 -m venv vllm-env
source vllm-env/bin/activate

# install vllm
pip install vllm

# if using unsloth models, also need
pip install transformers accelerate
```

### Upload the model
```bash
# option 1: scp from local machine
scp -r ./vedaz-qwen-merged/ user@your-vps-ip:/home/user/models/

# option 2: if model is on HuggingFace Hub
# just reference it directly in vllm, no upload needed

# option 3: use rsync for large models (resumable)
rsync -avP ./vedaz-qwen-merged/ user@your-vps-ip:/home/user/models/vedaz-qwen-merged/
```

---

## Step 4: Running vLLM Server

### Option A: Merged model (simplest)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/vedaz-qwen-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --dtype float16 \
    --gpu-memory-utilization 0.85
```

### Option B: Base model + LoRA adapter
```bash
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --lora-modules vedaz=/home/user/models/vedaz-qwen-lora \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --max-lora-rank 16 \
    --dtype float16
```

The server starts up and exposes an OpenAI-compatible API at `http://your-vps-ip:8000`.

### Key flags explained:
- `--gpu-memory-utilization 0.85`: use 85% of VRAM, leaves some headroom
- `--max-model-len 2048`: our chats are short, no need for 8k context
- `--dtype float16`: good balance of speed and quality for this model size
- `--enable-lora`: lets us hot-swap adapters without restarting the server

---

## Step 5: Testing the API

Once the server is running, test it with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/user/models/vedaz-qwen-merged",
    "messages": [
      {"role": "system", "content": "You are Vedaz AI Vedic astrologer. Compassionate, non-fatalistic."},
      {"role": "user", "content": "Meri shaadi kab hogi? DOB 5 March 1995, 3:15 PM, Delhi."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

Or from Python (this is what our app backend would use):

```python
from openai import OpenAI

# point to our vllm server instead of openai
client = OpenAI(
    base_url="http://your-vps-ip:8000/v1",
    api_key="not-needed"  # vllm doesn't check this by default
)

response = client.chat.completions.create(
    model="/home/user/models/vedaz-qwen-merged",
    messages=[
        {"role": "system", "content": "You are Vedaz's AI Vedic astrologer..."},
        {"role": "user", "content": "Mera career kab set hoga?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

The beauty here is that since vLLM exposes an OpenAI-compatible API, switching from OpenAI/Gemini to our self-hosted model in the app is literally just changing the `base_url`. No other code changes needed.

---

## Step 6: Running in Production

For a real deployment, we'd want a few more things:

### Use systemd to keep it running
```bash
sudo nano /etc/systemd/system/vllm-vedaz.service
```

```ini
[Unit]
Description=vLLM Vedaz Astrologer
After=network.target

[Service]
User=user
WorkingDirectory=/home/user
ExecStart=/home/user/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/vedaz-qwen-merged \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 2048 --dtype float16
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable vllm-vedaz
sudo systemctl start vllm-vedaz
sudo systemctl status vllm-vedaz  # check it's running
```

### Put nginx in front for HTTPS
```nginx
server {
    listen 443 ssl;
    server_name api.vedaz.ai;

    ssl_certificate /etc/letsencrypt/live/api.vedaz.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.vedaz.ai/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 120s;  # LLM generation can take a bit
    }
}
```

### Monitor GPU usage
```bash
# quick check
nvidia-smi

# continuous monitoring
watch -n 1 nvidia-smi

# or use nvitop (nicer UI)
pip install nvitop
nvitop
```

---

## Cost Estimate

For our use case (Qwen2.5-1.5B, serving Indian users mostly evenings/nights):

- **Vast.ai RTX 3090**: ~$0.20/hr × 24hr × 30 days = **~$144/month**
- **RunPod RTX 4090**: ~$0.44/hr × 24hr × 30 days = **~$317/month**
- With auto-scaling (spin down during low traffic 6am-2pm IST): **~$80-100/month**

For a startup, starting with a Vast.ai spot instance and manually scaling is reasonable. Once traffic is predictable, move to a reserved instance.

---

## What I'd Do Differently With More Time

1. **Quantize the model to GPTQ/AWQ** — would cut VRAM usage in half and allow running on cheaper GPUs (even an RTX 3060 12GB)
2. **Set up proper load testing** with locust or k6 to find the throughput ceiling
3. **Add a safety filter layer** between vLLM and the user — a lightweight classifier that checks responses before they go out, as a second line of defense beyond what the model learned
4. **Use Docker** for the deployment instead of bare metal — makes it reproducible across VPS providers
5. **Set up Prometheus + Grafana** for monitoring request latency, token throughput, and error rates
