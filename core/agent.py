import os
import torch
import transformers
from pathlib import Path
import yaml
import json
from typing import Dict, Any, List

# Default adapter path (fine-tuned LoRA)
DEFAULT_ADAPTER_PATH = str(Path(__file__).parent.parent / "outputs" / "cosmos-reason2-2b-safety-lora")

# Optional label lookup for demos (AI Hub sample layout). Off by default:
# it injects ground-truth labels into the prompt, so outputs produced with it
# must not be used to judge model accuracy.
DEFAULT_LABEL_DIR = Path(__file__).parent.parent.parent / "Sample" / "02.라벨링데이터"


class CosmosGuardianAgent:
    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Reason2-2B",
        device: str = None,
        use_label_grounding: bool = False,
        label_dir: str = None,
    ):
        self.model_id = model_id
        self.use_label_grounding = use_label_grounding
        self.label_dir = Path(label_dir) if label_dir else DEFAULT_LABEL_DIR
        self.adapter_loaded = False
        # Auto-detect GPU: prefer cuda:0, fallback to cpu
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = Path(__file__).parent.parent / "prompts" / "safety_inspector.yaml"
        with open(prompt_path, "r") as f:
            return yaml.safe_load(f)

    def load_model(self, adapter_path: str = None):
        print(f"Loading model {self.model_id} on {self.device}...")
        
        # Optimize for H100/3090 using bfloat16 if supported
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            self.model_id, 
            torch_dtype=compute_dtype, 
            device_map=self.device, 
            attn_implementation="sdpa",
            trust_remote_code=True
        )
        
        if adapter_path and os.path.exists(adapter_path):
            try:
                print(f"Integrating Safety Specialist Adapter from {adapter_path}...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.adapter_loaded = True
                print("LoRA adapter loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load adapter, using base model. Error: {e}")
            
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_id)
        
        # Balanced vision tokens for stability on 2B model
        self.processor.image_processor.min_pixels = 224 * 224
        self.processor.image_processor.max_pixels = 448 * 448 # Slightly higher for H100
        
        if adapter_path and not self.adapter_loaded:
            print(f"Adapter not found at {adapter_path}; running the base model (zero-shot).")
        print(f"Model loaded with {str(compute_dtype)}. Specialization: {'LoRA adapter' if self.adapter_loaded else 'Zero-shot'}")

    def analyze_media(self, media_path: str, safety_context: str = "General Industrial Safety"):
        import time
        start_time = time.time()
        
        if not self.model:
            yield {"stage": "model_loading", "detail": "Initializing NVIDIA Cosmos Reason 2..."}
            self.load_model(adapter_path=DEFAULT_ADAPTER_PATH)

        torch.cuda.empty_cache()
        media_type = "video" if Path(media_path).suffix.lower() in [".mp4", ".avi", ".mov", ".webm"] else "image"
        
        yield {"stage": "preprocessing", "detail": f"Extracting keyframes for rapid analysis..."}
        
        # --- OPTIONAL LABEL GROUNDING (demo only, off by default) ---
        grounding_data = ""
        video_name = Path(media_path).stem
        label_files = []
        if self.use_label_grounding and self.label_dir.exists():
            label_files = list(self.label_dir.rglob(f"{video_name}.json"))
        if label_files:
            try:
                with open(label_files[0], 'r', encoding='utf-8-sig') as f:
                    label_json = json.load(f)
                    meta = label_json.get("meta_information", {})
                    human = label_json.get("object_information_human", {})
                    facility = label_json.get("object_information_facility", {})
                    grounding_data = f"[Expert Ground Truth]: This scene is from a {meta.get('environment', 'factory')}. Event: {meta.get('event', 'safety event')}. Facility involved: {facility.get('facility_name', 'machine')}. Worker ID: {human.get('id', 'h0001')}."
                    print(f"Label grounding active for {video_name} (demo mode, not for evaluation)")
            except Exception as e:
                print(f"Grounding lookup failed: {e}")

        # --- INCIDENT FOCUS (Prompt Engineering) ---
        incident_instruction = ""
        if grounding_data:
            event_type = grounding_data.split("Event: ")[1].split(".")[0]
            incident_instruction = f"\n\nIMPORTANT: An active '{event_type}' incident is OCCURRING in this video. Analyze the physical sequence of this SPECIFIC incident, not just potential risks. Focus on the interaction between the worker and the {video_name}."

        user_prompt = self.prompts["user_prompt_template"].format(safety_context=safety_context) + incident_instruction

        content = []
        if media_type == "video":
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(media_path)
            frames = []
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Sample 6 frames - Optimal balance for 2B physical reasoning
                indices = [int(i * total_frames / 6) for i in range(6)]
                for i in range(total_frames):
                    ret, frame = cap.read()
                    if not ret: break
                    if i in indices:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 336x336 provides enough detail for 'jam' while keeping context manageable
                        img = Image.fromarray(frame_rgb).resize((336, 336))
                        frames.append(img)
                cap.release()
            
            content.append({
                "type": "video",
                "video": frames,
                "fps": 1.0,
            })
        else:
            content.append({
                "type": "image",
                "image": media_path,
            })
            
        content.append({"type": "text", "text": user_prompt})
        
        system_msg = self.prompts["system_instructions"]
        if grounding_data:
            system_msg += f"\n\nCRITICAL CONTEXT: {grounding_data}"

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        print(f"Input tokens: {inputs.input_ids.shape} | Preprocessing time: {time.time() - start_time:.2f}s")

        yield {"stage": "inference", "detail": f"Executing Chain-of-Thought ({self.device})..."}
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1536,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        full_output = ""
        last_yield_time = time.time()
        
        for new_text in streamer:
            full_output += new_text
            # Throttle yields to avoid overwhelming the network, but keep it feeling fast
            if time.time() - last_yield_time > 0.1:
                yield {"stage": "streaming_reasoning", "chunk": full_output}
                last_yield_time = time.time()
        
        thread.join()
        
        yield {"stage": "postprocessing", "detail": "Finalizing safety report..."}
        
        result = self._parse_output(full_output)
        print(f"Total cycle time: {time.time() - start_time:.2f}s")
        yield {"stage": "complete", "result": result}

    def _parse_output(self, output_text: str) -> Dict[str, Any]:
        import re
        try:
            # Find the true start of the JSON
            start_idx = output_text.find('{')
            if start_idx == -1:
                return {"raw_output": output_text, "error": "No JSON found"}
            
            clean_text = output_text[start_idx:].strip()

            def bulletproof_repair(text):
                # Remove common non-JSON trailing garbage
                text = text.split('```')[0].strip()
                
                # Close unquoted strings
                if text.count('"') % 2 != 0:
                    text += '"'
                
                # Remove trailing commas inside objects/arrays
                text = re.sub(r',\s*([\]}])', r'\1', text)
                # Remove trailing dangling commas or partial keys
                text = re.sub(r',\s*$', '', text)
                text = re.sub(r',\s*"\w+":\s*$', '', text)
                
                # Fix bracket stack
                stack = []
                for char in text:
                    if char == '{': stack.append('}')
                    elif char == '[': stack.append(']')
                    elif char == '}' or char == ']':
                        if stack and stack[-1] == char:
                            stack.pop()
                
                # Filter out the remaining closing brackets if they are already present
                # Actually, simpler to just count and append
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
                
                if open_braces > 0: text += '}' * open_braces
                if open_brackets > 0: text += ']' * open_brackets
                
                # Final check if it ends correctly, if not, force it
                if not text.endswith('}') and not text.endswith(']'):
                    text += '}'
                return text

            try:
                # Try raw first
                return json.loads(clean_text)
            except:
                # Try aggressive repair
                repaired = bulletproof_repair(clean_text)
                try:
                    return json.loads(repaired)
                except:
                    # Last resort: try to find the last valid object
                    last_obj_end = clean_text.rfind('}')
                    if last_obj_end != -1:
                        try:
                            return json.loads(clean_text[:last_obj_end+1])
                        except: pass
                    raise Exception("Final JSON extraction failed")
            
        except Exception as e:
            print(f"Error parsing JSON output: {e}")
            return {"raw_output": output_text, "error": "Parsing failed"}
