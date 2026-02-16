import requests
import os
import base64
import yaml

class Nvidia:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.api_cfg = self.config['nvidia_api']
        self.params = self.config['parameters']
        
        # Load the persona text from the path provided in config.yaml
        self.persona_path = self.params.get('system_prompt', "config.txt")
        self.persona = ""
        if os.path.exists(self.persona_path):
            with open(self.persona_path, "r") as f:
                self.persona = f.read().strip()
        else:
            self.persona = self.persona_path # Fallback to literal string

        self.supported_formats = {
            "png": ["image/png", "image_url"],
            "jpg": ["image/jpeg", "image_url"],
            "jpeg": ["image/jpeg", "image_url"],
            "webp": ["image/webp", "image_url"],
            "mp4": ["video/mp4", "video_url"],
            "webm": ["video/webm", "video_url"],
            "mov": ["video/mov", "video_url"]
        }

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _get_extension(self, filename):
        return os.path.splitext(filename)[1][1:].lower()

    def _encode_media(self, media_file):
        with open(media_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _prepare_content(self, query, media_files):
        # BRUTE FORCE: Inject persona and memory context into the query text
        injected_query = (
            f"SYSTEM INSTRUCTIONS (ACT AS THIS PERSONA): {self.persona}\n\n"
            f"CURRENT USER QUESTION: {query}"
        )
        
        if not media_files:
            return injected_query
        
        content = [{"type": "text", "text": injected_query}]
        for media_file in media_files:
            if not media_file: continue
            ext = self._get_extension(media_file)
            if ext not in self.supported_formats:
                continue
                
            m_type, m_key = self.supported_formats[ext]
            base64_data = self._encode_media(media_file)
            
            content.append({
                "type": m_key,
                m_key: {"url": f"data:{m_type};base64,{base64_data}"}
            })
        return content

    def ask(self, query, memory=None, media_files=None):
        if media_files is None:
            media_files = []
        if memory is None:
            memory = []
        
        # Prepare content with the injected persona
        current_turn_content = self._prepare_content(query, media_files)
        
        headers = {
            "Authorization": f"Bearer {self.api_cfg['key']}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not self.params['stream'] else "text/event-stream",
        }

        # Emptying the system role because we injected it into the user role
        messages = []
        messages.extend(memory)
        messages.append({"role": "user", "content": current_turn_content})

        payload = {
            "model": self.api_cfg['model'],
            "messages": messages,
            "max_tokens": self.params['max_tokens'],
            "temperature": self.params['temperature'],
            "top_p": self.params['top_p'],
            "frequency_penalty": self.params['frequency_penalty'],
            "presence_penalty": self.params['presence_penalty'],
            "stream": self.params['stream'],
        }

        try:
            response = requests.post(
                self.api_cfg['url'], 
                headers=headers, 
                json=payload, 
                timeout=self.params['timeout']
            )
            response.raise_for_status()
            
            if self.params['stream']:
                return response.iter_lines()
            else:
                data = response.json()
                return data['choices'][0]['message']['content']
                
        except Exception as e:
            return f"Nvidia API Error: {e}"