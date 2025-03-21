"""
Text-to-speech module for generating Hindi audio from English text.
"""
import os
import uuid
import logging
import tempfile
import threading
import queue
import asyncio
from typing import List

from gtts import gTTS
from googletrans import Translator

# Constants
MAX_CHUNK_SIZE = 1500
TRANSLATION_TIMEOUT = 60

# Set up logging
logger = logging.getLogger(__name__)

class TextToSpeechHindi:
    """Class for translating text to Hindi and generating speech."""
    
    def __init__(self):
        """Initialize translator and temp directory."""
        self.translator = Translator()
        self.temp_dir = tempfile.gettempdir()
        
    async def _translate_chunk(self, chunk: str) -> str:
        """Translate a single chunk of text asynchronously."""
        translation = await self.translator.translate(chunk, src='en', dest='hi')
        return translation.text
        
    def translate_to_hindi(self, text: str) -> str:
        """
        Translate text from English to Hindi.
        
        Args:
            text: English text to translate
            
        Returns:
            Hindi translation or empty string if translation fails
        """
        try:
            if not text:
                return ""
                
            # For short texts, try a simpler approach first
            if len(text) < 1000:
                try:
                    # Use alternative translation approach for shorter texts
                    from googletrans import Translator as SyncTranslator
                    sync_translator = SyncTranslator()
                    result = sync_translator.translate(text, src='en', dest='hi')
                    if hasattr(result, 'text') and result.text:
                        return result.text
                except Exception as e:
                    logger.warning(
                        f"Simple translation failed, falling back to async: {e}"
                    )
            
            # Continue with the chunked async approach for longer texts
            chunks = [
                text[i:i+MAX_CHUNK_SIZE] 
                for i in range(0, len(text), MAX_CHUNK_SIZE)
            ]
            
            # Use threading with explicit timeout
            result_queue = queue.Queue()
            translation_thread = threading.Thread(
                target=self._run_async_translation, 
                args=(chunks, result_queue)
            )
            translation_thread.daemon = True
            translation_thread.start()
            translation_thread.join(timeout=TRANSLATION_TIMEOUT)
            
            if not translation_thread.is_alive() and not result_queue.empty():
                translated_chunks = result_queue.get()
                if translated_chunks:
                    return " ".join(translated_chunks)
                
            logger.error("Translation failed or timed out")
            # Return empty instead of original to prevent English audio
            return ""
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return ""
        
    def _run_async_translation(self, chunks, result_queue):
        """Run async translation in a separate thread."""
        try:
            # Create async function for translations
            async def translate_all_chunks():
                tasks = [self._translate_chunk(chunk) for chunk in chunks]
                return await asyncio.gather(*tasks)
            
            # Create and run a new event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                translated = loop.run_until_complete(translate_all_chunks())
                result_queue.put(translated)
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Thread translation error: {e}")
            result_queue.put([])
        
    def generate_speech(self, text: str, output_file: str = None) -> str:
        """
        Generate speech from text in Hindi.
        
        Args:
            text: Hindi text to convert to speech
            output_file: Optional path for output file
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            if not output_file:
                # Create a unique filename
                filename = f"tts_{uuid.uuid4()}.mp3"
                output_file = os.path.join(self.temp_dir, filename)
                
            # Generate speech
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_file)
            
            return output_file
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None
            
    def generate_summary_audio(self, summary_text: str) -> str:
        """
        Generate Hindi audio from English summary.
        
        Args:
            summary_text: English text to translate and convert
            
        Returns:
            Path to audio file or None if failed
        """
        try:
            # Translate summary to Hindi
            logger.info("Starting translation to Hindi")
            hindi_text = self.translate_to_hindi(summary_text)
            
            # Verify we have Hindi text (add a debug log)
            is_likely_hindi = any(
                ord(char) > 2304 and ord(char) < 2432 
                for char in hindi_text[:100]
            )
            logger.info(
                f"Translation complete, text appears to be Hindi: {is_likely_hindi}"
            )
            
            # Only generate speech if we have Hindi text
            if hindi_text and is_likely_hindi:
                return self.generate_speech(hindi_text)
            elif hindi_text:
                # If we have text but it's not Hindi (likely English fallback)
                logger.error("Translation failed to produce Hindi text")
                return None
            return None
        except Exception as e:
            logger.error(f"Error in generate_summary_audio: {e}")
            return None