"""Text-to-speech module for generating Hindi audio from English text."""

import os
import uuid
import logging
import tempfile
import threading
import queue
import asyncio
import subprocess
from typing import List, Optional

from gtts import gTTS
from googletrans import Translator

# Constants
MAX_CHUNK_SIZE = 1500
TRANSLATION_TIMEOUT = 60
MAX_TTS_CHARS = 3000  # Maximum characters for TTS processing

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
        translation = await self.translator.translate(
            chunk, src='en', dest='hi'
        )
        return translation.text
        
    def translate_to_hindi(self, text: str) -> str:
        """Translate text from English to Hindi.
        
        Args:
            text: English text to translate
            
        Returns:
            Hindi translation or empty string if translation fails
        """
        try:
            if not text:
                return ""
                
            if len(text) < 1000:
                try:
                    from googletrans import Translator as SyncTranslator
                    sync_translator = SyncTranslator()
                    result = sync_translator.translate(
                        text, src='en', dest='hi'
                    )
                    if hasattr(result, 'text') and result.text:
                        return result.text
                except Exception as e:
                    logger.warning(
                        "Simple translation failed, falling back to async: "
                        f"{e}"
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
                
            logger.error("Translation failed or timed out.")
            # Return empty instead of original to prevent English audio
            return ""
            
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return ""
        
    def _run_async_translation(self, chunks: List[str], result_queue: queue.Queue) -> None:
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
        
    def generate_speech(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """Generate speech from text in Hindi."""
        try:
            if not text:
                logger.warning("No text provided for speech generation")
                return None
                
            if not output_file:
                # Create a unique filename
                filename = f"tts_{uuid.uuid4()}.mp3"
                output_file = os.path.join(self.temp_dir, filename)
            
            # For long texts, break into chunks and combine audio
            if len(text) > MAX_TTS_CHARS:
                logger.info(f"Text too long ({len(text)} chars), splitting into chunks")
                return self._generate_chunked_speech(text, output_file)
                
            # Generate speech
            tts = gTTS(text=text, lang='hi', slow=False)
            tts.save(output_file)
            
            # Verify the file was created successfully
            if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                logger.info(f"Generated audio file: {output_file}")
                return output_file
            else:
                logger.error("Audio file was not created or is too small")
                return None
                
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            return None
            
    def _generate_chunked_speech(self, text: str, output_file: str) -> Optional[str]:
        """Generate speech for long text by breaking into chunks."""
        try:
            # Split by sentences to avoid cutting words
            sentences = text.split('।')  # Hindi sentence delimiter
            if len(sentences) <= 1:
                # Fall back to periods if no Hindi delimiters
                sentences = text.split('.')
                
            chunks = []
            current_chunk = ""
            
            # Create logical chunks based on sentence boundaries
            for sentence in sentences:
                if sentence.strip():  # Skip empty sentences
                    if len(current_chunk) + len(sentence) <= MAX_TTS_CHARS:
                        current_chunk += sentence + (
                            '।' if '।' in text else '.'
                        )
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence + (
                            '।' if '।' in text else '.'
                        )
                        
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(current_chunk)
                
            # No valid chunks
            if not chunks:
                logger.error("No valid text chunks for TTS")
                return None
                
            # Generate temp audio files for each chunk
            temp_files = []
            for i, chunk in enumerate(chunks):
                try:
                    chunk_file = os.path.join(
                        self.temp_dir, f"chunk_{i}_{uuid.uuid4()}.mp3"
                    )
                    tts = gTTS(text=chunk, lang='hi', slow=False)
                    tts.save(chunk_file)
                    temp_files.append(chunk_file)
                except Exception as e:
                    logger.error(f"Error generating chunk {i}: {e}")
            
            # If no chunks were generated successfully, return None
            if not temp_files:
                return None
            
            # If only one chunk was created, just rename it
            if len(temp_files) == 1:
                try:
                    os.rename(temp_files[0], output_file)
                    return output_file
                except Exception as e:
                    logger.error(f"Error renaming single audio file: {e}")
                    return temp_files[0]  # Return the temp file as is
                
            # For multiple files, use ffmpeg if available to concatenate
            try:
                # Create a file list for ffmpeg
                list_file = os.path.join(self.temp_dir, f"list_{uuid.uuid4()}.txt")
                with open(list_file, 'w') as f:
                    for temp_file in temp_files:
                        f.write(f"file '{temp_file}'\n")
                
                # Use ffmpeg to concatenate
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    output_file
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up temp files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
                os.remove(list_file)
                
                return output_file
                
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.error(f"Error using ffmpeg to combine audio: {e}")
                
                # FFmpeg failed or isn't available - return the first chunk as fallback
                logger.warning("Returning first audio chunk only as fallback")
                first_file = temp_files[0]
                try:
                    os.rename(first_file, output_file)
                    
                    # Clean up other temp files
                    for temp_file in temp_files[1:]:
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
                    
                    return output_file
                except Exception:
                    return first_file  # Return the first temp file if rename fails
                    
        except Exception as e:
            logger.error(f"Error in chunked speech generation: {e}")
            return None
            
    def generate_summary_audio(self, summary_text: str) -> Optional[str]:
        """Generate Hindi audio from English summary."""
        try:
            # Translate summary to Hindi
            logger.info("Starting translation to Hindi.")
            hindi_text = self.translate_to_hindi(summary_text)
            
            # Verify we have Hindi text (add a debug log)
            is_likely_hindi = any(
                ord(char) > 2304 and ord(char) < 2432 
                for char in hindi_text[:100]
            )
            logger.info(
                "Translation complete, text appears to be Hindi: "
                f"{is_likely_hindi}"
            )
            
            # Only generate speech if we have Hindi text
            if hindi_text and is_likely_hindi:
                return self.generate_speech(hindi_text)
            elif hindi_text:
                # If we have text but it's not Hindi (likely English fallback)
                logger.error("Translation failed to produce Hindi text.")
                return None
            return None
        except Exception as e:
            logger.error(f"Error in generate_summary_audio: {e}")
            return None