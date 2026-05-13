import os
import time

from faster_whisper import BatchedInferencePipeline, WhisperModel
from nio import Event, RoomEncryptedAudio, RoomMessageText

from matrix_bot_common import MatrixBot


class WhisperBot(MatrixBot):
    name = "WhisperBot"

    def __init__(self):
        super().__init__()
        self.model_size = self.config.get("model_size", "medium")
        self.language = self.config.get("language", "ru")
        self.cpu_threads = int(self.config.get("cpu_threads", 8))
        self.batch_size = int(self.config.get("batch_size", 8))

        print(f"[{self.name}] Загрузка модели Whisper '{self.model_size}'...")
        self.model = WhisperModel(
            self.model_size, device="cpu", compute_type="int8", cpu_threads=self.cpu_threads
        )
        self.batched_model = BatchedInferencePipeline(model=self.model)

    async def on_start(self):
        self.client.add_event_callback(self._text_callback, RoomMessageText)
        self.client.add_event_callback(self._audio_callback, Event)

    async def _text_callback(self, room, event):
        if event.sender == self.client.user_id:
            return
        body = (event.body or "").strip()
        if body.lower() == "!whisper":
            await self.send_text(
                room.room_id,
                "🎙 Бот на связи! Пришлите аудио, и я превращу его в текст.",
            )

    async def _audio_callback(self, room, event):
        if not isinstance(event, RoomEncryptedAudio):
            return
        if event.sender == self.client.user_id:
            return

        print(f"[{self.name}] Получено аудио от {event.sender}")
        filename = f"temp_{event.event_id}.ogg"

        try:
            response = await self.client.download(event.url)
            if event.source.get("content", {}).get("file"):
                from nio.crypto import attachments
                file_info = event.source["content"]["file"]
                body = attachments.decrypt_attachment(
                    response.body,
                    file_info["key"]["k"],
                    file_info["hashes"]["sha256"],
                    file_info["iv"],
                )
            else:
                body = response.body
            with open(filename, "wb") as f:
                f.write(body)
        except Exception as e:
            print(f"Ошибка при скачивании/расшифровке: {e}")
            return

        try:
            start = time.perf_counter()
            segments, info = self.batched_model.transcribe(
                filename, language=self.language, batch_size=self.batch_size
            )
            text = "".join(segment.text for segment in segments).strip()
            elapsed = time.perf_counter() - start
            audio_duration = info.duration
            print(
                f"⏱ {elapsed:.2f}с на {audio_duration:.2f}с аудио "
                f"(x{elapsed / audio_duration:.2f} realtime)"
            )

            if text:
                await self.send_text(room.room_id, f"🎙 Распознано: {text}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)


if __name__ == "__main__":
    WhisperBot().main()
