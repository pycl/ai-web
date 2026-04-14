## запуск

1. Запустите PostgreSQL. (например, 
docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=root -e POSTGRES_DB=ai_web_db -e POSTGRES_USER=postgres postgres:16.9-alpine
https://docs.docker.com/engine/install/
)
2. Скопируйте `.env.example` в `.env`, затем при необходимости отредактируйте..
3. Примените миграции:
```bash
uv run alembic upgrade head
```
4. Запустите API:
```bash
uv run uvicorn app.main:app --reload --port <..порт..>
```
5. Порядок тестирования в Swagger:
   - `POST /users` — создать пользователя
   - `POST /users/{user_id}/api-keys` — создать API-ключ
   - `POST /users/{user_id}/sessions` — создать чат-сессию
   - `POST /chat` или `POST /chat/stream` — отправить сообщение
   - `GET /users/{user_id}/chat-history` — посмотреть историю


В текущем проекте реальным провайдером является **Groq**, модель по умолчанию — **`llama-3.1-8b-instant`**.

Причины выбора:
- Groq предоставляет бесплатный API, низкий порог входа
- Поддерживает как обычные ответы, так и потоковые

Примечания:
- Реальный режим включается через `LLM_MODE=real`
- Провайдер задаётся через `LLM_PROVIDER=groq`
- Базовый URL по умолчанию: `https://api.groq.com/openai/v1`

Проект поддерживает два режима:
- `mock`: используется локальный `MockLLM`
- `real`: вызовы к реальному внешнему LLM-провайдеру


На данный момент реализовано:
- Сохранён `MockLLM`
- Переключение между режимами `mock / real` через `.env`
- Подключён реальный провайдер Groq
- `/chat` поддерживает реальный провайдер
- `/chat/stream` поддерживает реальный провайдер
- История сохраняет метаданные модели и режима
- Единая обработка ошибок провайдера