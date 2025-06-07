import os
import json
import asyncio
import sqlite3
import base64
import tempfile
import hashlib
import requests
from datetime import datetime
import websockets
from dotenv import load_dotenv
import openai
from hkdf import Hkdf
from Crypto.Cipher import AES

# ────── Config ─────────────────────────────────────────────────────

# ────── SQLite (threads) ───────────────────────────────────────────
DB_PATH = "threads.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            phone_number TEXT PRIMARY KEY,
            thread_id    TEXT,
            created_at   TEXT
        )
    """)
    conn.commit(); conn.close()
init_db()

def get_or_create_thread(phone):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT thread_id FROM threads WHERE phone_number=?", (phone,))
    row = cur.fetchone()

    # Si hay un thread registrado, lo probamos
    if row:
        thread_id = row[0]
        try:
            # Prueba rápida para validar que el thread aún existe
            client.beta.threads.retrieve(thread_id=thread_id)
            conn.close()
            return thread_id
        except openai.NotFoundError:
            print(f"⚠️ Thread inválido para {phone}, creando uno nuevo...")

    # Si no hay o falló el anterior, creamos uno nuevo
    new_thread = client.beta.threads.create()
    cur.execute("REPLACE INTO threads (phone_number, thread_id, created_at) VALUES (?, ?, ?)",
                (phone, new_thread.id, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return new_thread.id

# ────── GPT helper ────────────────────────────────────────────────
async def askgpt(text, phone):
    th = get_or_create_thread(phone)
    client.beta.threads.messages.create(thread_id=th, role="user", content=text)
    run = client.beta.threads.runs.create(thread_id=th, assistant_id=ASSISTANT_ID)
    while run.status != "completed":
        await asyncio.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=th, run_id=run.id)
    msg = client.beta.threads.messages.list(thread_id=th).data[0]
    return msg.content[0].text.value

# ────── Imagen (tintado) ───────────────────────────────────────────
async def analizar_imagen_tintado(b64, phone):
    th = get_or_create_thread(phone)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(base64.b64decode(b64))
        img_path = tmp.name

    with open(img_path, "rb") as f:
        file_id = client.files.create(file=f, purpose="assistants").id

    client.beta.threads.messages.create(
        thread_id=th, role="user",
        content=[
            {
                "type": "text",
                "text": (
                    "Eres un experto en ACCIDENTE DE ANIMALES. "
                    "Analiza la imagen y recomienda LOS PRIMEROS AUXILIOS Y LUEGO DILE QUE LA VETERINARIA ESTA ABIERTO PARA EMERGENCIAS 24/7."
                )
            },
            {"type": "image_file", "image_file": {"file_id": file_id}}
        ]
    )

    run = client.beta.threads.runs.create(thread_id=th, assistant_id=ASSISTANT_ID)
    while run.status != "completed":
        await asyncio.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=th, run_id=run.id)

    msg = client.beta.threads.messages.list(thread_id=th).data[0]
    respuesta = msg.content[0].text.value

    try:
        os.remove(img_path)
    except Exception as e:
        print(f"⚠️ No se pudo borrar la imagen: {e}")

    return respuesta

# ────── Audio (WhatsApp → Whisper) ────────────────────────────────
async def procesar_audio(p):
    media_key = base64.b64decode(p["mediaKey"])
    enc = requests.get(f"https://mmg.whatsapp.net{p['directPath']}").content
    km = Hkdf(salt=None, input_key_material=media_key,
              hash=hashlib.sha256).expand(b"WhatsApp Audio Keys", 112)
    iv, key = km[:16], km[16:48]
    plain = AES.new(key, AES.MODE_CBC, iv).decrypt(enc[:-10])

    ogg = f"tmp_{p.get('chatID','')}.ogg"
    with open(ogg, "wb") as f:
        f.write(plain)

    with open(ogg, "rb") as audio_file:
        texto = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file).text

    try:
        os.remove(ogg)
    except Exception as e:
        print(f"⚠️ Limpieza .ogg falló: {e}")

    return texto

# ────── Extractor de texto ────────────────────────────────────────
def find_text(obj):
    if isinstance(obj, str):
        try:
            return find_text(json.loads(obj))
        except:
            return obj
    if not isinstance(obj, dict):
        return ""
    for k in ("payload", "text", "body", "conversation", "mensaje"):
        if k in obj:
            res = find_text(obj[k])
            if res:
                return res
    for k in ("message", "_data"):
        if k in obj:
            res = find_text(obj[k])
            if res:
                return res
    return ""

# ────── WebSocket Server ──────────────────────────────────────────
clientes = set()

async def enviar_a_todos(msg):
    vivos = set()
    for c in clientes:
        try:
            await c.send(msg)
            vivos.add(c)
        except:
            pass
    clientes.clear()
    clientes.update(vivos)

async def manejar_conexion(ws):
    print("🟢 Cliente:", ws.remote_address)
    clientes.add(ws)
    try:
        async for raw in ws:
            datos = json.loads(raw)
            payload = datos.get("payload")
            chat_id = datos.get("chatID", "desconocido")

            # payload puede venir como string JSON
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except:
                    pass
            if isinstance(payload, dict) and payload.get("chatID"):
                chat_id = payload["chatID"]

            tipo = payload.get("tipo") if isinstance(payload, dict) else None
            image_b64 = (
                payload.get("body")
                if isinstance(payload, dict) and payload.get("tipo") == "image"
                else None
            )
            # imagen por datos["message"]["_data"]
            if not image_b64:
                msgnode = datos.get("message") or {}
                if isinstance(msgnode, dict):
                    d = msgnode.get("_data", {})
                    if d.get("type") == "image" and d.get("body"):
                        image_b64 = d["body"]

            # -------- Flujos --------
            if tipo == "audio":
                print("🎙️ Audio → transcribiendo")
                texto = await procesar_audio(payload)
                respuesta = await askgpt(texto, chat_id)

            elif image_b64 and not find_text(payload).strip():
                print("🖼️ Solo imagen")
                respuesta = await analizar_imagen_tintado(image_b64, chat_id)

            elif image_b64:  # imagen + texto → solo analiza imagen
                print("🖼️ Imagen + texto (se analiza solo la imagen)")
                respuesta = await analizar_imagen_tintado(image_b64, chat_id)

            else:            # solo texto
                txt = find_text(payload if payload is not None else datos)
                print("📝 Texto extraído:", repr(txt))
                respuesta = await askgpt(txt, chat_id) if txt else "No entendí tu mensaje."

            out = {
                "event": "message",
                "payload": {"respuesta": respuesta},
                "from": chat_id,
                "chatID": chat_id
            }
            await enviar_a_todos(json.dumps(out))
            print("📤 →", chat_id, ":", repr(respuesta))

    finally:
        clientes.discard(ws)

# ────── Main ───────────────────────────────────────────────────────
async def main():
    server = await websockets.serve(manejar_conexion, "localhost", 8080)
    print("✅ WS en ws://localhost:8080")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
