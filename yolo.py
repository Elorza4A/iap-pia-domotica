import asyncio
import os
import threading
import time

import cv2
from dotenv import load_dotenv
from tapo import ApiClient
from ultralytics import YOLO

load_dotenv()

# --- Configuración de la cámara ---
USUARIO = os.environ["TAPO_USER"]
PASSWORD = os.environ["TAPO_PASS"]
IP_CAMARA = "192.168.1.160"
PUERTO = "554"
STREAM = "stream2"  # 'stream1' = alta resolución, 'stream2' = baja resolución

# Si la cámara IP no responde, se usa la cámara de la laptop.
# Se prueban estos índices en orden (0 = cámara integrada normalmente).
INDICES_CAMARA_LOCAL = (0, 1, 2)
TIMEOUT_CAMARA_IP = 10   # segundos máximos esperando el primer frame de la IP
TIMEOUT_CAMARA_LOCAL = 8 # segundos máximos esperando el primer frame de la laptop

# --- Configuración del enchufe Tapo (P100) ---
# Son las credenciales de tu cuenta Tapo (correo y contraseña de la app),
# no las de la cámara.
IP_ENCHUFE = "192.168.1.156"
TAPO_EMAIL = os.environ["TAPO_EMAIL"]
TAPO_PASSWORD = os.environ["TAPO_ACCOUNT_PASS"]

# --- Configuración del modelo ---
MODELO = "yolov8n.pt"   # n = más rápido; s/m/l/x = más preciso pero más lento
CONFIANZA = 0.5         # umbral mínimo de detección
CLASE_PERSONA = 0       # en COCO, la clase 0 es "person"

# --- Configuración de la detección periódica ---
INTERVALO_SEG = 5       # cada cuántos segundos se ejecuta YOLO
CONFIRMACIONES = 2      # lecturas consecutivas iguales para confirmar un cambio
                        # (evita falsos disparos si YOLO falla en una lectura;
                        #  con 2 y un intervalo de 5 s, el cambio se confirma en ~10 s;
                        #  pon 1 para disparar al instante)

# Forzar TCP en FFmpeg antes de inicializar VideoCapture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

RTSP_URL = f"rtsp://{USUARIO}:{PASSWORD}@{IP_CAMARA}:{PUERTO}/{STREAM}"


# =====================================================================
#  CONTROL DEL ENCHUFE
#  La librería tapo es asíncrona, así que cada acción se ejecuta en un
#  hilo aparte con su propio event loop. Así el video no se congela
#  mientras se conecta al enchufe y las corrutinas sí se ejecutan.
# =====================================================================
_lock_enchufe = threading.Lock()  # evita que dos acciones se pisen entre sí


async def _set_enchufe(encender: bool):
    client = ApiClient(TAPO_EMAIL, TAPO_PASSWORD)
    device = await client.p100(IP_ENCHUFE)

    info = await device.get_device_info()
    print(f"Conectado con éxito: {info.nickname} ({info.model})")

    if encender:
        await device.on()
        print("Enchufe encendido")
    else:
        await device.off()
        print("Enchufe apagado")


def _ejecutar_en_hilo(encender: bool):
    def tarea():
        with _lock_enchufe:
            try:
                asyncio.run(_set_enchufe(encender))
            except Exception as e:
                print(f"[ERROR] No se pudo controlar el enchufe: {e}")

    threading.Thread(target=tarea, daemon=True).start()


# =====================================================================
#  TRIGGERS
# =====================================================================
def trigger_presencia(num_personas):
    """Se llama UNA vez cuando se pasa de 'nadie' a 'hay personas'."""
    print(f"[TRIGGER] Presencia detectada: {num_personas} persona(s)")
    _ejecutar_en_hilo(True)  # encender el enchufe


def trigger_ausencia():
    """Se llama UNA vez cuando se pasa de 'hay personas' a 'nadie'."""
    print("[TRIGGER] Ausencia de personas")
    _ejecutar_en_hilo(False)  # apagar el enchufe


# =====================================================================


class LectorCamara:
    """Lee frames en un hilo aparte y conserva solo el último,
    para evitar que el buffer (RTSP o webcam) acumule retraso.

    `fuente` puede ser una URL RTSP o un índice de cámara local (0, 1, ...)."""

    def __init__(self, fuente):
        if isinstance(fuente, str):
            # URL RTSP: forzar FFmpeg (necesario para rtsp://) y limitar la
            # espera de conexión/lectura. Sin esto, OpenCV tarda 30 s en
            # rendirse cuando la cámara IP no responde, y el cambio a la
            # cámara de la laptop se sentiría como un cuelgue.
            self.cap = cv2.VideoCapture(
                fuente,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_CAMARA_IP * 1000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_CAMARA_IP * 1000,
                ],
            )
        else:
            # Índice de cámara local (webcam de la laptop)
            self.cap = cv2.VideoCapture(fuente)
        self.frame = None
        self.ok = False
        self.corriendo = True
        self.lock = threading.Lock()
        threading.Thread(target=self._leer, daemon=True).start()

    def _leer(self):
        while self.corriendo:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame
                self.ok = True
        self.cap.release()

    def abierto(self):
        return self.cap.isOpened()

    def ultimo(self):
        with self.lock:
            if not self.ok:
                return None
            return self.frame.copy()

    def cerrar(self):
        """Detiene el hilo lector. La captura la libera el propio hilo al
        salir, para no bloquear si `read()` está esperando a una cámara
        IP que no responde."""
        self.corriendo = False


def esperar_primer_frame(lector, timeout):
    """Espera hasta `timeout` segundos a que llegue un frame real.

    `isOpened()` puede devolver True aunque la cámara IP sea inalcanzable,
    así que la prueba definitiva es que llegue imagen."""
    inicio = time.time()
    while time.time() - inicio < timeout:
        if lector.ultimo() is not None:
            return True
        time.sleep(0.1)
    return False


def abrir_camara(fuente, descripcion, timeout):
    """Abre `fuente` y devuelve un LectorCamara listo, o None si falla."""
    lector = LectorCamara(fuente)
    if lector.abierto() and esperar_primer_frame(lector, timeout):
        print(f"Conectado a {descripcion}.")
        return lector

    lector.cerrar()
    print(f"No se pudo leer de {descripcion}.")
    return None


def main():
    print("Cargando modelo YOLO...")
    modelo = YOLO(MODELO)  # descarga los pesos la primera vez

    # 1) Intentar la cámara IP
    print(f"Intentando conectar a la cámara IP {IP_CAMARA}...")
    lector = abrir_camara(RTSP_URL, f"la cámara IP {IP_CAMARA}", TIMEOUT_CAMARA_IP)
    nombre_fuente = f"Tapo C200 {IP_CAMARA}"

    # 2) Si falla, usar la cámara de la laptop
    if lector is None:
        print("Cambiando a la cámara de la laptop...")
        for indice in INDICES_CAMARA_LOCAL:
            lector = abrir_camara(
                indice, f"la cámara local (índice {indice})", TIMEOUT_CAMARA_LOCAL
            )
            if lector is not None:
                nombre_fuente = "Cámara de la laptop"
                break

        if lector is None:
            print("Error: no se pudo abrir ni la cámara IP ni la cámara de la laptop.")
            return

    print(f"Fuente de video: {nombre_fuente}. "
          f"Detección cada {INTERVALO_SEG} s. Presiona 'q' para salir.")

    ultima_deteccion = 0.0      # momento de la última ejecución de YOLO
    cajas = []                  # últimas cajas detectadas: (x1, y1, x2, y2, conf)
    personas = 0                # personas en la última detección
    estado = None               # estado confirmado: True = hay personas, False = nadie
    candidato = None            # estado visto en las últimas lecturas, aún sin confirmar
    contador = 0                # lecturas consecutivas con el mismo candidato

    while True:
        frame = lector.ultimo()
        if frame is None:
            cv2.waitKey(10)
            continue

        ahora = time.time()
        if ahora - ultima_deteccion >= INTERVALO_SEG:
            ultima_deteccion = ahora

            resultados = modelo.predict(
                frame, classes=[CLASE_PERSONA], conf=CONFIANZA, verbose=False
            )
            boxes = resultados[0].boxes
            cajas = [
                (*map(int, b.xyxy[0].tolist()), float(b.conf[0])) for b in boxes
            ]
            personas = len(cajas)
            hay_personas = personas > 0

            # --- Lógica de cambio de estado ---
            if hay_personas == candidato:
                contador += 1
            else:
                candidato = hay_personas
                contador = 1

            if contador >= CONFIRMACIONES and hay_personas != estado:
                anterior = estado
                estado = hay_personas
                # Si anterior es None es la primera lectura al arrancar:
                # se fija el estado inicial sin disparar ningún trigger.
                if anterior is not None:
                    if estado:
                        trigger_presencia(personas)
                    else:
                        trigger_ausencia()

        # --- Dibujo sobre el video en vivo (con las últimas cajas) ---
        for x1, y1, x2, y2, conf in cajas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"persona {conf:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        texto_estado = {None: "Iniciando...", True: "HAY PERSONAS", False: "SIN PERSONAS"}[estado]
        cv2.putText(frame, f"Personas: {personas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, texto_estado, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"{nombre_fuente} - YOLO personas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    lector.cerrar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()