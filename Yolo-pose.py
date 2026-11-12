import asyncio
import math
import os
import threading
import time

import cv2
import tinytuya
from dotenv import load_dotenv
from tapo import ApiClient
from ultralytics import YOLO

load_dotenv()

# --- Configuración de la cámara ---
USUARIO = os.environ["TAPO_USER"]
PASSWORD = os.environ["TAPO_PASS"]
# IP de la cámara Tapo. Se puede sobrescribir con la variable de entorno TAPO_CAM_IP.
IP_CAMARA = os.getenv("TAPO_CAM_IP", "192.168.1.160")
PUERTO = "554"
STREAM = "stream2"  # 'stream1' = alta resolución, 'stream2' = baja resolución

# Si la cámara IP no responde, se usa la cámara de la laptop.
INDICES_CAMARA_LOCAL = (0, 1, 2)
TIMEOUT_CAMARA_IP = 10   # segundos máximos esperando el primer frame de la IP
TIMEOUT_CAMARA_LOCAL = 8 # segundos máximos esperando el primer frame de la laptop

# --- Configuración del enchufe Tapo (P100) ---
# Son las credenciales de tu cuenta Tapo (correo y contraseña de la app),
# no las de la cámara.
IP_ENCHUFE = "192.168.1.156"
TAPO_EMAIL = os.environ["TAPO_EMAIL"]
TAPO_PASSWORD = os.environ["TAPO_ACCOUNT_PASS"]

# --- Configuración del foco Tuya ---
# Estas credenciales se obtienen con `python -m tinytuya wizard`.
# Se leen del .env: TUYA_FOCO_IP, TUYA_FOCO_DEV_ID y TUYA_FOCO_KEY.
IP_FOCO = os.getenv("TUYA_FOCO_IP", "192.168.1.245")
FOCO_DEV_ID = os.environ["TUYA_FOCO_DEV_ID"]
FOCO_KEY = os.environ["TUYA_FOCO_KEY"]
FOCO_VERSIONES = (3.3, 3.4, 3.5)  # versiones de protocolo Tuya que se prueban

# --- Configuración del modelo (YOLOv8 Pose) ---
MODELO = "yolov8n-pose.pt"  # Modelo de estimación de poses
CONFIANZA = 0.5             # umbral mínimo de detección
CONF_KPT = 0.30             # confianza mínima para dar por visible un keypoint

# Índices de keypoints del modelo COCO (YOLOv8 Pose)
NARIZ = 0
HOMBRO_IZQ, HOMBRO_DER = 5, 6
CADERA_IZQ, CADERA_DER = 11, 12
RODILLA_IZQ, RODILLA_DER = 13, 14
TOBILLO_IZQ, TOBILLO_DER = 15, 16

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
#  CONTROL DEL FOCO (Tuya)
#  tinytuya es síncrono y puede tardar unos segundos en conectar, así que
#  también se ejecuta en un hilo aparte para no congelar el video.
#  Se prueba la última versión que funcionó y, si falla, las demás.
# =====================================================================
_lock_foco = threading.Lock()
_foco_version = None  # versión de protocolo Tuya que funcionó


def _set_foco(encender: bool):
    global _foco_version

    if _foco_version in FOCO_VERSIONES:
        versiones = [_foco_version] + [v for v in FOCO_VERSIONES if v != _foco_version]
    else:
        versiones = list(FOCO_VERSIONES)

    for version in versiones:
        try:
            foco = tinytuya.BulbDevice(
                dev_id=FOCO_DEV_ID,
                address=IP_FOCO,
                local_key=FOCO_KEY,
                version=version,
                connection_timeout=3,
                connection_retry_limit=1,
                connection_retry_delay=1,
            )
            estado = foco.status()
            if "Error" in estado:
                continue

            _foco_version = version
            resultado = foco.turn_on() if encender else foco.turn_off()
            if isinstance(resultado, dict) and "Error" in resultado:
                print(f"[ERROR] El foco (v{version}) respondió: {resultado['Error']}")
                return

            print("Foco encendido" if encender else "Foco apagado")
            return
        except Exception as e:
            print(f"[AVISO] No se pudo usar el foco con la versión {version}: {e}")

    print("[ERROR] No se pudo controlar el foco Tuya "
          "(revisa TUYA_FOCO_IP, TUYA_FOCO_DEV_ID y TUYA_FOCO_KEY)")


def _ejecutar_foco_en_hilo(encender: bool):
    def tarea():
        with _lock_foco:
            _set_foco(encender)

    threading.Thread(target=tarea, daemon=True).start()


# =====================================================================
#  TRIGGERS
#  Sentado -> enchufe encendido / foco apagado
#  De pie -> foco encendido / enchufe apagado
#  Nadie  -> ambos apagados
# =====================================================================
def trigger_de_pie(num_personas):
    """Se llama UNA vez cuando se pasa a 'alguien de pie'."""
    print(f"[TRIGGER] Persona de pie detectada: {num_personas} persona(s)")
    _ejecutar_en_hilo(False)       # apagar el enchufe
    _ejecutar_foco_en_hilo(True)   # encender el foco


def trigger_sentado(num_personas):
    """Se llama UNA vez cuando se pasa a 'alguien sentado'."""
    print(f"[TRIGGER] Persona sentada detectada: {num_personas} persona(s)")
    _ejecutar_en_hilo(True)        # encender el enchufe
    _ejecutar_foco_en_hilo(False)  # apagar el foco


def trigger_ausencia():
    """Se llama UNA vez cuando se pasa de 'alguien' a 'nadie'."""
    print("[TRIGGER] Ya no hay nadie")
    _ejecutar_en_hilo(False)       # apagar el enchufe
    _ejecutar_foco_en_hilo(False)  # apagar el foco


# =====================================================================


class LectorCamara:
    def __init__(self, fuente):
        if isinstance(fuente, str):
            self.cap = cv2.VideoCapture(
                fuente,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_CAMARA_IP * 1000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_CAMARA_IP * 1000,
                ],
            )
        else:
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
        self.corriendo = False


def esperar_primer_frame(lector, timeout):
    inicio = time.time()
    while time.time() - inicio < timeout:
        if lector.ultimo() is not None:
            return True
        time.sleep(0.1)
    return False


def abrir_camara(fuente, descripcion, timeout):
    lector = LectorCamara(fuente)
    if lector.abierto() and esperar_primer_frame(lector, timeout):
        print(f"Conectado a {descripcion}.")
        return lector

    lector.cerrar()
    print(f"No se pudo leer de {descripcion}.")
    return None


def _conf_keypoint(kpts, confs, idx):
    """Confianza de un keypoint (si no hay confs, se usa la coordenada)."""
    if confs is None:
        return 1.0 if (kpts[idx][0] > 0 and kpts[idx][1] > 0) else 0.0
    return float(confs[idx])


def _punto(kpts, confs, idx):
    """Devuelve (x, y) si el keypoint es visible y confiable; si no, None."""
    if _conf_keypoint(kpts, confs, idx) < CONF_KPT:
        return None
    x, y = float(kpts[idx][0]), float(kpts[idx][1])
    if x <= 0 and y <= 0:
        return None
    return (x, y)


def _punto_medio(p1, p2):
    """Promedia dos puntos; si solo uno es visible, devuelve ese."""
    if p1 is not None and p2 is not None:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    return p1 if p1 is not None else p2


def _angulo(a, b, c):
    """Ángulo (en grados) formado en el punto b por los segmentos a-b y c-b."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    coseno = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    coseno = max(-1.0, min(1.0, coseno))
    return math.degrees(math.acos(coseno))


def clasificar_postura(kpts, confs, bbox):
    """
    Clasifica la postura combinando varias señales, para que funcione
    también cuando la persona está parcialmente oculta (por ejemplo,
    sentada detrás de un escritorio o de una mesa).

    Señales usadas, de más a menos fiable:
      1. Ángulo de la rodilla (cadera-rodilla-tobillo).
      2. Extensión vertical cadera -> rodilla / tobillo.
      3. Si la cadera está oculta: forma de la caja (alto/ancho).
      4. Alto de la caja respecto al ancho de hombros y espacio bajo la cadera.
      5. Longitud del torso normalizada por el ancho de hombros.

    Devuelve: "De pie", "Sentado", "Sentado (ocluido)" o "No visible".
    """
    hombros = _punto_medio(
        _punto(kpts, confs, HOMBRO_IZQ), _punto(kpts, confs, HOMBRO_DER)
    )
    caderas = _punto_medio(
        _punto(kpts, confs, CADERA_IZQ), _punto(kpts, confs, CADERA_DER)
    )
    rodillas = _punto_medio(
        _punto(kpts, confs, RODILLA_IZQ), _punto(kpts, confs, RODILLA_DER)
    )
    tobillos = _punto_medio(
        _punto(kpts, confs, TOBILLO_IZQ), _punto(kpts, confs, TOBILLO_DER)
    )
    nariz = _punto(kpts, confs, NARIZ)

    if hombros is None:
        # A veces la cara se ve aunque los hombros estén ocultos.
        if nariz is None and caderas is None:
            return "No visible"
        hombros = nariz if nariz is not None else caderas

    # --- Escala corporal: independiente de la distancia a la cámara ---
    escala = None
    hombro_izq = _punto(kpts, confs, HOMBRO_IZQ)
    hombro_der = _punto(kpts, confs, HOMBRO_DER)
    if hombro_izq is not None and hombro_der is not None:
        ancho_hombros = abs(hombro_izq[0] - hombro_der[0])
        if ancho_hombros > 2:
            escala = ancho_hombros

    alto_caja = ancho_caja = None
    if bbox is not None:
        ancho_caja = max(float(bbox[2] - bbox[0]), 1.0)
        alto_caja = max(float(bbox[3] - bbox[1]), 1.0)
        if escala is None:
            # Sin hombros fiables: aproximar el ancho de hombros con la caja.
            escala = max(ancho_caja * 0.55, alto_caja * 0.25, 1.0)
    if escala is None or escala < 1:
        escala = 1.0

    # 1) Ángulo de la rodilla: pierna extendida -> de pie; flexionada -> sentado.
    angulos_rodilla = []
    for cadera_idx, rodilla_idx, tobillo_idx in (
        (CADERA_IZQ, RODILLA_IZQ, TOBILLO_IZQ),
        (CADERA_DER, RODILLA_DER, TOBILLO_DER),
    ):
        cadera_p = _punto(kpts, confs, cadera_idx)
        rodilla_p = _punto(kpts, confs, rodilla_idx)
        tobillo_p = _punto(kpts, confs, tobillo_idx)
        if cadera_p and rodilla_p and tobillo_p:
            angulo = _angulo(cadera_p, rodilla_p, tobillo_p)
            if angulo is not None:
                angulos_rodilla.append(angulo)

    if angulos_rodilla:
        angulo = sum(angulos_rodilla) / len(angulos_rodilla)
        if angulo >= 150:
            return "De pie"
        if angulo <= 115:
            return "Sentado"

    # 2) Extensión vertical de las piernas respecto al ancho de hombros.
    if caderas is not None and rodillas is not None:
        extension = (rodillas[1] - caderas[1]) / escala
        if extension >= 0.55:
            return "De pie"
        if extension <= 0.30:
            return "Sentado"

    if caderas is not None and tobillos is not None:
        extension = (tobillos[1] - caderas[1]) / escala
        if extension >= 1.25:
            return "De pie"
        if extension <= 0.65:
            return "Sentado"

    # 3) Cadera oculta: hay un objeto tapando la parte inferior del cuerpo.
    #    Si la caja es claramente más alta que ancha, probablemente sigue de pie
    #    detrás del objeto; si es ancha y baja, está sentado tras él.
    if caderas is None:
        if alto_caja is not None and ancho_caja is not None:
            relacion = alto_caja / ancho_caja
            if relacion >= 2.2:
                return "De pie"
        return "Sentado (ocluido)"

    # 4) Cadera visible pero piernas ocultas: usar proporciones de la caja.
    if alto_caja is not None and ancho_caja is not None:
        alto_normalizado = alto_caja / escala
        if alto_normalizado >= 3.2:
            return "De pie"
        if alto_normalizado <= 2.4:
            return "Sentado"

        espacio_bajo_cadera = (bbox[3] - caderas[1]) / alto_caja
        if espacio_bajo_cadera <= 0.12:
            return "Sentado"
        if espacio_bajo_cadera >= 0.30:
            return "De pie"

    # 5) Último recurso: torso alargado respecto al ancho de hombros.
    torso = (caderas[1] - hombros[1]) / escala
    if torso >= 1.15:
        return "De pie"
    return "Sentado"


def main():
    print("Cargando modelo YOLOv8 Pose...")
    modelo = YOLO(MODELO)

    print(f"Intentando conectar a la cámara IP {IP_CAMARA}...")
    lector = abrir_camara(RTSP_URL, f"la cámara IP {IP_CAMARA}", TIMEOUT_CAMARA_IP)
    nombre_fuente = f"Cámara IP {IP_CAMARA}"

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

    ultima_deteccion = 0.0
    cajas = []
    keypoints_list = []
    posturas = []
    personas = 0
    estado = None               # "de pie", "sentado", "nadie" (confirmado)
    candidato = None            # estado visto en las últimas lecturas, aún sin confirmar
    contador = 0                # lecturas consecutivas con el mismo candidato

    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # Cabeza
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Torso y brazos
        (5, 11), (6, 12)                         # Tronco superior
    ]

    while True:
        frame = lector.ultimo()
        if frame is None:
            cv2.waitKey(10)
            continue

        ahora = time.time()
        if ahora - ultima_deteccion >= INTERVALO_SEG:
            ultima_deteccion = ahora

            resultados = modelo.predict(
                frame, conf=CONFIANZA, verbose=False
            )
            boxes = resultados[0].boxes
            cajas = [
                (*map(int, b.xyxy[0].tolist()), float(b.conf[0])) for b in boxes
            ]

            keypoints_list = []
            keypoints_conf = None
            if resultados[0].keypoints is not None:
                kp = resultados[0].keypoints
                if kp.xy is not None:
                    keypoints_list = kp.xy.cpu().numpy()
                if kp.conf is not None:
                    keypoints_conf = kp.conf.cpu().numpy()

            posturas = []
            for i, kpts in enumerate(keypoints_list):
                confs = (
                    keypoints_conf[i]
                    if keypoints_conf is not None and i < len(keypoints_conf)
                    else None
                )
                bbox = cajas[i][:4] if i < len(cajas) else None
                posturas.append(clasificar_postura(kpts, confs, bbox))

            personas = len(cajas)

            # --- Lógica de cambio de estado (de pie / sentado / nadie) ---
            hay_de_pie = any(postura == "De pie" for postura in posturas)
            hay_sentado = any(
                postura in ("Sentado", "Sentado (ocluido)") for postura in posturas
            )

            if hay_de_pie:
                estado_actual = "de pie"
            elif hay_sentado:
                estado_actual = "sentado"
            else:
                estado_actual = "nadie"

            if estado_actual == candidato:
                contador += 1
            else:
                candidato = estado_actual
                contador = 1

            if contador >= CONFIRMACIONES and estado_actual != estado:
                anterior = estado
                estado = estado_actual
                # Si anterior es None es la primera lectura al arrancar:
                # se fija el estado inicial sin disparar ningún trigger.
                if anterior is not None:
                    if estado == "de pie":
                        trigger_de_pie(personas)
                    elif estado == "sentado":
                        trigger_sentado(personas)
                    else:
                        trigger_ausencia()

        # --- Dibujo sobre el video en vivo ---
        for i, (x1, y1, x2, y2, conf) in enumerate(cajas):
            postura_actual = posturas[i] if i < len(posturas) else "Analizando..."

            if postura_actual == "De pie":
                color_caja = (0, 255, 0)      # Verde
            elif postura_actual == "Sentado":
                color_caja = (0, 255, 255)    # Amarillo
            elif postura_actual == "Sentado (ocluido)":
                color_caja = (0, 165, 255)    # Naranja: sentado tras un objeto
            else:
                color_caja = (150, 150, 150)  # Gris: no visible

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
            cv2.putText(frame, f"{postura_actual} ({conf:.2f})", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)

            if i < len(keypoints_list):
                kpts = keypoints_list[i]
                
                for p1, p2 in SKELETON_PAIRS:
                    if p1 < len(kpts) and p2 < len(kpts):
                        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                for px, py in kpts:
                    px, py = int(px), int(py)
                    if px > 0 and py > 0:
                        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

        texto_estado = {
            None: "Iniciando...",
            "de pie": "DE PIE: foco encendido",
            "sentado": "SENTADO: enchufe encendido",
            "nadie": "NADIE: todo apagado",
        }[estado]
        cv2.putText(frame, f"Personas detectadas: {personas}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, texto_estado, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(f"{nombre_fuente} - YOLO Pose (de pie / sentado)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    lector.cerrar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()