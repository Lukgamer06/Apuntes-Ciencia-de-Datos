const API_URL = (window.EGG_API_URL || 'https://eggsofialucas.duckdns.org/api').replace(/\/$/, '');
const FRAME_INTERVAL_MS = 180;

const video = document.querySelector('#camera');
const overlay = document.querySelector('#overlay');
const overlayContext = overlay.getContext('2d');
const placeholder = document.querySelector('#cameraPlaceholder');
const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const motorButton = document.querySelector('#motorButton');
const motorLabel = document.querySelector('#motorLabel');
const motorTransport = document.querySelector('#motorTransport');
const statusMessage = document.querySelector('#statusMessage');
const connectionBadge = document.querySelector('#connectionBadge');
const modelStatus = document.querySelector('#modelStatus');
const detectionCount = document.querySelector('#detectionCount');
const confidence = document.querySelector('#confidence');
const frameRate = document.querySelector('#frameRate');
const livePill = document.querySelector('.live-pill');

let stream = null;
let running = false;
let processing = false;
let lastFrameAt = 0;
let analyzedFrames = 0;
let motorEnabled = false;

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle('error', isError);
}

function setConnection(connected, message) {
  connectionBadge.textContent = message;
  connectionBadge.className = `badge ${connected ? 'ok' : 'error'}`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

async function checkBackend() {
  try {
    const health = await requestJson('/health');
    setConnection(true, 'Backend conectado');
    motorButton.disabled = false;
    modelStatus.textContent = health.model_exists
      ? `Modelo: ${health.model_path}`
      : 'Modelo: falta best.pt';
    motorTransport.textContent = health.motor_transport === 'esp32'
      ? 'ESP32 conectado por HTTP.'
      : 'El control queda en simulacion hasta configurar el ESP32.';
    if (!health.model_exists) setStatus('Backend conectado, pero falta el modelo entrenado best.pt.', true);
  } catch (error) {
    setConnection(false, 'Backend desconectado');
    motorButton.disabled = true;
    modelStatus.textContent = 'Modelo: sin respuesta';
  }
}

function drawDetections(payload) {
  overlay.width = payload.width;
  overlay.height = payload.height;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  const detections = payload.detections || [];
  detectionCount.textContent = String(detections.length);
  confidence.textContent = detections.length
    ? `${Math.round(Math.max(...detections.map((item) => item.confidence)) * 100)}%`
    : '--';

  overlayContext.lineWidth = Math.max(3, payload.width / 360);
  overlayContext.font = `${Math.max(14, payload.width / 48)}px 'DM Mono', monospace`;
  detections.forEach((item) => {
    const box = item.box;
    const width = box.x2 - box.x1;
    const height = box.y2 - box.y1;
    overlayContext.strokeStyle = '#d5f06f';
    overlayContext.strokeRect(box.x1, box.y1, width, height);
    const label = `${item.class_name} ${Math.round(item.confidence * 100)}%`;
    const labelWidth = overlayContext.measureText(label).width + 14;
    overlayContext.fillStyle = '#d5f06f';
    overlayContext.fillRect(box.x1, Math.max(0, box.y1 - 27), labelWidth, 27);
    overlayContext.fillStyle = '#10150e';
    overlayContext.fillText(label, box.x1 + 7, Math.max(18, box.y1 - 8));
  });
}

async function processFrame() {
  if (!running || processing || video.readyState < 2) return;
  processing = true;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  try {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.72));
    if (!blob) throw new Error('No se pudo preparar el frame.');
    const formData = new FormData();
    formData.append('file', blob, 'live-frame.jpg');
    const payload = await requestJson('/predict-frame', { method: 'POST', body: formData });
    drawDetections(payload);
    analyzedFrames += 1;
    const now = performance.now();
    if (lastFrameAt) frameRate.textContent = `${Math.round(1000 / (now - lastFrameAt))} FPS de analisis`;
    lastFrameAt = now;
    if (payload.detections.length) setStatus(`${payload.detections.length} objeto(s) detectado(s) en el frame actual.`);
    else setStatus('Video activo. Sin detecciones en el frame actual.');
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    processing = false;
  }
}

function liveLoop() {
  if (!running) return;
  processFrame();
  window.setTimeout(() => requestAnimationFrame(liveLoop), FRAME_INTERVAL_MS);
}

async function startVideo() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    running = true;
    placeholder.hidden = true;
    startButton.disabled = true;
    stopButton.disabled = false;
    livePill.classList.add('active');
    lastFrameAt = 0;
    frameRate.textContent = 'Iniciando analisis...';
    setStatus('Video en vivo iniciado.');
    liveLoop();
  } catch (error) {
    setStatus(`No se pudo acceder a la webcam: ${error.message}`, true);
  }
}

function stopVideo() {
  running = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  placeholder.hidden = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  livePill.classList.remove('active');
  frameRate.textContent = '0 FPS de analisis';
  setStatus('Video detenido.');
}

async function toggleMotor() {
  motorButton.disabled = true;
  try {
    const payload = await requestJson('/motor', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !motorEnabled }),
    });
    motorEnabled = payload.enabled;
    motorButton.classList.toggle('on', motorEnabled);
    motorButton.classList.toggle('off', !motorEnabled);
    motorLabel.textContent = motorEnabled ? 'Motor encendido' : 'Motor apagado';
    motorTransport.textContent = payload.transport === 'esp32'
      ? 'Orden enviada al ESP32 por HTTP.'
      : 'Orden registrada en simulacion.';
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
  } finally {
    motorButton.disabled = false;
  }
}

startButton.addEventListener('click', startVideo);
stopButton.addEventListener('click', stopVideo);
motorButton.addEventListener('click', toggleMotor);
checkBackend();
window.setInterval(checkBackend, 10000);
