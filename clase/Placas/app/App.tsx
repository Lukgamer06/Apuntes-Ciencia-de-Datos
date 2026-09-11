import { CameraView, useCameraPermissions } from "expo-camera";
import { useState } from "react";
import { ActivityIndicator, Button, Image, SafeAreaView, StyleSheet, Text, View } from "react-native";

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://192.168.1.100:8000";

type Plate = { text: string; confidence: number; ocr_confidence?: number };

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [camera, setCamera] = useState<CameraView | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{ plates: Plate[] } | null>(null);
  const [loading, setLoading] = useState(false);

  if (!permission) return <View style={styles.center}><ActivityIndicator /></View>;
  if (!permission.granted) {
    return <View style={styles.center}><Text style={styles.title}>Se necesita acceso a la camara</Text><Button title="Permitir camara" onPress={requestPermission} /></View>;
  }

  const capture = async () => {
    if (!camera) return;
    setLoading(true);
    setResult(null);
    try {
      const photo = await camera.takePictureAsync({ quality: 0.85, skipProcessing: false });
      if (!photo?.uri) return;
      setPreview(photo.uri);
      const body = new FormData();
      body.append("file", { uri: photo.uri, name: "plate.jpg", type: "image/jpeg" } as unknown as Blob);
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json() as { plates: Plate[] };
      setResult(payload);
    } catch (error) {
      setResult({ plates: [{ text: error instanceof Error ? error.message : "No se pudo procesar la imagen", confidence: 0 }] });
    } finally {
      setLoading(false);
    }
  };

  return <SafeAreaView style={styles.container}>
    <View style={styles.cameraFrame}><CameraView ref={setCamera} style={StyleSheet.absoluteFill} facing="back" /></View>
    {preview && <Image source={{ uri: preview }} style={styles.preview} />}
    <View style={styles.controls}><Button title={loading ? "Procesando..." : "Capturar y leer placa"} onPress={capture} disabled={loading} /></View>
    {result && <View style={styles.result}><Text style={styles.resultTitle}>Resultado</Text>{result.plates.map((plate, index) => <Text key={`${plate.text}-${index}`} style={styles.plate}>{plate.text || "Texto no detectado"} ({Math.round(plate.confidence * 100)}%)</Text>)}</View>}
  </SafeAreaView>;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#101820", padding: 16, gap: 16 },
  center: { flex: 1, alignItems: "center", justifyContent: "center", gap: 16, padding: 24 },
  title: { fontSize: 18, textAlign: "center" },
  cameraFrame: { flex: 1, minHeight: 320, overflow: "hidden", borderRadius: 12, backgroundColor: "#202b33" },
  preview: { width: 100, height: 75, borderRadius: 8, alignSelf: "center" },
  controls: { paddingHorizontal: 24 },
  result: { padding: 16, borderRadius: 12, backgroundColor: "#e9f5ef" },
  resultTitle: { fontSize: 16, fontWeight: "700", marginBottom: 8 },
  plate: { fontSize: 20, fontWeight: "700", color: "#133c2b" },
});
