import { CameraView, useCameraPermissions, FlashMode } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Button,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";

// ============================================================
// NO TOCAR: conexión con el backend en AWS
// ============================================================
const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://192.168.1.100:8000";
const PREDICT_URL = `${API_URL.replace(/\/$/, "")}/predict`;
const REQUEST_TIMEOUT_MS = 60000;
// ============================================================

type SimitSummary = {
  total?: number | null;
  total_fines?: number | null;
  total_agreements?: number | null;
  payable_total?: number | null;
  payable_fines_count?: number | null;
};

type SimitOffender = {
  document_type?: string | null;
  document_number?: string | null;
  name?: string | null;
};

type SimitInfraction = {
  code?: string | null;
  description?: string | null;
  amount?: number | null;
};

type SimitPaymentProjection = {
  description?: string | null;
  amount?: number | null;
  date?: string | null;
  days?: number | null;
  instructions?: string | null;
};

type SimitFine = {
  ticket_number?: string | null;
  status?: string | null;
  is_comparendo?: boolean;
  plate?: string | null;
  traffic_authority?: string | null;
  department?: string | null;
  amount?: number | null;
  amount_payable?: number | null;
  infraction_date?: string | null;
  offender?: SimitOffender | null;
  infractions?: SimitInfraction[];
  payment_projections?: SimitPaymentProjection[];
};

type SimitAgreement = {
  description?: string | null;
  amount?: number | null;
  [key: string]: unknown;
};

type SimitCourse = {
  description?: string | null;
  [key: string]: unknown;
};

type SimitResult = {
  found?: boolean;
  document_number?: string;
  is_plate?: boolean;
  clear?: boolean;
  summary?: SimitSummary | null;
  fines?: SimitFine[];
  payment_agreements?: SimitAgreement[];
  driving_courses?: SimitCourse[];
  error?: string;
  data?: SimitResult;
};

type Plate = {
  text: string;
  confidence?: number;
  ocr_confidence?: number;
  simit?: SimitResult;
};

type HistoryItem = {
  plate: string;
  simit_data?: SimitResult | null;
  last_checked_at?: string | null;
  created_at?: string;
};

const formatMoney = (value?: number | null) =>
  value == null ? "No disponible" : `$${value.toLocaleString("es-CO")} COP`;

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [camera, setCamera] = useState<CameraView | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{ plates: Plate[] } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [facing, setFacing] = useState<"back" | "front">("back");
  const [flash, setFlash] = useState<FlashMode>("off");
  const [tab, setTab] = useState<"camera" | "history">("camera");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  useEffect(() => {
    if (tab !== "history") return;
    setHistoryLoading(true);
    fetch(`${API_URL.replace(/\/$/, "")}/history`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "No se pudo cargar el historial.");
        setHistory(payload.items ?? []);
      })
      .catch((error) => setErrorMessage(error instanceof Error ? error.message : String(error)))
      .finally(() => setHistoryLoading(false));
  }, [tab]);

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#4fd1a5" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Se necesita acceso a la cámara</Text>
        <Text style={styles.subtitle}>
          La app usa la cámara para fotografiar la placa y enviarla a
          reconocer.
        </Text>
        <Button title="Permitir cámara" onPress={requestPermission} />
      </View>
    );
  }

  const loadHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_URL.replace(/\/$/, "")}/history`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "No se pudo cargar el historial.");
      setHistory(payload.items ?? []);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setHistoryLoading(false);
    }
  };

  const reset = () => {
    setPreview(null);
    setResult(null);
    setErrorMessage(null);
  };

  const processImage = async (uri: string, name = "plate.jpg", type = "image/jpeg") => {
    setLoading(true);
    setResult(null);
    setErrorMessage(null);
    try {
      setPreview(uri);

      const body = new FormData();
      // NO TOCAR: el backend espera exactamente el campo `file`
      body.append("file", {
        uri,
        name,
        type,
      } as unknown as Blob);

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      let response: Response;
      try {
        // NO TOCAR: petición al backend
        response = await fetch(PREDICT_URL, {
          method: "POST",
          body,
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
      }

      const responseText = await response.text();
      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status} ${response.statusText}: ${
            responseText || "sin respuesta"
          }`
        );
      }
      const payload = JSON.parse(responseText) as { plates: Plate[] };
      setResult(payload);
      if (!payload.plates?.length) {
        setErrorMessage("No se detectó ninguna placa en la imagen. Intenta de nuevo, más cerca y con buena luz.");
      } else {
        setErrorMessage(`Placa recibida: ${payload.plates.map((plate) => plate.text).join(", ")}`);
      }
    } catch (error) {
      const message =
        error instanceof Error && error.name === "AbortError"
          ? `Tiempo de espera agotado después de ${
              REQUEST_TIMEOUT_MS / 1000
            } segundos. Verifica tu conexión.`
          : error instanceof Error
          ? error.message
          : String(error);
      console.error("Error de conexión o procesamiento:", error);
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  };

  const capture = async () => {
    if (!camera || loading) return;
    setLoading(true);
    setResult(null);
    setErrorMessage(null);
    try {
      const photo = await camera.takePictureAsync({
        quality: 0.85,
        skipProcessing: false,
      });
      if (!photo?.uri) {
        throw new Error("La cámara no devolvió una URI para la imagen.");
      }
      setLoading(false);
      await processImage(photo.uri);
    } catch (error) {
      setLoading(false);
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const selectFromGallery = async () => {
    if (loading) return;
    try {
      const selection = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.85,
      });

      if (selection.canceled || !selection.assets[0]?.uri) return;

      const asset = selection.assets[0];
      await processImage(
        asset.uri,
        asset.fileName ?? "plate.jpg",
        asset.mimeType ?? "image/jpeg",
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const testBackend = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await fetch(`${API_URL.replace(/\/$/, "")}/health`);
      const responseText = await response.text();
      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status} ${response.statusText}: ${
            responseText || "sin respuesta"
          }`
        );
      }
      setErrorMessage("Conexion con el backend disponible ✅");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("No se pudo conectar con el backend:", error);
      setErrorMessage(`No se pudo conectar con el backend ❌\n${message}`);
    } finally {
      setLoading(false);
    }
  };

  const refreshHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_URL.replace(/\/$/, "")}/history/refresh`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "No se pudo actualizar el historial.");
      await loadHistory();
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
      setHistoryLoading(false);
    }
  };

  const removeHistory = async (plate: string) => {
    try {
      const response = await fetch(`${API_URL.replace(/\/$/, "")}/history/${plate}`, { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "No se pudo eliminar la placa.");
      setHistory((items) => items.filter((item) => item.plate !== plate));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const confidenceColor = (value: number) => {
    if (value >= 0.85) return "#1f9d55";
    if (value >= 0.6) return "#b8860b";
    return "#c0392b";
  };

  const normalizeSimitResult = (simit?: SimitResult | null): SimitResult | null => {
    if (!simit) return null;
    return simit.data ?? simit;
  };

  // ------------------------------------------------------------
  // Renderiza el resultado de SIMIT de forma legible (sin JSON crudo)
  // ------------------------------------------------------------
  const renderSimitStatus = (simit?: SimitResult | null) => {
    const normalizedSimit = normalizeSimitResult(simit);
    if (!normalizedSimit) {
      return <Text style={styles.detailText}>Consulta pendiente.</Text>;
    }
    if (normalizedSimit.error) {
      return (
        <Text style={styles.simitError}>
          ⚠️ No se pudo consultar SIMIT: {normalizedSimit.error}
        </Text>
      );
    }
    if (normalizedSimit.found === false) {
      return (
        <Text style={styles.detailText}>
          No se encontró información en SIMIT para esta placa.
        </Text>
      );
    }

    const fines = normalizedSimit.fines ?? [];
    const agreements = normalizedSimit.payment_agreements ?? [];
    const courses = normalizedSimit.driving_courses ?? [];
    const summary = normalizedSimit.summary;
    const hasBalance = (summary?.total ?? 0) > 0 ||
      (summary?.total_fines ?? 0) > 0 ||
      (summary?.total_agreements ?? 0) > 0 ||
      (summary?.payable_total ?? 0) > 0;
    const isClean = normalizedSimit.clear === true ||
      (normalizedSimit.found === true && !hasBalance && fines.length === 0 && agreements.length === 0 && courses.length === 0);

    return (
      <View style={styles.simit}>
        <Text
          style={[
            styles.simitStatus,
            isClean ? styles.clearStatus : styles.pendingStatus,
          ]}
        >
          {isClean ? "✅ Está limpio en el SIMIT" : "⚠️ Tiene pendientes en SIMIT"}
        </Text>

        {summary && (
          <View style={styles.summaryBlock}>
            <Text style={styles.sectionLabel}>Totales de la cuenta:</Text>
            <Text style={styles.detailText}>• Total adeudado: {formatMoney(summary.total)}</Text>
            <Text style={styles.detailText}>• Multas: {formatMoney(summary.total_fines)}</Text>
            <Text style={styles.detailText}>• Acuerdos de pago: {formatMoney(summary.total_agreements)}</Text>
            <Text style={styles.detailText}>• Total a pagar hoy: {formatMoney(summary.payable_total)}</Text>
            {summary.payable_fines_count != null && (
              <Text style={styles.detailText}>
                • Multas pagables: {summary.payable_fines_count}
              </Text>
            )}
          </View>
        )}

        {fines.length > 0 && (
          <View style={styles.summaryBlock}>
            <Text style={styles.sectionLabel}>Comparendos y multas pendientes:</Text>
            {fines.map((fine, idx) => (
              <View key={`${fine.ticket_number ?? idx}`} style={styles.fine}>
                <Text style={styles.detailText}>
                  • {fine.is_comparendo ? "Comparendo" : "Multa"}{" "}
                  {fine.ticket_number ?? "sin número"}
                  {fine.traffic_authority ? ` — ${fine.traffic_authority}` : ""}
                </Text>
                <Text style={styles.detailSub}>
                  {fine.infraction_date ? `Fecha: ${fine.infraction_date}. ` : ""}
                  A pagar: {formatMoney(fine.amount_payable ?? fine.amount)}
                </Text>
              </View>
            ))}
          </View>
        )}

        {agreements.length > 0 && (
          <View style={styles.summaryBlock}>
            <Text style={styles.sectionLabel}>
              Acuerdos de pago: {agreements.length}
            </Text>
            {agreements.map((agreement, idx) => (
              <Text key={idx} style={styles.detailSub}>
                • {agreement.description ?? "Acuerdo de pago"}
                {agreement.amount != null ? ` — ${formatMoney(agreement.amount)}` : ""}
              </Text>
            ))}
          </View>
        )}

        {courses.length > 0 && (
          <View style={styles.summaryBlock}>
            <Text style={styles.sectionLabel}>
              Cursos viales: {courses.length}
            </Text>
            {courses.map((course, idx) => (
              <Text key={idx} style={styles.detailSub}>
                • {course.description ?? "Curso vial"}
              </Text>
            ))}
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.tabs}>
        <Pressable style={[styles.tab, tab === "camera" && styles.activeTab]} onPress={() => setTab("camera")}>
          <Text style={styles.tabText}>Cámara</Text>
        </Pressable>
        <Pressable style={[styles.tab, tab === "history" && styles.activeTab]} onPress={() => setTab("history")}>
          <Text style={styles.tabText}>Historial</Text>
        </Pressable>
      </View>
      {tab === "camera" && (!preview ? (
        <View style={styles.cameraFrame}>
          <CameraView
            ref={setCamera}
            style={StyleSheet.absoluteFill}
            facing={facing}
            flash={flash}
          />
          <View style={styles.cameraOverlay}>
            <Pressable
              style={styles.iconButton}
              onPress={() =>
                setFacing((f) => (f === "back" ? "front" : "back"))
              }
            >
              <Text style={styles.iconButtonText}>🔄</Text>
            </Pressable>
            <Pressable
              style={styles.iconButton}
              onPress={() =>
                setFlash((f) => (f === "off" ? "on" : "off"))
              }
            >
              <Text style={styles.iconButtonText}>
                {flash === "off" ? "⚡️" : "⚡️✅"}
              </Text>
            </Pressable>
          </View>
        </View>
      ) : (
        <Image source={{ uri: preview }} style={styles.previewFull} />
      ))}

      {tab === "camera" && <View style={styles.controls}>
        {!preview ? (
          <>
            <Pressable
              style={[styles.secondaryButton, loading && styles.disabled]}
              onPress={testBackend}
              disabled={loading}
            >
              <Text style={styles.secondaryButtonText}>
                Probar conexión con el servidor
              </Text>
            </Pressable>
            <Pressable
              style={[styles.primaryButton, loading && styles.disabled]}
              onPress={capture}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color="#101820" />
              ) : (
                <Text style={styles.primaryButtonText}>
                  Capturar y leer placa
                </Text>
              )}
            </Pressable>
            <Pressable
              style={[styles.secondaryButton, loading && styles.disabled]}
              onPress={selectFromGallery}
              disabled={loading}
            >
              <Text style={styles.secondaryButtonText}>
                Elegir foto de la galería
              </Text>
            </Pressable>
          </>
        ) : (
          <>
            <Pressable
              style={[styles.secondaryButton, loading && styles.disabled]}
              onPress={reset}
              disabled={loading}
            >
              <Text style={styles.secondaryButtonText}>Tomar otra foto</Text>
            </Pressable>
            {loading && (
              <View style={styles.loadingRow}>
                <ActivityIndicator color="#4fd1a5" />
                <Text style={styles.loadingText}>Procesando imagen...</Text>
              </View>
            )}
          </>
        )}
      </View>}

      {errorMessage && (
        <View style={styles.error}>
          <Text style={styles.errorText}>{errorMessage}</Text>
        </View>
      )}

      {tab === "camera" && result && result.plates?.length > 0 && (
        <View style={styles.result}>
          <Text style={styles.resultTitle}>Placa recibida</Text>
          {result.plates.map((plate, index) => (
            <View key={`${plate.text}-${index}`} style={styles.plateBlock}>
              <View style={styles.plateRow}>
                <Text style={styles.plate}>
                  Placa: {plate.text || "Texto no detectado"}
                </Text>
                {plate.confidence != null && <Text
                  style={[styles.confidence, { color: confidenceColor(plate.confidence) }]}
                >{Math.round(plate.confidence * 100)}%</Text>}
              </View>
              {plate.simit && renderSimitStatus(plate.simit)}
            </View>
          ))}
        </View>
      )}
      {tab === "history" && <ScrollView style={styles.history} contentContainerStyle={styles.historyContent}>
        <View style={styles.historyHeader}>
          <Text style={styles.resultTitle}>Historial de consultas</Text>
          <Pressable style={styles.refreshButton} onPress={refreshHistory} disabled={historyLoading}>
            <Text style={styles.refreshText}>{historyLoading ? "Actualizando..." : "Actualizar datos"}</Text>
          </Pressable>
        </View>
        {history.map((item) => (
          <View key={item.plate} style={styles.historyItem}>
            <View style={styles.plateRow}>
              <Text style={styles.plate}>Placa: {item.plate}</Text>
              <Pressable onPress={() => removeHistory(item.plate)}>
                <Text style={styles.deleteText}>Eliminar</Text>
              </Pressable>
            </View>
            <Text style={styles.detailText}>
              Última consulta: {item.last_checked_at ? new Date(item.last_checked_at).toLocaleString() : "Pendiente"}
            </Text>
            {renderSimitStatus(item.simit_data)}
          </View>
        ))}
        {!historyLoading && history.length === 0 && <Text style={styles.loadingText}>No hay placas guardadas.</Text>}
      </ScrollView>}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#101820", padding: 16, gap: 12 },
  tabs: { flexDirection: "row", gap: 8 },
  tab: { flex: 1, paddingVertical: 10, alignItems: "center", borderRadius: 10, backgroundColor: "#26333c" },
  activeTab: { backgroundColor: "#4fd1a5" },
  tabText: { color: "#e6ecf0", fontWeight: "700" },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
    padding: 24,
    backgroundColor: "#101820",
  },
  title: { fontSize: 18, fontWeight: "700", textAlign: "center", color: "#fff" },
  subtitle: {
    fontSize: 14,
    textAlign: "center",
    color: "#a9b4bd",
    paddingHorizontal: 12,
  },
  cameraFrame: {
    flex: 1,
    minHeight: 320,
    overflow: "hidden",
    borderRadius: 16,
    backgroundColor: "#202b33",
  },
  cameraOverlay: {
    position: "absolute",
    top: 12,
    right: 12,
    gap: 8,
  },
  iconButton: {
    backgroundColor: "rgba(0,0,0,0.45)",
    borderRadius: 20,
    width: 40,
    height: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  iconButtonText: { fontSize: 16 },
  previewFull: {
    flex: 1,
    minHeight: 320,
    borderRadius: 16,
    resizeMode: "cover",
  },
  controls: { gap: 10 },
  primaryButton: {
    backgroundColor: "#4fd1a5",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  primaryButtonText: { color: "#101820", fontWeight: "700", fontSize: 16 },
  secondaryButton: {
    backgroundColor: "#26333c",
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
  },
  secondaryButtonText: { color: "#e6ecf0", fontWeight: "600", fontSize: 14 },
  disabled: { opacity: 0.5 },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },
  loadingText: { color: "#a9b4bd", fontSize: 13 },
  result: { padding: 16, borderRadius: 12, backgroundColor: "#e9f5ef" },
  history: { flex: 1 },
  historyContent: { gap: 12, paddingBottom: 24 },
  historyHeader: { gap: 10 },
  refreshButton: { backgroundColor: "#4fd1a5", borderRadius: 10, paddingVertical: 12, alignItems: "center" },
  refreshText: { color: "#101820", fontWeight: "700" },
  historyItem: { padding: 14, borderRadius: 12, backgroundColor: "#e9f5ef", gap: 6 },
  deleteText: { color: "#b42318", fontWeight: "700" },
  error: { padding: 16, borderRadius: 12, backgroundColor: "#ffe9e9" },
  errorText: { color: "#8b1e1e", fontSize: 14 },
  resultTitle: { fontSize: 16, fontWeight: "700", marginBottom: 8, color: "#133c2b" },
  plateRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 4,
  },
  plateBlock: { borderTopWidth: 1, borderTopColor: "#c9e2d5", paddingTop: 8, marginTop: 8 },
  plate: { fontSize: 20, fontWeight: "700", color: "#133c2b" },
  confidence: { fontSize: 16, fontWeight: "700" },
  simit: { marginTop: 8, gap: 4 },
  simitStatus: { fontSize: 15, fontWeight: "700" },
  clearStatus: { color: "#1f9d55" },
  pendingStatus: { color: "#b8860b" },
  summaryBlock: { marginTop: 6, gap: 2 },
  sectionLabel: { fontSize: 13, fontWeight: "700", color: "#133c2b", marginBottom: 2 },
  detailText: { fontSize: 13, color: "#315443" },
  detailSub: { fontSize: 12, color: "#4a6357", marginLeft: 8, marginBottom: 4 },
  fine: { marginTop: 4 },
  simitError: { marginTop: 8, color: "#8b1e1e", fontSize: 13 },
});