import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001';
const REFRESH_INTERVAL = 5000;
const DIVIDER_CROSSING_BAND_PX = 120;

const COLORS = {
  picked: '#2ec27e',
  rejected: '#f6ad55',
  touch: '#6ea8fe',
  cold: '#ff6b6b',
  hot: '#5eead4',
  trap: '#ff9f43',
  ink: '#10233f',
  panel: '#16325c',
  panelSoft: '#1e4278',
  line: '#355f96',
  text: '#f4f7fb',
  muted: '#9bb3d1',
  accent: '#f3d27a',
  divider: '#ff4d6d',
  overlay: 'rgba(9, 20, 39, 0.72)'
};

const DEFAULT_CONFIG = {
  version: '3.0',
  rtsp_url: '0',
  calibration_timestamp: new Date().toISOString(),
  global_settings: {
    store_open_time: '00:00',
    store_close_time: '23:59',
    interaction_friction_window: 10,
    decision_window: 5,
    divider_detection_width: 120,
    motion_sensitivity: 70,
    min_hand_size: 120
  },
  rois: []
};

function useAnalytics(refreshInterval = REFRESH_INTERVAL) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [resetMessage, setResetMessage] = useState('');

  const fetchData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/analytics/summary`);
      setData(response.data);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message || 'Failed to fetch analytics');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, refreshInterval);
    return () => clearInterval(timer);
  }, [fetchData, refreshInterval]);

  const resetAnalytics = useCallback(async () => {
    await axios.post(`${API_BASE_URL}/api/analytics/reset`);
    setResetMessage('Dashboard history reset');
    await fetchData();
  }, [fetchData]);

  return { data, loading, error, lastUpdated, resetMessage, refetch: fetchData, resetAnalytics };
}

function useCalibrationConfig() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/calibration-config`);
      setConfig(response.data || DEFAULT_CONFIG);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to load calibration config');
    } finally {
      setLoading(false);
    }
  }, []);

  const save = useCallback(async (nextConfig) => {
    const response = await axios.post(`${API_BASE_URL}/api/calibration-config`, nextConfig);
    setConfig(response.data.config);
    return response.data.config;
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { config, setConfig, loading, error, load, save };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function getDividerX(roi) {
  if (!roi.tripwire || roi.tripwire.length !== 2) {
    return roi.x + Math.round(roi.width / 2);
  }
  return roi.tripwire[0][0];
}

function setDividerX(roi, dividerX) {
  const x = clamp(Math.round(dividerX), roi.x + 10, roi.x + roi.width - 10);
  return {
    ...roi,
    tripwire: [
      [x, roi.y],
      [x, roi.y + roi.height]
    ]
  };
}

function createZoneFromRect(rect, index) {
  const roi = {
    zone_id: `Zone_${index + 1}`,
    x: Math.round(rect.x),
    y: Math.round(rect.y),
    width: Math.max(40, Math.round(rect.width)),
    height: Math.max(40, Math.round(rect.height)),
    shelf_side: 'right'
  };
  return setDividerX(roi, roi.x + Math.round(roi.width / 2));
}

function normalizeRect(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  return { x, y, width, height };
}

function pointInRect(point, roi) {
  return (
    point.x >= roi.x &&
    point.x <= roi.x + roi.width &&
    point.y >= roi.y &&
    point.y <= roi.y + roi.height
  );
}

function createDefaultLivePass() {
  return {
    status: 'Watching live feed',
    activity: 'No movement',
    motionPixels: 0,
    lastDirection: 'none',
    totalPasses: 0,
    passesToShelf: 0,
    exitsToCustomer: 0,
    takenCount: 0,
    putBackCount: 0,
    activeTimerMs: 0,
    centroid: null,
    motionBox: null,
    motionPoints: []
  };
}

function createDefaultZoneRuntime() {
  return {
    previousMotionSide: null,
    previousCentroid: null,
    detectionCooldownUntil: 0,
    interactionState: 'idle',
    decisionDeadline: 0,
    statusHoldUntil: 0,
    activityHoldUntil: 0,
    lastActivity: 'No movement'
  };
}

function Header({ mode, setMode, lastUpdated, isConnected }) {
  return (
    <header style={styles.header}>
      <div style={styles.headerBrand}>
        <div style={styles.headerBadge}>StoreSense</div>
        <div>
          <h1 style={styles.headerTitle}>Retail analytics + web calibration</h1>
          <p style={styles.headerSubtitle}>Dashboard and divider setup in one place</p>
        </div>
      </div>

      <div style={styles.headerActions}>
        <div style={styles.segmentedControl}>
          <button
            type="button"
            onClick={() => setMode('dashboard')}
            style={mode === 'dashboard' ? styles.segmentActive : styles.segmentButton}
          >
            Dashboard
          </button>
          <button
            type="button"
            onClick={() => setMode('calibration')}
            style={mode === 'calibration' ? styles.segmentActive : styles.segmentButton}
          >
            Calibration
          </button>
        </div>

        <div style={styles.connectionWrap}>
          <span style={{ ...styles.statusDot, background: isConnected ? COLORS.picked : COLORS.cold }} />
          <span>{isConnected ? 'Backend live' : 'Backend offline'}</span>
          {lastUpdated && <span style={styles.headerTimestamp}>Updated {lastUpdated.toLocaleTimeString()}</span>}
        </div>
      </div>
    </header>
  );
}

function SummaryCards({ summary }) {
  if (!summary) return null;

  const cards = [
    ['Zones', summary.total_zones, COLORS.touch],
    ['Interactions', summary.total_interactions, COLORS.accent],
    ['Picked', summary.total_taken, COLORS.picked],
    ['Rejected', summary.total_put_back, COLORS.rejected]
  ];

  return (
    <div style={styles.cardsGrid}>
      {cards.map(([label, value, color]) => (
        <div key={label} style={styles.metricCard}>
          <div style={styles.metricLabel}>{label}</div>
          <div style={{ ...styles.metricValue, color }}>{value}</div>
        </div>
      ))}
    </div>
  );
}

function ZoneComparisonChart({ zones }) {
  const chartData = (zones || []).map((zone) => ({
    name: zone.zone_id,
    Picked: zone.total_taken,
    Rejected: zone.total_put_back,
    Touch: zone.total_touch
  }));

  return (
    <section style={styles.panel}>
      <h3 style={styles.panelTitle}>Zone performance</h3>
      {chartData.length === 0 ? (
        <div style={styles.emptyState}>No zone data yet</div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 20, right: 16, left: 0, bottom: 48 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.line} />
            <XAxis dataKey="name" angle={-25} textAnchor="end" tick={{ fill: COLORS.muted, fontSize: 12 }} />
            <YAxis tick={{ fill: COLORS.muted }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend wrapperStyle={{ color: COLORS.text }} />
            <Bar dataKey="Picked" fill={COLORS.picked} radius={[8, 8, 0, 0]} />
            <Bar dataKey="Rejected" fill={COLORS.rejected} radius={[8, 8, 0, 0]} />
            <Bar dataKey="Touch" fill={COLORS.touch} radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </section>
  );
}

function ConversionChart({ zones }) {
  const totals = useMemo(() => {
    const list = zones || [];
    return {
      picked: list.reduce((sum, z) => sum + z.total_taken, 0),
      rejected: list.reduce((sum, z) => sum + z.total_put_back, 0),
      touch: list.reduce((sum, z) => sum + z.total_touch, 0)
    };
  }, [zones]);

  const data = [
    { name: 'Picked', value: totals.picked, color: COLORS.picked },
    { name: 'Rejected', value: totals.rejected, color: COLORS.rejected },
    { name: 'Touch', value: totals.touch, color: COLORS.touch }
  ].filter((item) => item.value > 0);

  return (
    <section style={styles.panel}>
      <h3 style={styles.panelTitle}>Interaction mix</h3>
      {data.length === 0 ? (
        <div style={styles.emptyState}>No interaction data yet</div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={data} dataKey="value" innerRadius={56} outerRadius={92} paddingAngle={2}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </section>
  );
}

function AlertsPanel({ alerts }) {
  return (
    <section style={styles.panel}>
      <h3 style={styles.panelTitle}>Alerts</h3>
      {!alerts || alerts.length === 0 ? (
        <div style={styles.emptyState}>All zones look stable</div>
      ) : (
        <div style={styles.alertList}>
          {alerts.map((alert, index) => (
            <div key={`${alert.zone_id}-${index}`} style={styles.alertItem}>
              <strong>{alert.type.replace(/_/g, ' ')}</strong>
              <span style={styles.alertText}>{alert.message}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ZoneTable({ zones }) {
  return (
    <section style={styles.panel}>
      <h3 style={styles.panelTitle}>Zone details</h3>
      {!zones || zones.length === 0 ? (
        <div style={styles.emptyState}>No configured zones yet</div>
      ) : (
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Zone</th>
                <th style={styles.th}>Status</th>
                <th style={styles.th}>Interactions</th>
                <th style={styles.th}>Picked</th>
                <th style={styles.th}>Rejected</th>
                <th style={styles.th}>Neglect</th>
              </tr>
            </thead>
            <tbody>
              {zones.map((zone) => (
                <tr key={zone.zone_id}>
                  <td style={styles.td}>{zone.zone_id}</td>
                  <td style={styles.td}>{zone.status}</td>
                  <td style={styles.td}>{zone.total_interactions}</td>
                  <td style={{ ...styles.td, color: COLORS.picked }}>{zone.total_taken}</td>
                  <td style={{ ...styles.td, color: COLORS.rejected }}>{zone.total_put_back}</td>
                  <td style={styles.td}>{zone.neglect_rate?.toFixed?.(1) ?? zone.neglect_rate}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function DashboardView({ data, loading, error, refetch, resetAnalytics, resetMessage }) {
  if (loading && !data) {
    return <div style={styles.centerState}>Loading analytics...</div>;
  }

  if (error && !data) {
    return (
      <div style={styles.centerState}>
        <div style={styles.errorBox}>Failed to load analytics: {error}</div>
        <button type="button" onClick={refetch} style={styles.primaryButton}>Retry</button>
      </div>
    );
  }

  return (
    <main style={styles.main}>
      <section style={styles.panel}>
        <div style={styles.actionRow}>
          <button type="button" onClick={refetch} style={styles.secondaryButton}>Refresh dashboard</button>
          <button type="button" onClick={resetAnalytics} style={styles.ghostDangerButton}>Reset dashboard data</button>
        </div>
        {resetMessage && <div style={styles.successBox}>{resetMessage}</div>}
      </section>
      <SummaryCards summary={data?.summary} />
      <div style={styles.gridTwo}>
        <ZoneComparisonChart zones={data?.zones} />
        <ConversionChart zones={data?.zones} />
      </div>
      <AlertsPanel alerts={data?.alerts} />
      <ZoneTable zones={data?.zones} />
    </main>
  );
}

function CalibrationStudio({ config, setConfig, loading, error, save, reload }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const analysisCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const previousFrameRef = useRef(null);
  const zoneRuntimeRef = useRef({});

  const [localConfig, setLocalConfig] = useState(config || DEFAULT_CONFIG);
  const [selectedZoneIndex, setSelectedZoneIndex] = useState(0);
  const [isAddingZone, setIsAddingZone] = useState(false);
  const [draftStart, setDraftStart] = useState(null);
  const [draftRect, setDraftRect] = useState(null);
  const [cameraError, setCameraError] = useState('');
  const [saveState, setSaveState] = useState('');
  const [historyState, setHistoryState] = useState('');
  const [sessionEvents, setSessionEvents] = useState([]);
  const [livePassByZone, setLivePassByZone] = useState({});
  
  // Phone camera state
  const [cameraSource, setCameraSource] = useState('laptop'); // 'laptop' or 'phone'
  const [phoneIp, setPhoneIp] = useState('');
  const [phonePort, setPhonePort] = useState('8080');
  const [isConnectingPhone, setIsConnectingPhone] = useState(false);
  const [phoneConnected, setPhoneConnected] = useState(false);
  const [phoneError, setPhoneError] = useState('');
  const phoneImgRef = useRef(null);

  useEffect(() => {
    setLocalConfig(config || DEFAULT_CONFIG);
    setSelectedZoneIndex(0);
  }, [config]);

  // Function to stop laptop camera stream
  const stopLaptopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  // Function to start laptop camera
  const startLaptopCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setCameraError('');
    } catch (err) {
      setCameraError(err.message || 'Failed to access laptop camera');
    }
  }, []);

  // Function to connect to phone camera
  const connectPhoneCamera = useCallback(async () => {
    if (!phoneIp.trim()) {
      setPhoneError('Please enter phone IP address');
      return;
    }
    
    setIsConnectingPhone(true);
    setPhoneError('');
    setPhoneConnected(false);
    
    // Stop laptop camera first
    stopLaptopCamera();
    
    // IP Webcam provides MJPEG stream at /video endpoint
    const phoneStreamUrl = `http://${phoneIp.trim()}:${phonePort}/video`;
    
    // For MJPEG streams, we can use an img element directly
    // But for video analysis, we need to draw it to canvas
    // We'll use a hidden img element and draw frames to video canvas
    try {
      // Test connection first by trying to fetch a single frame
      const testUrl = `http://${phoneIp.trim()}:${phonePort}/photo.jpg`;
      await fetch(testUrl, { mode: 'no-cors' });
      
      // If we get here, the phone is likely reachable
      // Set the source for drawing
      setCameraSource('phone');
      setPhoneConnected(true);
      setPhoneError('');
      
      // Store the URL for the analysis loop to use
      if (phoneImgRef.current) {
        phoneImgRef.current.crossOrigin = 'anonymous';
        phoneImgRef.current.src = phoneStreamUrl;
      }
    } catch (err) {
      setPhoneError(`Failed to connect: ${err.message || 'Check IP and ensure IP Webcam is running'}`);
      setPhoneConnected(false);
      setCameraSource('laptop');
      startLaptopCamera();
    } finally {
      setIsConnectingPhone(false);
    }
  }, [phoneIp, phonePort, stopLaptopCamera, startLaptopCamera]);

  // Function to switch back to laptop camera
  const switchToLaptopCamera = useCallback(() => {
    if (phoneImgRef.current) {
      phoneImgRef.current.src = '';
    }
    setCameraSource('laptop');
    setPhoneConnected(false);
    setPhoneError('');
    startLaptopCamera();
  }, [startLaptopCamera]);

  useEffect(() => {
    setLivePassByZone((prev) => {
      const next = {};
      (localConfig.rois || []).forEach((roi) => {
        next[roi.zone_id] = prev[roi.zone_id] || createDefaultLivePass();
      });
      return next;
    });

    zoneRuntimeRef.current = (localConfig.rois || []).reduce((acc, roi) => {
      acc[roi.zone_id] = zoneRuntimeRef.current[roi.zone_id] || createDefaultZoneRuntime();
      return acc;
    }, {});
  }, [localConfig.rois]);

  useEffect(() => {
    setSelectedZoneIndex((prev) => {
      if ((localConfig.rois || []).length === 0) {
        return 0;
      }
      return Math.min(prev, localConfig.rois.length - 1);
    });
  }, [localConfig.rois]);

  const selectedRoi = localConfig.rois[selectedZoneIndex] || localConfig.rois[0] || null;
  const selectedZoneId = selectedRoi?.zone_id || 'Main_Zone';
  const livePass = livePassByZone[selectedZoneId] || createDefaultLivePass();

  const updateZoneLivePass = useCallback((zoneId, updater) => {
    setLivePassByZone((prev) => {
      const current = prev[zoneId] || createDefaultLivePass();
      return {
        ...prev,
        [zoneId]: updater(current)
      };
    });
  }, []);

  // Initialize camera on mount - start with laptop camera
  useEffect(() => {
    let active = true;
    async function initCamera() {
      if (cameraSource === 'laptop') {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: 'user',
              width: { ideal: 1280 },
              height: { ideal: 720 }
            },
            audio: false
          });
          if (!active) return;
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(() => {});
          }
          setCameraError('');
        } catch (err) {
          setCameraError(err.message || 'Failed to access laptop camera');
        }
      }
    }
    initCamera();

    return () => {
      active = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current;
    const phoneImg = phoneImgRef.current;
    const canvas = canvasRef.current;
    const analysisCanvas = analysisCanvasRef.current;
    
    // Get dimensions from active source
    let sourceWidth, sourceHeight;
    if (cameraSource === 'phone' && phoneImg && phoneImg.naturalWidth > 0) {
      sourceWidth = phoneImg.naturalWidth;
      sourceHeight = phoneImg.naturalHeight;
    } else if (video && video.videoWidth > 0) {
      sourceWidth = video.videoWidth;
      sourceHeight = video.videoHeight;
    } else {
      return;
    }
    
    if (!canvas || !analysisCanvas) return;
    
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;
    analysisCanvas.width = sourceWidth;
    analysisCanvas.height = sourceHeight;

    setLocalConfig((prev) => {
      return {
        ...prev,
        rtsp_url: '0',
        rois: (prev.rois || []).map((roi) => {
          const nextRoi = {
            ...roi,
            x: clamp(roi.x, 0, Math.max(0, sourceWidth - 40)),
            y: clamp(roi.y, 0, Math.max(0, sourceHeight - 40)),
            width: clamp(roi.width, 40, sourceWidth),
            height: clamp(roi.height, 40, sourceHeight)
          };
          return setDividerX(nextRoi, getDividerX(roi));
        })
      };
    });
  }, [cameraSource]);

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rois = localConfig.rois || [];
    rois.forEach((roi, index) => {
      const isSelected = index === selectedZoneIndex;

      ctx.fillStyle = isSelected ? 'rgba(243, 210, 122, 0.08)' : 'rgba(110, 168, 254, 0.05)';
      ctx.fillRect(roi.x, roi.y, roi.width, roi.height);

      ctx.strokeStyle = isSelected ? COLORS.accent : 'rgba(244,247,251,0.55)';
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);

      // Zone label (shelf side removed per user request)
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px sans-serif';
      ctx.fillText(roi.zone_id, roi.x + 8, Math.max(22, roi.y - 8));
    });

    const roi = localConfig.rois[selectedZoneIndex];
    if (roi) {
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px sans-serif';
      ctx.fillText('Live pass detection active', 16, 28);
      if (livePass.activeTimerMs > 0) {
        ctx.fillStyle = COLORS.accent;
        ctx.fillText(`Decision window: ${(livePass.activeTimerMs / 1000).toFixed(1)}s`, 16, 50);
      }

      if (livePass.centroid) {
        if (livePass.motionBox) {
          ctx.strokeStyle = 'rgba(110, 168, 254, 0.9)';
          ctx.lineWidth = 2;
          ctx.strokeRect(
            livePass.motionBox.x,
            livePass.motionBox.y,
            livePass.motionBox.width,
            livePass.motionBox.height
          );
        }

        if (livePass.motionPoints && livePass.motionPoints.length > 0) {
          ctx.fillStyle = '#7dd3fc';
          livePass.motionPoints.forEach((point) => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
            ctx.fill();
          });
        }

        ctx.fillStyle = COLORS.touch;
        ctx.beginPath();
        ctx.arc(livePass.centroid.x, livePass.centroid.y, 7, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (draftRect && isAddingZone) {
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = '#7dd3fc';
      ctx.lineWidth = 2;
      ctx.strokeRect(draftRect.x, draftRect.y, draftRect.width, draftRect.height);
      ctx.setLineDash([]);
    }
  }, [draftRect, isAddingZone, livePass.activeTimerMs, livePass.centroid, livePass.motionBox, livePass.motionPoints, localConfig.rois, selectedZoneIndex, syncCanvasSize]);

  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  const getPointer = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: Math.round((event.clientX - rect.left) * scaleX),
      y: Math.round((event.clientY - rect.top) * scaleY)
    };
  };

  const updateSelectedZone = useCallback((updater) => {
    setLocalConfig((prev) => {
      if (!prev.rois[selectedZoneIndex]) return prev;
      const nextRois = [...prev.rois];
      nextRois[selectedZoneIndex] = updater(nextRois[selectedZoneIndex]);
      return { ...prev, rois: nextRois };
    });
  }, [selectedZoneIndex]);

  const handlePointerDown = (event) => {
    const point = getPointer(event);
    if (isAddingZone) {
      setDraftStart(point);
      setDraftRect({ x: point.x, y: point.y, width: 0, height: 0 });
      return;
    }

    const rois = localConfig.rois || [];
    for (let index = rois.length - 1; index >= 0; index -= 1) {
      const roi = rois[index];
      if (pointInRect(point, roi)) {
        setSelectedZoneIndex(index);
        return;
      }
    }
  };

  const handlePointerMove = (event) => {
    const point = getPointer(event);

    if (isAddingZone && draftStart) {
      setDraftRect(normalizeRect(draftStart, point));
      return;
    }
  };

  const handlePointerUp = () => {
    if (isAddingZone && draftRect && draftRect.width >= 40 && draftRect.height >= 40) {
      setLocalConfig((prev) => {
        const nextZone = createZoneFromRect(draftRect, prev.rois.length);
        return { ...prev, rois: [...prev.rois, nextZone] };
      });
      setSelectedZoneIndex(localConfig.rois.length);
    }

    setDraftStart(null);
    setDraftRect(null);
    setIsAddingZone(false);
  };

  const stopDecisionWindow = useCallback((zoneId) => {
    const runtime = zoneRuntimeRef.current[zoneId] || createDefaultZoneRuntime();
    runtime.interactionState = 'idle';
    runtime.decisionDeadline = 0;
    zoneRuntimeRef.current[zoneId] = runtime;
    updateZoneLivePass(zoneId, (prev) => ({ ...prev, activeTimerMs: 0 }));
  }, [updateZoneLivePass]);

  const pushSessionEvent = useCallback((eventType, zoneId) => {
    setSessionEvents((prev) => ([
      ...prev,
      {
        timestamp: Math.floor(Date.now() / 1000),
        zone_id: zoneId,
        event: eventType
      }
    ]));
  }, []);

  const startDecisionWindow = useCallback((zoneId) => {
    stopDecisionWindow(zoneId);
    const durationMs = (Number(localConfig.global_settings.decision_window) || 5) * 1000;
    const runtime = zoneRuntimeRef.current[zoneId] || createDefaultZoneRuntime();
    runtime.interactionState = 'decision';
    runtime.statusHoldUntil = Date.now() + 1500;
    runtime.decisionDeadline = Date.now() + durationMs;
    zoneRuntimeRef.current[zoneId] = runtime;

    updateZoneLivePass(zoneId, (prev) => ({
      ...prev,
      activeTimerMs: durationMs,
      status: `Object passed to customer side - waiting ${durationMs / 1000}s for put-back`
    }));
  }, [localConfig.global_settings.decision_window, stopDecisionWindow, updateZoneLivePass]);

  const setStableActivity = useCallback((zoneId, nextActivity, holdMs = 600) => {
    const runtime = zoneRuntimeRef.current[zoneId] || createDefaultZoneRuntime();
    const now = Date.now();
    if (nextActivity !== runtime.lastActivity) {
      runtime.lastActivity = nextActivity;
      runtime.activityHoldUntil = now + holdMs;
      zoneRuntimeRef.current[zoneId] = runtime;
      updateZoneLivePass(zoneId, (prev) => ({ ...prev, activity: nextActivity }));
      return;
    }

    if (now >= runtime.activityHoldUntil) {
      updateZoneLivePass(zoneId, (prev) => ({ ...prev, activity: nextActivity }));
    }
    zoneRuntimeRef.current[zoneId] = runtime;
  }, [updateZoneLivePass]);

  const resetLiveSession = useCallback(() => {
    previousFrameRef.current = null;
    zoneRuntimeRef.current = {};
    setSessionEvents([]);
    setHistoryState('');
    setSelectedZoneIndex(0);
    setIsAddingZone(false);
    setDraftStart(null);
    setDraftRect(null);
    setLivePassByZone((prev) => {
      const next = {};
      Object.keys(prev).forEach((zoneId) => {
        next[zoneId] = createDefaultLivePass();
      });
      return next;
    });
  }, []);

  const exportSessionToDashboard = useCallback(async () => {
    if (sessionEvents.length === 0) {
      setHistoryState('No taken/put-back events to add yet');
      return;
    }

    try {
      await axios.post(`${API_BASE_URL}/api/telemetry`, { events: sessionEvents });
      setHistoryState(`Added ${sessionEvents.length} events to dashboard history`);
      setSessionEvents([]);
    } catch (err) {
      setHistoryState(`Failed to add history: ${err.message || 'Unknown error'}`);
    }
  }, [sessionEvents]);

  useEffect(() => {
    let rafId = 0;

    const detectMotionPass = () => {
      const video = videoRef.current;
      const phoneImg = phoneImgRef.current;
      const analysisCanvas = analysisCanvasRef.current;
      const rois = localConfig.rois || [];
      
      // Determine which source to use
      const sourceElement = cameraSource === 'phone' ? phoneImg : video;
      const isReady = cameraSource === 'phone' 
        ? (phoneImg && phoneImg.complete && phoneImg.naturalWidth > 0)
        : (video && video.readyState >= 2);
      
      if (!sourceElement || !analysisCanvas || rois.length === 0 || !isReady) {
        rafId = requestAnimationFrame(detectMotionPass);
        return;
      }

      const ctx = analysisCanvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        rafId = requestAnimationFrame(detectMotionPass);
        return;
      }

      ctx.drawImage(sourceElement, 0, 0, analysisCanvas.width, analysisCanvas.height);
      const frame = ctx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
      const pixels = frame.data;
      const previous = previousFrameRef.current;

      const motionSensitivity = Number(localConfig.global_settings.motion_sensitivity) || 70;
      const minHandSize = Number(localConfig.global_settings.min_hand_size) || 120;
      const dividerBandWidth = Number(localConfig.global_settings.divider_detection_width) || DIVIDER_CROSSING_BAND_PX;
      const now = Date.now();

      if (previous && previous.length === pixels.length) {
        rois.forEach((roi) => {
          const roiX1 = roi.x;
          const roiY1 = roi.y;
          const roiX2 = roi.x + roi.width;
          const roiY2 = roi.y + roi.height;
          let zoneChanged = 0;
          let zoneSumX = 0;
          let zoneSumY = 0;
          let zoneMinX = analysisCanvas.width;
          let zoneMaxX = 0;
          let zoneMinY = analysisCanvas.height;
          let zoneMaxY = 0;

          for (let i = 0; i < pixels.length; i += 16) {
            const diff =
              Math.abs(pixels[i] - previous[i]) +
              Math.abs(pixels[i + 1] - previous[i + 1]) +
              Math.abs(pixels[i + 2] - previous[i + 2]);

            if (diff > motionSensitivity) {
              const pixelIndex = i / 4;
              const x = pixelIndex % analysisCanvas.width;
              const y = Math.floor(pixelIndex / analysisCanvas.width);
              if (x < roiX1 || x > roiX2 || y < roiY1 || y > roiY2) {
                continue;
              }
              zoneChanged += 1;
              zoneSumX += x;
              zoneSumY += y;
              if (x < zoneMinX) zoneMinX = x;
              if (x > zoneMaxX) zoneMaxX = x;
              if (y < zoneMinY) zoneMinY = y;
              if (y > zoneMaxY) zoneMaxY = y;
            }
          }

          const runtime = zoneRuntimeRef.current[roi.zone_id] || createDefaultZoneRuntime();

          if (runtime.interactionState === 'decision' && runtime.decisionDeadline > 0) {
            const remaining = Math.max(0, runtime.decisionDeadline - now);
            if (remaining <= 0) {
              runtime.interactionState = 'idle';
              runtime.decisionDeadline = 0;
              runtime.statusHoldUntil = now + 1800;
              updateZoneLivePass(roi.zone_id, (prev) => ({
                ...prev,
                activeTimerMs: 0,
                takenCount: prev.takenCount + 1,
                status: 'TAKEN +1 - object not returned within decision window',
                lastDirection: 'taken confirmed'
              }));
              pushSessionEvent('TAKEN', roi.zone_id || 'Main_Zone');
            } else {
              updateZoneLivePass(roi.zone_id, (prev) => ({
                ...prev,
                activeTimerMs: remaining,
                status: `Decision window active - ${(remaining / 1000).toFixed(1)}s remaining`
              }));
            }
          }

          if (zoneChanged > minHandSize) {
            const centroid = {
              x: Math.round(zoneSumX / zoneChanged),
              y: Math.round(zoneSumY / zoneChanged)
            };
            const motionBox = {
              x: zoneMinX,
              y: zoneMinY,
              width: Math.max(1, zoneMaxX - zoneMinX),
              height: Math.max(1, zoneMaxY - zoneMinY)
            };
            const motionPoints = [
              centroid,
              { x: zoneMinX, y: centroid.y },
              { x: zoneMaxX, y: centroid.y },
              { x: centroid.x, y: zoneMinY },
              { x: centroid.x, y: zoneMaxY },
              { x: zoneMinX, y: zoneMinY },
              { x: zoneMaxX, y: zoneMinY },
              { x: zoneMinX, y: zoneMaxY },
              { x: zoneMaxX, y: zoneMaxY }
            ];
            const dividerX = getDividerX(roi);
            const side = centroid.x < dividerX ? 'left' : 'right';

            updateZoneLivePass(roi.zone_id, (prev) => ({
              ...prev,
              motionPixels: zoneChanged,
              centroid,
              motionBox,
              motionPoints
            }));
            setStableActivity(roi.zone_id, `Motion detected on ${side} side`, 700);

            const crossedDivider = runtime.previousCentroid
              ? (((runtime.previousCentroid.x - dividerX) * (centroid.x - dividerX)) <= 0 ||
                 (motionBox.x <= dividerX && motionBox.x + motionBox.width >= dividerX))
              : (motionBox.x <= dividerX && motionBox.x + motionBox.width >= dividerX);
            const nearDivider =
              motionBox.x <= dividerX + dividerBandWidth &&
              motionBox.x + motionBox.width >= dividerX - dividerBandWidth;

            if (
              runtime.previousMotionSide &&
              runtime.previousMotionSide !== side &&
              crossedDivider &&
              nearDivider &&
              now > runtime.detectionCooldownUntil
            ) {
              const movingToShelf = side === roi.shelf_side;
              runtime.detectionCooldownUntil = now + 1200;
              runtime.statusHoldUntil = now + 1200;
              runtime.previousMotionSide = side;

              if (movingToShelf) {
                if (runtime.interactionState === 'decision') {
                  stopDecisionWindow(roi.zone_id);
                  updateZoneLivePass(roi.zone_id, (prev) => ({
                    ...prev,
                    motionPixels: zoneChanged,
                    centroid,
                    totalPasses: prev.totalPasses + 1,
                    lastDirection: 'returned to shelf',
                    passesToShelf: prev.passesToShelf + 1,
                    putBackCount: prev.putBackCount + 1,
                    status: 'PUT BACK +1 - object returned within 5s',
                    motionBox,
                    motionPoints
                  }));
                  pushSessionEvent('PUT_BACK', roi.zone_id || 'Main_Zone');
                  setStableActivity(roi.zone_id, 'Return motion detected', 1400);
                } else {
                  updateZoneLivePass(roi.zone_id, (prev) => ({
                    ...prev,
                    motionPixels: zoneChanged,
                    centroid,
                    totalPasses: prev.totalPasses + 1,
                    lastDirection: 'to shelf',
                    passesToShelf: prev.passesToShelf + 1,
                    status: 'PASS DETECTED -> shelf side',
                    motionBox,
                    motionPoints
                  }));
                  setStableActivity(roi.zone_id, 'Shelf-side pass detected', 1200);
                }
              } else {
                startDecisionWindow(roi.zone_id);
                updateZoneLivePass(roi.zone_id, (prev) => ({
                  ...prev,
                  motionPixels: zoneChanged,
                  centroid,
                  lastDirection: 'to customer',
                  exitsToCustomer: prev.exitsToCustomer + 1,
                  motionBox,
                  motionPoints
                }));
                setStableActivity(roi.zone_id, 'Customer-side pass detected', 1200);
              }
            } else {
              runtime.previousMotionSide = side;
            }

            runtime.previousCentroid = centroid;
          } else {
            updateZoneLivePass(roi.zone_id, (prev) => {
              const shouldHoldStatus =
                runtime.interactionState === 'decision' ||
                now < runtime.statusHoldUntil;

              return {
                ...prev,
                motionPixels: zoneChanged,
                centroid: null,
                motionBox: null,
                motionPoints: [],
                activeTimerMs: runtime.interactionState === 'decision' && runtime.decisionDeadline > now ? runtime.decisionDeadline - now : 0,
                status: shouldHoldStatus ? prev.status : 'Watching live feed'
              };
            });
            runtime.previousCentroid = null;

            if (zoneChanged > 0) {
              if (now >= runtime.activityHoldUntil) {
                setStableActivity(roi.zone_id, 'Small movement ignored', 500);
              }
            } else if (now >= runtime.activityHoldUntil) {
              setStableActivity(roi.zone_id, 'No movement', 500);
            }
          }

          zoneRuntimeRef.current[roi.zone_id] = runtime;
        });
      }

      previousFrameRef.current = new Uint8ClampedArray(pixels);
      rafId = requestAnimationFrame(detectMotionPass);
    };

    rafId = requestAnimationFrame(detectMotionPass);
    return () => cancelAnimationFrame(rafId);
  }, [cameraSource, localConfig.global_settings.divider_detection_width, localConfig.global_settings.min_hand_size, localConfig.global_settings.motion_sensitivity, localConfig.rois, pushSessionEvent, setStableActivity, startDecisionWindow, stopDecisionWindow, updateZoneLivePass]);

  const handleSave = async () => {
    setSaveState('Saving...');
    try {
      const saved = await save(localConfig);
      setConfig(saved);
      setLocalConfig(saved);
      setSaveState('Saved to config.json');
    } catch (err) {
      setSaveState(`Save failed: ${err.message || 'Unknown error'}`);
    }
  };

  const deleteSelectedZone = () => {
    if (selectedZoneIndex < 0 || !localConfig.rois[selectedZoneIndex]) {
      return;
    }
    setLocalConfig((prev) => ({
      ...prev,
      rois: prev.rois.filter((_, index) => index !== selectedZoneIndex)
    }));
    setSelectedZoneIndex((prev) => Math.max(0, prev - 1));
    resetLiveSession();
  };

  return (
    <main style={styles.main}>
      <div style={styles.gridCalibration}>
        <section style={{ ...styles.panel, ...styles.videoPanel, padding: 0, overflow: 'hidden' }}>
          <div style={styles.stageHeader}>
            <div>
              <h3 style={styles.panelTitle}>Live calibration</h3>
              <p style={styles.stageHint}>Select camera source below. Add zones, choose one zone to tune, and detect passes.</p>
            </div>
            <div style={styles.stageMeta}>Camera: {cameraSource === 'laptop' ? 'Laptop webcam' : `Phone (${phoneIp})`}</div>
          </div>

          {/* Camera Source Controls */}
          <div style={styles.cameraSourcePanel}>
            <div style={styles.cameraSourceRow}>
              <span style={styles.cameraSourceLabel}>Camera source:</span>
              <div style={styles.segmentedControl}>
                <button
                  type="button"
                  onClick={switchToLaptopCamera}
                  style={cameraSource === 'laptop' ? styles.segmentActive : styles.segmentButton}
                  disabled={isConnectingPhone}
                >
                  Laptop
                </button>
                <button
                  type="button"
                  onClick={() => { setCameraSource('phone'); setPhoneConnected(false); }}
                  style={cameraSource === 'phone' ? styles.segmentActive : styles.segmentButton}
                  disabled={isConnectingPhone}
                >
                  Phone
                </button>
              </div>
              {cameraSource === 'phone' && phoneConnected && (
                <span style={styles.connectionStatus}>
                  <span style={styles.connectedDot} />
                  Connected
                </span>
              )}
            </div>
            
            {cameraSource === 'phone' && (
              <>
                <div style={styles.phoneHelpText}>
                  Install "IP Webcam" app on your Android phone, start the server, and enter the IP shown in the app below.
                </div>
                <div style={styles.phoneInputRow}>
                  <label style={styles.phoneInputLabel}>
                    Phone IP:
                    <input
                      style={styles.phoneInput}
                      type="text"
                      placeholder="192.168.1.100"
                      value={phoneIp}
                      onChange={(e) => setPhoneIp(e.target.value)}
                      disabled={isConnectingPhone}
                    />
                  </label>
                  <label style={styles.phoneInputLabel}>
                    Port:
                    <input
                      style={styles.phoneInputSmall}
                      type="text"
                      placeholder="8080"
                      value={phonePort}
                      onChange={(e) => setPhonePort(e.target.value)}
                      disabled={isConnectingPhone}
                    />
                  </label>
                  <button
                    type="button"
                    style={phoneConnected ? styles.disconnectButton : styles.connectButton}
                    onClick={phoneConnected ? switchToLaptopCamera : connectPhoneCamera}
                    disabled={isConnectingPhone || (!phoneConnected && !phoneIp.trim())}
                  >
                    {isConnectingPhone ? 'Connecting...' : phoneConnected ? 'Disconnect' : 'Connect'}
                  </button>
                </div>
                {phoneError && <div style={styles.phoneErrorText}>{phoneError}</div>}
                {phoneConnected && <div style={styles.phoneSuccessText}>Phone camera connected successfully!</div>}
              </>
            )}
          </div>

          <div style={styles.videoStage}>
            {/* Laptop webcam video element */}
            <video
              ref={videoRef}
              style={{ ...styles.video, display: cameraSource === 'laptop' ? 'block' : 'none' }}
              muted
              playsInline
              onLoadedMetadata={syncCanvasSize}
            />
            {/* Phone camera img element (MJPEG stream) */}
            <img
              ref={phoneImgRef}
              alt="Phone camera feed"
              style={{ ...styles.video, display: cameraSource === 'phone' ? 'block' : 'none' }}
              crossOrigin="anonymous"
              onLoad={syncCanvasSize}
              onError={() => setPhoneError('Failed to load phone camera stream')}
            />
            <canvas
              ref={canvasRef}
              style={styles.canvas}
              onMouseDown={handlePointerDown}
              onMouseMove={handlePointerMove}
              onMouseUp={handlePointerUp}
              onMouseLeave={handlePointerUp}
            />
            <canvas ref={analysisCanvasRef} style={styles.hiddenCanvas} />
            {!cameraError && <div style={styles.videoOverlayTag}>{isAddingZone ? 'Drag to draw a new zone' : 'Click a zone to select it'}</div>}
            {cameraError && <div style={styles.cameraError}>{cameraError}</div>}
          </div>
        </section>

        <section style={styles.sideStack}>
          <section style={{ ...styles.panel, marginBottom: 0 }}>
            <h3 style={styles.panelTitle}>Store settings</h3>
            <div style={styles.formGrid}>
              <label style={styles.label}>Open time
                <input
                  style={styles.input}
                  type="time"
                  value={localConfig.global_settings.store_open_time}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, store_open_time: e.target.value }
                  }))}
                />
              </label>
              <label style={styles.label}>Close time
                <input
                  style={styles.input}
                  type="time"
                  value={localConfig.global_settings.store_close_time}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, store_close_time: e.target.value }
                  }))}
                />
              </label>
              <label style={styles.label}>Friction window
                <input
                  style={styles.input}
                  type="number"
                  min="1"
                  max="60"
                  value={localConfig.global_settings.interaction_friction_window}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, interaction_friction_window: Number(e.target.value) || 10 }
                  }))}
                />
              </label>
              <label style={styles.label}>Decision window
                <input
                  style={styles.input}
                  type="number"
                  min="1"
                  max="30"
                  value={localConfig.global_settings.decision_window}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, decision_window: Number(e.target.value) || 5 }
                  }))}
                />
              </label>
              <label style={styles.label}>Divider detection width
                <input
                  type="range"
                  min="20"
                  max="220"
                  value={localConfig.global_settings.divider_detection_width}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, divider_detection_width: Number(e.target.value) || 120 }
                  }))}
                />
                <span style={styles.helperText}>{localConfig.global_settings.divider_detection_width}px</span>
              </label>
              <label style={styles.label}>Motion sensitivity
                <input
                  type="range"
                  min="20"
                  max="180"
                  value={localConfig.global_settings.motion_sensitivity}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, motion_sensitivity: Number(e.target.value) || 70 }
                  }))}
                />
                <span style={styles.helperText}>{localConfig.global_settings.motion_sensitivity}</span>
              </label>
              <label style={styles.label}>Minimum hand size
                <input
                  type="range"
                  min="20"
                  max="600"
                  value={localConfig.global_settings.min_hand_size}
                  onChange={(e) => setLocalConfig((prev) => ({
                    ...prev,
                    global_settings: { ...prev.global_settings, min_hand_size: Number(e.target.value) || 120 }
                  }))}
                />
                <span style={styles.helperText}>{localConfig.global_settings.min_hand_size} motion pixels</span>
              </label>
            </div>
          </section>

          <section style={{ ...styles.panel, marginBottom: 0 }}>
            <div style={styles.sideHeaderRow}>
              <h3 style={styles.panelTitle}>Zone setup</h3>
              <div style={styles.actionRow}>
                <button type="button" style={styles.secondaryButton} onClick={() => setIsAddingZone(true)}>
                  Add zone
                </button>
                <button type="button" style={styles.ghostDangerButton} onClick={deleteSelectedZone} disabled={!selectedRoi}>
                  Delete zone
                </button>
              </div>
            </div>

            {!selectedRoi ? (
              <div style={styles.emptyState}>Waiting for camera frame...</div>
            ) : (
              <div style={styles.formStack}>
                <label style={styles.label}>Zone ID
                  <input
                    style={styles.input}
                    value={selectedRoi.zone_id}
                    onChange={(e) => updateSelectedZone((roi) => ({ ...roi, zone_id: e.target.value }))}
                  />
                </label>

                <div style={styles.helperCard}>
                  <strong>Controls</strong>
                  <span>Use Add zone, then drag on the camera to create a zone.</span>
                  <span>Click any zone in the frame or list to select it.</span>
                  <span>Object detection triggers when hand exits the zone boundary.</span>
                </div>
              </div>
            )}
          </section>

          <section style={{ ...styles.panel, marginBottom: 0 }}>
            <div style={styles.sideHeaderRow}>
              <h3 style={styles.panelTitle}>Zones in frame</h3>
              <span style={styles.stageMeta}>{localConfig.rois.length} total</span>
            </div>
            <div style={styles.zoneList}>
              {localConfig.rois.length === 0 ? (
                <div style={styles.emptyState}>No zones yet. Click Add zone.</div>
              ) : localConfig.rois.map((roi, index) => {
                const zoneStats = livePassByZone[roi.zone_id] || createDefaultLivePass();
                return (
                  <button
                    key={`${roi.zone_id}-${index}`}
                    type="button"
                    onClick={() => setSelectedZoneIndex(index)}
                    style={index === selectedZoneIndex ? styles.zoneListItemActive : styles.zoneListItem}
                  >
                    <span>{roi.zone_id}</span>
                    <small>{zoneStats.totalPasses} pass | {zoneStats.takenCount} taken | {zoneStats.putBackCount} back</small>
                  </button>
                );
              })}
            </div>
          </section>

          <section style={{ ...styles.panel, marginBottom: 0 }}>
            <div style={styles.sideHeaderRow}>
              <h3 style={styles.panelTitle}>Live pass detection</h3>
              <span style={styles.stageMeta}>{livePass.lastDirection}</span>
            </div>
            <div style={styles.formStack}>
              <div style={styles.helperCard}>
                <strong>{livePass.status}</strong>
                <span>Activity: {livePass.activity}</span>
                <span>Motion pixels: {livePass.motionPixels}</span>
                <span>Total passes: {livePass.totalPasses}</span>
                <span>Passes to shelf: {livePass.passesToShelf}</span>
                <span>Exits to customer: {livePass.exitsToCustomer}</span>
                <span>Taken: {livePass.takenCount}</span>
                <span>Put back: {livePass.putBackCount}</span>
                <span>Timer: {livePass.activeTimerMs > 0 ? `${(livePass.activeTimerMs / 1000).toFixed(1)}s` : 'idle'}</span>
                <span>History queue: {sessionEvents.length}</span>
              </div>
            </div>
          </section>

          <section style={styles.panel}>
            <div style={styles.actionRow}>
              <button type="button" style={styles.primaryButton} onClick={handleSave} disabled={loading}>
                Save calibration
              </button>
              <button type="button" style={styles.secondaryButton} onClick={exportSessionToDashboard}>
                Add live data to dashboard
              </button>
              <button type="button" style={styles.ghostDangerButton} onClick={resetLiveSession}>
                Reset live data
              </button>
              <button type="button" style={styles.secondaryButton} onClick={reload}>
                Reload
              </button>
            </div>
            {error && <div style={styles.errorBox}>Config load error: {error}</div>}
            {saveState && <div style={styles.successBox}>{saveState}</div>}
            {historyState && <div style={styles.successBox}>{historyState}</div>}
          </section>
        </section>
      </div>
    </main>
  );
}

function App() {
  const [mode, setMode] = useState('dashboard');
  const analytics = useAnalytics();
  const calibration = useCalibrationConfig();
  const connected = !analytics.error || !calibration.error;

  return (
    <div style={styles.app}>
      <Header
        mode={mode}
        setMode={setMode}
        lastUpdated={analytics.lastUpdated}
        isConnected={connected}
      />

      {mode === 'dashboard' ? (
        <DashboardView
          data={analytics.data}
          loading={analytics.loading}
          error={analytics.error}
          resetAnalytics={analytics.resetAnalytics}
          resetMessage={analytics.resetMessage}
          refetch={analytics.refetch}
        />
      ) : (
        <CalibrationStudio
          config={calibration.config}
          setConfig={calibration.setConfig}
          loading={calibration.loading}
          error={calibration.error}
          save={calibration.save}
          reload={calibration.load}
        />
      )}
    </div>
  );
}

const tooltipStyle = {
  backgroundColor: COLORS.panel,
  border: `1px solid ${COLORS.line}`,
  borderRadius: 12,
  color: COLORS.text
};

const styles = {
  app: {
    minHeight: '100vh',
    background: 'linear-gradient(180deg, #0b1b32 0%, #12294a 55%, #16325c 100%)',
    color: COLORS.text,
    fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 24,
    padding: '22px 28px',
    borderBottom: `1px solid ${COLORS.line}`,
    background: 'rgba(11, 27, 50, 0.8)',
    backdropFilter: 'blur(12px)',
    flexWrap: 'wrap'
  },
  headerBrand: { display: 'flex', alignItems: 'center', gap: 18 },
  headerBadge: {
    padding: '10px 14px',
    borderRadius: 999,
    background: COLORS.accent,
    color: COLORS.ink,
    fontWeight: 800,
    letterSpacing: 0.4
  },
  headerTitle: { margin: 0, fontSize: 28 },
  headerSubtitle: { margin: '4px 0 0', color: COLORS.muted },
  headerActions: { display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' },
  connectionWrap: {
    display: 'flex', alignItems: 'center', gap: 10, color: COLORS.muted, background: COLORS.panel,
    border: `1px solid ${COLORS.line}`, borderRadius: 999, padding: '10px 14px'
  },
  statusDot: { width: 10, height: 10, borderRadius: '50%' },
  headerTimestamp: { color: COLORS.text },
  segmentedControl: {
    display: 'inline-flex',
    background: COLORS.panel,
    border: `1px solid ${COLORS.line}`,
    borderRadius: 999,
    padding: 4,
    gap: 4
  },
  segmentButton: {
    border: 'none',
    background: 'transparent',
    color: COLORS.muted,
    padding: '10px 14px',
    borderRadius: 999,
    cursor: 'pointer'
  },
  segmentActive: {
    border: 'none',
    background: COLORS.accent,
    color: COLORS.ink,
    padding: '10px 14px',
    borderRadius: 999,
    cursor: 'pointer',
    fontWeight: 700
  },
  main: { maxWidth: 1500, margin: '0 auto', padding: 24 },
  cardsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: 16, marginBottom: 20 },
  metricCard: { background: COLORS.panel, border: `1px solid ${COLORS.line}`, borderRadius: 18, padding: 20 },
  metricLabel: { color: COLORS.muted, marginBottom: 8 },
  metricValue: { fontSize: 36, fontWeight: 800 },
  gridTwo: { display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20, marginBottom: 20 },
  panel: { background: COLORS.panel, border: `1px solid ${COLORS.line}`, borderRadius: 20, padding: 20, marginBottom: 20 },
  panelTitle: { margin: 0, fontSize: 20 },
  emptyState: { color: COLORS.muted, padding: '18px 0' },
  alertList: { display: 'grid', gap: 12 },
  alertItem: { display: 'grid', gap: 4, background: COLORS.panelSoft, borderLeft: `4px solid ${COLORS.trap}`, padding: 14, borderRadius: 12 },
  alertText: { color: COLORS.muted },
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', color: COLORS.muted, padding: '12px 10px', borderBottom: `1px solid ${COLORS.line}` },
  td: { padding: '14px 10px', borderBottom: `1px solid rgba(53,95,150,0.45)` },
  centerState: { display: 'grid', placeItems: 'center', minHeight: '60vh', gap: 16 },
  errorBox: { background: 'rgba(255,107,107,0.12)', color: '#ffd9d9', border: '1px solid rgba(255,107,107,0.35)', borderRadius: 14, padding: 14 },
  successBox: { marginTop: 12, background: 'rgba(46,194,126,0.12)', color: '#d9ffef', border: '1px solid rgba(46,194,126,0.35)', borderRadius: 14, padding: 14 },
  primaryButton: { border: 'none', background: COLORS.accent, color: COLORS.ink, borderRadius: 12, padding: '12px 16px', fontWeight: 700, cursor: 'pointer' },
  secondaryButton: { border: `1px solid ${COLORS.line}`, background: 'transparent', color: COLORS.text, borderRadius: 12, padding: '12px 16px', fontWeight: 700, cursor: 'pointer' },
  ghostDangerButton: { border: `1px solid rgba(255,107,107,0.35)`, background: 'transparent', color: '#ffd9d9', borderRadius: 10, padding: '8px 12px', cursor: 'pointer' },
  gridCalibration: { display: 'grid', gridTemplateColumns: 'minmax(720px, 1.7fr) minmax(360px, 0.95fr)', gap: 20, alignItems: 'start' },
  videoPanel: { position: 'sticky', top: 24, marginBottom: 0 },
  sideStack: { display: 'grid', gap: 20, alignContent: 'start' },
  stageHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, padding: 20, borderBottom: `1px solid ${COLORS.line}` },
  stageHint: { margin: '6px 0 0', color: COLORS.muted },
  stageMeta: { color: COLORS.accent, fontSize: 13, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.6 },
  videoStage: { position: 'relative', aspectRatio: '16 / 9', background: '#091427', overflow: 'hidden' },
  video: { width: '100%', height: '100%', objectFit: 'cover', display: 'block' },
  canvas: { position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair' },
  hiddenCanvas: { display: 'none' },
  videoOverlayTag: { position: 'absolute', left: 16, bottom: 16, background: COLORS.overlay, color: COLORS.text, padding: '10px 14px', borderRadius: 999, fontSize: 13 },
  cameraError: { position: 'absolute', inset: 16, display: 'grid', placeItems: 'center', textAlign: 'center', background: COLORS.overlay, borderRadius: 16, padding: 20 },
  formGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 },
  formStack: { display: 'grid', gap: 12 },
  label: { display: 'grid', gap: 8, color: COLORS.muted, fontSize: 14 },
  input: { width: '100%', borderRadius: 12, border: `1px solid ${COLORS.line}`, background: COLORS.panelSoft, color: COLORS.text, padding: '11px 12px' },
  helperText: { color: COLORS.muted, fontSize: 13 },
  helperCard: { display: 'grid', gap: 6, background: COLORS.panelSoft, borderRadius: 14, padding: 14, color: COLORS.muted },
  sideHeaderRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: 10 },
  zoneList: { display: 'grid', gap: 10 },
  zoneListItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, padding: '12px 14px', borderRadius: 12, border: `1px solid ${COLORS.line}`, background: 'transparent', color: COLORS.text, cursor: 'pointer' },
  zoneListItemActive: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, padding: '12px 14px', borderRadius: 12, border: `1px solid ${COLORS.accent}`, background: 'rgba(243,210,122,0.12)', color: COLORS.text, cursor: 'pointer' },
  actionRow: { display: 'flex', gap: 12, flexWrap: 'wrap' },
  // Phone camera styles
  cameraSourcePanel: { 
    padding: '12px 20px', 
    borderBottom: `1px solid ${COLORS.line}`, 
    background: COLORS.panelSoft 
  },
  cameraSourceRow: { 
    display: 'flex', 
    alignItems: 'center', 
    gap: 16, 
    flexWrap: 'wrap' 
  },
  cameraSourceLabel: { 
    color: COLORS.muted, 
    fontSize: 14, 
    fontWeight: 600 
  },
  phoneInputRow: { 
    display: 'flex', 
    alignItems: 'flex-end', 
    gap: 12, 
    marginTop: 12, 
    flexWrap: 'wrap' 
  },
  phoneInputLabel: { 
    display: 'flex', 
    flexDirection: 'column', 
    gap: 4, 
    color: COLORS.muted, 
    fontSize: 12 
  },
  phoneInput: { 
    width: 160, 
    borderRadius: 8, 
    border: `1px solid ${COLORS.line}`, 
    background: COLORS.panel, 
    color: COLORS.text, 
    padding: '8px 10px', 
    fontSize: 14 
  },
  phoneInputSmall: { 
    width: 70, 
    borderRadius: 8, 
    border: `1px solid ${COLORS.line}`, 
    background: COLORS.panel, 
    color: COLORS.text, 
    padding: '8px 10px', 
    fontSize: 14 
  },
  connectButton: { 
    border: 'none', 
    background: COLORS.picked, 
    color: COLORS.ink, 
    borderRadius: 8, 
    padding: '8px 16px', 
    fontWeight: 700, 
    cursor: 'pointer',
    fontSize: 14
  },
  disconnectButton: { 
    border: `1px solid ${COLORS.cold}`, 
    background: 'transparent', 
    color: COLORS.cold, 
    borderRadius: 8, 
    padding: '8px 16px', 
    fontWeight: 700, 
    cursor: 'pointer',
    fontSize: 14
  },
  phoneErrorText: { 
    color: '#ff6b6b', 
    fontSize: 12, 
    marginTop: 8 
  },
  phoneSuccessText: { 
    color: COLORS.picked, 
    fontSize: 12, 
    marginTop: 8 
  },
  phoneHelpText: { 
    color: COLORS.muted, 
    fontSize: 12, 
    marginTop: 8,
    marginBottom: 4,
    lineHeight: 1.4
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    color: COLORS.picked,
    fontSize: 13,
    fontWeight: 600
  },
  connectedDot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    background: COLORS.picked
  }
};

const styleSheet = document.createElement('style');
styleSheet.textContent = `
  * { box-sizing: border-box; }
  body { margin: 0; background: #0b1b32; }
  button, input { font: inherit; }
  @media (max-width: 1100px) {
    .storesense-grid-two { grid-template-columns: 1fr; }
  }
  @media (max-width: 1080px) {
    body { overflow-x: hidden; }
  }
`;
if (!document.head.querySelector('style[data-storesense-app="true"]')) {
  styleSheet.setAttribute('data-storesense-app', 'true');
  document.head.appendChild(styleSheet);
}

export default App;
