import React, { useState, useEffect, useMemo } from 'react';
import VideoFeed from './components/VideoFeed';
import { MetricsDashboard } from './components/MetricsDashboard';
import RecordingGallery from './components/RecordingGallery';
import { Moon, Sun, Activity, Film, Camera } from 'lucide-react';
import { useTheme } from './ThemeContext';

const MAX_HISTORY = 100;

function AppContent() {
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [playbackMetrics, setPlaybackMetrics] = useState([]);
  const [playbackTime, setPlaybackTime] = useState(0); // Current video play time in seconds
  const [viewMode, setViewMode] = useState('live'); // 'live' or 'playback'
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState([]);
  const [playingFile, setPlayingFile] = useState(null);
  const { isDark, toggleTheme } = useTheme();

  const hostname = window.location.hostname;
  const baseUrl = `http://${hostname}:8000`;

  useEffect(() => {
    fetchRecordings();
    
    const wsUrl = `ws://${hostname}:8000/ws/metrics`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('Connected to metrics WebSocket');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      if (viewMode !== 'live') return;

      try {
        const data = JSON.parse(event.data);
        const timestamp = new Date().getTime();
        
        setMetricsHistory(prev => {
          const newHistory = [...prev, { ...data, time: timestamp }];
          if (newHistory.length > MAX_HISTORY) {
            return newHistory.slice(newHistory.length - MAX_HISTORY);
          }
          return newHistory;
        });
      } catch (e) {
        console.error("Failed to parse websocket data", e);
      }
    };

    ws.onclose = () => {
      console.log('Disconnected from WebSocket');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [viewMode]);

  const fetchRecordings = async () => {
    try {
      const res = await fetch(`${baseUrl}/recordings`);
      const data = await res.json();
      setRecordings(data.recordings || []);
    } catch (e) {
      console.error("Failed to fetch recordings", e);
    }
  };

  const handleStartRecord = async () => {
    try {
      const res = await fetch(`${baseUrl}/record/start`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'started') {
        setIsRecording(true);
      }
    } catch (e) {
      console.error("Failed to start recording", e);
    }
  };

  const handleStopRecord = async () => {
    try {
      const res = await fetch(`${baseUrl}/record/stop`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'stopped') {
        setIsRecording(false);
        fetchRecordings();
      }
    } catch (e) {
      console.error("Failed to stop recording", e);
    }
  };

  const handlePlayRecording = async (filename) => {
    try {
      const res = await fetch(`${baseUrl}/recordings/${filename}/metrics`);
      const metricsData = await res.json();
      
      if (Array.isArray(metricsData)) {
        setPlaybackMetrics(metricsData);
        setPlayingFile(filename);
        setViewMode('playback');
        setPlaybackTime(0);
      } else {
        alert("Metrics data for this recording not found.");
      }
    } catch (e) {
      console.error("Failed to load playback data", e);
    }
  };

  const handleReplayRecordingOsc = async (filename) => {
    try {
      await fetch(`${baseUrl}/recordings/${filename}/osc/replay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: 1.0, loop: false })
      });
    } catch (e) {
      console.error("Failed to replay recording OSC", e);
    }
  };

  const handleBackToLive = () => {
    setViewMode('live');
    setPlayingFile(null);
    setPlaybackMetrics([]);
    setPlaybackTime(0);
  };

  // Sync metrics with playback time
  const activeMetricsData = useMemo(() => {
    if (viewMode === 'live') {
      return metricsHistory;
    } else {
      if (playbackMetrics.length === 0) return [];
      
      // Calculate data based on current video time
      const baseTime = playbackMetrics[0].time;
      const targetTime = baseTime + (playbackTime * 1000);
      
      // Filter data up to the current playback time
      const filtered = playbackMetrics.filter(m => m.time <= targetTime);
      
      // We only want to show a window of data (like the live view)
      if (filtered.length > MAX_HISTORY) {
        return filtered.slice(filtered.length - MAX_HISTORY);
      }
      return filtered;
    }
  }, [viewMode, metricsHistory, playbackMetrics, playbackTime]);

  return (
    <div className="min-h-screen p-4 lg:p-6 transition-colors duration-300">
      <div className="max-w-[1600px] mx-auto h-full flex flex-col gap-6">
        
        {/* Header Area */}
        <header className="flex justify-between items-center glass-card py-3 px-6 rounded-2xl">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-lg">
                <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
                <h1 className="text-xl md:text-2xl text-slate-800 dark:text-slate-100 mb-0 leading-none">
                    Realtime Dance Aesthetics
                </h1>
                <p className="text-xs text-slate-500 font-mono mt-1">
                    {viewMode === 'live' ? 'Live motion analytics dashboard' : `Playback: ${playingFile}`}
                </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {viewMode === 'playback' && (
                <button 
                    onClick={handleBackToLive}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 rounded-full text-xs font-bold transition-all"
                >
                    <Camera className="w-3 h-3" />
                    BACK TO LIVE
                </button>
            )}
            <div className={`badge ${isConnected ? 'badge-active' : 'badge-inactive'} hidden md:flex items-center gap-2`}>
                <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} ${isConnected ? 'animate-pulse' : ''}`}></span>
                {isConnected ? 'WS CONNECTED' : 'WS DISCONNECTED'}
            </div>
            <button 
                onClick={toggleTheme}
                className="p-2 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors shadow-sm"
            >
                {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[800px] xl:h-[calc(100vh-140px)] items-stretch">
          
          {/* Left Column: Video Feed & Library */}
          <div className="lg:col-span-4 xl:col-span-5 flex flex-col gap-6 h-full">
              <div className="flex-1 relative overflow-hidden rounded-2xl shadow-glass dark:shadow-glass-dark group">
                  <VideoFeed 
                    url={`${baseUrl}/video_feed`} 
                    isRecording={isRecording}
                    onRecordStart={handleStartRecord}
                    onRecordStop={handleStopRecord}
                    isPlayback={viewMode === 'playback'}
                    playbackUrl={`${baseUrl}/recordings/${playingFile}`}
                    onBackToLive={handleBackToLive}
                    onPlaybackTimeUpdate={setPlaybackTime}
                  />
              </div>

              {/* Recording Library Section */}
              <div className="h-[280px] glass-card flex flex-col p-0 overflow-hidden">
                  <div className="flex items-center justify-between p-4 border-b border-white/5 bg-white/5">
                      <div className="flex items-center gap-2">
                          <Film className="w-5 h-5 text-blue-400" />
                          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-widest">Library</h2>
                      </div>
                      <span className="text-[10px] font-mono text-slate-500 bg-black/30 px-2 py-1 rounded-full border border-white/5">
                          {recordings.length} CLIPS
                      </span>
                  </div>
                  <RecordingGallery 
                    recordings={recordings} 
                    onPlay={handlePlayRecording} 
                    onReplayOsc={handleReplayRecordingOsc}
                  />
              </div>
          </div>
          
          {/* Right Column: 3x3 Dashboard */}
          <div className="lg:col-span-8 xl:col-span-7 h-full flex flex-col bg-slate-100/30 dark:bg-slate-800/20 backdrop-blur-sm rounded-2xl border border-slate-200/50 dark:border-slate-700/50 p-2 shadow-inner">
             <MetricsDashboard dataHistory={activeMetricsData} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default AppContent;
