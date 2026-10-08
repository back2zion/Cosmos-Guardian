import React, { useState, useRef, useEffect } from 'react';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Play, 
  Upload, 
  Activity, 
  ChevronRight,
  Brain,
  Video,
  Settings,
  HelpCircle,
  X,
  Cpu,
  Layers,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

const API_BASE = 'http://localhost:8888';

function App() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [context, setContext] = useState("Industrial Warehouse Safety");
  const [isLive, setIsLive] = useState(false);
  
  const [status, setStatus] = useState(null);
  const [streamingText, setStreamingText] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  
  const [config, setConfig] = useState({
    samplingFps: 4,
    confidenceThreshold: 70,
    reportDetail: 'high',
    autoCapture: true
  });
  
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setVideoUrl(URL.createObjectURL(selectedFile));
      setResult(null);
      setIsLive(false);
      stopWebcam();
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsLive(false);
  };

  const toggleLive = async () => {
    if (isLive) {
      stopWebcam();
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsLive(true);
        setVideoUrl(null);
        setFile(null);
      } catch (err) {
        console.error("Error accessing webcam", err);
      }
    }
  };

  // Status Handler for Streams
  const processStream = async (response) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.stage === 'complete') {
              setResult(data.result);
              setStatus(null);
              setStreamingText(""); // Clear once structured result is in
            } else if (data.stage === 'streaming_reasoning') {
              setStreamingText(data.chunk);
              setStatus("Reasoning Active...");
            } else {
              setStatus(data.detail);
            }
          } catch (e) {
            console.error("Error parsing stream line", e);
          }
        }
      }
    }
  };

  // Auto-analysis loop for Live Mode
  useEffect(() => {
    let interval;
    if (isLive && !analyzing) {
      interval = setInterval(() => {
        captureAndAnalyze();
      }, 15000); // 15 seconds loop for UX comfort
    }
    return () => clearInterval(interval);
  }, [isLive, analyzing]);

  const captureAndAnalyze = async () => {
    if (!isLive || analyzing) return;

    setAnalyzing(true);
    const stream = streamRef.current;
    const recorder = new MediaRecorder(stream);
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: 'video/webm' });
      const liveFile = new File([blob], "live_capture.webm", { type: 'video/webm' });
      
      const formData = new FormData();
      formData.append('file', liveFile);
      formData.append('context', context + " (Live Feed)");

      try {
        const response = await fetch(`${API_BASE}/analyze`, {
          method: 'POST',
          body: formData
        });
        await processStream(response);
      } catch (error) {
        console.error("Live analysis failed", error);
        setStatus("Live Feed Error");
      } finally {
        setAnalyzing(false);
      }
    };

    recorder.start();
    setTimeout(() => recorder.stop(), 3000); // Capture 3 seconds of video
  };

  const startAnalysis = async () => {
    if (!file && !isLive) return;
    if (isLive) {
      captureAndAnalyze();
      return;
    }
    
    setAnalyzing(true);
    setResult(null);
    setStatus("Initiating Expert Logic...");

    const formData = new FormData();
    formData.append('file', file);
    formData.append('context', context);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
      });
      await processStream(response);
    } catch (error) {
      console.error("Analysis failed", error);
      setStatus("System Error. Please check backend.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen font-sans text-zinc-900">
      {/* Background Glow */}
      <div className="fixed top-0 left-0 w-full h-full pointer-events-none opacity-10">
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-blue-400 rounded-full blur-[140px] -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-[800px] h-[800px] bg-emerald-400 rounded-full blur-[140px] translate-y-1/2 -translate-x-1/2" />
      </div>

      {/* Header */}
      <header className="fixed top-0 z-50 w-full border-b border-zinc-200 glass bg-white/40 backdrop-blur-xl">
        <div className="container flex items-center justify-between px-6 py-4 mx-auto">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600 rounded-lg shadow-lg shadow-blue-500/30">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-zinc-900">COSMOS <span className="text-blue-600">GUARDIAN</span></h1>
              <div className="flex items-center gap-2 text-[10px] text-zinc-500 font-mono">
                <span className="flex w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                NVIDIA COSMOS REASON 2 ACTIVE
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-6 text-zinc-600">
            <button 
              onClick={() => setShowSettings(true)}
              className="flex items-center gap-2 text-sm hover:text-blue-600 transition-colors font-bold"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>
            <button 
              onClick={() => setShowHelp(true)}
              className="flex items-center gap-2 text-sm hover:text-blue-600 transition-colors font-bold"
            >
              <HelpCircle className="w-4 h-4" />
              Help
            </button>
          </nav>
        </div>
      </header>

      <main className="container pt-32 pb-20 mx-auto px-6 grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10">
        
        {/* Left Column: Input & Video */}
        <div className="lg:col-span-8 space-y-6">
          <div className="relative aspect-video rounded-3xl overflow-hidden glass border border-white bg-white/20 shadow-2xl shadow-zinc-200/50 group">
            {videoUrl ? (
              <video 
                src={videoUrl} 
                controls 
                className="w-full h-full object-cover"
              />
            ) : isLive ? (
              <video 
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="w-full h-full object-cover scale-x-[-1]"
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full p-12 text-center text-zinc-500">
                <Video className="w-16 h-16 mb-4 opacity-20" />
                <p className="text-lg">No video feed detected</p>
                <p className="text-sm opacity-60">Upload a system footprint or camera stream to begin analysis</p>
              </div>
            )}

            {/* Overlay Status */}
            <div className="absolute top-6 left-6 flex gap-2">
              <span className="px-4 py-1.5 text-[10px] font-bold tracking-widest bg-white/80 backdrop-blur-md rounded-full border border-zinc-200 text-zinc-800 uppercase flex items-center gap-2 shadow-sm">
                <span className={`w-2 h-2 rounded-full ${analyzing ? 'bg-amber-500 animate-pulse' : (isLive ? 'bg-emerald-500' : 'bg-zinc-400')}`} />
                {analyzing ? 'Analyzing Frame...' : isLive ? 'Live Feed Active' : 'Standby'}
              </span>
            </div>
            
            <AnimatePresence>
              {analyzing && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 bg-white/60 backdrop-blur-[6px] pointer-events-none flex flex-col items-center justify-center p-8 text-center"
                >
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
                    className="mb-6 relative"
                  >
                    <div className="absolute inset-0 bg-blue-500/20 blur-2xl rounded-full" />
                    <Activity className="w-16 h-16 text-blue-600 relative z-10" />
                  </motion.div>
                  
                  <motion.h4 
                    key={status}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-xl font-extrabold text-zinc-900 tracking-tight mb-2"
                  >
                    {status || "Initiating Cosmos Reason..."}
                  </motion.h4>
                  
                  <div className="flex gap-1.5 mt-4">
                    {[0, 1, 2, 3].map(i => (
                      <motion.div
                        key={i}
                        animate={{ 
                          scale: [1, 1.4, 1],
                          opacity: [0.4, 1, 0.4]
                        }}
                        transition={{ 
                          repeat: Infinity, 
                          duration: 1.5, 
                          delay: i * 0.2 
                        }}
                        className="w-2.5 h-2.5 bg-blue-600 rounded-full"
                      />
                    ))}
                  </div>
                  
                  <p className="mt-8 text-[10px] text-zinc-500 font-bold uppercase tracking-[0.3em]">
                    NVIDIA COSMOS VLM ENGINE
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <button 
              onClick={() => fileInputRef.current.click()}
              className="flex items-center justify-center gap-4 p-5 rounded-2xl bg-white border border-zinc-200 text-zinc-700 shadow-sm hover:shadow-md hover:border-blue-200 transition-all group"
            >
              <div className="p-2 bg-blue-50 rounded-lg group-hover:bg-blue-100 transition-colors">
                <Upload className="w-5 h-5 text-blue-600" />
              </div>
              <div className="text-left">
                <div className="text-sm font-bold">Upload Media</div>
                <div className="text-[10px] text-zinc-400 font-medium uppercase">Select Footprint</div>
              </div>
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="video/*,image/*"
              onChange={handleFileChange}
            />

            <button 
              onClick={toggleLive}
              className={`flex items-center justify-center gap-4 p-5 rounded-2xl border transition-all shadow-sm ${
                isLive 
                ? 'bg-emerald-50 border-emerald-200 text-emerald-700 shadow-emerald-100' 
                : 'bg-white border-zinc-200 text-zinc-700 hover:shadow-md hover:border-emerald-200'
              }`}
            >
              <div className={`p-2 rounded-lg transition-colors ${isLive ? 'bg-emerald-100' : 'bg-zinc-100'}`}>
                <Video className={`w-5 h-5 ${isLive ? 'text-emerald-600' : 'text-zinc-500'}`} />
              </div>
              <div className="text-left">
                <div className="text-sm font-bold">{isLive ? 'Stop Feed' : 'Start Feed'}</div>
                <div className="text-[10px] text-zinc-400 font-medium uppercase">Live Stream</div>
              </div>
            </button>

            <button 
              onClick={startAnalysis}
              disabled={(!file && !isLive) || analyzing}
              className={`flex items-center justify-center gap-4 p-5 rounded-2xl border transition-all shadow-lg ${
                (!file && !isLive) || analyzing 
                ? 'bg-zinc-50 border-zinc-200 text-zinc-300 shadow-none' 
                : 'bg-blue-600 border-blue-700 text-white hover:bg-blue-700 shadow-blue-200'
              }`}
            >
              <Brain className="w-5 h-5" />
              <div className="text-left">
                <div className="text-sm font-bold uppercase tracking-tight">Run Reason 2</div>
                <div className="text-[10px] opacity-70 font-medium uppercase">Execute Logic</div>
              </div>
            </button>
          </div>
        </div>

        {/* Right Column: Analysis Panel */}
        <div className="lg:col-span-4 space-y-6">
          
          {/* Safety Score */}
          <div className="p-8 rounded-3xl bg-white border border-zinc-200 shadow-xl shadow-zinc-200/40">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-[10px] font-black uppercase tracking-widest text-zinc-400">Environment Integrity</h3>
              <div className={`w-2 h-2 rounded-full ${result && result.overall_safety_score < 70 ? 'bg-amber-500' : 'bg-emerald-500'}`} />
            </div>
            
            <div className="flex items-baseline gap-2">
              <span className="text-6xl font-black text-zinc-900 leading-none">{result ? result.overall_safety_score : '--'}</span>
              <span className="text-zinc-400 font-bold text-xl">pt</span>
            </div>
            
            <div className="mt-8 h-2 w-full bg-zinc-100 rounded-full overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${result ? result.overall_safety_score : 0}%` }}
                className={`h-full ${result && result.overall_safety_score < 70 ? 'bg-amber-500' : 'bg-blue-600'}`}
              />
            </div>
          </div>

          {/* Hazard List */}
          <div className="p-8 rounded-3xl bg-zinc-50/50 backdrop-blur-md border border-zinc-200 min-h-[480px] shadow-inner relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 opacity-50" />
            
            {/* NEW: Emergency Flash Summary */}
            {result?.hazards_detected?.some(h => h.severity === 'Critical' || h.severity === 'High') && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 p-6 bg-red-600 rounded-2xl shadow-xl shadow-red-200 border border-red-500 text-white"
              >
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 animate-pulse" />
                  <span className="text-[10px] font-black uppercase tracking-widest">Immediate Alert</span>
                </div>
                <h4 className="text-lg font-black leading-tight">
                  {result.hazards_detected.find(h => h.severity === 'Critical' || h.severity === 'High').type}
                </h4>
                <p className="mt-2 text-xs font-medium text-red-100 opacity-90">
                  {result.hazards_detected.find(h => h.severity === 'Critical' || h.severity === 'High').dynamic_prediction}
                </p>
              </motion.div>
            )}

            <h3 className="text-[10px] font-black uppercase tracking-widest text-zinc-400 mb-8 relative z-10">
              {streamingText ? 'Live Reasoning Trace' : 'Detected Physics Hazards'}
            </h3>
            
            <div className="space-y-5 relative z-10">
              {streamingText ? (
                <div className="p-6 rounded-2xl bg-white border border-zinc-200 shadow-sm font-mono text-[11px] leading-relaxed text-zinc-600 max-h-[400px] overflow-y-auto custom-scrollbar">
                  <div className="flex items-center gap-2 text-blue-600 mb-4 font-sans font-black uppercase tracking-widest text-[9px]">
                    <span className="w-1.5 h-1.5 bg-blue-600 rounded-full animate-ping" />
                    VLM Stream Active
                  </div>
                  <ReactMarkdown>{streamingText}</ReactMarkdown>
                </div>
              ) : result?.error ? (
                <div className="p-5 rounded-2xl bg-red-50 border border-red-100 text-center space-y-3">
                  <AlertTriangle className="w-8 h-8 text-red-500 mx-auto" />
                  <p className="text-xs font-bold text-red-600 uppercase">Analysis Error</p>
                  <p className="text-[10px] text-red-400 font-medium leading-relaxed">
                    The model was unable to generate a structured reasoning trace. 
                    This can happen with very long videos or complex scenes. 
                    Please try a shorter clip or different context.
                  </p>
                  {result.raw_output && (
                    <div className="mt-4 p-3 bg-white/50 rounded-xl text-left border border-red-50">
                      <p className="text-[9px] font-black text-red-300 uppercase mb-1">Raw Output Sample</p>
                      <p className="text-[9px] text-zinc-400 truncate italic">"{result.raw_output.slice(0, 100)}..."</p>
                    </div>
                  )}
                </div>
              ) : result?.hazards_detected?.length > 0 ? (
                result.hazards_detected.map((hazard, i) => (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: i * 0.1 }}
                    key={i} 
                    className="p-5 rounded-2xl bg-white border border-zinc-200 shadow-sm space-y-4"
                  >
                    <div className="flex items-center justify-between">
                      <span className={`text-[9px] font-black px-2.5 py-1 rounded-full uppercase tracking-tighter ${
                        hazard.severity === 'Critical' ? 'bg-red-50 text-red-600 border border-red-100' :
                        hazard.severity === 'High' ? 'bg-amber-50 text-amber-600 border border-amber-100' :
                        'bg-zinc-100 text-zinc-500'
                      }`}>
                        {hazard.severity} RISK
                      </span>
                      <AlertTriangle className={`w-4 h-4 ${hazard.severity === 'Critical' ? 'text-red-500' : 'text-amber-500'}`} />
                    </div>
                    <div>
                      <h4 className="font-extrabold text-sm text-zinc-900 mb-1.5">{hazard.type}</h4>
                      <p className="text-xs text-zinc-500 leading-relaxed font-medium">{hazard.description}</p>
                    </div>
                    <div className="pt-4 border-t border-zinc-100">
                      <p className="text-[10px] font-bold text-blue-600 mb-2 flex items-center gap-2">
                        <Brain className="w-3 h-3" /> REASONING TRACE
                      </p>
                      <div className="text-[10px] text-zinc-900 font-medium leading-relaxed bg-zinc-50 p-4 rounded-xl border border-zinc-100 prose prose-sm max-w-none relative overflow-hidden group">
                        <div className="absolute top-0 left-0 w-1 h-full bg-blue-500/20 group-hover:bg-blue-500 transition-colors" />
                        <ReactMarkdown>
                          {hazard.reasoning || hazard.description || "The model identifies a critical safety breach based on spatial proximity and physical risk factors observed in the scene."}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-20 text-zinc-300 space-y-4">
                  <Shield className="w-12 h-12 opacity-20" />
                  <p className="text-xs font-bold uppercase tracking-widest">Awaiting Input</p>
                </div>
              )}
            </div>
            
            {result?.recommendation && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 p-5 rounded-2xl bg-zinc-900 border border-zinc-800 shadow-xl"
              >
                <p className="text-[9px] font-black text-blue-400 uppercase tracking-widest mb-1.5 flex items-center gap-2">
                   Countermeasure Plan
                </p>
                <p className="text-xs text-white font-medium leading-relaxed">{result.recommendation}</p>
              </motion.div>
            )}
          </div>

        </div>
      </main>

      {/* Footer / Meta */}
      <footer className="container mx-auto px-6 py-12 border-t border-zinc-200 text-[10px] text-zinc-400 flex justify-between items-center uppercase font-bold tracking-widest leading-none">
        <div className="flex items-center gap-4">
          <span>COSMOS GUARDIAN UNIT #7155</span>
          <span className="w-1 h-1 bg-zinc-200 rounded-full" />
          <span>LATENCY 400MS</span>
        </div>
        <div className="flex items-center gap-4">
          <span>NVIDIA RTX 3090</span>
          <span className="w-1 h-1 bg-zinc-200 rounded-full" />
          <span className="text-zinc-600">DESIGNED BY DOOIL KWAK</span>
        </div>
      </footer>

      {/* Settings Modal */}
      <AnimatePresence>
        {showSettings && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              onClick={() => setShowSettings(false)}
              className="absolute inset-0 bg-zinc-900/40 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-lg bg-white rounded-3xl shadow-2xl overflow-hidden border border-zinc-200"
            >
              <div className="p-8 border-b border-zinc-100 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-zinc-100 rounded-xl text-zinc-600">
                    <Settings className="w-5 h-5" />
                  </div>
                  <h2 className="text-xl font-black tracking-tight">System Configuration</h2>
                </div>
                <button onClick={() => setShowSettings(false)} className="p-2 hover:bg-zinc-100 rounded-full transition-colors text-zinc-400">
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="p-8 space-y-8">
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] text-blue-600">
                    <Cpu className="w-3 h-3" /> Hardware & Engine
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-2xl bg-zinc-50 border border-zinc-100">
                      <div className="text-[10px] text-zinc-400 font-bold uppercase mb-1">Model Unit</div>
                      <div className="text-sm font-black text-zinc-800">Cosmos Reason 2B</div>
                    </div>
                    <div className="p-4 rounded-2xl bg-zinc-50 border border-zinc-100">
                      <div className="text-[10px] text-zinc-400 font-bold uppercase mb-1">Inference Engine</div>
                      <div className="text-sm font-black text-zinc-800">HF/Transformers</div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.2em] text-blue-600">
                    <Layers className="w-3 h-3" /> Model Parameters
                  </div>
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="text-xs font-bold text-zinc-600 uppercase">Confidence Threshold</label>
                        <span className="text-xs font-black text-blue-600">{config.confidenceThreshold}%</span>
                      </div>
                      <input 
                        type="range" 
                        value={config.confidenceThreshold}
                        onChange={(e) => setConfig({...config, confidenceThreshold: parseInt(e.target.value)})}
                        className="w-full accent-blue-600"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <label className="text-xs font-bold text-zinc-600 uppercase">Sampling Frequency</label>
                        <span className="text-xs font-black text-blue-600">{config.samplingFps} FPS</span>
                      </div>
                      <input 
                        type="range" 
                        min="1" max="8"
                        value={config.samplingFps}
                        onChange={(e) => setConfig({...config, samplingFps: parseInt(e.target.value)})}
                        className="w-full accent-blue-600"
                      />
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="p-6 bg-zinc-50 border-t border-zinc-100 flex justify-end">
                <button 
                  onClick={() => setShowSettings(false)}
                  className="px-6 py-2.5 bg-zinc-900 text-white text-xs font-black uppercase tracking-widest rounded-xl shadow-lg hover:bg-zinc-800 transition-all"
                >
                  Save Configuration
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* Help Modal */}
      <AnimatePresence>
        {showHelp && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              onClick={() => setShowHelp(false)}
              className="absolute inset-0 bg-zinc-900/40 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="relative w-full max-w-lg bg-white rounded-3xl shadow-2xl overflow-hidden border border-zinc-200"
            >
              <div className="p-8 border-b border-zinc-100 flex items-center justify-between bg-blue-600 text-white">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-white/20 rounded-xl">
                    <HelpCircle className="w-5 h-5" />
                  </div>
                  <h2 className="text-xl font-black tracking-tight uppercase">User Guide</h2>
                </div>
                <button onClick={() => setShowHelp(false)} className="p-2 hover:bg-white/10 rounded-full transition-colors">
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="p-8 space-y-6">
                <div className="space-y-3">
                  <h4 className="text-xs font-black uppercase text-zinc-400 tracking-widest">How to use</h4>
                  <ul className="space-y-4">
                    <li className="flex gap-4">
                      <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-[10px] flex items-center justify-center font-black shrink-0">1</div>
                      <div className="text-xs text-zinc-600 font-medium leading-relaxed prose prose-sm max-w-none">
                        <ReactMarkdown>
                          Upload the prepared **Sample Media** (MP4/PNG) or activate the live webcam feed.
                        </ReactMarkdown>
                      </div>
                    </li>
                    <li className="flex gap-4">
                      <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-[10px] flex items-center justify-center font-black shrink-0">2</div>
                      <div className="text-xs text-zinc-600 font-medium leading-relaxed prose prose-sm max-w-none">
                        <ReactMarkdown>
                          Click the **'Run Reason 2'** button to execute the AI expert's logic reasoning process.
                        </ReactMarkdown>
                      </div>
                    </li>
                    <li className="flex gap-4">
                      <div className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 text-[10px] flex items-center justify-center font-black shrink-0">3</div>
                      <div className="text-xs text-zinc-600 font-medium leading-relaxed prose prose-sm max-w-none">
                        <ReactMarkdown>
                          Check the **Reasoning Trace** in the right panel to understand the logical basis for why the AI determined the situation to be hazardous.
                        </ReactMarkdown>
                      </div>
                    </li>
                  </ul>
                </div>

                <div className="p-5 rounded-2xl bg-zinc-50 border border-zinc-100 space-y-3">
                  <div className="flex items-center gap-2 text-[10px] font-black uppercase text-blue-600">
                    <Info className="w-3 h-3" /> Pro Tip
                  </div>
                  <div className="text-[11px] text-zinc-500 font-medium leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown>
                      NVIDIA Cosmos Reason 2 goes beyond simple object recognition to understand **physical properties** such as weight, speed, and trajectory. Actively utilize Chain-of-Thought reasoning for the most critical "preventive measures" in the field.
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
              
              <div className="p-6 bg-zinc-50 border-t border-zinc-100 text-center">
                <p className="text-[9px] font-bold text-zinc-400 uppercase tracking-widest">
                  NVIDIA Cosmos Cookoff Competition • Built by Dooil Kwak
                </p>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
