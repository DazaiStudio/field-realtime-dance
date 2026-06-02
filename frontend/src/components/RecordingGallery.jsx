import React, { useState } from 'react';
import { Play, Download, Trash2, X, Film, Radio } from 'lucide-react';

const RecordingGallery = ({ recordings, onPlay, onReplayOsc, onDelete }) => {
  if (!recordings || recordings.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-8">
        <Film className="w-12 h-12 mb-4 opacity-20" />
        <p className="font-mono text-sm">No recordings found.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="grid grid-cols-1 gap-3">
        {recordings.map((file, idx) => (
          <div 
            key={idx} 
            className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors group"
          >
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center text-blue-400">
                    <Film className="w-5 h-5" />
                </div>
                <div>
                    <p className="text-sm font-medium text-slate-200 truncate max-w-[150px]" title={file}>
                        {file}
                    </p>
                    <p className="text-[10px] text-slate-500 font-mono">MP4 Video</p>
                </div>
            </div>
            
            <div className="flex items-center gap-2">
                <button 
                    onClick={() => onPlay(file)}
                    className="p-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors"
                    title="Play"
                >
                    <Play className="w-4 h-4 fill-current" />
                </button>
                <button
                    onClick={() => onReplayOsc?.(file)}
                    className="p-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white transition-colors"
                    title="Replay OSC"
                >
                    <Radio className="w-4 h-4" />
                </button>
                <a 
                    href={`http://${window.location.hostname}:8000/recordings/${file}`}
                    download
                    className="p-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white transition-colors"
                    title="Download"
                >
                    <Download className="w-4 h-4" />
                </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export const VideoPlayerModal = ({ filename, onClose }) => {
    if (!filename) return null;
    const url = `http://${window.location.hostname}:8000/recordings/${filename}`;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <div className="relative w-full max-w-4xl bg-slate-900 rounded-3xl overflow-hidden shadow-2xl border border-white/10">
                <div className="flex items-center justify-between p-4 border-b border-white/5">
                    <h3 className="text-lg font-medium text-white flex items-center gap-2">
                        <Film className="w-5 h-5 text-blue-400" />
                        Playback: {filename}
                    </h3>
                    <button 
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-white/10 text-slate-400 transition-colors"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>
                
                <div className="aspect-video bg-black">
                    <video 
                        src={url} 
                        controls 
                        autoPlay 
                        className="w-full h-full"
                    />
                </div>
            </div>
        </div>
    );
};

export default RecordingGallery;
