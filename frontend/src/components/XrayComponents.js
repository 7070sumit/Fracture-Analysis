import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadIcon, ActivityIcon, BoneIcon, CheckCircleIcon, WarningIcon, XIcon } from '@phosphor-icons/react';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { cn } from '@/lib/utils';

export const XrayViewer = ({ 
  image, 
  gradCamImage, 
  className,
  onImageLoad,
  isProcessing = false
}) => {
  const [zoom, setZoom] = useState(1);
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [showGradCam, setShowGradCam] = useState(true);
  const containerRef = useRef(null);

  return (
    <div className={cn("relative bg-[#18181b] rounded-md border border-[#27272a] overflow-hidden", className)}>
      {/* Tech Corners */}
      <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-[#3b82f6]/50 z-10" />
      <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-[#3b82f6]/50 z-10" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-[#3b82f6]/50 z-10" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-[#3b82f6]/50 z-10" />

      {/* Image Container */}
      <div 
        ref={containerRef}
        className="relative w-full h-full min-h-[400px] flex items-center justify-center p-8"
        data-testid="xray-viewer-container"
      >
        {!image && !isProcessing && (
          <div className="text-center text-[#52525b]">
            <UploadIcon size={64} className="mx-auto mb-4 opacity-50" />
            <p className="font-mono text-sm">No X-ray image loaded</p>
          </div>
        )}

        {image && (
          <>
            <img
              src={image}
              alt="X-ray"
              className="relative z-0 max-w-full max-h-full object-contain"
              style={{ transform: `scale(${zoom})` }}
              onLoad={onImageLoad}
              data-testid="xray-image"
            />
            
            {/* Scanline effect */}
            <div className="absolute inset-0 scanlines pointer-events-none z-20" />
            
            {/* Grad-CAM Overlay */}
            {gradCamImage && showGradCam && (
              <motion.img
                initial={{ opacity: 0 }}
                animate={{ opacity: overlayOpacity }}
                src={gradCamImage}
                alt="Grad-CAM"
                className="absolute inset-0 m-auto max-w-full max-h-full object-contain z-10"
                style={{ transform: `scale(${zoom})`, mixBlendMode: 'screen' }}
                data-testid="gradcam-overlay"
              />
            )}
            
            {/* Processing Scan Line */}
            {isProcessing && (
              <motion.div
                className="absolute left-0 right-0 h-0.5 bg-[#3b82f6] shadow-lg shadow-[#3b82f6]/50 z-30"
                animate={{
                  top: ['0%', '100%', '0%'],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              />
            )}
          </>
        )}
      </div>

      {/* Controls */}
      {image && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 glass-panel px-4 py-2 rounded-full flex items-center gap-4 z-30">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#a1a1aa] font-mono">Zoom</span>
            <Slider
              value={[zoom]}
              onValueChange={(v) => setZoom(v[0])}
              min={1}
              max={3}
              step={0.1}
              className="w-24"
              data-testid="zoom-slider"
            />
            <span className="text-xs text-[#fafafa] font-mono w-12">{zoom.toFixed(1)}x</span>
          </div>
          
          {gradCamImage && (
            <>
              <div className="w-px h-6 bg-white/10" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-[#a1a1aa] font-mono">Heatmap</span>
                <Slider
                  value={[overlayOpacity]}
                  onValueChange={(v) => setOverlayOpacity(v[0])}
                  min={0}
                  max={1}
                  step={0.1}
                  className="w-24"
                  data-testid="heatmap-opacity-slider"
                />
                <Button
                  size="sm"
                  variant={showGradCam ? "default" : "outline"}
                  onClick={() => setShowGradCam(!showGradCam)}
                  className="h-6 px-2 text-xs"
                  data-testid="toggle-heatmap-button"
                >
                  {showGradCam ? 'ON' : 'OFF'}
                </Button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export const ConfidenceGauge = ({ confidence, prediction }) => {
  const isFracture = prediction === 'Fracture';
  const angle = (confidence * 180) - 90;
  
  return (
    <div className="relative w-48 h-24 mx-auto" data-testid="confidence-gauge">
      {/* Gauge Background */}
      <svg className="w-full h-full" viewBox="0 0 200 100">
        <path
          d="M 10 90 A 80 80 0 0 1 190 90"
          fill="none"
          stroke="#27272a"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d="M 10 90 A 80 80 0 0 1 190 90"
          fill="none"
          stroke={isFracture ? '#ef4444' : '#10b981'}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${confidence * 251.2} 251.2`}
          className="transition-all duration-1000"
        />
      </svg>
      
      {/* Needle */}
      <motion.div
        className="absolute bottom-2 left-1/2 w-1 h-16 bg-white origin-bottom"
        style={{ transformOrigin: 'bottom center' }}
        initial={{ rotate: -90 }}
        animate={{ rotate: angle }}
        transition={{ duration: 1, ease: 'easeOut' }}
      />
      
      {/* Center dot */}
      <div className="absolute bottom-2 left-1/2 w-3 h-3 bg-white rounded-full transform -translate-x-1/2" />
      
      {/* Percentage */}
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-6">
        <span className="text-2xl font-mono font-bold text-[#fafafa]" data-testid="confidence-percentage">
          {(confidence * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
};

export const StatusBadge = ({ prediction, confidence }) => {
  const isFracture = prediction === 'Fracture';
  
  return (
    <div 
      className={cn(
        "inline-flex items-center gap-2 px-4 py-2 rounded-full border",
        isFracture 
          ? "bg-red-500/10 border-red-500/30 text-red-400" 
          : "bg-green-500/10 border-green-500/30 text-green-400"
      )}
      data-testid="status-badge"
    >
      <motion.div
        className={cn(
          "w-2 h-2 rounded-full",
          isFracture ? "bg-red-500" : "bg-green-500"
        )}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
      <span className="font-mono text-sm font-medium" data-testid="status-text">
        {prediction}
      </span>
    </div>
  );
};

export const MetricCard = ({ label, value, icon: Icon, trend }) => {
  return (
    <div className="bg-[#18181b] border border-[#27272a] rounded-md p-6" data-testid="metric-card">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[#a1a1aa] text-sm font-body">{label}</p>
          <p className="text-3xl font-bold font-heading text-[#fafafa] mt-2" data-testid="metric-value">
            {value}
          </p>
          {trend && (
            <p className="text-[#2dd4bf] text-xs font-mono mt-1" data-testid="metric-trend">
              {trend}
            </p>
          )}
        </div>
        {Icon && (
          <div className="bg-[#3b82f6]/10 p-3 rounded-md">
            <Icon size={24} className="text-[#3b82f6]" />
          </div>
        )}
      </div>
    </div>
  );
};
