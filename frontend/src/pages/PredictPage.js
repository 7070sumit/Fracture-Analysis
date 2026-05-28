import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { UploadIcon, XIcon, WarningIcon } from '@phosphor-icons/react';
import axios from 'axios';
import { XrayViewer, ConfidenceGauge, StatusBadge } from '@/components/XrayComponents';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const PredictPage = () => {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setImageFile(file);
      setResult(null);
      
      const reader = new FileReader();
      reader.onload = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false,
    maxSize: 10485760
  });

  const handlePredict = async () => {
    if (!imageFile) {
      toast.error('Please upload an X-ray image first');
      return;
    }

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await axios.post(`${API}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
      toast.success('Analysis complete!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error(error.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClear = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#fafafa]">
      {/* Header */}
      <header className="border-b border-[#27272a] bg-[#09090b]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-[#3b82f6]/10 p-2 rounded-md">
                <UploadIcon size={24} className="text-[#3b82f6]" />
              </div>
              <div>
                <h1 className="text-xl font-heading font-bold tracking-tight" data-testid="page-title">
                  Fracture Risk Prediction
                </h1>
                <p className="text-xs text-[#a1a1aa] font-mono">Multimodal AI Risk Assessment</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Upload & Viewer */}
          <div className="space-y-6">
            {/* Upload Zone */}
            {!imagePreview && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-[#18181b] border-2 border-dashed border-[#27272a] rounded-md p-12 hover:border-[#3b82f6]/50 transition-colors cursor-pointer"
                {...getRootProps()}
                data-testid="upload-dropzone"
              >
                <input {...getInputProps()} />
                <div className="text-center">
                  <UploadIcon size={64} className="mx-auto mb-4 text-[#3b82f6]" weight="duotone" />
                  <h3 className="text-lg font-heading font-semibold mb-2">
                    {isDragActive ? 'Drop X-ray here' : 'Upload X-ray Image'}
                  </h3>
                  <p className="text-sm text-[#a1a1aa] font-body">
                    Drag & drop or click to select
                  </p>
                  <p className="text-xs text-[#52525b] font-mono mt-2">
                    Supports: JPG, PNG (Max 10MB)
                  </p>
                </div>
              </motion.div>
            )}

            {/* X-ray Viewer */}
            {imagePreview && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-4"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm text-[#a1a1aa] font-mono">X-ray Viewer</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleClear}
                    className="text-[#a1a1aa] hover:text-[#fafafa]"
                    data-testid="clear-button"
                  >
                    <XIcon size={16} className="mr-2" />
                    Clear
                  </Button>
                </div>
                <XrayViewer
                  image={imagePreview}
                  gradCamImage={result?.grad_cam_image}
                  isProcessing={isProcessing}
                  className="h-[500px]"
                />
              </motion.div>
            )}

            {/* Action Button */}
            {imagePreview && !result && (
              <Button
                onClick={handlePredict}
                disabled={isProcessing}
                className="w-full bg-[#3b82f6] hover:bg-[#3b82f6]/90 text-white font-heading py-6 text-lg"
                data-testid="analyze-button"
              >
                {isProcessing ? (
                  <>
                    <div className="animate-spin mr-2 h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                    Assessing Risk...
                  </>
                ) : (
                  'Assess Fracture Risk'
                )}
              </Button>
            )}
          </div>

          {/* Right Panel - Results */}
          <div className="space-y-6">
            {result ? (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="space-y-6"
              >
                {/* Status Badge */}
                <div className="bg-[#18181b] border border-[#27272a] rounded-md p-6">
                  <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">RISK ASSESSMENT</h3>
                  <div className="flex justify-center mb-6">
                    <StatusBadge prediction={result.prediction} confidence={result.confidence} />
                  </div>
                  <ConfidenceGauge confidence={result.confidence} prediction={result.prediction} />
                </div>

                {/* Probabilities */}
                <div className="bg-[#18181b] border border-[#27272a] rounded-md p-6">
                  <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">RISK PROBABILITY BREAKDOWN</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="font-body text-[#fafafa]">Low Risk (Normal)</span>
                        <span className="font-mono text-[#10b981]" data-testid="normal-probability">
                          {(result.probabilities[0] * 100).toFixed(2)}%
                        </span>
                      </div>
                      <Progress value={result.probabilities[0] * 100} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="font-body text-[#fafafa]">High Risk (Fracture)</span>
                        <span className="font-mono text-[#ef4444]" data-testid="fracture-probability">
                          {(result.probabilities[1] * 100).toFixed(2)}%
                        </span>
                      </div>
                      <Progress value={result.probabilities[1] * 100} className="h-2 bg-[#ef4444]/20" />
                    </div>
                  </div>
                </div>

                {/* Interpretation */}
                <div className="bg-[#18181b] border border-[#27272a] rounded-md p-6">
                  <div className="flex items-start gap-3">
                    <WarningIcon size={20} className="text-[#f59e0b] mt-1" weight="duotone" />
                    <div className="w-full">
                      <h3 className="text-sm font-semibold text-[#fafafa] mb-2">Clinical Note</h3>
                      <p className="text-sm text-[#a1a1aa] font-body leading-relaxed">
                        This AI-assisted diagnosis is for reference only. The Grad-CAM heatmap highlights visual regions 
                        that influenced the prediction, while the SHAP chart quantifies the mathematical impact of the patient's clinical history. Always consult with a qualified radiologist for final diagnosis.
                      </p>
                      
                      {result.shap_image && (
                        <div className="mt-6 pt-4 border-t border-[#27272a]">
                          <h4 className="text-xs font-semibold text-[#fafafa] mb-3 uppercase tracking-wider font-mono">Clinical Feature Impact (SHAP)</h4>
                          <img 
                            src={result.shap_image} 
                            alt="SHAP Clinical Explainability" 
                            className="w-full rounded-md border border-[#27272a] bg-white/5"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* New Analysis Button */}
                <Button
                  onClick={handleClear}
                  variant="outline"
                  className="w-full border-[#27272a] hover:bg-[#27272a] font-heading"
                  data-testid="new-analysis-button"
                >
                  New Analysis
                </Button>
              </motion.div>
            ) : (
              <div className="bg-[#18181b] border border-[#27272a] rounded-md p-12 text-center">
                <div className="text-[#52525b]">
                  <p className="font-mono text-sm">Upload an X-ray image to begin analysis</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default PredictPage;
