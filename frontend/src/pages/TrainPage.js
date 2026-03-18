import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ActivityIcon, CheckCircleIcon, WarningIcon, TrendUpIcon } from '@phosphor-icons/react';
import axios from 'axios';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { toast } from 'sonner';
import { MetricCard } from '@/components/XrayComponents';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TrainPage = () => {
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 10,
    batch_size: 16,
    learning_rate: 0.0001
  });
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [pollingInterval, setPollingInterval] = useState(null);

  useEffect(() => {
    fetchDatasetInfo();
    fetchModelMetrics();
  }, []);

  useEffect(() => {
    if (trainingStatus?.is_training) {
      const interval = setInterval(() => {
        fetchTrainingStatus();
      }, 2000);
      setPollingInterval(interval);
      return () => clearInterval(interval);
    } else if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  }, [trainingStatus?.is_training]);

  const fetchDatasetInfo = async () => {
    try {
      const response = await axios.get(`${API}/dataset-info`);
      setDatasetInfo(response.data);
    } catch (error) {
      console.error('Error fetching dataset info:', error);
    }
  };

  const fetchModelMetrics = async () => {
    try {
      const response = await axios.get(`${API}/model-metrics`);
      setModelMetrics(response.data);
    } catch (error) {
      console.error('Error fetching model metrics:', error);
    }
  };

  const fetchTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API}/training-status`);
      setTrainingStatus(response.data);
      
      if (response.data.status === 'completed') {
        toast.success('Training completed successfully!');
        fetchModelMetrics();
      } else if (response.data.status.startsWith('error')) {
        toast.error('Training failed: ' + response.data.status);
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  const handleStartTraining = async () => {
    try {
      const response = await axios.post(`${API}/train`, trainingConfig);
      toast.success('Training started!');
      setTimeout(() => fetchTrainingStatus(), 1000);
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Failed to start training');
    }
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#fafafa]">
      {/* Header */}
      <header className="border-b border-[#27272a] bg-[#09090b]/95 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-[#3b82f6]/10 p-2 rounded-md">
                <ActivityIcon size={24} className="text-[#3b82f6]" />
              </div>
              <div>
                <h1 className="text-xl font-heading font-bold tracking-tight" data-testid="train-page-title">
                  Model Training
                </h1>
                <p className="text-xs text-[#a1a1aa] font-mono">Train & Evaluate</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Panel - Training Configuration */}
          <div className="space-y-6">
            {/* Dataset Info */}
            {datasetInfo && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-[#18181b] border border-[#27272a] rounded-md p-6"
              >
                <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">DATASET SUMMARY</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">Training Samples</p>
                    <p className="text-2xl font-mono font-bold text-[#fafafa]" data-testid="train-samples">
                      {datasetInfo.train.total}
                    </p>
                    <div className="text-xs text-[#a1a1aa] mt-1 font-mono">
                      <span className="text-[#10b981]">Normal: {datasetInfo.train.normal}</span>
                      {' | '}
                      <span className="text-[#ef4444]">Fractured: {datasetInfo.train.fractured}</span>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">Validation Samples</p>
                    <p className="text-2xl font-mono font-bold text-[#fafafa]" data-testid="val-samples">
                      {datasetInfo.val.total}
                    </p>
                    <div className="text-xs text-[#a1a1aa] mt-1 font-mono">
                      <span className="text-[#10b981]">Normal: {datasetInfo.val.normal}</span>
                      {' | '}
                      <span className="text-[#ef4444]">Fractured: {datasetInfo.val.fractured}</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Training Configuration */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-[#18181b] border border-[#27272a] rounded-md p-6"
            >
              <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">TRAINING CONFIGURATION</h3>
              <div className="space-y-4">
                <div>
                  <Label className="text-xs text-[#a1a1aa] font-mono">Epochs</Label>
                  <Input
                    type="number"
                    value={trainingConfig.epochs}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                    disabled={trainingStatus?.is_training}
                    className="mt-1 bg-[#09090b] border-[#27272a] font-mono"
                    data-testid="epochs-input"
                  />
                </div>
                <div>
                  <Label className="text-xs text-[#a1a1aa] font-mono">Batch Size</Label>
                  <Input
                    type="number"
                    value={trainingConfig.batch_size}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
                    disabled={trainingStatus?.is_training}
                    className="mt-1 bg-[#09090b] border-[#27272a] font-mono"
                    data-testid="batch-size-input"
                  />
                </div>
                <div>
                  <Label className="text-xs text-[#a1a1aa] font-mono">Learning Rate</Label>
                  <Input
                    type="number"
                    step="0.00001"
                    value={trainingConfig.learning_rate}
                    onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
                    disabled={trainingStatus?.is_training}
                    className="mt-1 bg-[#09090b] border-[#27272a] font-mono"
                    data-testid="learning-rate-input"
                  />
                </div>
              </div>

              <Button
                onClick={handleStartTraining}
                disabled={trainingStatus?.is_training}
                className="w-full mt-6 bg-[#3b82f6] hover:bg-[#3b82f6]/90 text-white font-heading py-6"
                data-testid="start-training-button"
              >
                {trainingStatus?.is_training ? 'Training in Progress...' : 'Start Training'}
              </Button>
            </motion.div>

            {/* Current Model Metrics */}
            {modelMetrics && !modelMetrics.error && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-[#18181b] border border-[#27272a] rounded-md p-6"
              >
                <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">CURRENT MODEL PERFORMANCE</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">Accuracy</p>
                    <p className="text-2xl font-mono font-bold text-[#10b981]" data-testid="model-accuracy">
                      {(modelMetrics.accuracy * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">F1 Score</p>
                    <p className="text-2xl font-mono font-bold text-[#3b82f6]" data-testid="model-f1">
                      {(modelMetrics.f1_score * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">Precision</p>
                    <p className="text-2xl font-mono font-bold text-[#2dd4bf]" data-testid="model-precision">
                      {(modelMetrics.precision * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-[#52525b] mb-1">Recall</p>
                    <p className="text-2xl font-mono font-bold text-[#f59e0b]" data-testid="model-recall">
                      {(modelMetrics.recall * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </div>

          {/* Right Panel - Training Progress */}
          <div className="space-y-6">
            {trainingStatus && trainingStatus.is_training && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="bg-[#18181b] border border-[#27272a] rounded-md p-6"
              >
                <h3 className="text-sm text-[#a1a1aa] font-mono mb-4">TRAINING PROGRESS</h3>
                
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="font-body text-[#fafafa]">Overall Progress</span>
                      <span className="font-mono text-[#3b82f6]" data-testid="progress-percentage">
                        {trainingStatus.progress.toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={trainingStatus.progress} className="h-2" />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-[#52525b] mb-1">Current Epoch</p>
                      <p className="text-2xl font-mono font-bold text-[#fafafa]" data-testid="current-epoch">
                        {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-[#52525b] mb-1">Status</p>
                      <p className="text-sm font-mono text-[#2dd4bf] capitalize" data-testid="training-status">
                        {trainingStatus.status}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-[#52525b] mb-1">Training Loss</p>
                      <p className="text-xl font-mono font-bold text-[#ef4444]" data-testid="train-loss">
                        {trainingStatus.train_loss.toFixed(4)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-[#52525b] mb-1">Val Accuracy</p>
                      <p className="text-xl font-mono font-bold text-[#10b981]" data-testid="val-accuracy">
                        {(trainingStatus.val_accuracy * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  {/* Animated indicator */}
                  <div className="flex items-center gap-2 text-sm text-[#a1a1aa]">
                    <motion.div
                      className="w-2 h-2 bg-[#3b82f6] rounded-full"
                      animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                    <span className="font-mono">Training in progress...</span>
                  </div>
                </div>
              </motion.div>
            )}

            {!trainingStatus?.is_training && (
              <div className="bg-[#18181b] border border-[#27272a] rounded-md p-12 text-center">
                <div className="text-[#52525b]">
                  <ActivityIcon size={64} className="mx-auto mb-4 opacity-50" />
                  <p className="font-mono text-sm">No active training session</p>
                  <p className="font-body text-xs mt-2">Configure parameters and start training</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default TrainPage;
