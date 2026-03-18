import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { BoneIcon, UploadIcon, ActivityIcon, BrainIcon, ArrowRightIcon } from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';

const HomePage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: BrainIcon,
      title: 'AI-Powered Detection',
      description: 'Deep learning model trained on bone X-ray images using transfer learning with ResNet50'
    },
    {
      icon: ActivityIcon,
      title: 'Grad-CAM Visualization',
      description: 'Visual explanations showing which regions influenced the fracture prediction'
    },
    {
      icon: UploadIcon,
      title: 'Instant Analysis',
      description: 'Upload X-ray images and receive predictions with confidence scores in seconds'
    }
  ];

  return (
    <div className="min-h-screen bg-[#09090b] text-[#fafafa] overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#3b82f6]/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[#2dd4bf]/5 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <header className="relative border-b border-[#27272a] bg-[#09090b]/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-[#3b82f6]/10 p-2 rounded-md">
                <BoneIcon size={28} className="text-[#3b82f6]" weight="duotone" />
              </div>
              <div>
                <h1 className="text-xl font-heading font-bold tracking-tight" data-testid="app-title">
                  BoneFractureAI
                </h1>
                <p className="text-xs text-[#a1a1aa] font-mono">Medical Imaging Analysis</p>
              </div>
            </div>
            <nav className="flex items-center gap-4">
              <Button
                variant="ghost"
                onClick={() => navigate('/predict')}
                className="text-[#a1a1aa] hover:text-[#fafafa]"
                data-testid="nav-predict"
              >
                Predict
              </Button>
              <Button
                variant="ghost"
                onClick={() => navigate('/train')}
                className="text-[#a1a1aa] hover:text-[#fafafa]"
                data-testid="nav-train"
              >
                Train Model
              </Button>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative container mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto text-center"
        >
          {/* Animated Scanner Effect */}
          <div className="relative inline-block mb-8">
            <div className="relative">
              <BoneIcon size={120} className="text-[#3b82f6]" weight="duotone" />
              <motion.div
                className="absolute inset-0 border-2 border-[#3b82f6]"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 0, 0.5],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
              />
            </div>
          </div>

          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-heading font-bold tracking-tighter mb-6" data-testid="hero-title">
            Detect Bone Fractures
            <br />
            <span className="text-[#3b82f6]">with AI Precision</span>
          </h1>

          <p className="text-lg sm:text-xl text-[#a1a1aa] font-body mb-8 max-w-2xl mx-auto">
            Advanced deep learning system for analyzing X-ray images and identifying bone fractures 
            with visual explanations powered by Grad-CAM technology.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Button
              onClick={() => navigate('/predict')}
              className="bg-[#3b82f6] hover:bg-[#3b82f6]/90 text-white font-heading px-8 py-6 text-lg group"
              data-testid="cta-analyze"
            >
              Start Analysis
              <ArrowRightIcon size={20} className="ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button
              onClick={() => navigate('/train')}
              variant="outline"
              className="border-[#27272a] hover:bg-[#27272a] font-heading px-8 py-6 text-lg"
              data-testid="cta-train"
            >
              Train Model
            </Button>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="grid grid-cols-3 gap-8 max-w-3xl mx-auto mt-20"
        >
          {[
            { label: 'Accuracy', value: '95%', color: '#10b981' },
            { label: 'Inference Time', value: '<2s', color: '#3b82f6' },
            { label: 'Model Type', value: 'ResNet50', color: '#2dd4bf' },
          ].map((stat, idx) => (
            <div key={idx} className="text-center" data-testid={`stat-${idx}`}>
              <div className="text-3xl font-mono font-bold" style={{ color: stat.color }}>
                {stat.value}
              </div>
              <div className="text-sm text-[#a1a1aa] font-body mt-1">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="relative container mx-auto px-6 py-20">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-3xl sm:text-4xl font-heading font-bold text-center mb-12">
            How It Works
          </h2>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                className="bg-[#18181b] border border-[#27272a] rounded-md p-6 hover:border-[#3b82f6]/50 transition-colors"
                data-testid={`feature-${idx}`}
              >
                <div className="bg-[#3b82f6]/10 p-3 rounded-md w-fit mb-4">
                  <feature.icon size={32} className="text-[#3b82f6]" weight="duotone" />
                </div>
                <h3 className="text-xl font-heading font-semibold mb-3">{feature.title}</h3>
                <p className="text-[#a1a1aa] font-body text-sm leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-[#27272a] mt-20">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center text-sm text-[#52525b] font-mono">
            <p>BoneFractureAI © 2026 | Medical AI Decision Support System</p>
            <p className="mt-2">For research and educational purposes</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
