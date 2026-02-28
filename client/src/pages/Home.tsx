import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { predictionRequestSchema, trainRequestSchema, type PredictionRequest, type TrainRequest } from "@shared/schema";
import { api } from "@shared/routes";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Brain, Home as HomeIcon, Code, Settings2, Activity, Calculator, ArrowRight, Layers, Zap, Target, TrendingDown, Database, Cpu, Languages } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";

const CODE_SNIPPET = `class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # 1. Input Layer: 3 features (sqft, age, rooms)
        # 2. Hidden Layer 1: 64 neurons + ReLU
        # 3. Hidden Layer 2: 32 neurons + ReLU
        # 4. Output Layer: 1 neuron (price prediction)
        return self.network(x)

# Why ReLU?
# Linear operations alone (W1x + b1) only result in linear transformations.
# ReLU (Rectified Linear Unit) introduces non-linearity, allowing the
# network to learn complex patterns like diminishing returns on square footage.

# Optimizer: Adam (Adaptive Moment Estimation)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()`;

const t = {
  zh: {
    headerTitle: "深度神經網路架構",
    headerSub: "House Price Intelligence - Deep Learning Model",
    tabPredict: "Predict",
    tabTrain: "Training Setup",
    tabPerformance: "Performance",
    tabModel: "Model Summary",
    // Predict tab
    modelInputs: "模型參數 (Model Inputs)",
    modelInputsDesc: "輸入房屋特徵：坪數、屋齡、房間數",
    sqft: "坪數 (Square Footage)",
    age: "屋齡 (House Age)",
    rooms: "房間數 (Rooms)",
    predicting: "預測中...",
    predict: "開始預測 (Predict)",
    resultLabel: "預測連續數值 (價格)",
    resultSuccess: "深度神經網路模型已定義，預測成功",
    resultPlaceholder: "請輸入參數以進行估價",
    // Train tab
    trainTitle: "訓練模擬 (Training Setup)",
    trainDesc: "調整數據生成參數與訓練輪數",
    samples: "數據樣本數 (Samples)",
    noise: "雜訊程度 (Noise)",
    epochs: "訓練輪數 (Epochs)",
    training: "模型訓練中 (Training...)",
    retrain: "重新訓練模型 (Retrain Model)",
    // Performance tab
    lossCurveTitle: "損失函數收斂 (Loss Curve)",
    lossCurveDesc: "MSE 隨訓練輪數下降的情況 — 曲線越低代表模型預測越準確",
    lossFunction: "損失函數 (Loss Function)",
    lossFunctionDesc: "Mean Squared Error — 計算預測值與實際值之差的平方平均，對較大誤差施以更重的懲罰",
    optimizer: "優化器 (Optimizer)",
    optimizerDesc: "Adaptive Moment Estimation — 自動為每個參數調整學習率，結合動量與自適應學習率的優點",
    batchSize: "批次大小 (Batch Size)",
    batchSizeDesc: "每次更新權重時使用 32 筆資料，平衡訓練速度與梯度估計的穩定性",
    trainingPipeline: "訓練流程 (Training Pipeline)",
    pipelineSteps: [
      { step: "1", title: "資料生成", desc: "根據公式產生合成房價資料，加入高斯雜訊模擬真實世界的不確定性" },
      { step: "2", title: "特徵正規化", desc: "sqft ÷ 1000、age ÷ 10、price ÷ 100,000 — 將數值縮放至相近範圍，加速收斂" },
      { step: "3", title: "前向傳播", desc: "輸入經過 3 層隱藏層（64→32→16 神經元），每層後接 ReLU 激活函數" },
      { step: "4", title: "損失計算", desc: "用 MSE 衡量預測價格與真實價格的差距" },
      { step: "5", title: "反向傳播", desc: "計算每個權重對損失的梯度（偏導數），用鏈式法則逐層回傳" },
      { step: "6", title: "參數更新", desc: "Adam 優化器依據梯度更新 2,881 個可訓練參數" },
    ],
    reluTitle: "為什麼需要 ReLU 激活函數？",
    reluSubtitle: "非線性是深度學習的核心",
    reluExplain: "如果只使用線性運算（加法和乘法），無論堆疊多少層，最終結果仍等同於一個單層線性變換：",
    reluConclusion: "ReLU 引入非線性，讓網絡能夠學習複雜的非線性關係，例如房價與坪數之間可能存在的邊際效益遞減。",
    reluComment: "// 正值保留，負值歸零 — 簡單卻有效的非線性轉換",
    // Model Summary tab
    overviewTitle: "模型概述 (What This Model Does)",
    overviewBody: "這是一個迴歸型深度神經網路 (Deep Neural Network for Regression)，目標是根據三個房屋特徵預測房價。模型透過學習合成資料中的非線性模式，將輸入特徵映射到連續的價格數值。",
    formulaTitle: "資料生成公式 (Pricing Formula)",
    formulaDesc: "模型訓練所用的合成資料，根據以下公式產生",
    inputRanges: "輸入範圍 (Input Ranges)",
    normalization: "正規化方式 (Normalization)",
    formulaNote: "加入高斯雜訊 (σ = base_price × noise_level) 模擬真實世界的變異。最低價格為 $10,000 以防止負值。",
    archTitle: "網路架構 (Network Architecture)",
    archDesc: "從輸入到輸出的完整資料流",
    depthTitle: "深度 (Depth)",
    depthDesc: "3 層隱藏層讓模型能逐層提取越來越抽象的特徵 — 第一層可能學會「大房子較貴」，後續層則能捕捉特徵間的交互作用。",
    activationTitle: "激活函數 (Activation)",
    activationDesc: "ReLU(x) = max(0, x) 引入非線性，使多層網路不會退化為單層線性模型。輸出層不加激活函數，讓價格預測可為任意正值。",
    bottleneckTitle: "瓶頸結構 (Bottleneck)",
    bottleneckDesc: "神經元數遞減（64→32→16→1），迫使模型壓縮資訊、提取最重要的特徵，類似自動編碼器的編碼過程。",
  },
  en: {
    headerTitle: "Deep Neural Network Architecture",
    headerSub: "House Price Intelligence - Deep Learning Model",
    tabPredict: "Predict",
    tabTrain: "Training Setup",
    tabPerformance: "Performance",
    tabModel: "Model Summary",
    // Predict tab
    modelInputs: "Model Inputs",
    modelInputsDesc: "Enter house features: square footage, age, and number of rooms",
    sqft: "Square Footage",
    age: "House Age (years)",
    rooms: "Number of Rooms",
    predicting: "Predicting...",
    predict: "Predict Price",
    resultLabel: "Predicted Price",
    resultSuccess: "Deep neural network prediction successful",
    resultPlaceholder: "Enter parameters to get a price estimate",
    // Train tab
    trainTitle: "Training Setup",
    trainDesc: "Adjust data generation parameters and training epochs",
    samples: "Data Samples",
    noise: "Noise Level",
    epochs: "Training Epochs",
    training: "Training model...",
    retrain: "Retrain Model",
    // Performance tab
    lossCurveTitle: "Loss Convergence",
    lossCurveDesc: "MSE decreasing over training epochs — lower curve means more accurate predictions",
    lossFunction: "Loss Function",
    lossFunctionDesc: "Mean Squared Error — averages the squared difference between predicted and actual values, penalizing larger errors more heavily",
    optimizer: "Optimizer",
    optimizerDesc: "Adaptive Moment Estimation — automatically adjusts learning rate per parameter, combining momentum with adaptive learning rates",
    batchSize: "Batch Size",
    batchSizeDesc: "Uses 32 samples per weight update, balancing training speed with gradient estimation stability",
    trainingPipeline: "Training Pipeline",
    pipelineSteps: [
      { step: "1", title: "Data Generation", desc: "Generate synthetic house price data using the pricing formula, adding Gaussian noise to simulate real-world uncertainty" },
      { step: "2", title: "Feature Normalization", desc: "sqft ÷ 1000, age ÷ 10, price ÷ 100,000 — scale values to similar ranges for faster convergence" },
      { step: "3", title: "Forward Pass", desc: "Input flows through 3 hidden layers (64→32→16 neurons), each followed by ReLU activation" },
      { step: "4", title: "Loss Calculation", desc: "MSE measures the gap between predicted and actual house prices" },
      { step: "5", title: "Backpropagation", desc: "Compute gradients (partial derivatives) for each weight using the chain rule, propagating layer by layer" },
      { step: "6", title: "Parameter Update", desc: "Adam optimizer updates all 2,881 trainable parameters based on computed gradients" },
    ],
    reluTitle: "Why Do We Need ReLU Activation?",
    reluSubtitle: "Non-linearity is the core of deep learning",
    reluExplain: "Using only linear operations (addition and multiplication), no matter how many layers are stacked, the result is equivalent to a single linear transformation:",
    reluConclusion: "ReLU introduces non-linearity, enabling the network to learn complex relationships such as diminishing returns on square footage.",
    reluComment: "// Keeps positive values, zeroes out negatives — simple yet effective non-linearity",
    // Model Summary tab
    overviewTitle: "What This Model Does",
    overviewBody: "This is a regression-based Deep Neural Network that predicts house prices from three input features. It learns non-linear patterns from synthetic data to map input features to a continuous price value.",
    formulaTitle: "Pricing Formula",
    formulaDesc: "Synthetic training data is generated using the following formula",
    inputRanges: "Input Ranges",
    normalization: "Normalization",
    formulaNote: "Gaussian noise (σ = base_price × noise_level) is added to simulate real-world variance. A price floor of $10,000 prevents negative values.",
    archTitle: "Network Architecture",
    archDesc: "Complete data flow from input to output",
    depthTitle: "Depth",
    depthDesc: "3 hidden layers allow the model to extract progressively more abstract features — the first layer might learn \"bigger houses cost more\", while deeper layers capture feature interactions.",
    activationTitle: "Activation",
    activationDesc: "ReLU(x) = max(0, x) introduces non-linearity, preventing multi-layer networks from collapsing into a single linear model. No activation on the output layer allows unbounded price predictions.",
    bottleneckTitle: "Bottleneck",
    bottleneckDesc: "Decreasing neuron counts (64→32→16→1) force the model to compress information and extract the most important features, similar to an autoencoder's encoding process.",
  },
};

type Lang = "zh" | "en";

export default function Home() {
  const { toast } = useToast();
  const [prediction, setPrediction] = useState<number | null>(null);
  const [lossCurve, setLossCurve] = useState<string>("/loss_curve.png");
  const [lang, setLang] = useState<Lang>("zh");
  const l = t[lang];

  const predictForm = useForm<PredictionRequest>({
    resolver: zodResolver(predictionRequestSchema),
    defaultValues: { sqft: 2000, age: 10, rooms: 3 }
  });

  const trainForm = useForm<TrainRequest>({
    resolver: zodResolver(trainRequestSchema),
    defaultValues: { samples: 5000, noise: 0.1, epochs: 50 }
  });

  const predictMutation = useMutation({
    mutationFn: async (data: PredictionRequest) => {
      const res = await apiRequest("POST", api.predict.path, data);
      return res.json();
    },
    onSuccess: (data) => setPrediction(data.predictedPrice),
    onError: () => toast({ variant: "destructive", title: "Prediction failed" })
  });

  const trainMutation = useMutation({
    mutationFn: async (data: TrainRequest) => {
      const res = await apiRequest("POST", api.train.path, data);
      return res.json();
    },
    onSuccess: (data) => {
      const newUrl = `/loss_curve.png?t=${new Date().getTime()}`;
      setLossCurve(newUrl);
      toast({ title: "Model retrained!", description: data.message });
    },
    onError: () => toast({ variant: "destructive", title: "Training failed" })
  });

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 lg:p-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <header className="relative text-center space-y-2">
          {/* Language Toggle */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setLang(lang === "zh" ? "en" : "zh")}
            className="absolute top-0 right-0 gap-2 rounded-full"
          >
            <Languages className="w-4 h-4" />
            {lang === "zh" ? "EN" : "中文"}
          </Button>

          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold tracking-tight sm:text-5xl flex items-center justify-center gap-3"
          >
            <Brain className="w-10 h-10 text-primary" /> {l.headerTitle}
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground text-lg"
          >
            {l.headerSub}
          </motion.p>
        </header>

        <Tabs defaultValue="predict" className="w-full">
          <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 mb-8">
            <TabsTrigger value="predict" className="gap-2"><HomeIcon className="w-4 h-4" /> {l.tabPredict}</TabsTrigger>
            <TabsTrigger value="train" className="gap-2"><Settings2 className="w-4 h-4" /> {l.tabTrain}</TabsTrigger>
            <TabsTrigger value="analytics" className="gap-2"><Activity className="w-4 h-4" /> {l.tabPerformance}</TabsTrigger>
            <TabsTrigger value="code" className="gap-2"><Code className="w-4 h-4" /> {l.tabModel}</TabsTrigger>
          </TabsList>

          <TabsContent value="predict">
            <div className="grid md:grid-cols-2 gap-8">
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle>{l.modelInputs}</CardTitle>
                  <CardDescription>{l.modelInputsDesc}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Form {...predictForm}>
                    <form onSubmit={predictForm.handleSubmit((data) => predictMutation.mutate(data))} className="space-y-4">
                      <FormField control={predictForm.control} name="sqft" render={({ field }) => (
                        <FormItem>
                          <FormLabel>{l.sqft}</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="age" render={({ field }) => (
                        <FormItem>
                          <FormLabel>{l.age}</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="rooms" render={({ field }) => (
                        <FormItem>
                          <FormLabel>{l.rooms}</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <Button type="submit" className="w-full h-12 rounded-xl shadow-lg shadow-primary/20" disabled={predictMutation.isPending}>
                        {predictMutation.isPending ? l.predicting : l.predict}
                      </Button>
                    </form>
                  </Form>
                </CardContent>
              </Card>

              <Card className="flex flex-col items-center justify-center p-8 bg-primary/5 border-dashed border-2 rounded-3xl min-h-[300px]">
                <AnimatePresence mode="wait">
                  {prediction !== null ? (
                    <motion.div
                      key="result"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="text-center space-y-4"
                    >
                      <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">{l.resultLabel}</p>
                      <div className="text-5xl sm:text-6xl font-bold text-primary tracking-tighter">
                        {formatCurrency(prediction)}
                      </div>
                      <div className="bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        {l.resultSuccess}
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="placeholder"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center text-muted-foreground"
                    >
                      <Calculator className="w-16 h-16 mx-auto mb-4 opacity-10" />
                      <p className="font-medium">{l.resultPlaceholder}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="train">
            <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl">
              <CardHeader>
                <CardTitle>{l.trainTitle}</CardTitle>
                <CardDescription>{l.trainDesc}</CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...trainForm}>
                  <form onSubmit={trainForm.handleSubmit((data) => trainMutation.mutate(data))} className="space-y-8">
                    <FormField control={trainForm.control} name="samples" render={({ field }) => (
                      <FormItem>
                        <div className="flex justify-between items-center mb-2">
                          <FormLabel className="text-base">{l.samples}</FormLabel>
                          <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider min={100} max={10000} step={100} value={[field.value]} onValueChange={(v) => field.onChange(v[0])} />
                        </FormControl>
                      </FormItem>
                    )} />
                    <FormField control={trainForm.control} name="noise" render={({ field }) => (
                      <FormItem>
                        <div className="flex justify-between items-center mb-2">
                          <FormLabel className="text-base">{l.noise}</FormLabel>
                          <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">{(field.value * 100).toFixed(0)}%</span>
                        </div>
                        <FormControl>
                          <Slider min={0} max={1} step={0.05} value={[field.value]} onValueChange={(v) => field.onChange(v[0])} />
                        </FormControl>
                      </FormItem>
                    )} />
                    <FormField control={trainForm.control} name="epochs" render={({ field }) => (
                      <FormItem>
                        <div className="flex justify-between items-center mb-2">
                          <FormLabel className="text-base">{l.epochs}</FormLabel>
                          <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider min={10} max={200} step={10} value={[field.value]} onValueChange={(v) => field.onChange(v[0])} />
                        </FormControl>
                      </FormItem>
                    )} />
                    <Button type="submit" variant="secondary" className="w-full h-12 rounded-xl" disabled={trainMutation.isPending}>
                      {trainMutation.isPending ? l.training : l.retrain}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics">
            <div className="space-y-8">
              {/* Loss Curve */}
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><TrendingDown className="w-5 h-5 text-primary" /> {l.lossCurveTitle}</CardTitle>
                  <CardDescription>{l.lossCurveDesc}</CardDescription>
                </CardHeader>
                <CardContent className="bg-white p-4">
                  <img
                    key={lossCurve}
                    src={lossCurve}
                    alt="Loss Curve"
                    className="w-full h-auto rounded-xl border border-border/20"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = "none";
                    }}
                  />
                </CardContent>
              </Card>

              {/* Performance Summary */}
              <div className="grid md:grid-cols-3 gap-6">
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 text-center space-y-2">
                    <Target className="w-8 h-8 text-blue-500 mx-auto" />
                    <p className="text-sm font-medium text-muted-foreground">{l.lossFunction}</p>
                    <p className="text-xl font-bold">MSE</p>
                    <p className="text-xs text-muted-foreground">{l.lossFunctionDesc}</p>
                  </CardContent>
                </Card>
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 text-center space-y-2">
                    <Zap className="w-8 h-8 text-amber-500 mx-auto" />
                    <p className="text-sm font-medium text-muted-foreground">{l.optimizer}</p>
                    <p className="text-xl font-bold">Adam</p>
                    <p className="text-xs text-muted-foreground">{l.optimizerDesc}</p>
                  </CardContent>
                </Card>
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 text-center space-y-2">
                    <Database className="w-8 h-8 text-green-500 mx-auto" />
                    <p className="text-sm font-medium text-muted-foreground">{l.batchSize}</p>
                    <p className="text-xl font-bold">32</p>
                    <p className="text-xs text-muted-foreground">{l.batchSizeDesc}</p>
                  </CardContent>
                </Card>
              </div>

              {/* Training Details */}
              <div className="grid lg:grid-cols-2 gap-8">
                <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                  <CardHeader>
                    <CardTitle>{l.trainingPipeline}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {l.pipelineSteps.map((item) => (
                      <div key={item.step} className="flex gap-3 items-start">
                        <span className="flex-shrink-0 w-7 h-7 rounded-full bg-primary/10 text-primary text-sm font-bold flex items-center justify-center">{item.step}</span>
                        <div>
                          <p className="text-sm font-semibold">{item.title}</p>
                          <p className="text-xs text-muted-foreground">{item.desc}</p>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                  <CardHeader>
                    <CardTitle>{l.reluTitle}</CardTitle>
                    <CardDescription>{l.reluSubtitle}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      {l.reluExplain}
                    </p>
                    <div className="bg-muted p-4 rounded-xl font-mono text-xs">
                      f(x) = W2(W1·x + b1) + b2<br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (W2·W1)·x + (W2·b1 + b2)<br/>
                      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= W'·x + b'
                    </div>
                    <p className="text-sm font-medium">
                      {l.reluConclusion}
                    </p>
                    <div className="bg-muted p-4 rounded-xl font-mono text-xs">
                      ReLU(x) = max(0, x)<br/>
                      <span className="text-muted-foreground">{l.reluComment}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="code">
            <div className="space-y-8">
              {/* What This Model Does */}
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Brain className="w-5 h-5 text-primary" /> {l.overviewTitle}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {l.overviewBody}
                  </p>
                </CardContent>
              </Card>

              {/* Data & Pricing Formula */}
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Database className="w-5 h-5 text-green-500" /> {l.formulaTitle}</CardTitle>
                  <CardDescription>{l.formulaDesc}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="bg-muted p-4 rounded-xl font-mono text-sm">
                    price = $50,000 + (sqft × $150) − (age × $500) + (rooms × $10,000) + noise
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4 text-sm">
                    <div className="space-y-2">
                      <p className="font-semibold">{l.inputRanges}</p>
                      <ul className="text-muted-foreground space-y-1 text-xs">
                        <li>sqft: 500 – 5,000 sq ft</li>
                        <li>age: 0 – 100 years</li>
                        <li>rooms: 1 – 10</li>
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <p className="font-semibold">{l.normalization}</p>
                      <ul className="text-muted-foreground space-y-1 text-xs">
                        <li>sqft ÷ 1,000 → ~[0.5, 5.0]</li>
                        <li>age ÷ 10 → ~[0, 10]</li>
                        <li>price ÷ 100,000 → ~[0.1, 10]</li>
                      </ul>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {l.formulaNote}
                  </p>
                </CardContent>
              </Card>

              {/* Architecture Diagram */}
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Layers className="w-5 h-5 text-violet-500" /> {l.archTitle}</CardTitle>
                  <CardDescription>{l.archDesc}</CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Visual flow */}
                  <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 mb-6">
                    {[
                      { label: "Input", sub: "3 features", color: "bg-blue-100 text-blue-700 border-blue-200" },
                      { label: "Linear", sub: "3→64", color: "bg-violet-100 text-violet-700 border-violet-200" },
                      { label: "ReLU", sub: "", color: "bg-amber-100 text-amber-700 border-amber-200" },
                      { label: "Linear", sub: "64→32", color: "bg-violet-100 text-violet-700 border-violet-200" },
                      { label: "ReLU", sub: "", color: "bg-amber-100 text-amber-700 border-amber-200" },
                      { label: "Linear", sub: "32→16", color: "bg-violet-100 text-violet-700 border-violet-200" },
                      { label: "ReLU", sub: "", color: "bg-amber-100 text-amber-700 border-amber-200" },
                      { label: "Linear", sub: "16→1", color: "bg-green-100 text-green-700 border-green-200" },
                    ].map((block, i, arr) => (
                      <React.Fragment key={i}>
                        <div className={`px-3 py-2 rounded-xl border text-xs font-semibold text-center ${block.color}`}>
                          <div>{block.label}</div>
                          {block.sub && <div className="text-[10px] font-normal opacity-70">{block.sub}</div>}
                        </div>
                        {i < arr.length - 1 && <ArrowRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />}
                      </React.Fragment>
                    ))}
                  </div>

                  {/* Layer detail table */}
                  <div className="bg-slate-950 rounded-xl p-4 font-mono text-sm text-blue-200 space-y-1">
                    <div className="text-slate-400 text-xs mb-2">HousePriceNN(</div>
                    <div className="pl-4">(layer1): Linear(in_features=3, out_features=64)   <span className="text-slate-500">  # 3×64 + 64 = 256 params</span></div>
                    <div className="pl-4">(relu1):  ReLU()</div>
                    <div className="pl-4">(layer2): Linear(in_features=64, out_features=32)  <span className="text-slate-500"> # 64×32 + 32 = 2,080 params</span></div>
                    <div className="pl-4">(relu2):  ReLU()</div>
                    <div className="pl-4">(layer3): Linear(in_features=32, out_features=16)  <span className="text-slate-500"> # 32×16 + 16 = 528 params</span></div>
                    <div className="pl-4">(relu3):  ReLU()</div>
                    <div className="pl-4">(output): Linear(in_features=16, out_features=1)   <span className="text-slate-500"> # 16×1 + 1 = 17 params</span></div>
                    <div className="text-slate-400 text-xs">)</div>
                    <div className="pt-3 border-t border-slate-800 flex flex-wrap gap-x-8 gap-y-1 text-slate-400 text-xs">
                      <div>Total params: <span className="text-white font-semibold">2,881</span></div>
                      <div>Trainable: <span className="text-white font-semibold">2,881</span></div>
                      <div>Model size: <span className="text-white font-semibold">~11 KB</span></div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Key Concepts */}
              <div className="grid md:grid-cols-3 gap-6">
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 space-y-2">
                    <Cpu className="w-7 h-7 text-violet-500" />
                    <p className="font-semibold text-sm">{l.depthTitle}</p>
                    <p className="text-xs text-muted-foreground">{l.depthDesc}</p>
                  </CardContent>
                </Card>
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 space-y-2">
                    <Zap className="w-7 h-7 text-amber-500" />
                    <p className="font-semibold text-sm">{l.activationTitle}</p>
                    <p className="text-xs text-muted-foreground">{l.activationDesc}</p>
                  </CardContent>
                </Card>
                <Card className="border-border/50 shadow-lg rounded-3xl">
                  <CardContent className="pt-6 space-y-2">
                    <Target className="w-7 h-7 text-blue-500" />
                    <p className="font-semibold text-sm">{l.bottleneckTitle}</p>
                    <p className="text-xs text-muted-foreground">{l.bottleneckDesc}</p>
                  </CardContent>
                </Card>
              </div>

              {/* PyTorch Code */}
              <Card className="bg-slate-950 text-slate-50 border-none rounded-3xl overflow-hidden shadow-2xl">
                <CardHeader className="border-b border-slate-800/50">
                  <CardTitle className="text-slate-50 flex items-center gap-2"><Code className="w-5 h-5 text-blue-400" /> PyTorch Implementation</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <pre className="text-xs sm:text-sm font-mono overflow-x-auto p-6 bg-slate-950/50 leading-relaxed">
                    <code className="text-blue-200">{CODE_SNIPPET}</code>
                  </pre>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
