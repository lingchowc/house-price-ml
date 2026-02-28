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
import { Brain, Home as HomeIcon, Code, Settings2, Activity, Calculator } from "lucide-react";
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

export default function Home() {
  const { toast } = useToast();
  const [prediction, setPrediction] = useState<number | null>(null);
  const [lossCurve, setLossCurve] = useState<string>("/loss_curve.png");

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
      // Force refresh the image by appending a unique timestamp
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
        <header className="text-center space-y-2">
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold tracking-tight sm:text-5xl flex items-center justify-center gap-3"
          >
            <Brain className="w-10 h-10 text-primary" /> 深度神經網路架構
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground text-lg"
          >
            House Price Intelligence - Deep Learning Model
          </motion.p>
        </header>

        <Tabs defaultValue="predict" className="w-full">
          <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 mb-8">
            <TabsTrigger value="predict" className="gap-2"><HomeIcon className="w-4 h-4" /> Predict</TabsTrigger>
            <TabsTrigger value="train" className="gap-2"><Settings2 className="w-4 h-4" /> Training Setup</TabsTrigger>
            <TabsTrigger value="analytics" className="gap-2"><Activity className="w-4 h-4" /> Performance</TabsTrigger>
            <TabsTrigger value="code" className="gap-2"><Code className="w-4 h-4" /> Model Summary</TabsTrigger>
          </TabsList>

          <TabsContent value="predict">
            <div className="grid md:grid-cols-2 gap-8">
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle>模型參數 (Model Inputs)</CardTitle>
                  <CardDescription>輸入房屋特徵：坪數、屋齡、房間數</CardDescription>
                </CardHeader>
                <CardContent>
                  <Form {...predictForm}>
                    <form onSubmit={predictForm.handleSubmit((data) => predictMutation.mutate(data))} className="space-y-4">
                      <FormField control={predictForm.control} name="sqft" render={({ field }) => (
                        <FormItem>
                          <FormLabel>坪數 (Square Footage)</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="age" render={({ field }) => (
                        <FormItem>
                          <FormLabel>屋齡 (House Age)</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="rooms" render={({ field }) => (
                        <FormItem>
                          <FormLabel>房間數 (Rooms)</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <Button type="submit" className="w-full h-12 rounded-xl shadow-lg shadow-primary/20" disabled={predictMutation.isPending}>
                        {predictMutation.isPending ? "預測中..." : "開始預測 (Predict)"}
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
                      <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">預測連續數值 (價格)</p>
                      <div className="text-5xl sm:text-6xl font-bold text-primary tracking-tighter">
                        {formatCurrency(prediction)}
                      </div>
                      <div className="bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        深度神經網路模型已定義，預測成功
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
                      <p className="font-medium">請輸入參數以進行估價</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="train">
            <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl">
              <CardHeader>
                <CardTitle>訓練模擬 (Training Setup)</CardTitle>
                <CardDescription>調整數據生成參數與訓練輪數</CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...trainForm}>
                  <form onSubmit={trainForm.handleSubmit((data) => trainMutation.mutate(data))} className="space-y-8">
                    <FormField control={trainForm.control} name="samples" render={({ field }) => (
                      <FormItem>
                        <div className="flex justify-between items-center mb-2">
                          <FormLabel className="text-base">數據樣本數 (Samples)</FormLabel>
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
                          <FormLabel className="text-base">雜訊程度 (Noise/Pain)</FormLabel>
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
                          <FormLabel className="text-base">訓練輪數 (Epochs)</FormLabel>
                          <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider min={10} max={200} step={10} value={[field.value]} onValueChange={(v) => field.onChange(v[0])} />
                        </FormControl>
                      </FormItem>
                    )} />
                    <Button type="submit" variant="secondary" className="w-full h-12 rounded-xl" disabled={trainMutation.isPending}>
                      {trainMutation.isPending ? "模型訓練中 (Training...)" : "重新訓練模型 (Retrain Model)"}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics">
            <div className="grid lg:grid-cols-2 gap-8">
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle>損失函數收斂 (Loss Curve)</CardTitle>
                  <CardDescription>MSE 隨訓練輪數下降的情況</CardDescription>
                </CardHeader>
                <CardContent className="bg-white p-4">
                  <img 
                    key={lossCurve} 
                    src={lossCurve} 
                    alt="Loss Curve" 
                    className="w-full h-auto rounded-xl border border-border/20" 
                    onError={(e) => {
                      // Fallback if image doesn't exist yet
                      (e.target as HTMLImageElement).src = "https://placehold.co/600x400?text=Loss+Curve+Loading...";
                    }}
                  />
                </CardContent>
              </Card>

              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle>為什麼需要 ReLU 激活函數？</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    如果只使用線性運算（加法和乘法），無論堆疊多少層，最終結果仍等同於一個單層線性變換：
                  </p>
                  <div className="bg-muted p-4 rounded-xl font-mono text-xs">
                    f(x) = W2(W1·x + b1) + b2<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (W2·W1)·x + (W2·b1 + b2)<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= W'·x + b'
                  </div>
                  <p className="text-sm font-medium">
                    ReLU 引入非線性，讓網絡能夠學習複雜的非線性關係，例如房價與坪數之間可能存在的邊際效益遞減。
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="code">
            <div className="space-y-8">
              <Card className="bg-slate-950 text-slate-50 border-none rounded-3xl overflow-hidden shadow-2xl">
                <CardHeader className="border-b border-slate-800/50">
                  <CardTitle className="text-slate-50 flex items-center justify-between">
                    <span>模型摘要 (Model Summary)</span>
                    <span className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-400">HousePriceNN</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-6 space-y-4 font-mono text-sm text-blue-200">
                  <div className="space-y-1">
                    <div>(layer1): Linear(in_features=3, out_features=64)</div>
                    <div>(relu1): ReLU()</div>
                    <div>(layer2): Linear(in_features=64, out_features=32)</div>
                    <div>(relu2): ReLU()</div>
                    <div>(layer3): Linear(in_features=32, out_features=16)</div>
                    <div>(relu3): ReLU()</div>
                    <div>(output_layer): Linear(in_features=16, out_features=1)</div>
                  </div>
                  <div className="pt-4 border-t border-slate-800 flex justify-between text-slate-400">
                    <div>總參數數量: <span className="text-white">2,881</span></div>
                    <div>可訓練參數: <span className="text-white">2,881</span></div>
                  </div>
                </CardContent>
              </Card>

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


function RefreshCcw(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  )
}
