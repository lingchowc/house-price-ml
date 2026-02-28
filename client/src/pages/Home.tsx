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
        return self.network(x)

# Optimizer: Adam (Adaptive Moment Estimation)
# Chosen because it computes adaptive learning rates for each parameter,
# combining the benefits of AdaGrad and RMSProp for faster convergence.
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() # Best for continuous regression`;

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
      setLossCurve(`${data.lossCurveUrl}?t=${Date.now()}`);
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
            className="text-4xl font-bold tracking-tight sm:text-5xl"
          >
            House Price Intelligence
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground text-lg"
          >
            Deep Neural Network Regression with PyTorch
          </motion.p>
        </header>

        <Tabs defaultValue="predict" className="w-full">
          <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 mb-8">
            <TabsTrigger value="predict" className="gap-2"><HomeIcon className="w-4 h-4" /> Predict</TabsTrigger>
            <TabsTrigger value="train" className="gap-2"><Settings2 className="w-4 h-4" /> Training Setup</TabsTrigger>
            <TabsTrigger value="analytics" className="gap-2"><Activity className="w-4 h-4" /> Performance</TabsTrigger>
            <TabsTrigger value="code" className="gap-2"><Code className="w-4 h-4" /> Architecture</TabsTrigger>
          </TabsList>

          <TabsContent value="predict">
            <div className="grid md:grid-cols-2 gap-8">
              <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
                <CardHeader>
                  <CardTitle>Estimation Parameters</CardTitle>
                  <CardDescription>Enter house features to generate a valuation</CardDescription>
                </CardHeader>
                <CardContent>
                  <Form {...predictForm}>
                    <form onSubmit={predictForm.handleSubmit((data) => predictMutation.mutate(data))} className="space-y-4">
                      <FormField control={predictForm.control} name="sqft" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Square Footage</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="age" render={({ field }) => (
                        <FormItem>
                          <FormLabel>House Age (Years)</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <FormField control={predictForm.control} name="rooms" render={({ field }) => (
                        <FormItem>
                          <FormLabel>Number of Rooms</FormLabel>
                          <FormControl><Input type="number" {...field} className="h-12 rounded-xl" /></FormControl>
                          <FormMessage />
                        </FormItem>
                      )} />
                      <Button type="submit" className="w-full h-12 rounded-xl shadow-lg shadow-primary/20" disabled={predictMutation.isPending}>
                        {predictMutation.isPending ? "Calculating..." : "Generate Estimate"}
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
                      <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Estimated Market Value</p>
                      <div className="text-5xl sm:text-6xl font-bold text-primary tracking-tighter">
                        {formatCurrency(prediction)}
                      </div>
                      <p className="text-xs text-muted-foreground italic">Based on current deep learning weights</p>
                    </motion.div>
                  ) : (
                    <motion.div 
                      key="placeholder"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="text-center text-muted-foreground"
                    >
                      <Brain className="w-16 h-16 mx-auto mb-4 opacity-10" />
                      <p className="font-medium">Enter parameters to start valuation</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="train">
            <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl">
              <CardHeader>
                <CardTitle>Retraining Simulation</CardTitle>
                <CardDescription>Adjust data generation parameters to retrain the neural network</CardDescription>
              </CardHeader>
              <CardContent>
                <Form {...trainForm}>
                  <form onSubmit={trainForm.handleSubmit((data) => trainMutation.mutate(data))} className="space-y-8">
                    <FormField control={trainForm.control} name="samples" render={({ field }) => (
                      <FormItem>
                        <div className="flex justify-between items-center mb-2">
                          <FormLabel className="text-base">Data Samples</FormLabel>
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
                          <FormLabel className="text-base">Noise Level (Uncertainty)</FormLabel>
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
                          <FormLabel className="text-base">Training Epochs</FormLabel>
                          <span className="text-sm font-mono font-bold text-primary bg-primary/10 px-2 py-1 rounded">{field.value}</span>
                        </div>
                        <FormControl>
                          <Slider min={10} max={200} step={10} value={[field.value]} onValueChange={(v) => field.onChange(v[0])} />
                        </FormControl>
                      </FormItem>
                    )} />
                    <Button type="submit" variant="secondary" className="w-full h-12 rounded-xl" disabled={trainMutation.isPending}>
                      {trainMutation.isPending ? (
                        <span className="flex items-center gap-2"><RefreshCcw className="w-4 h-4 animate-spin" /> Retraining Weights...</span>
                      ) : "Re-Initialize & Train Model"}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics">
            <Card className="border-border/50 shadow-xl shadow-black/5 rounded-3xl overflow-hidden">
              <CardHeader>
                <CardTitle>Loss Convergence</CardTitle>
                <CardDescription>Visualization of Mean Squared Error (MSE) over training iterations</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col items-center bg-white p-6">
                <img key={lossCurve} src={lossCurve} alt="Loss Curve" className="max-w-full h-auto rounded-xl border border-border/20 shadow-sm" />
                <div className="mt-6 p-4 bg-muted/30 rounded-xl text-sm text-muted-foreground max-w-2xl text-center italic">
                  "The loss curve represents the model's 'pain' or error. As it decreases, the model is successfully learning the underlying patterns of the property market."
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="code">
            <Card className="bg-slate-950 text-slate-50 border-none rounded-3xl overflow-hidden shadow-2xl">
              <CardHeader className="border-b border-slate-800/50">
                <CardTitle className="text-slate-50 flex items-center gap-2 font-mono"><Code className="w-5 h-5 text-blue-400" /> model.py</CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <pre className="text-sm font-mono overflow-x-auto p-6 bg-slate-950/50 leading-relaxed">
                  <code className="text-blue-200">{CODE_SNIPPET}</code>
                </pre>
              </CardContent>
            </Card>
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
