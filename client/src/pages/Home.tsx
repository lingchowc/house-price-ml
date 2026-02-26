import React, { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { predictionRequestSchema, type PredictionRequest } from "@shared/schema";
import { usePredict } from "@/hooks/use-predict";
import { motion, AnimatePresence } from "framer-motion";
import { Building2, Calculator, AlertCircle, RefreshCcw, SquareAsterisk, DoorClosed, CalendarDays } from "lucide-react";

export default function Home() {
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  
  const { mutate: predictPrice, isPending, error, reset } = usePredict();

  const form = useForm<PredictionRequest>({
    resolver: zodResolver(predictionRequestSchema),
    defaultValues: {
      sqft: 2000,
      age: 10,
      rooms: 3,
    },
  });

  // Watch values to clear result if user starts typing again
  useEffect(() => {
    const subscription = form.watch(() => {
      if (predictedPrice !== null) {
        setPredictedPrice(null);
        reset();
      }
    });
    return () => subscription.unsubscribe();
  }, [form, predictedPrice, reset]);

  const onSubmit = (data: PredictionRequest) => {
    setPredictedPrice(null);
    predictPrice(data, {
      onSuccess: (response) => {
        setPredictedPrice(response.predictedPrice);
      },
    });
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="min-h-screen relative flex items-center justify-center p-4 sm:p-6 overflow-hidden bg-background">
      {/* Subtle background grid and gradients */}
      <div className="absolute inset-0 bg-grid-pattern opacity-[0.4] mix-blend-multiply pointer-events-none" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-[500px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="w-full max-w-lg relative z-10"
      >
        <div className="bg-card/80 backdrop-blur-2xl border border-border/50 shadow-2xl shadow-black/5 rounded-3xl overflow-hidden flex flex-col">
          
          {/* Header */}
          <div className="p-8 sm:p-10 pb-6 text-center border-b border-border/40 bg-gradient-to-b from-muted/30 to-transparent">
            <div className="w-14 h-14 bg-primary/5 text-primary rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-inner">
              <Building2 className="w-7 h-7" />
            </div>
            <h1 className="text-3xl font-display font-semibold tracking-tight text-foreground mb-2">
              Property Valuation
            </h1>
            <p className="text-muted-foreground text-sm font-medium">
              Enter the property details below to generate an AI-driven market estimate.
            </p>
          </div>

          {/* Form Content */}
          <div className="p-8 sm:p-10">
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              
              <div className="space-y-5">
                {/* Square Footage Input */}
                <div className="space-y-2 relative group">
                  <label htmlFor="sqft" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground ml-1">
                    Square Footage
                  </label>
                  <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors">
                      <SquareAsterisk className="w-5 h-5" />
                    </div>
                    <input
                      id="sqft"
                      type="number"
                      {...form.register("sqft")}
                      className="w-full h-14 pl-12 pr-4 bg-background border border-border rounded-xl text-foreground font-medium placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-sm"
                      placeholder="e.g. 2500"
                    />
                  </div>
                  {form.formState.errors.sqft && (
                    <p className="text-sm text-destructive font-medium mt-1 flex items-center gap-1.5">
                      <AlertCircle className="w-4 h-4" />
                      {form.formState.errors.sqft.message}
                    </p>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-5">
                  {/* Age Input */}
                  <div className="space-y-2 relative group">
                    <label htmlFor="age" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground ml-1">
                      Age (Years)
                    </label>
                    <div className="relative">
                      <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors">
                        <CalendarDays className="w-5 h-5" />
                      </div>
                      <input
                        id="age"
                        type="number"
                        {...form.register("age")}
                        className="w-full h-14 pl-12 pr-4 bg-background border border-border rounded-xl text-foreground font-medium placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-sm"
                        placeholder="e.g. 15"
                      />
                    </div>
                    {form.formState.errors.age && (
                      <p className="text-sm text-destructive font-medium mt-1 flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4" />
                        {form.formState.errors.age.message}
                      </p>
                    )}
                  </div>

                  {/* Rooms Input */}
                  <div className="space-y-2 relative group">
                    <label htmlFor="rooms" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground ml-1">
                      Total Rooms
                    </label>
                    <div className="relative">
                      <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors">
                        <DoorClosed className="w-5 h-5" />
                      </div>
                      <input
                        id="rooms"
                        type="number"
                        {...form.register("rooms")}
                        className="w-full h-14 pl-12 pr-4 bg-background border border-border rounded-xl text-foreground font-medium placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all shadow-sm"
                        placeholder="e.g. 4"
                      />
                    </div>
                    {form.formState.errors.rooms && (
                      <p className="text-sm text-destructive font-medium mt-1 flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4" />
                        {form.formState.errors.rooms.message}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {error && (
                <div className="p-4 rounded-xl bg-destructive/10 text-destructive text-sm font-medium flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <p>{error.message}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={isPending}
                className="relative w-full h-14 bg-primary text-primary-foreground font-semibold rounded-xl shadow-lg shadow-primary/20 hover:shadow-xl hover:shadow-primary/30 hover:-translate-y-0.5 active:translate-y-0 active:shadow-md transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed disabled:transform-none overflow-hidden group"
              >
                <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
                <span className="relative flex items-center justify-center gap-2">
                  {isPending ? (
                    <>
                      <RefreshCcw className="w-5 h-5 animate-spin" />
                      Analyzing Data...
                    </>
                  ) : (
                    <>
                      <Calculator className="w-5 h-5" />
                      Calculate Estimate
                    </>
                  )}
                </span>
              </button>
            </form>
          </div>

          {/* Result Section */}
          <AnimatePresence>
            {predictedPrice !== null && (
              <motion.div
                initial={{ opacity: 0, height: 0, scale: 0.95 }}
                animate={{ opacity: 1, height: "auto", scale: 1 }}
                exit={{ opacity: 0, height: 0, scale: 0.95 }}
                transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                className="border-t border-border/50 bg-primary/5 overflow-hidden"
              >
                <div className="p-8 sm:p-10 text-center">
                  <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                    Estimated Market Value
                  </p>
                  <motion.div 
                    initial={{ y: 10, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.2, duration: 0.4 }}
                    className="text-5xl sm:text-6xl font-display font-bold tracking-tighter text-foreground"
                  >
                    {formatCurrency(predictedPrice)}
                  </motion.div>
                  <p className="mt-4 text-sm text-muted-foreground max-w-xs mx-auto">
                    This is an algorithmic estimate based on provided parameters and synthetic market data patterns.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
        
        <p className="text-center text-xs text-muted-foreground/60 mt-6 font-medium">
          Powered by Synthetic Intelligence Models
        </p>
      </motion.div>
    </div>
  );
}
