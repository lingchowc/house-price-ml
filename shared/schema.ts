import { z } from "zod";

export const predictionRequestSchema = z.object({
  sqft: z.coerce.number().min(100),
  age: z.coerce.number().min(0),
  rooms: z.coerce.number().min(1),
});

export const trainRequestSchema = z.object({
  samples: z.coerce.number().min(100).max(10000).default(5000),
  noise: z.coerce.number().min(0).max(1).default(0.1),
  epochs: z.coerce.number().min(10).max(200).default(50),
});

export type PredictionRequest = z.infer<typeof predictionRequestSchema>;
export type TrainRequest = z.infer<typeof trainRequestSchema>;

export type PredictionResponse = {
  predictedPrice: number;
};
