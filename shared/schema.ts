import { z } from "zod";

export const predictionRequestSchema = z.object({
  sqft: z.coerce.number().min(100, "Square footage must be at least 100").max(20000, "Square footage must be less than 20000"),
  age: z.coerce.number().min(0, "Age must be at least 0").max(200, "Age must be less than 200"),
  rooms: z.coerce.number().min(1, "Must have at least 1 room").max(50, "Must have less than 50 rooms"),
});

export type PredictionRequest = z.infer<typeof predictionRequestSchema>;

export type PredictionResponse = {
  predictedPrice: number;
};
