import { z } from 'zod';
import { predictionRequestSchema, trainRequestSchema } from './schema';

export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

export const api = {
  predict: {
    method: 'POST' as const,
    path: '/api/predict' as const,
    input: predictionRequestSchema,
    responses: {
      200: z.object({ predictedPrice: z.number() }),
      400: errorSchemas.validation,
      500: errorSchemas.internal,
    },
  },
  train: {
    method: 'POST' as const,
    path: '/api/train' as const,
    input: trainRequestSchema,
    responses: {
      200: z.object({ success: z.boolean(), message: z.string(), lossCurveUrl: z.string() }),
      400: errorSchemas.validation,
      500: errorSchemas.internal,
    },
  },
};

export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}
