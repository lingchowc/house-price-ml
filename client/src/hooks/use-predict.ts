import { useMutation } from "@tanstack/react-query";
import { api } from "@shared/routes";
import type { PredictionRequest } from "@shared/schema";
import { z } from "zod";

export function usePredict() {
  return useMutation({
    mutationFn: async (data: PredictionRequest) => {
      // Coerce numeric inputs for safety before sending
      const coercedData = {
        sqft: Number(data.sqft),
        age: Number(data.age),
        rooms: Number(data.rooms),
      };
      
      const validated = api.predict.input.parse(coercedData);
      
      const res = await fetch(api.predict.path, {
        method: api.predict.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
      });

      if (!res.ok) {
        let errorMessage = "Failed to predict price";
        try {
          const errorData = await res.json();
          // Try to parse as validation error first
          const parsedError = api.predict.responses[400].safeParse(errorData);
          if (parsedError.success) {
            errorMessage = parsedError.data.message;
          } else {
            errorMessage = errorData.message || errorMessage;
          }
        } catch {
          // Fallback to generic message if not JSON
        }
        throw new Error(errorMessage);
      }

      const responseData = await res.json();
      return api.predict.responses[200].parse(responseData);
    },
  });
}
