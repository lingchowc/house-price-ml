import type { Express } from "express";
import type { Server } from "http";
import { api } from "@shared/routes";
import { z } from "zod";
import { exec } from "child_process";
import util from "util";

const execAsync = util.promisify(exec);

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  app.post(api.predict.path, async (req, res) => {
    try {
      const input = api.predict.input.parse(req.body);
      
      // Execute the python script with the input parameters
      const { stdout, stderr } = await execAsync(`python predict.py ${input.sqft} ${input.age} ${input.rooms}`);
      
      try {
        const result = JSON.parse(stdout);
        if (result.error) {
          return res.status(500).json({ message: result.error });
        }
        res.status(200).json(result);
      } catch (parseError) {
        console.error("Failed to parse python output:", stdout, stderr);
        res.status(500).json({ message: "Failed to parse model output" });
      }
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0].message,
          field: err.errors[0].path.join('.'),
        });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  return httpServer;
}
