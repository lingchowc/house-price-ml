import type { Express } from "express";
import type { Server } from "http";
import { api } from "@shared/routes";
import { z } from "zod";
import { exec } from "child_process";
import util from "util";
import path from "path";

const execAsync = util.promisify(exec);
const venvPython = path.join(process.cwd(), ".venv", "bin", "python");

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  app.post(api.predict.path, async (req, res) => {
    try {
      const input = api.predict.input.parse(req.body);
      const { stdout } = await execAsync(`${venvPython} predict.py ${input.sqft} ${input.age} ${input.rooms}`);
      const result = JSON.parse(stdout);
      if (result.error) return res.status(500).json({ message: result.error });
      res.status(200).json(result);
    } catch (err) {
      res.status(500).json({ message: "Prediction failed" });
    }
  });

  app.post(api.train.path, async (req, res) => {
    try {
      const input = api.train.input.parse(req.body);
      await execAsync(`${venvPython} train.py ${input.samples} ${input.noise} ${input.epochs}`);
      res.status(200).json({ 
        success: true, 
        message: "Model retrained successfully",
        lossCurveUrl: `/loss_curve.png?t=${Date.now()}`
      });
    } catch (err) {
      res.status(500).json({ message: "Training failed" });
    }
  });

  return httpServer;
}
