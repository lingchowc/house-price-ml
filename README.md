# House Price ML Predictor

Interactive house price prediction app with:
- React + TypeScript frontend
- Express API backend
- PyTorch model inference/training scripts

## Live Demo
- GitHub Pages: https://lingchowc.github.io/house-price-ml/
- Repository: https://github.com/lingchowc/house-price-ml

## Features
- Predict house prices from:
  - Square footage
  - House age
  - Number of rooms
- Training controls (samples, noise, epochs)
- Loss-curve visualization
- English / Chinese UI toggle

## Project Structure
- `client/`: frontend app (Vite + React)
- `server/`: Express server and API routes
- `train.py`: model training script
- `predict.py`: model inference script
- `house_model.pth`: trained model weights

## Local Development

### 1. Install Node dependencies
```bash
npm install
```

### 2. Install Python dependencies
Use your preferred Python environment, then install:
```bash
pip install torch matplotlib
```

### 3. Run the app
```bash
npm run dev
```

Open: `http://localhost:3000`

## Production Build
```bash
npm run build
npm start
```

## GitHub Pages Deployment Notes
- Deployment is automated by `.github/workflows/deploy-pages.yml`.
- The Pages build runs in static mode (`VITE_STATIC_MODEL=true`).
- In static mode:
  - Price prediction works in-browser.
  - Retraining requires the backend server (not available on GitHub Pages).

## API Endpoints (Backend Mode)
- `POST /api/predict`
- `POST /api/train`

